"""
Prediction tuning and evaluation utilities.

This module provides:
1) Lightweight post-processing tweaks for predicted moves
2) Offline A/B evaluation against realized prices
3) Report generation for ongoing drift monitoring
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class TweakConfig:
    enabled: bool = True
    neutral_band_pct: float = 1.0
    min_confidence_for_direction: float = 0.82
    use_williams_conflict_brake: bool = True
    max_upside_cap_pct: float = 6.0
    max_downside_cap_pct: float = -6.0
    bias_correction_pct: float = -1.5


DEFAULT_TWEAK_CONFIG = TweakConfig()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def direction_from_change_pct(change_pct: float, neutral_band_pct: float = 0.0) -> str:
    if abs(change_pct) <= neutral_band_pct:
        return "NEUTRAL"
    return "BULLISH" if change_pct > 0 else "BEARISH"


def _derive_base_price(predictions: List[Dict]) -> Optional[float]:
    if not predictions:
        return None
    first = predictions[0]
    direct_current = first.get("current_price")
    if direct_current is not None:
        try:
            current_val = float(direct_current)
            if current_val > 0:
                return current_val
        except (TypeError, ValueError):
            pass
    p = float(first.get("predicted_price", 0) or 0)
    u = float(first.get("upside_potential", 0) or 0)
    denom = 1 + (u / 100.0)
    if p <= 0 or abs(denom) < 1e-8:
        return None
    return p / denom


def get_live_tweak_config() -> TweakConfig:
    """
    Load tweak config from environment with safe defaults.
    Set PREDICTION_TWEAKS_ENABLED=0 for immediate rollback.
    """
    # Default OFF to preserve raw model behavior unless explicitly enabled.
    enabled = os.getenv("PREDICTION_TWEAKS_ENABLED", "0").strip() not in {"0", "false", "False"}
    return TweakConfig(
        enabled=enabled,
        neutral_band_pct=float(os.getenv("PRED_TWEAK_NEUTRAL_BAND_PCT", str(DEFAULT_TWEAK_CONFIG.neutral_band_pct))),
        min_confidence_for_direction=float(
            os.getenv("PRED_TWEAK_MIN_CONFIDENCE", str(DEFAULT_TWEAK_CONFIG.min_confidence_for_direction))
        ),
        use_williams_conflict_brake=os.getenv("PRED_TWEAK_WILLIAMS_BRAKE", "1").strip() not in {"0", "false", "False"},
        max_upside_cap_pct=float(os.getenv("PRED_TWEAK_MAX_UPSIDE_CAP_PCT", str(DEFAULT_TWEAK_CONFIG.max_upside_cap_pct))),
        max_downside_cap_pct=float(
            os.getenv("PRED_TWEAK_MAX_DOWNSIDE_CAP_PCT", str(DEFAULT_TWEAK_CONFIG.max_downside_cap_pct))
        ),
        bias_correction_pct=float(os.getenv("PRED_TWEAK_BIAS_CORRECTION_PCT", str(DEFAULT_TWEAK_CONFIG.bias_correction_pct))),
    )


PREDICTION_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "prediction_logs" / "prediction_log.json"


def compute_per_symbol_bias(symbol: str, lookback_n: int = 20) -> float:
    """
    Compute adaptive bias correction from recent evaluated predictions.
    Returns a correction value to ADD to the raw prediction (positive = nudge upward).
    """
    if not PREDICTION_LOG_PATH.exists():
        return 0.0
    try:
        with open(PREDICTION_LOG_PATH, "r") as f:
            log = json.load(f)
    except Exception:
        return 0.0

    evaluated = [
        p for p in log
        if p.get("symbol") == symbol
        and p.get("evaluated")
        and p.get("predicted_change_pct") is not None
        and p.get("actual_change_pct") is not None
    ]
    if len(evaluated) < 5:
        return 0.0

    # Take last N evaluated predictions
    evaluated.sort(key=lambda x: x.get("prediction_date", ""))
    recent = evaluated[-lookback_n:]

    # signed_bias > 0 means model has been predicting too high (bullish bias)
    # signed_bias < 0 means model has been predicting too low (bearish bias)
    signed_bias = sum(
        p["predicted_change_pct"] - p["actual_change_pct"] for p in recent
    ) / len(recent)

    # Partial correction (50%) to avoid overcorrecting
    return round(-0.5 * signed_bias, 4)


def apply_prediction_tweaks(predictions: List[Dict], config: TweakConfig) -> List[Dict]:
    """
    Apply tiny, reversible post-processing tweaks to prediction paths.
    Keeps original values in raw_* fields for auditability.
    """
    if not predictions:
        return predictions
    if not config.enabled:
        return predictions

    out = deepcopy(predictions)
    base_price = _derive_base_price(out)

    # Use adaptive per-symbol bias if available, else fall back to static config
    symbol = None
    for p in out:
        symbol = p.get("symbol") or p.get("ticker")
        if symbol:
            break
    adaptive_bias = compute_per_symbol_bias(symbol) if symbol else 0.0
    bias_correction = adaptive_bias if adaptive_bias != 0.0 else config.bias_correction_pct

    for pred in out:
        raw_up = float(pred.get("upside_potential", 0) or 0)
        conf = float(pred.get("confidence", 0) or 0)
        williams = str(pred.get("williams_signal", "") or "").upper()

        adjusted = raw_up + bias_correction
        pred["adaptive_bias_correction"] = adaptive_bias
        adjusted = _clamp(adjusted, config.max_downside_cap_pct, config.max_upside_cap_pct)

        if conf < config.min_confidence_for_direction:
            adjusted = 0.0

        if config.use_williams_conflict_brake and williams in {"UP", "DOWN"}:
            williams_conf = float(pred.get("williams_confidence", 0) or 0)
            pred_dir = "UP" if adjusted > 0 else ("DOWN" if adjusted < 0 else "NEUTRAL")
            if (pred_dir == "UP" and williams == "DOWN") or (pred_dir == "DOWN" and williams == "UP"):
                # Only dampen (not zero) when Williams confidence is meaningful
                if williams_conf > 0.60:
                    adjusted *= 0.5
                # Low-confidence Williams signal: leave prediction unchanged

        if abs(adjusted) < config.neutral_band_pct:
            adjusted = 0.0

        pred["raw_upside_potential"] = raw_up
        pred["upside_potential"] = round(adjusted, 2)
        pred["tweaked_direction"] = direction_from_change_pct(adjusted, neutral_band_pct=0.0)

        if base_price and base_price > 0:
            raw_price = float(pred.get("predicted_price", 0) or 0)
            pred["raw_predicted_price"] = raw_price
            pred["predicted_price"] = round(max(0.01, base_price * (1 + adjusted / 100.0)), 2)

    return out


def _fetch_actual_on_or_after(symbol: str, date_str: str, cache: Dict[Tuple[str, int, int], List[Dict]]) -> Tuple[Optional[float], Optional[str]]:
    """
    Fetch actual close on evaluation date (or next trading day) via PSX endpoint.
    """
    # Fast path: use locally cached historical file first.
    local_key = (symbol, 0, 0)
    if local_key not in cache:
        local_path = Path(__file__).resolve().parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
        local_rows: List[Dict] = []
        if local_path.exists():
            try:
                with open(local_path, "r") as f:
                    raw = json.load(f)
                # Ensure Date/Close shape and sort
                local_rows = sorted(
                    [{"Date": r.get("Date"), "Close": float(r.get("Close"))} for r in raw if r.get("Date") and r.get("Close") is not None],
                    key=lambda r: r["Date"],
                )
            except Exception:
                local_rows = []
        cache[local_key] = local_rows

    local_rows = cache.get(local_key, [])
    if local_rows:
        local_hit = [r for r in local_rows if r["Date"] >= date_str]
        if local_hit:
            chosen = local_hit[0]
            return float(chosen["Close"]), chosen["Date"]

    # Fallback path: fetch from PSX API.
    from backend.stock_analyzer_fixed import fetch_month_data, parse_html_table

    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    months_to_try = [(target.year, target.month)]
    next_month = (target.replace(day=1) + timedelta(days=32)).replace(day=1)
    months_to_try.append((next_month.year, next_month.month))

    rows: List[Dict] = []
    for year, month in months_to_try:
        key = (symbol, year, month)
        if key not in cache:
            html = fetch_month_data(symbol, month, year)
            parsed = parse_html_table(html or "") if html else []
            cache[key] = sorted(parsed, key=lambda r: r["Date"])
        rows.extend(cache[key])

    if not rows:
        return None, None
    rows = sorted(rows, key=lambda r: r["Date"])
    hit = [r for r in rows if r["Date"] >= date_str]
    chosen = hit[0] if hit else rows[-1]
    return float(chosen["Close"]), chosen["Date"]


def evaluate_prediction_log(log_path: str, config: Optional[TweakConfig] = None, latest_per_symbol: bool = True) -> Dict:
    """
    Evaluate saved predictions against realized prices.
    """
    with open(log_path, "r") as f:
        entries = json.load(f)

    if latest_per_symbol:
        latest = {}
        for p in entries:
            sym = p["symbol"]
            t = datetime.fromisoformat(p["prediction_date"])
            if sym not in latest or t > latest[sym][0]:
                latest[sym] = (t, p)
        eval_entries = [v[1] for v in latest.values()]
    else:
        eval_entries = list(entries)

    if config is None:
        config = TweakConfig(enabled=False)

    cache: Dict[Tuple[str, int, int], List[Dict]] = {}
    rows = []
    for p in sorted(eval_entries, key=lambda x: x["symbol"]):
        sym = p["symbol"]
        curr = float(p["current_price"])
        raw_change = float(p.get("predicted_change_pct", 0) or 0)
        williams = p.get("williams_signal")
        conf = float(p.get("confidence", 0) or 0)

        synthetic_pred = [{"predicted_price": float(p["predicted_price"]), "upside_potential": raw_change, "confidence": conf, "williams_signal": williams}]
        tweaked = apply_prediction_tweaks(synthetic_pred, config)[0] if config.enabled else synthetic_pred[0]
        pred_price = float(tweaked.get("predicted_price", p["predicted_price"]))
        pred_change = float(tweaked.get("upside_potential", raw_change))
        pred_dir = direction_from_change_pct(pred_change, neutral_band_pct=0.0)

        actual_price, actual_date = _fetch_actual_on_or_after(sym, p["evaluation_date"], cache)
        if actual_price is None:
            continue

        actual_change = (actual_price - curr) / curr * 100.0
        actual_dir = direction_from_change_pct(actual_change, neutral_band_pct=0.0)
        # NEUTRAL counts as correct only if actual move was small (within neutral band)
        if pred_dir == "NEUTRAL":
            dir_correct = abs(actual_change) <= config.neutral_band_pct
        else:
            dir_correct = pred_dir == actual_dir
        err_pct = (pred_price - actual_price) / actual_price * 100.0

        rows.append(
            {
                "symbol": sym,
                "evaluation_date": p["evaluation_date"],
                "actual_date_used": actual_date,
                "predicted_price": round(pred_price, 2),
                "actual_price": round(actual_price, 2),
                "predicted_change_pct": round(pred_change, 2),
                "actual_change_pct": round(actual_change, 2),
                "predicted_direction": pred_dir,
                "actual_direction": actual_dir,
                "direction_correct": dir_correct,
                "error_pct": round(err_pct, 4),
                "abs_error_pct": round(abs(err_pct), 4),
                "confidence": conf,
                "williams_signal": williams,
            }
        )

    total = len(rows)
    if total == 0:
        return {
            "config": asdict(config),
            "n": 0,
            "direction_accuracy_pct": None,
            "mae_pct": None,
            "signed_bias_pct": None,
            "rows": [],
        }

    direction_accuracy = 100.0 * sum(1 for r in rows if r["direction_correct"]) / total
    mae = sum(r["abs_error_pct"] for r in rows) / total
    signed_bias = sum(r["error_pct"] for r in rows) / total

    # Segment views
    high_conf = [r for r in rows if r["confidence"] >= 0.85]
    williams_agree = [
        r
        for r in rows
        if (r["williams_signal"] == "UP" and r["predicted_direction"] == "BULLISH")
        or (r["williams_signal"] == "DOWN" and r["predicted_direction"] == "BEARISH")
    ]
    williams_conflict = [
        r
        for r in rows
        if (r["williams_signal"] == "UP" and r["predicted_direction"] == "BEARISH")
        or (r["williams_signal"] == "DOWN" and r["predicted_direction"] == "BULLISH")
    ]

    def _acc(subset: List[Dict]) -> Optional[float]:
        if not subset:
            return None
        return round(100.0 * sum(1 for x in subset if x["direction_correct"]) / len(subset), 2)

    return {
        "config": asdict(config),
        "n": total,
        "direction_accuracy_pct": round(direction_accuracy, 2),
        "mae_pct": round(mae, 4),
        "signed_bias_pct": round(signed_bias, 4),
        "high_confidence_accuracy_pct": _acc(high_conf),
        "williams_agree_accuracy_pct": _acc(williams_agree),
        "williams_conflict_accuracy_pct": _acc(williams_conflict),
        "rows": rows,
    }


def run_ab(log_path: str) -> Dict:
    """
    Run baseline vs small tweak candidates and select the best config.
    """
    baseline_cfg = TweakConfig(enabled=False)
    baseline = evaluate_prediction_log(log_path, baseline_cfg, latest_per_symbol=True)

    candidates = [
        TweakConfig(enabled=True, neutral_band_pct=0.8, min_confidence_for_direction=0.80, bias_correction_pct=-1.0),
        TweakConfig(enabled=True, neutral_band_pct=1.0, min_confidence_for_direction=0.82, bias_correction_pct=-1.5),
        TweakConfig(enabled=True, neutral_band_pct=1.2, min_confidence_for_direction=0.85, bias_correction_pct=-2.0),
    ]

    trials = [{"name": "baseline", "metrics": baseline}]
    for i, cfg in enumerate(candidates, start=1):
        metrics = evaluate_prediction_log(log_path, cfg, latest_per_symbol=True)
        trials.append({"name": f"candidate_{i}", "metrics": metrics})

    # KPI gates
    base_acc = baseline["direction_accuracy_pct"] or 0.0
    base_mae = baseline["mae_pct"] or 1e9
    base_bias_mag = abs(baseline["signed_bias_pct"] or 1e9)

    gated = []
    for t in trials[1:]:
        m = t["metrics"]
        if m["direction_accuracy_pct"] is None or m["mae_pct"] is None or m["signed_bias_pct"] is None:
            continue
        acc_gain = m["direction_accuracy_pct"] - base_acc
        mae_impr = (base_mae - m["mae_pct"]) / base_mae if base_mae > 0 else 0
        bias_better = abs(m["signed_bias_pct"]) < base_bias_mag
        if acc_gain >= 5.0 and mae_impr >= 0.10 and bias_better:
            gated.append((t, acc_gain, mae_impr))

    if gated:
        best = sorted(gated, key=lambda x: (x[1], x[2]), reverse=True)[0][0]
    else:
        # fallback score if strict gates not met
        def score(t):
            m = t["metrics"]
            if m["direction_accuracy_pct"] is None or m["mae_pct"] is None or m["signed_bias_pct"] is None:
                return -1e9
            return (m["direction_accuracy_pct"] - base_acc) * 2.0 + (base_mae - m["mae_pct"]) - abs(m["signed_bias_pct"])

        best = sorted(trials[1:], key=score, reverse=True)[0] if len(trials) > 1 else trials[0]

    return {
        "generated_at": datetime.now().isoformat(),
        "baseline": baseline,
        "trials": trials,
        "selected": best,
    }


def write_ab_report(log_path: str, output_path: Optional[str] = None) -> str:
    result = run_ab(log_path)
    if output_path is None:
        output_path = str(Path(log_path).parent / "prediction_tuning_report.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return output_path


def drift_snapshot(log_path: str) -> Dict:
    """
    Lightweight periodic monitoring snapshot for drift checks.
    """
    baseline = evaluate_prediction_log(log_path, TweakConfig(enabled=False), latest_per_symbol=False)
    live_cfg = get_live_tweak_config()
    tuned = evaluate_prediction_log(log_path, live_cfg, latest_per_symbol=False)
    return {
        "generated_at": datetime.now().isoformat(),
        "live_config": asdict(live_cfg),
        "baseline": {k: baseline[k] for k in ["n", "direction_accuracy_pct", "mae_pct", "signed_bias_pct"]},
        "tuned": {k: tuned[k] for k in ["n", "direction_accuracy_pct", "mae_pct", "signed_bias_pct"]},
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    log_file = root / "data" / "prediction_logs" / "prediction_log.json"
    if not log_file.exists():
        raise SystemExit(f"Prediction log not found: {log_file}")
    report = write_ab_report(str(log_file))
    print(f"Saved A/B report to: {report}")
