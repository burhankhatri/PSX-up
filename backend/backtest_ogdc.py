#!/usr/bin/env python3
"""
OGDC 21-day backtest: BASELINE predictions (already in cache) vs
ENRICHED predictions (baseline + worldmonitor overlay) vs ACTUALS.

The baseline predictions in `data/OGDC_research_predictions_2026.json`
were generated 2026-04-01 for days 1..21. Today is 2026-04-24 and we have
real OGDC closes for trading days 1..17 of that horizon. We use that
window to score:
    1. baseline error vs actuals (the existing model)
    2. enriched error vs actuals (existing model + worldmonitor overlay)

The overlay's signal scores are computed POINT-IN-TIME for each prediction
day — i.e. the overlay applied for prediction day=k uses VIX/Asian/GDELT/
quake data only as it would have been known on the eve of day=k.

This script does NOT touch the WebSocket pipeline or retrain the model.
It is a pure post-process benchmark of the overlay's value-add.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.worldmonitor_signals import (
    fetch_vix, fetch_extended_asian_indices,
    fetch_gdelt_topic_timeline, GDELT_TOPIC_PACKS,
    fetch_pakistan_region_quakes,
    fetch_usd_pkr, fetch_crude_prices,
)
from backend.external_features import _to_naive_datetime
from backend.markov_regime import compute_markov_regime_signal
from backend.worldmonitor_overlay import (
    collapse_snapshot, apply_worldmonitor_overlay, DEFAULT_WEIGHTS,
)


SYMBOL = "OGDC"  # default; override via CLI: `python backend/backtest_ogdc.py LUCK`


def _paths_for(symbol: str):
    return (
        ROOT / "data" / f"{symbol}_research_predictions_2026.json",
        ROOT / "data" / f"{symbol}_historical_with_indicators.json",
    )


PRED_FILE, HIST_FILE = _paths_for(SYMBOL)


def load_predictions() -> Tuple[List[Dict], datetime, float]:
    """Load the cached prediction set + the issue date + the current_price baseline."""
    payload = json.loads(PRED_FILE.read_text())
    daily = payload["daily_predictions"]
    issue_date = pd.to_datetime(payload["generated_at"]).normalize()
    # Find the current_price the model used (closest historical to issue_date)
    hist = pd.DataFrame(json.loads(HIST_FILE.read_text()))
    hist["Date"] = _to_naive_datetime(hist["Date"]).dt.normalize()
    prior = hist[hist["Date"] <= issue_date]
    current_price = float(prior.iloc[-1]["Close"]) if len(prior) else float(daily[0]["predicted_price"])
    return daily, issue_date, current_price


def load_actuals(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Actual OGDC closes between start_date and end_date inclusive."""
    hist = pd.DataFrame(json.loads(HIST_FILE.read_text()))
    hist["Date"] = _to_naive_datetime(hist["Date"]).dt.normalize()
    mask = (hist["Date"] >= start_date) & (hist["Date"] <= end_date)
    return hist.loc[mask, ["Date", "Close"]].reset_index(drop=True)


def load_history_for_markov() -> pd.DataFrame:
    return pd.DataFrame(json.loads(HIST_FILE.read_text())).assign(
        Date=lambda d: _to_naive_datetime(d["Date"]).dt.normalize()
    )


def synthetic_baseline_geo_adjustments(daily_preds: List[Dict],
                                         current_price: float) -> List[Dict]:
    """Reverse-engineer the per-day adjustment from cached predictions.

    The cached `daily_predictions` were post-geo-overlay already in production.
    For backtest baseline = "as-is", we treat each predicted price as the
    baseline — and the adjustment block becomes the IDENTITY (delta 0). The
    overlay then strictly ADDS its own delta on top.

    This is the right baseline for an A/B comparison: BASELINE = cached
    predictions; ENRICHED = cached + overlay delta.
    """
    out = []
    for p in daily_preds:
        out.append({
            "day": int(p["day"]),
            "raw_adjustment": 0.0,
            "deterministic_adjustment": 0.0,
            "ai_trajectory_adjustment": 0.0,
            "blend_weight": 0.0,
            "capped_adjustment": 0.0,
            "percentage": 0.0,
            "event_impacts": [],
        })
    return out


def fetch_signal_history() -> Dict[str, pd.DataFrame]:
    """One-shot pull of all signal histories. Cached on disk under data/external_cache/."""
    return {
        "vix": fetch_vix(period="1y"),
        "asian": fetch_extended_asian_indices(period="1y"),
        "quakes": fetch_pakistan_region_quakes(min_magnitude=5.0, lookback_days=120),
        "gdelt_pk_conflict": fetch_gdelt_topic_timeline(
            "pk_conflict", GDELT_TOPIC_PACKS["pk_conflict"], timespan="60d"),
        "gdelt_regional_war": fetch_gdelt_topic_timeline(
            "regional_war", GDELT_TOPIC_PACKS["regional_war"], timespan="60d"),
        "pkr": fetch_usd_pkr(period="1y"),
        "crude": fetch_crude_prices(period="6mo"),
    }


def asof_row(df: pd.DataFrame, date_col: str, asof: pd.Timestamp) -> Optional[pd.Series]:
    """Last row in df with date_col <= asof. None if df is empty or no match."""
    if df is None or df.empty:
        return None
    d = df.copy()
    d[date_col] = _to_naive_datetime(d[date_col]).dt.normalize()
    mask = d[date_col] <= asof
    if not mask.any():
        return None
    return d.loc[mask].iloc[-1]


def asof_series(df: pd.DataFrame, date_col: str, value_col: str,
                asof: pd.Timestamp, lookback: int = 30) -> pd.Series:
    """Return the time-series of `value_col` ending at `asof` (last `lookback` rows)."""
    if df is None or df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)
    d = df.copy()
    d[date_col] = _to_naive_datetime(d[date_col]).dt.normalize()
    sub = d[d[date_col] <= asof].tail(lookback)
    return pd.Series(sub[value_col].values)


@dataclass
class DayResult:
    day: int
    date: str
    actual_close: Optional[float]
    baseline_pred: float
    enriched_pred: float
    baseline_err_pct: Optional[float]
    enriched_err_pct: Optional[float]
    actual_change_pct: Optional[float]
    baseline_pred_change_pct: float
    enriched_pred_change_pct: float
    baseline_dir_correct: Optional[bool]
    enriched_dir_correct: Optional[bool]
    overlay_delta_pct: float
    snapshot: Dict[str, float]


def run_backtest(weights: Optional[Dict[str, float]] = None) -> Dict:
    daily, issue_date, current_price = load_predictions()
    actuals = load_actuals(issue_date, issue_date + timedelta(days=45))
    actuals_by_date = {row["Date"]: float(row["Close"]) for _, row in actuals.iterrows()}

    print(f"[backtest] Symbol: {SYMBOL}")
    print(f"[backtest] Issue date: {issue_date.date()}, current_price: {current_price:.2f}")
    print(f"[backtest] Predicted days: {len(daily)}, actuals available: {len(actuals)}")

    signals = fetch_signal_history()
    print(f"[backtest] Signal history: vix={len(signals['vix'])}, asian={len(signals['asian'])}, "
          f"quakes={len(signals['quakes'])}, gdelt_pk={len(signals['gdelt_pk_conflict'])}, "
          f"gdelt_rg={len(signals['gdelt_regional_war'])}")

    hist_df = load_history_for_markov()

    base_adjustments = synthetic_baseline_geo_adjustments(daily, current_price)

    # Build prediction-date index. The cached predictions have explicit "date"
    # fields — use them to align with actuals. For days that aren't actual
    # trading days (weekends/holidays), we'll skip the actual-error measure
    # but still compute the overlay (the model itself extrapolates).
    results: List[DayResult] = []

    # Base pred prices by day
    baseline_pred_prices = [float(p["predicted_price"]) for p in daily]

    # Walk each day, compute point-in-time overlay
    enriched_adjustments_full: List[Dict] = []
    snapshots: List[Dict[str, float]] = []
    prev_actual_for_dir = current_price  # for direction-check baseline

    for i, p in enumerate(daily):
        pred_date = pd.to_datetime(p["date"]).normalize()
        # asof for signals = day BEFORE the prediction (we only know yesterday's data)
        asof = pred_date - pd.Timedelta(days=1)

        vix_row = asof_row(signals["vix"], "date", asof)
        asian_row = asof_row(signals["asian"], "date", asof)
        pkr_row = asof_row(signals["pkr"], "date", asof)
        brent_row = asof_row(signals["crude"], "date", asof)

        gdelt_pk_tone = asof_series(signals["gdelt_pk_conflict"], "date", "gdelt_pk_conflict_tone", asof, 30)
        gdelt_pk_vol  = asof_series(signals["gdelt_pk_conflict"], "date", "gdelt_pk_conflict_vol",  asof, 30)
        gdelt_rg_tone = asof_series(signals["gdelt_regional_war"], "date", "gdelt_regional_war_tone", asof, 30)
        gdelt_rg_vol  = asof_series(signals["gdelt_regional_war"], "date", "gdelt_regional_war_vol",  asof, 30)

        # Markov: walk-forward — use ticker closes up to asof
        prior_closes = hist_df.loc[hist_df["Date"] <= asof, "Close"]
        markov_sig = compute_markov_regime_signal(prior_closes) if len(prior_closes) >= 250 else None
        markov_score = float(markov_sig.signal_score) if markov_sig else 0.0

        snapshot = collapse_snapshot(
            vix_row=vix_row,
            asian_row=asian_row,
            gdelt_pk_tone=gdelt_pk_tone,
            gdelt_pk_vol=gdelt_pk_vol,
            gdelt_regional_tone=gdelt_rg_tone,
            gdelt_regional_vol=gdelt_rg_vol,
            quake_df=signals["quakes"],
            asof_date=asof,
            markov_signal_score=markov_score,
            ticker_close_prices=prior_closes,
            pkr_row=pkr_row,
            brent_row=brent_row,
        )
        snapshots.append(snapshot.as_dict())

        # Apply overlay JUST for this day (single-element list)
        single = [base_adjustments[i]]
        enriched_single = apply_worldmonitor_overlay(single, snapshot, symbol=SYMBOL, weights=weights)
        enriched_adjustments_full.append(enriched_single[0])

    # Now produce side-by-side prediction prices
    for i, p in enumerate(daily):
        baseline_price = baseline_pred_prices[i]
        delta_frac = float(enriched_adjustments_full[i]["capped_adjustment"])
        enriched_price = round(baseline_price * (1 + delta_frac), 4)

        pred_date = pd.to_datetime(p["date"]).normalize()
        actual = actuals_by_date.get(pred_date)
        if actual is None:
            # Use last available actual <= pred_date for direction comparison only
            prior = [v for d, v in actuals_by_date.items() if d <= pred_date]
            actual_close = None
            actual_chg = None
            base_err = None
            enriched_err = None
            base_dir = None
            enriched_dir = None
        else:
            actual_close = actual
            actual_chg = (actual - prev_actual_for_dir) / prev_actual_for_dir * 100.0
            base_pred_chg = (baseline_price - prev_actual_for_dir) / prev_actual_for_dir * 100.0
            enr_pred_chg  = (enriched_price - prev_actual_for_dir) / prev_actual_for_dir * 100.0
            base_err = abs(baseline_price - actual) / actual * 100.0
            enriched_err = abs(enriched_price - actual) / actual * 100.0
            # Direction = sign of change vs prior actual
            def _sgn(x): return 1 if x > 0.05 else (-1 if x < -0.05 else 0)
            base_dir = _sgn(base_pred_chg) == _sgn(actual_chg) if _sgn(actual_chg) != 0 else None
            enriched_dir = _sgn(enr_pred_chg) == _sgn(actual_chg) if _sgn(actual_chg) != 0 else None

        # For change-pct columns, fall back to "vs current_price" if we don't have prior actual
        prior_for_chg = prev_actual_for_dir
        results.append(DayResult(
            day=int(p["day"]),
            date=p["date"],
            actual_close=actual_close,
            baseline_pred=baseline_price,
            enriched_pred=enriched_price,
            baseline_err_pct=base_err,
            enriched_err_pct=enriched_err,
            actual_change_pct=actual_chg,
            baseline_pred_change_pct=(baseline_price - prior_for_chg) / prior_for_chg * 100.0,
            enriched_pred_change_pct=(enriched_price - prior_for_chg) / prior_for_chg * 100.0,
            baseline_dir_correct=base_dir,
            enriched_dir_correct=enriched_dir,
            overlay_delta_pct=float(enriched_adjustments_full[i]["worldmonitor_delta_pct"]),
            snapshot=snapshots[i],
        ))
        if actual is not None:
            prev_actual_for_dir = actual

    # Aggregate metrics
    measured = [r for r in results if r.actual_close is not None]
    base_errs = [r.baseline_err_pct for r in measured]
    enr_errs  = [r.enriched_err_pct for r in measured]
    base_dirs = [r.baseline_dir_correct for r in measured if r.baseline_dir_correct is not None]
    enr_dirs  = [r.enriched_dir_correct for r in measured if r.enriched_dir_correct is not None]

    def _safe_mean(xs): return float(np.mean(xs)) if xs else None
    def _safe_max(xs):  return float(np.max(xs)) if xs else None
    def _hit(xs):       return float(sum(1 for x in xs if x)) / len(xs) if xs else None

    summary = {
        "symbol": SYMBOL,
        "issue_date": issue_date.isoformat(),
        "actuals_available_days": len(measured),
        "baseline_mae_pct": _safe_mean(base_errs),
        "enriched_mae_pct": _safe_mean(enr_errs),
        "mae_improvement_pp": (_safe_mean(base_errs) or 0) - (_safe_mean(enr_errs) or 0),
        "baseline_max_err_pct": _safe_max(base_errs),
        "enriched_max_err_pct": _safe_max(enr_errs),
        "baseline_dir_accuracy": _hit(base_dirs),
        "enriched_dir_accuracy": _hit(enr_dirs),
        "weights_used": weights or DEFAULT_WEIGHTS,
    }
    return {"summary": summary, "rows": [asdict(r) for r in results]}


def print_table(report: Dict) -> None:
    s = report["summary"]
    print()
    print("=" * 78)
    print(f"OGDC 21-DAY BACKTEST — issued {s['issue_date'][:10]}, {s['actuals_available_days']} actual days")
    print("=" * 78)
    print(f"{'Day':>3}  {'Date':<11}  {'Actual':>8}  {'Base':>8}  {'Enr':>8}  "
          f"{'BaseErr%':>8}  {'EnrErr%':>8}  {'Δ%pp':>6}  {'BaseDir':>7}  {'EnrDir':>6}")
    print("-" * 78)
    for r in report["rows"]:
        actual = f"{r['actual_close']:.2f}" if r['actual_close'] is not None else "  -  "
        base_err = f"{r['baseline_err_pct']:.2f}" if r['baseline_err_pct'] is not None else "  -  "
        enr_err  = f"{r['enriched_err_pct']:.2f}"  if r['enriched_err_pct']  is not None else "  -  "
        bdir = "✓" if r['baseline_dir_correct'] else ("✗" if r['baseline_dir_correct'] is False else "-")
        edir = "✓" if r['enriched_dir_correct'] else ("✗" if r['enriched_dir_correct'] is False else "-")
        print(f"{r['day']:>3}  {r['date']:<11}  {actual:>8}  {r['baseline_pred']:>8.2f}  "
              f"{r['enriched_pred']:>8.2f}  {base_err:>8}  {enr_err:>8}  "
              f"{r['overlay_delta_pct']:>+6.2f}  {bdir:>7}  {edir:>6}")
    print("-" * 78)
    if s['baseline_mae_pct'] is None:
        print(f"No measured days in horizon — pure forward forecast.")
    else:
        print(f"Baseline MAE: {s['baseline_mae_pct']:.3f}%   Enriched MAE: {s['enriched_mae_pct']:.3f}%   "
              f"Improvement: {s['mae_improvement_pp']:+.3f}pp")
        if s['baseline_dir_accuracy'] is not None:
            print(f"Baseline dir-acc: {s['baseline_dir_accuracy']*100:.1f}%   "
                  f"Enriched dir-acc: {s['enriched_dir_accuracy']*100:.1f}%")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SYMBOL = sys.argv[1].upper()
        PRED_FILE, HIST_FILE = _paths_for(SYMBOL)
    rep = run_backtest()
    print_table(rep)
    out = ROOT / "data" / f"{SYMBOL}_worldmonitor_backtest.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"[backtest] Saved: {out.relative_to(ROOT)}")
