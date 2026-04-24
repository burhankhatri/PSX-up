#!/usr/bin/env python3
"""
Walk-forward backtest harness — slides a 21-day window across historical
price data at monthly intervals, measures overlay predictive value across
many regime-windows rather than the single point-estimate of the cached
predictions.

Design choices (honest tradeoffs):

- **Baseline = flat (today's close for all 21 days).** We're not testing
  the research model here — we're testing whether the overlay's DELTA is
  predictive. Flat is the correct null hypothesis: "no information."

- **Only uses signals that the ablation study validated**: momentum_20d,
  markov_regime, hormuz (Brent-only, no GDELT because we'd need historical
  GDELT for every start_date and the API rate-limits aggressively). This
  means walk-forward numbers are slightly MORE conservative than the single
  point estimate, because they exclude Hormuz's GDELT component.

- **Monthly stride across 3 years** — gives ~36 windows per ticker.
  Distributions over 36 windows are statistically meaningful; single point
  estimates are not.

Reports: median, IQR, P5, P95, worst-case MAE delta vs flat baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.external_features import _to_naive_datetime
from backend.markov_regime import compute_markov_regime_signal
from backend.worldmonitor_overlay import (
    collapse_snapshot, apply_worldmonitor_overlay,
    DEFAULT_WEIGHTS, WorldmonitorSnapshot,
)
from backend.worldmonitor_signals import fetch_crude_prices


# -------------------------------------------------------------------------
# Per-ticker historical price loader (uses existing cached JSON in data/)
# -------------------------------------------------------------------------

def load_ticker_history(symbol: str) -> pd.DataFrame:
    path = ROOT / "data" / f"{symbol}_historical_with_indicators.json"
    if not path.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.load(open(path)))
    df["Date"] = _to_naive_datetime(df["Date"]).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


# -------------------------------------------------------------------------
# Lean snapshot: only the signals the ablation study validated
# -------------------------------------------------------------------------

def _as_of_brent_row(brent_df: pd.DataFrame, asof: pd.Timestamp) -> Optional[pd.Series]:
    if brent_df is None or brent_df.empty:
        return None
    d = brent_df.copy()
    d["date"] = _to_naive_datetime(d["date"])
    m = d["date"] <= asof
    return d.loc[m].iloc[-1] if m.any() else None


def lean_snapshot(close_prices: pd.Series, asof: pd.Timestamp,
                   brent_df: Optional[pd.DataFrame]) -> WorldmonitorSnapshot:
    """Build a snapshot using only the 3 validated signals: momentum, Markov,
    Hormuz-lite (Brent return only; no GDELT). Returns a snapshot with other
    signal scores set to 0 so they can't contribute to the delta.
    """
    markov = compute_markov_regime_signal(close_prices).signal_score if len(close_prices) >= 250 else 0.0
    brent_row = _as_of_brent_row(brent_df, asof)

    from backend.worldmonitor_overlay import momentum_score_from_prices, _safe
    # Hormuz-lite: only the Brent-confirmation component (the GDELT parts
    # would need historical GDELT API calls; skipping for walk-forward).
    brent_confirm = 0.0
    if brent_row is not None:
        b1 = _safe(brent_row.get("brent_change_1d", 0.0))
        b5 = _safe(brent_row.get("brent_change_5d", 0.0))
        brent_confirm = max(0.0, min(1.0, max(b1 / 0.03, b5 / 0.05)))

    return WorldmonitorSnapshot(
        vix_score=0.0,
        extended_asian_score=0.0,
        gdelt_tone_score=0.0,
        gdelt_vol_score=0.0,
        markov_score=float(np.clip(markov, -1.0, 1.0)),
        quake_score=0.0,
        momentum_score=momentum_score_from_prices(close_prices),
        pkr_fx_score=0.0,
        hormuz_risk_score=0.35 * brent_confirm,  # only the brent component
    )


# -------------------------------------------------------------------------
# Single-window evaluator
# -------------------------------------------------------------------------

def evaluate_window(hist: pd.DataFrame, symbol: str, start_date: pd.Timestamp,
                    horizon_days: int, brent_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """For a given start_date, simulate a horizon-day-ahead prediction using
    only data <= start_date, and score against actuals for the next N trading
    days.

    Returns dict with per-day errors and summary, or None if insufficient data.
    """
    prior = hist[hist["Date"] <= start_date]
    if len(prior) < 260:
        return None

    # Find the next `horizon_days` trading days after start_date
    future = hist[hist["Date"] > start_date].head(horizon_days)
    if len(future) < 10:
        return None

    last_close = float(prior["Close"].iloc[-1])
    snap = lean_snapshot(prior["Close"], start_date, brent_df)

    # Build a zero-baseline adjustment block (overlay is the entire prediction)
    base_adjs = [{"day": d, "capped_adjustment": 0.0, "percentage": 0.0, "event_impacts": []}
                 for d in range(1, horizon_days + 1)]
    enriched = apply_worldmonitor_overlay(base_adjs, snap, symbol=symbol)

    flat_errs, overlay_errs = [], []
    flat_dir_hits, overlay_dir_hits, dir_total = 0, 0, 0
    for d in range(min(horizon_days, len(future))):
        actual = float(future["Close"].iloc[d])
        flat_pred = last_close
        delta = float(enriched[d]["capped_adjustment"])
        overlay_pred = last_close * (1.0 + delta)

        flat_errs.append(abs(flat_pred - actual) / actual * 100.0)
        overlay_errs.append(abs(overlay_pred - actual) / actual * 100.0)

        # Direction check: does sign(delta) match sign(actual - last_close)?
        actual_dir = actual - last_close
        if abs(actual_dir) > 0.001 * last_close:  # non-trivial move
            dir_total += 1
            if (delta > 0) == (actual_dir > 0):
                overlay_dir_hits += 1
            if False == (actual_dir > 0):  # flat predicts 0 → never correct on non-trivial moves
                flat_dir_hits += 1

    return {
        "symbol": symbol,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "n_days_measured": len(flat_errs),
        "flat_mae": float(np.mean(flat_errs)),
        "overlay_mae": float(np.mean(overlay_errs)),
        "mae_improvement_pp": float(np.mean(flat_errs) - np.mean(overlay_errs)),
        "overlay_dir_acc": overlay_dir_hits / dir_total if dir_total > 0 else None,
        "dir_sample_size": dir_total,
        "snap_momentum": snap.momentum_score,
        "snap_markov": snap.markov_score,
    }


# -------------------------------------------------------------------------
# Multi-window + multi-ticker sweep
# -------------------------------------------------------------------------

def sweep_ticker(symbol: str, horizon_days: int = 21, stride_days: int = 30,
                  lookback_years: int = 3, brent_df: Optional[pd.DataFrame] = None) -> List[Dict]:
    hist = load_ticker_history(symbol)
    if hist.empty:
        print(f"  {symbol}: no history, skipping")
        return []

    latest = hist["Date"].max()
    earliest = max(hist["Date"].min(), latest - pd.Timedelta(days=365 * lookback_years))

    # Need horizon_days of future data — stop strides at (latest - horizon_days + 1)
    stop = latest - pd.Timedelta(days=horizon_days + 2)

    results = []
    start = earliest + pd.Timedelta(days=260)  # need at least 260 prior days for Markov
    while start <= stop:
        res = evaluate_window(hist, symbol, start, horizon_days, brent_df)
        if res is not None:
            results.append(res)
        start += pd.Timedelta(days=stride_days)

    return results


def summarize(results: List[Dict], label: str) -> Dict:
    if not results:
        return {"label": label, "n_windows": 0}
    improvements = np.array([r["mae_improvement_pp"] for r in results])
    overlay_mae = np.array([r["overlay_mae"] for r in results])
    flat_mae = np.array([r["flat_mae"] for r in results])
    dir_acc_values = [r["overlay_dir_acc"] for r in results if r["overlay_dir_acc"] is not None]
    summary = {
        "label": label,
        "n_windows": len(results),
        "flat_mae_median": float(np.median(flat_mae)),
        "overlay_mae_median": float(np.median(overlay_mae)),
        "improvement_median_pp": float(np.median(improvements)),
        "improvement_p25_pp": float(np.percentile(improvements, 25)),
        "improvement_p75_pp": float(np.percentile(improvements, 75)),
        "improvement_p05_pp": float(np.percentile(improvements, 5)),
        "improvement_p95_pp": float(np.percentile(improvements, 95)),
        "improvement_worst_pp": float(np.min(improvements)),
        "improvement_best_pp": float(np.max(improvements)),
        "pct_windows_overlay_wins": float(np.mean(improvements > 0) * 100),
        "mean_dir_acc": float(np.mean(dir_acc_values) * 100) if dir_acc_values else None,
    }
    return summary


def print_summary(summary: Dict) -> None:
    s = summary
    if s["n_windows"] == 0:
        print(f"  {s['label']}: no windows"); return
    print(f"  {s['label']:<10}  n={s['n_windows']:>3}  "
          f"flatMAE={s['flat_mae_median']:>5.2f}%  "
          f"overlayMAE={s['overlay_mae_median']:>5.2f}%  "
          f"Δmed={s['improvement_median_pp']:>+6.2f}pp  "
          f"Δiqr[{s['improvement_p25_pp']:+5.2f},{s['improvement_p75_pp']:+5.2f}]  "
          f"worst={s['improvement_worst_pp']:>+6.2f}pp  "
          f"wins={s['pct_windows_overlay_wins']:.0f}%  "
          f"dir={('%.1f%%' % s['mean_dir_acc']) if s['mean_dir_acc'] is not None else '-'}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else [
        "OGDC", "PPL", "MARI", "LUCK", "DGKC", "FCCL",
        "HBL", "UBL", "FFC", "EFERT", "HUBC", "SYS", "INDU", "CNERGY",
    ]
    print(f"[walkforward] Fetching Brent history once (cached)...")
    brent = fetch_crude_prices(period="5y")
    print(f"[walkforward] Brent history rows: {len(brent)}")
    print()
    print(f"Walk-forward sweep: {len(tickers)} tickers, 21-day horizon, 30-day stride, 3y lookback")
    print("=" * 120)

    all_summaries: List[Dict] = []
    all_results_global: List[Dict] = []
    for t in tickers:
        results = sweep_ticker(t, brent_df=brent)
        summary = summarize(results, t)
        all_summaries.append(summary)
        all_results_global.extend(results)
        print_summary(summary)

    print()
    print("=" * 120)
    print("AGGREGATE (all tickers, all windows):")
    agg = summarize(all_results_global, "ALL")
    print_summary(agg)

    out = ROOT / "data" / "walkforward_results.json"
    out.write_text(json.dumps({
        "summaries": all_summaries,
        "aggregate": agg,
        "results": all_results_global,
    }, indent=2, default=str))
    print(f"\n[walkforward] Saved: {out.relative_to(ROOT)}")
