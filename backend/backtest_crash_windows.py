#!/usr/bin/env python3
"""
Crash-window sanity test — does the overlay help or hurt during the PSX
drawdowns we actually care about?

Episodes (all dates inferred from the codebase's own comments + public history):
- 2020-03 COVID crash (KSE-100 -30%+)
- 2022-07 PKR plunge / IMF stalemate
- 2025-05 India-Pakistan mini-war (KSE-100 -12.6% → +15.76% on ceasefire)
- 2026-03 oil shock per Mettis Global article

For each episode: run the overlay across the crash + recovery window,
compare overlay vs flat baseline MAE and direction accuracy on the steepest
down days and the recovery days.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.backtest_walkforward import (
    load_ticker_history, lean_snapshot, evaluate_window,
)
from backend.worldmonitor_signals import fetch_crude_prices


CRASH_EPISODES = [
    # (label, start, end, tickers_to_test)
    ("2020-03 COVID + oil crash", "2020-02-24", "2020-04-17",
     ["OGDC", "PPL", "MARI", "LUCK", "HUBC", "SYS"]),
    ("2022-07 PKR / IMF stalemate", "2022-06-20", "2022-08-12",
     ["OGDC", "PPL", "LUCK", "HUBC"]),
    ("2024-08 summer pullback", "2024-08-01", "2024-09-20",
     ["OGDC", "PPL", "LUCK", "FFC", "UBL"]),
    ("2025-05 India-Pak mini-war", "2025-05-01", "2025-05-30",
     ["OGDC", "PPL", "LUCK", "MARI", "FFC", "HUBC"]),
    ("2026-03 oil shock (cited in Mettis)", "2026-03-02", "2026-04-10",
     ["OGDC", "PPL", "LUCK", "HBL", "UBL", "FFC"]),
]


def evaluate_episode(label: str, start: str, end: str, tickers: List[str],
                     brent_df: pd.DataFrame) -> Dict:
    print(f"\n── {label} ({start} → {end}) ──")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    per_ticker = []
    for t in tickers:
        hist = load_ticker_history(t)
        if hist.empty:
            print(f"  {t}: no history, skipping"); continue
        # The "window" is: at each trading day in [start, end-21], predict
        # 21 days forward and measure against the actual close path.
        prior = hist[hist["Date"] <= start_ts]
        if len(prior) < 260:
            print(f"  {t}: insufficient history before {start} ({len(prior)} days)"); continue

        # For crash episodes we care about a single-shot forecast at the
        # START of the crash window — would the overlay have warned us?
        res = evaluate_window(hist, t, start_ts, horizon_days=21, brent_df=brent_df)
        if res is None:
            print(f"  {t}: no actuals in window"); continue

        # Also measure what actually happened in the crash
        future = hist[hist["Date"] > start_ts].head(21)
        actual_return = (future["Close"].iloc[-1] / prior["Close"].iloc[-1] - 1) * 100 if len(future) == 21 else None

        act_str = f"{actual_return:+.1f}%" if actual_return is not None else "  n/a"
        print(f"  {t:<8}  flatMAE={res['flat_mae']:>5.2f}%  overlayMAE={res['overlay_mae']:>5.2f}%  "
              f"Δ={res['mae_improvement_pp']:>+6.2f}pp  actual21d={act_str}  "
              f"momo={res['snap_momentum']:+.2f}  mkv={res['snap_markov']:+.2f}")
        per_ticker.append({
            "ticker": t, **res, "actual_21d_return_pct": actual_return,
        })

    if not per_ticker:
        return {"label": label, "n": 0}
    improvements = np.array([r["mae_improvement_pp"] for r in per_ticker])
    return {
        "label": label, "start": start, "end": end,
        "n": len(per_ticker),
        "median_improvement_pp": float(np.median(improvements)),
        "mean_improvement_pp": float(np.mean(improvements)),
        "pct_wins": float(np.mean(improvements > 0) * 100),
        "worst": float(np.min(improvements)),
        "best": float(np.max(improvements)),
        "per_ticker": per_ticker,
    }


if __name__ == "__main__":
    print(f"[crash] Fetching Brent history...")
    brent = fetch_crude_prices(period="max")
    print(f"[crash] Brent rows: {len(brent)}  range: {brent['date'].min()} → {brent['date'].max()}")

    all_eps = []
    for label, start, end, tickers in CRASH_EPISODES:
        rep = evaluate_episode(label, start, end, tickers, brent)
        all_eps.append(rep)

    print()
    print("=" * 80)
    print("CRASH-EPISODE SUMMARY")
    print("=" * 80)
    for e in all_eps:
        if e["n"] == 0:
            print(f"  {e['label']:<45}  (no data)")
            continue
        print(f"  {e['label']:<45}  n={e['n']}  median={e['median_improvement_pp']:+6.2f}pp  "
              f"wins={e['pct_wins']:.0f}%  worst={e['worst']:+.2f}pp")

    out = ROOT / "data" / "crash_backtest_results.json"
    out.write_text(json.dumps(all_eps, indent=2, default=str))
    print(f"\n[crash] Saved: {out.relative_to(ROOT)}")
