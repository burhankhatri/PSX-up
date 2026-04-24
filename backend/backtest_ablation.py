#!/usr/bin/env python3
"""
Ablation study: for each of the 9 overlay signals, zero its weight and
re-run the backtest. Compare to the full-weights baseline. A signal that
HURTS when removed contributes positively (earns its weight); a signal
that HELPS when removed is dead weight or actively harmful.

Usage: ./venv/bin/python backend/backtest_ablation.py [TICKER]
Default ticker: OGDC.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.backtest_ogdc as bk
from backend.worldmonitor_overlay import DEFAULT_WEIGHTS


def run_with_weights(weights):
    # bk.run_backtest accepts a weights override
    return bk.run_backtest(weights=weights)


def summarize(rep):
    s = rep["summary"]
    return {
        "mae": s.get("enriched_mae_pct"),
        "dir_acc": s.get("enriched_dir_accuracy"),
    }


def main(symbol: str = "OGDC"):
    bk.SYMBOL = symbol
    bk.PRED_FILE, bk.HIST_FILE = bk._paths_for(symbol)

    print(f"Ablation study for {symbol}")
    print("=" * 70)

    # Full-weights baseline
    full_rep = run_with_weights(DEFAULT_WEIGHTS)
    full = summarize(full_rep)
    baseline_mae = full_rep["summary"]["baseline_mae_pct"]
    print(f"BASELINE (no overlay):     MAE={baseline_mae:.3f}%")
    print(f"FULL (all signals on):     MAE={full['mae']:.3f}%  dir-acc={full['dir_acc']*100:.1f}%")
    print()
    print(f"{'Signal ablated':<22}  {'MAE%':>7}  {'ΔMAE vs full':>14}  {'ΔMAE vs baseline':>17}  {'verdict':>22}")
    print("-" * 95)

    results = []
    for signal in DEFAULT_WEIGHTS:
        w = dict(DEFAULT_WEIGHTS)
        w[signal] = 0.0
        rep = run_with_weights(w)
        summ = summarize(rep)
        d_full = summ["mae"] - full["mae"]
        d_base = summ["mae"] - baseline_mae
        if d_full > 0.02:
            verdict = "✓ earns its weight"
        elif d_full < -0.02:
            verdict = "✗ net-negative (drop!)"
        else:
            verdict = "  ~noise"
        results.append({"signal": signal, "mae": summ["mae"], "delta_full": d_full, "delta_base": d_base, "verdict": verdict})
        print(f"{signal:<22}  {summ['mae']:>6.3f}  {d_full:>+13.3f}pp  {d_base:>+16.3f}pp  {verdict:>22}")

    # Also: ZERO overlay (all signals off)
    zero_w = {k: 0.0 for k in DEFAULT_WEIGHTS}
    zero_rep = run_with_weights(zero_w)
    zero = summarize(zero_rep)
    print("-" * 95)
    print(f"{'ALL off':<22}  {zero['mae']:>6.3f}  {zero['mae'] - full['mae']:>+13.3f}pp  {zero['mae'] - baseline_mae:>+16.3f}pp  (sanity check)")

    print()
    print("INTERPRETATION")
    print("-" * 95)
    earns = [r for r in results if r["delta_full"] > 0.02]
    noise = [r for r in results if -0.02 <= r["delta_full"] <= 0.02]
    harms = [r for r in results if r["delta_full"] < -0.02]
    print(f"  earns its weight: {[r['signal'] for r in earns]}")
    print(f"  noise (cut):      {[r['signal'] for r in noise]}")
    print(f"  net-negative:     {[r['signal'] for r in harms]}")

    return {
        "symbol": symbol,
        "baseline_mae": baseline_mae,
        "full_mae": full["mae"],
        "zero_mae": zero["mae"],
        "per_signal": results,
    }


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "OGDC"
    main(sym)
