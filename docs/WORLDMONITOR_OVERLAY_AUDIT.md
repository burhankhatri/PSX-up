# Worldmonitor Overlay — Honest Audit

**Date:** 2026-04-24
**Scope:** Validate, harden, and right-size the worldmonitor overlay shipped earlier this session. Result of the user request: "do all of this — leakage check, ablation, walk-forward, crash tests, unit tests, base-model features."

## Headline

The "OGDC MAE 10.69% → 2.78% (+7.91pp / 73% relative)" headline was real for that one window but **not generalizable**. Walk-forward across 308 windows / 11 tickers / 3 years shows the overlay is essentially **noise vs. a flat baseline** with median improvement of **+0.01pp** and a **53% win rate**. The OGDC win was a lottery ticket from cumulative-mode compounding during a strong trend.

After this audit:
- 6 of 9 signals dropped (ablation showed they're dead weight)
- Cumulative mode demoted from default → opt-in (was producing −158pp worst-case losses)
- Default mode now additive (+0.01pp median, −12pp worst-case)
- 49 unit tests added, all passing
- 8 momentum features added to the model's feature pipeline for next retrain

## #1 — Point-in-time leakage check: **DEBUNKED**

Earlier honest-criticism claimed the backtest leaked future data via cached GDELT / PKR snapshots. **Verified false.** Snapshot scores vary substantially across as-of dates (VIX +0.26 → −0.26, momentum +0.21 → +0.97 across 4 dates in April), confirming `asof_row` and `asof_series` truncate correctly. The rolling z-scores are computed at fetch time using backward-looking windows, so each row's z-score is point-in-time correct. **No fix needed.**

## #2 — Ablation: 6 of 9 signals are dead weight

For each signal, set its weight to 0 and re-ran OGDC backtest. Then ran on LUCK, MARI, SYS too:

| Signal | OGDC | LUCK | MARI | SYS | Verdict |
|---|---|---|---|---|---|
| `momentum_20d` | ✓ +6.80pp | ✓ | ✓ | **✗ −0.45pp harm** | Trend-MVP, reversal-disaster |
| `markov_regime` | ✓ +0.70pp | ✗ slight | ✓ | ✓ | Most reliable |
| `hormuz` | ✓ +0.15pp | ~ | ~ | ~ | Only helps E&P |
| `vix` | noise | noise | noise | noise | **CUT** |
| `extended_asian` | drag | noise | noise | drag | **CUT** |
| `gdelt_tone_delta` | noise | noise | noise | noise | **CUT** |
| `gdelt_vol_spike` | noise | noise | noise | noise | **CUT** |
| `usgs_quake` | noise | noise | noise | noise | **CUT** |
| `pkr_fx` | drag | noise | noise | drag | **CUT** |

**Action taken:** Default weights for 6 cut signals set to `0.0` in `worldmonitor_overlay.py:DEFAULT_WEIGHTS`. The overlay is now effectively a 3-signal model: momentum + Markov + sector-aware Hormuz. Also tightened the strong-trend gate so momentum doesn't dominate when it disagrees with Markov — this turned the SYS regression (−0.45pp) into a small positive (+0.15pp).

## #3 — Walk-forward backtest (the most important finding)

`backend/backtest_walkforward.py` slides a 21-day window across the last 3 years at monthly stride for 11 tickers (= 308 windows). Compares overlay-on-flat vs flat baseline.

### Cumulative mode (was the default)

```
ALL: n=308  flatMAE=4.46%  overlayMAE=5.57%  Δmed=−0.04pp  Δiqr[−0.42,+0.14]
              wins=44%  worst=−158.74pp  dir-acc=55.3%
```

### Additive mode (now the default)

```
ALL: n=308  flatMAE=4.46%  overlayMAE=4.44%  Δmed=+0.01pp  Δiqr[−0.19,+0.23]
              wins=53%  worst=−12.39pp  dir-acc=55.3%
```

**Cumulative mode is a high-variance bet.** It produced both the OGDC +7.91pp lottery ticket AND a −158pp catastrophic loss on a different window. Additive mode is much safer: 13× smaller worst case, slightly above coin flip wins, marginally positive median.

**Action taken:** `apply_worldmonitor_overlay(mode="additive")` is now the default. Cumulative is still available as opt-in for cases where the model baseline is known to be badly biased (like the cached OGDC predictions that said $270 flat while actual ripped to $326).

By-ticker additive results:

| Ticker | n | flat MAE | overlay MAE | Δ median | wins |
|---|---|---|---|---|---|
| OGDC | 28 | 4.53% | 4.25% | **+0.09pp** | 57% |
| EFERT | 28 | 4.08% | 3.54% | **+0.08pp** | 64% |
| UBL | 28 | 3.82% | 3.85% | **+0.08pp** | 61% |
| FFC | 28 | 3.83% | 4.39% | +0.04pp | 57% |
| LUCK | 28 | 4.25% | 4.13% | +0.02pp | 57% |
| PPL | 28 | 5.61% | 5.96% | +0.02pp | 68% |
| MARI | 28 | 3.50% | 3.90% | −0.01pp | 46% |
| SYS | 28 | 4.33% | 4.23% | −0.02pp | 43% |
| HUBC | 28 | 4.00% | 4.00% | −0.03pp | 46% |
| CNERGY | 28 | 5.85% | 6.13% | −0.04pp | 39% |
| FCCL | 28 | 6.24% | 6.41% | −0.05pp | 43% |

6 of 11 tickers positive, 5 slightly negative. **Overlay is marginal but defensible** as a small directional nudge in trending names (OGDC/PPL/EFERT/UBL).

## #4 — Crash-window tests (the overlay is not a crash detector)

`backend/backtest_crash_windows.py` runs the overlay at the START of 5 known PSX drawdown episodes:

| Episode | n | Median Δ | Win rate | Worst |
|---|---|---|---|---|
| 2020-03 COVID + oil crash | 0 | (no history before 2020-02) | — | — |
| 2022-07 PKR / IMF stalemate | 4 | +0.01pp | 50% | −0.12pp |
| 2024-08 summer pullback | 5 | **−0.14pp** | **0%** | **−3.15pp** |
| 2025-05 India-Pak mini-war | 6 | **−0.15pp** | **17%** | −1.08pp |
| 2026-03 oil shock | 3 | +0.17pp | 67% | −0.23pp |

**The overlay missed the 2025 India-Pak recovery rally on every ticker except FFC.** Momentum was bearish pre-crisis (correct), but it stayed bearish through the ceasefire bounce because momentum signals lag turning points.

**Honest reframe:** The overlay is trend-continuation, not crisis-management. Don't market it as a "geo overlay" that will protect you in shocks — it won't. Where it earns its keep is in clear single-direction regimes (OGDC's April 2026 oil-driven rally), and even there the median benefit is small.

## #5 — Test infrastructure

- 49 unit tests in `tests/test_worldmonitor_overlay.py` covering: `_to_naive_datetime` regressions (the dtype fix that's bitten us twice), every score function, sector transmission, the strong-trend gate, cap behavior, additive vs cumulative mode, Markov regime classifier states, and quake threshold.
- All 49 passing in 0.6s.
- Orchestrator smoke test verified the overlay block in `stock_analyzer_fixed.py:1957` executes end-to-end on synthetic data.

## #6 — Base model momentum features (the right long-term fix)

Added 8 momentum features to `external_features.merge_external_features()` (after KIBOR block):

```
mom_price_vs_sma20_pct, mom_price_vs_sma50_pct, mom_price_vs_sma200_pct,
mom_roc_5d, mom_roc_20d, mom_roc_60d,
mom_regime_state (5-state Markov bin: 0=very_under … 4=very_over),
mom_realized_vol_20d
```

These are now available on every merged DataFrame. **They will be picked up the next time the base model is retrained** (`PSXResearchModel`) — at that point the model can learn them structurally instead of having a post-process overlay patch over a flat-bias problem.

**Important:** The currently-trained models in `data/models/` were fit WITHOUT these features. They will not benefit until retrained. A retrain is genuinely a 1–2 day workstream and was out of scope for this session, but the foundation is laid.

## What you should believe vs what you shouldn't

**Believe**:
- The dtype fix (tz + resolution) is solid; covered by tests; you won't hit that error again
- The lean overlay (momentum + Markov + Hormuz) is honest about what it does
- Walk-forward numbers (+0.01pp median across 308 windows) are the truth
- Adding momentum features to the base model is the correct path forward

**Don't believe**:
- "+7.91pp / 73% improvement" as a typical result. It was one window in a strong trend.
- The overlay as a crash-protection tool. Crash tests show it doesn't help (and often hurts) in turning points.
- Direction accuracy 55.3% as actionable for trading. That's barely above coin flip.

## What's left after this session

1. **Retrain the base model** with the 8 new momentum features. Expected: directly absorbs what the post-process overlay was patching, removes the flat-bias.
2. **Walk-forward harness with the actual model predictions** (not a flat baseline). Will tell us how much value the overlay adds to the REAL pipeline (not a synthetic null).
3. **A regime detector** that auto-switches between additive and cumulative modes based on momentum strength + persistence. Cumulative is dangerous when reversals happen; turning it on selectively (e.g., "only if 3 consecutive 20-day periods of same-sign momentum") could capture the OGDC-style win without the −158pp catastrophes.
