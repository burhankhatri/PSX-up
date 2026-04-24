#!/usr/bin/env python3
"""
Markov regime classifier for valuation/momentum-based mean reversion signal.

Premise: classify a stock's *current state* into one of K discrete regimes
based on price relative to long trend (over/fair/under-valued), then use
historical state-transition probabilities to forecast the expected regime
shift over the next H days. The expected forward return is the probability-
weighted mean return across destination regimes.

Why this is useful for PSX prediction:
- Pure mean-reversion signals (e.g. z-score below -2 → buy) ignore the fact
  that a deeply discounted stock can stay discounted. Markov chains reflect
  the *empirical* probability of regime change at each level — learned
  from the stock's own history rather than assumed.
- Output is a single bounded scalar in [-1, +1] that drops cleanly into
  our existing geo overlay's `x_factor` aggregation pattern.

Implementation:
- 5-state classifier on (price - SMA_200) / SMA_200  →  bins:
    very_under (-inf, -0.15], under (-0.15, -0.05],
    fair (-0.05, 0.05], over (0.05, 0.15], very_over (0.15, +inf)
- Transition matrix = empirical 1-step transitions over a training window.
- Forward H-step matrix = T^H by matrix power.
- Expected forward return = Σ P(state_H | state_now) * mean_return_in_state.
- Final scalar signal = clamp(expected_return * scale, -1.0, +1.0).

Defaults pick H=7 to match our day-7 horizon. Caller can override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# State definitions
STATE_NAMES = ["very_under", "under", "fair", "over", "very_over"]
STATE_BINS = [-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf]
N_STATES = len(STATE_NAMES)


def _classify_state(deviations: np.ndarray) -> np.ndarray:
    """Bin deviations (price - sma) / sma into one of N_STATES integer states."""
    # np.digitize returns 1..N; subtract 1 for 0-indexed states. Clip just in case.
    states = np.digitize(deviations, STATE_BINS[1:-1])
    return np.clip(states, 0, N_STATES - 1)


def _compute_transition_matrix(states: np.ndarray) -> np.ndarray:
    """Empirical 1-step state transition matrix with Laplace smoothing."""
    T = np.ones((N_STATES, N_STATES), dtype=float)  # +1 smoothing
    for i in range(len(states) - 1):
        T[states[i], states[i + 1]] += 1.0
    T = T / T.sum(axis=1, keepdims=True)
    return T


def _compute_state_returns(deviations: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Mean next-step return, conditioned on the state at time t."""
    states = _classify_state(deviations)
    means = np.zeros(N_STATES, dtype=float)
    for s in range(N_STATES):
        mask = states[:-1] == s  # state at t, return at t+1
        if mask.any():
            means[s] = float(np.nanmean(returns[1:][mask]))
        else:
            means[s] = 0.0
    return means


@dataclass(frozen=True)
class MarkovRegimeSignal:
    current_state: str
    deviation_pct: float
    transition_probs_horizon: List[float]
    expected_forward_return_pct: float
    signal_score: float          # in [-1.0, +1.0]
    sample_size: int
    horizon_days: int

    def to_dict(self) -> dict:
        return {
            "current_state": self.current_state,
            "deviation_pct": round(self.deviation_pct, 4),
            "transition_probs_horizon": [round(p, 4) for p in self.transition_probs_horizon],
            "expected_forward_return_pct": round(self.expected_forward_return_pct, 4),
            "signal_score": round(self.signal_score, 4),
            "sample_size": self.sample_size,
            "horizon_days": self.horizon_days,
        }


def neutral_markov_signal(horizon_days: int = 7) -> MarkovRegimeSignal:
    return MarkovRegimeSignal(
        current_state="fair",
        deviation_pct=0.0,
        transition_probs_horizon=[0.0] * N_STATES,
        expected_forward_return_pct=0.0,
        signal_score=0.0,
        sample_size=0,
        horizon_days=horizon_days,
    )


def compute_markov_regime_signal(close_prices: pd.Series,
                                   sma_window: int = 200,
                                   horizon_days: int = 7,
                                   min_history: int = 250,
                                   signal_scale: float = 20.0) -> MarkovRegimeSignal:
    """Compute Markov regime signal for the latest day in `close_prices`.

    Args:
        close_prices: pd.Series of daily close prices, indexed in chrono order.
        sma_window: trend baseline window (default 200d).
        horizon_days: forward horizon for state-transition projection.
        min_history: refuse to return non-neutral signal below this.
        signal_scale: maps expected return → [-1, +1] scalar. 20.0 means
                      a 5% expected forward return → score 1.0.
    """
    if close_prices is None or len(close_prices) < min_history:
        return neutral_markov_signal(horizon_days)

    closes = pd.Series(close_prices).astype(float).reset_index(drop=True)
    sma = closes.rolling(sma_window).mean()
    dev = ((closes - sma) / sma).dropna()

    if len(dev) < 50:
        return neutral_markov_signal(horizon_days)

    returns = closes.pct_change().fillna(0.0).values
    dev_arr = dev.values
    states = _classify_state(dev_arr)

    T = _compute_transition_matrix(states)
    state_means = _compute_state_returns(dev_arr, returns[-len(dev_arr):])

    # Forward horizon: T^H gives P(state_H | state_now)
    T_h = np.linalg.matrix_power(T, horizon_days)
    current_state_idx = int(states[-1])
    probs_h = T_h[current_state_idx]

    # Expected H-day cumulative return ~ sum(P(s_H) * mean_return_in_s) * H
    # (mean_return_in_s is per-day; over H days assume IID within state)
    expected_per_day = float(np.dot(probs_h, state_means))
    expected_forward_return = expected_per_day * horizon_days

    score = max(-1.0, min(1.0, expected_forward_return * signal_scale))

    return MarkovRegimeSignal(
        current_state=STATE_NAMES[current_state_idx],
        deviation_pct=float(dev_arr[-1]),
        transition_probs_horizon=probs_h.tolist(),
        expected_forward_return_pct=expected_forward_return * 100.0,
        signal_score=score,
        sample_size=int(len(dev_arr)),
        horizon_days=horizon_days,
    )


def compute_markov_regime_features_series(close_prices: pd.Series,
                                            sma_window: int = 200,
                                            horizon_days: int = 7,
                                            roll_train_window: int = 500,
                                            signal_scale: float = 20.0) -> pd.DataFrame:
    """Walk-forward Markov signal for EVERY day in `close_prices`.

    Each day t uses ONLY data up to and including t to compute its signal —
    no look-ahead. Returns DataFrame of length = len(close_prices) with cols:
      regime_state_idx, regime_deviation_pct, regime_expected_return_pct,
      regime_signal_score
    """
    closes = pd.Series(close_prices).astype(float).reset_index(drop=True)
    n = len(closes)
    out = {
        "regime_state_idx": np.full(n, np.nan),
        "regime_deviation_pct": np.full(n, np.nan),
        "regime_expected_return_pct": np.full(n, np.nan),
        "regime_signal_score": np.full(n, np.nan),
    }
    min_t = max(sma_window + 50, roll_train_window // 2)
    for t in range(min_t, n):
        window = closes.iloc[max(0, t - roll_train_window):t + 1]
        sig = compute_markov_regime_signal(
            window,
            sma_window=sma_window,
            horizon_days=horizon_days,
            min_history=min(min_t, len(window)),
            signal_scale=signal_scale,
        )
        out["regime_state_idx"][t] = STATE_NAMES.index(sig.current_state)
        out["regime_deviation_pct"][t] = sig.deviation_pct
        out["regime_expected_return_pct"][t] = sig.expected_forward_return_pct
        out["regime_signal_score"][t] = sig.signal_score
    return pd.DataFrame(out)
