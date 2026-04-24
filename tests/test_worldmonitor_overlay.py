"""Unit tests for the worldmonitor overlay: snapshot math, sector transmission,
cap behavior, and the strong-trend gate. Uses only synthetic data — no network,
no yfinance, no GDELT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.worldmonitor_overlay import (
    WorldmonitorSnapshot, DEFAULT_WEIGHTS,
    apply_worldmonitor_overlay, collapse_snapshot,
    vix_score_from_row, extended_asian_score_from_row,
    gdelt_tone_score, gdelt_vol_score, quake_score_from_df,
    momentum_score_from_prices, pkr_fx_score_from_row,
    hormuz_risk_score_from_sources,
    _sector_of, _transmission, TRANSMISSION_HORMUZ, TRANSMISSION_PKR,
    SECTOR_MAP, _overlay_cap_for_day,
)
from backend.markov_regime import (
    compute_markov_regime_signal, STATE_NAMES, neutral_markov_signal,
)
from backend.external_features import _to_naive_datetime


# ─── _to_naive_datetime: the whole reason the user hit dtype errors ──

class TestToNaiveDatetime:
    def test_tz_naive_passthrough(self):
        s = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]))
        out = _to_naive_datetime(s)
        assert str(out.dtype) == "datetime64[ns]"

    def test_tz_aware_stripped(self):
        s = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]).tz_localize("UTC"))
        out = _to_naive_datetime(s)
        assert str(out.dtype) == "datetime64[ns]"
        # Tz should be gone
        assert getattr(out.dt, "tz", None) is None

    def test_second_resolution_upcast_to_ns(self):
        s = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]).astype("datetime64[s]"))
        out = _to_naive_datetime(s)
        assert str(out.dtype) == "datetime64[ns]"

    def test_strings_get_parsed(self):
        s = pd.Series(["2026-01-01", "2026-01-02"])
        out = _to_naive_datetime(s)
        assert str(out.dtype) == "datetime64[ns]"

    def test_merge_asof_with_mixed_resolutions(self):
        """Regression: us + s + tz-aware all flow through the helper and merge."""
        a = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-03"]))
        b = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]).astype("datetime64[s]"))
        c = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]).tz_localize("UTC"))
        a, b, c = _to_naive_datetime(a), _to_naive_datetime(b), _to_naive_datetime(c)
        merged = pd.merge_asof(
            pd.DataFrame({"d": a.sort_values(), "x": [1, 2]}),
            pd.DataFrame({"d": c.sort_values(), "y": [10, 20]}),
            on="d", direction="backward",
        )
        assert len(merged) == 2


# ─── WorldmonitorSnapshot + score functions ──

class TestSignalScores:
    def test_neutral_snapshot_zero(self):
        snap = WorldmonitorSnapshot(0, 0, 0, 0, 0, 0, 0)
        for k, v in snap.as_dict().items():
            assert v == 0.0

    def test_vix_score_above_25(self):
        row = pd.Series({"vix_zscore_60d": 2.0, "vix_above_25": 1, "vix_above_30": 0})
        s = vix_score_from_row(row)
        assert 0.5 < s < 1.0  # positive = bearish

    def test_vix_score_calm_market(self):
        row = pd.Series({"vix_zscore_60d": -1.5, "vix_above_25": 0, "vix_above_30": 0})
        s = vix_score_from_row(row)
        assert s < 0  # negative = risk-on

    def test_vix_score_none_row(self):
        assert vix_score_from_row(None) == 0.0

    def test_gdelt_tone_improving(self):
        """3-day mean better (less negative) than 14-day baseline → +."""
        series = pd.Series([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
                             -2, -2, -2])  # last 3 less negative
        s = gdelt_tone_score(series)
        assert s > 0

    def test_gdelt_tone_worsening(self):
        series = pd.Series([-1] * 11 + [-5, -5, -5])
        s = gdelt_tone_score(series)
        assert s < 0

    def test_gdelt_tone_insufficient_data(self):
        assert gdelt_tone_score(pd.Series([1, 2, 3])) == 0.0

    def test_gdelt_vol_spike(self):
        series = pd.Series([1.0] * 11 + [5.0, 5.0, 5.0])
        s = gdelt_vol_score(series)
        assert s > 0  # spike = bearish

    def test_gdelt_vol_flat(self):
        series = pd.Series([1.0] * 14)
        s = gdelt_vol_score(series)
        assert s == 0.0

    def test_extended_asian_gain(self):
        row = pd.Series({"extended_asian_avg_return": 0.02, "extended_asian_risk_off": 0.0})
        s = extended_asian_score_from_row(row)
        assert s > 0

    def test_extended_asian_loss(self):
        row = pd.Series({"extended_asian_avg_return": -0.03, "extended_asian_risk_off": 1.0})
        s = extended_asian_score_from_row(row)
        assert s < 0

    def test_momentum_strong_uptrend(self):
        # Need >=25 history; price 10%+ above 20d SMA → score saturates at +1.
        prices = pd.Series([100.0] * 30 + [115.0])
        assert momentum_score_from_prices(prices) > 0.9

    def test_momentum_strong_downtrend(self):
        prices = pd.Series([110.0] * 30 + [99.0])
        assert momentum_score_from_prices(prices) < -0.9

    def test_momentum_flat(self):
        prices = pd.Series([100.0] * 25)
        assert momentum_score_from_prices(prices) == pytest.approx(0.0, abs=0.001)

    def test_momentum_insufficient_history(self):
        assert momentum_score_from_prices(pd.Series([100.0, 101.0])) == 0.0

    def test_pkr_score_weakening(self):
        row = pd.Series({"usdpkr_zscore_30d": 2.0, "usdpkr_zscore_90d": 2.5,
                          "usdpkr_weakening_streak": 5.0})
        s = pkr_fx_score_from_row(row)
        assert s > 0

    def test_hormuz_risk_calm(self):
        """All signals calm → score near 0."""
        vol = pd.Series([1.0] * 30)
        tone = pd.Series([0.0] * 30)
        brent = pd.Series({"brent_change_1d": 0.0, "brent_change_5d": 0.0})
        s = hormuz_risk_score_from_sources(vol, tone, brent)
        assert s == pytest.approx(0.0, abs=0.05)

    def test_hormuz_risk_abqaiq_like(self):
        """Big vol spike + negative tone + big brent move → high score."""
        vol = pd.Series([1.0] * 20 + [8.0, 8.0, 8.0])
        tone = pd.Series([-1.0] * 20 + [-5.0] * 3)
        brent = pd.Series({"brent_change_1d": 0.08, "brent_change_5d": 0.12})
        s = hormuz_risk_score_from_sources(vol, tone, brent)
        assert s > 0.5

    def test_hormuz_score_always_non_negative(self):
        """Hormuz is supply-risk only — no such thing as negative Hormuz."""
        vol = pd.Series([10.0] * 30)  # very negative values technically
        tone = pd.Series([5.0] * 30)  # positive tone (rare but valid)
        brent = pd.Series({"brent_change_1d": -0.05, "brent_change_5d": -0.08})
        s = hormuz_risk_score_from_sources(vol, tone, brent)
        assert s >= 0.0


# ─── Sector transmission + cap behavior ──

class TestSectorTransmission:
    def test_ogdc_in_upstream_ep(self):
        assert _sector_of("OGDC") == "ep"

    def test_luck_in_cement(self):
        assert _sector_of("LUCK") == "cem"

    def test_unknown_symbol_returns_other(self):
        assert _sector_of("ZZZZ") == "other"

    def test_hormuz_transmission_ep_positive(self):
        assert _transmission(TRANSMISSION_HORMUZ, "OGDC") > 0

    def test_hormuz_transmission_cement_negative(self):
        assert _transmission(TRANSMISSION_HORMUZ, "LUCK") < 0

    def test_pkr_transmission_tech_positive(self):
        """Tech exports → PKR weakness benefits them."""
        assert _transmission(TRANSMISSION_PKR, "SYS") > 0

    def test_pkr_transmission_auto_strongly_negative(self):
        """Autos get hit by PKR weakness (imported CKD kits)."""
        assert _transmission(TRANSMISSION_PKR, "INDU") < -0.5

    def test_overlay_cap_grows_with_horizon(self):
        """21-day forecast can correct more than a 1-day forecast."""
        assert _overlay_cap_for_day(1, "OGDC") < _overlay_cap_for_day(21, "OGDC")


# ─── apply_worldmonitor_overlay: end-to-end ──

def _zero_adjs(n: int):
    return [{"day": d, "capped_adjustment": 0.0, "percentage": 0.0, "event_impacts": []}
            for d in range(1, n + 1)]


class TestApplyOverlay:
    def test_neutral_snapshot_yields_tiny_delta(self):
        snap = WorldmonitorSnapshot(0, 0, 0, 0, 0, 0, 0)
        out = apply_worldmonitor_overlay(_zero_adjs(5), snap, symbol="OGDC")
        for day in out:
            assert abs(day["worldmonitor_delta_pct"]) < 0.1

    def test_strong_momentum_boosts_ep(self):
        """Strong momentum + agreeing Markov → upstream E&P gets positive overlay."""
        snap = WorldmonitorSnapshot(
            vix_score=0, extended_asian_score=0, gdelt_tone_score=0,
            gdelt_vol_score=0, markov_score=0.3, quake_score=0,
            momentum_score=0.8, pkr_fx_score=0.0, hormuz_risk_score=0.0,
        )
        out = apply_worldmonitor_overlay(_zero_adjs(7), snap, symbol="OGDC")
        # Day 1 should be bullish
        assert out[0]["worldmonitor_delta_pct"] > 0

    def test_weak_momentum_disagreeing_markov_attenuates(self):
        """Weak momentum + markov disagreement → overlay is near zero."""
        snap = WorldmonitorSnapshot(
            vix_score=0, extended_asian_score=0, gdelt_tone_score=0,
            gdelt_vol_score=0, markov_score=0.2, quake_score=0,
            momentum_score=-0.1, pkr_fx_score=0.0, hormuz_risk_score=0.0,
        )
        out = apply_worldmonitor_overlay(_zero_adjs(5), snap, symbol="LUCK")
        # Should be damped — momentum attenuator = 0.25
        assert abs(out[0]["worldmonitor_delta_pct"]) < 0.5

    def test_hormuz_shock_bullish_for_ep(self):
        snap = WorldmonitorSnapshot(
            vix_score=0, extended_asian_score=0, gdelt_tone_score=0,
            gdelt_vol_score=0, markov_score=0, quake_score=0,
            momentum_score=0, pkr_fx_score=0.0, hormuz_risk_score=0.8,
        )
        out = apply_worldmonitor_overlay(_zero_adjs(3), snap, symbol="OGDC")
        assert out[0]["worldmonitor_delta_pct"] > 0  # EP factor is +1.0

    def test_hormuz_shock_bearish_for_cement(self):
        snap = WorldmonitorSnapshot(
            vix_score=0, extended_asian_score=0, gdelt_tone_score=0,
            gdelt_vol_score=0, markov_score=0, quake_score=0,
            momentum_score=0, pkr_fx_score=0.0, hormuz_risk_score=0.8,
        )
        out = apply_worldmonitor_overlay(_zero_adjs(3), snap, symbol="LUCK")
        assert out[0]["worldmonitor_delta_pct"] < 0  # cement factor is -0.4

    def test_horizon_cap_applied(self):
        """Even an extreme snapshot can't produce more than the day cap."""
        extreme_snap = WorldmonitorSnapshot(
            vix_score=-1, extended_asian_score=1, gdelt_tone_score=1,
            gdelt_vol_score=-1, markov_score=1, quake_score=0,
            momentum_score=1, pkr_fx_score=1, hormuz_risk_score=1,
        )
        out = apply_worldmonitor_overlay(_zero_adjs(21), extreme_snap, symbol="OGDC")
        for day in out:
            cap = _overlay_cap_for_day(day["day"], "OGDC")
            assert abs(day["capped_adjustment"]) <= cap + 1e-9

    def test_additive_is_default(self):
        """After lean-overlay refactor, additive mode is the default (safer)."""
        snap = WorldmonitorSnapshot(0, 0, 0, 0, 0.5, 0, 0.8, 0, 0)
        additive = apply_worldmonitor_overlay(_zero_adjs(7), snap, symbol="OGDC")
        cumulative = apply_worldmonitor_overlay(_zero_adjs(7), snap, symbol="OGDC", mode="cumulative")
        # By day 7, cumulative should be much bigger than additive
        assert cumulative[-1]["worldmonitor_delta_pct"] > additive[-1]["worldmonitor_delta_pct"]

    def test_cumulative_mode_compounds(self):
        """Cumulative mode with strong gate open: day-k delta grows with k."""
        snap = WorldmonitorSnapshot(0, 0, 0, 0, 0.3, 0, 0.8, 0, 0)
        out = apply_worldmonitor_overlay(_zero_adjs(14), snap, symbol="OGDC", mode="cumulative")
        deltas = [d["worldmonitor_delta_pct"] for d in out]
        # Monotonically increasing (before cap kicks in)
        assert deltas[-1] > deltas[0]


# ─── Markov regime ──

class TestMarkovRegime:
    def test_neutral_signal_has_zero_score(self):
        sig = neutral_markov_signal()
        assert sig.signal_score == 0.0
        assert sig.current_state == "fair"

    def test_insufficient_history_returns_neutral(self):
        prices = pd.Series([100.0] * 50)
        sig = compute_markov_regime_signal(prices)
        assert sig.signal_score == 0.0

    def test_uptrending_stock_classified_correctly(self):
        """Price rising linearly → ends up in over/very_over state."""
        prices = pd.Series(range(100, 500))  # 400 days rising
        sig = compute_markov_regime_signal(prices.astype(float))
        assert sig.current_state in ("over", "very_over")

    def test_range_bound_stock_is_fair(self):
        """Oscillating around a mean → fair state."""
        t = np.arange(300)
        prices = pd.Series(100 + 2 * np.sin(t * 0.1))
        sig = compute_markov_regime_signal(prices)
        assert sig.current_state == "fair"

    def test_probabilities_sum_to_one(self):
        prices = pd.Series(100 + np.random.RandomState(42).randn(500).cumsum())
        sig = compute_markov_regime_signal(prices)
        assert sig.transition_probs_horizon == pytest.approx([1.0], abs=0.01) or \
               abs(sum(sig.transition_probs_horizon) - 1.0) < 0.01


# ─── quake score ──

class TestQuakeScore:
    def test_no_quakes_returns_zero(self):
        assert quake_score_from_df(None, pd.Timestamp("2026-04-01")) == 0.0
        assert quake_score_from_df(pd.DataFrame(), pd.Timestamp("2026-04-01")) == 0.0

    def test_small_quake_below_threshold_returns_zero(self):
        df = pd.DataFrame({"date": ["2026-03-30"], "quake_count": [1], "max_magnitude": [5.0]})
        assert quake_score_from_df(df, pd.Timestamp("2026-04-01")) == 0.0

    def test_m6_quake_returns_positive(self):
        df = pd.DataFrame({"date": ["2026-03-30"], "quake_count": [1], "max_magnitude": [6.0]})
        assert quake_score_from_df(df, pd.Timestamp("2026-04-01")) > 0.3

    def test_old_quake_outside_window_returns_zero(self):
        df = pd.DataFrame({"date": ["2026-01-01"], "quake_count": [1], "max_magnitude": [7.0]})
        assert quake_score_from_df(df, pd.Timestamp("2026-04-01")) == 0.0
