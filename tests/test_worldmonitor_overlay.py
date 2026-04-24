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


# ─── fetch_snapshot_for_weights: the hot-path speedup ─────────────────────────
# The overlay block used to fire 7 synchronous network calls every analyze_stock
# invocation, even though ablation zeroed 6 of 9 weights. These tests pin the
# gating + budget + fail-open contract so we don't regress back to a 15s hang.

from backend.worldmonitor_overlay import fetch_snapshot_for_weights


class TestFetchSnapshotGating:
    """Prove that zero-weighted signals skip their network fetch."""

    def _mk_call_tracker(self):
        calls = []
        def make(name):
            def f():
                calls.append(name)
                return pd.DataFrame()
            return f
        fake = {name: make(name) for name in
                ("vix", "extended_asian", "gdelt_pk", "gdelt_regional",
                 "quakes", "pkr", "brent")}
        return calls, fake

    def test_default_weights_call_only_hormuz_inputs(self):
        """Default DEFAULT_WEIGHTS zero 6 of 9 → only brent + gdelt_regional fetch."""
        calls, fake = self._mk_call_tracker()
        prices = pd.Series(np.linspace(100, 110, 260))
        snap = fetch_snapshot_for_weights(
            symbol="OGDC", close_prices=prices, fetchers=fake,
        )
        assert snap is not None
        # Only hormuz-relevant fetchers should have fired.
        assert set(calls) == {"gdelt_regional", "brent"}

    def test_all_external_weights_zero_calls_nothing(self):
        """When even hormuz=0, no fetchers are called; local-only snapshot."""
        calls, fake = self._mk_call_tracker()
        w = dict(DEFAULT_WEIGHTS)
        for k in ("hormuz", "vix", "extended_asian", "gdelt_tone_delta",
                  "gdelt_vol_spike", "usgs_quake", "pkr_fx"):
            w[k] = 0.0
        prices = pd.Series(np.linspace(100, 110, 260))
        snap = fetch_snapshot_for_weights(
            symbol="OGDC", close_prices=prices, weights=w, fetchers=fake,
        )
        assert snap is not None
        assert calls == []

    def test_all_weights_nonzero_calls_every_fetcher(self):
        """If user override turns every signal back on, every fetcher runs."""
        calls, fake = self._mk_call_tracker()
        w = {k: 1.0 for k in DEFAULT_WEIGHTS}
        prices = pd.Series(np.linspace(100, 110, 260))
        snap = fetch_snapshot_for_weights(
            symbol="OGDC", close_prices=prices, weights=w, fetchers=fake,
        )
        assert snap is not None
        assert set(calls) == {
            "vix", "extended_asian", "gdelt_pk", "gdelt_regional",
            "quakes", "pkr", "brent",
        }

    def test_fetcher_exception_fails_open(self):
        """A raising fetcher must not crash the overlay — snapshot still returns."""
        def boom():
            raise RuntimeError("network dead")
        fake = {k: boom for k in
                ("vix", "extended_asian", "gdelt_pk", "gdelt_regional",
                 "quakes", "pkr", "brent")}
        prices = pd.Series(np.linspace(100, 110, 260))
        w = {k: 1.0 for k in DEFAULT_WEIGHTS}
        snap = fetch_snapshot_for_weights(
            symbol="OGDC", close_prices=prices, weights=w, fetchers=fake,
        )
        assert snap is not None
        # Momentum is local-only — should still be valid.
        assert isinstance(snap.momentum_score, float)


class TestFetchSnapshotBudget:
    """The budget must kill a slow fetcher instead of blocking the prediction."""

    def test_slow_fetcher_is_cut_by_budget(self):
        import time as _t

        def slow():
            _t.sleep(2.0)
            return pd.DataFrame()

        def fast():
            return pd.DataFrame()

        fake = {k: fast for k in
                ("vix", "extended_asian", "gdelt_pk", "quakes", "pkr", "brent")}
        fake["gdelt_regional"] = slow  # hormuz input hangs

        prices = pd.Series(np.linspace(100, 110, 260))
        w = {k: 1.0 for k in DEFAULT_WEIGHTS}

        t0 = _t.time()
        snap = fetch_snapshot_for_weights(
            symbol="OGDC", close_prices=prices, weights=w, fetchers=fake,
            budget_seconds=0.3,
        )
        elapsed = _t.time() - t0

        # Must NOT have waited the full 2s for slow().
        assert elapsed < 1.5, f"budget ignored, took {elapsed:.2f}s"
        assert snap is not None  # partial snapshot returned

    def test_default_weights_budget_is_trivial(self):
        """Under default weights, only 2 fast fetchers run — well under 1s."""
        import time as _t

        def fast():
            return pd.DataFrame()

        fake = {k: fast for k in
                ("vix", "extended_asian", "gdelt_pk", "gdelt_regional",
                 "quakes", "pkr", "brent")}
        prices = pd.Series(np.linspace(100, 110, 260))

        t0 = _t.time()
        snap = fetch_snapshot_for_weights(
            symbol="OGDC", close_prices=prices, fetchers=fake,
        )
        elapsed = _t.time() - t0

        assert snap is not None
        assert elapsed < 1.0, f"default path too slow: {elapsed:.2f}s"


# ─── yfinance disk cache ──────────────────────────────────────────────────────
# yfinance used to fetch on every process start (no persistent cache). These
# tests pin the TTL-based CSV cache so warm-path is ~10ms instead of 7-14s.

import os
import time as _time_mod

from backend.worldmonitor_signals import _read_df_cache, _write_df_cache


class TestYfinanceDiskCache:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.worldmonitor_signals.CACHE_DIR", tmp_path)
        assert _read_df_cache("nope.csv", max_age_seconds=3600) is None

    def test_write_then_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.worldmonitor_signals.CACHE_DIR", tmp_path)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "x": [1.5, 2.5],
        })
        _write_df_cache("roundtrip.csv", df)
        out = _read_df_cache("roundtrip.csv", max_age_seconds=3600)
        assert out is not None
        assert len(out) == 2
        assert float(out["x"].iloc[1]) == 2.5
        # date should survive roundtrip as datetime
        assert pd.api.types.is_datetime64_any_dtype(out["date"])

    def test_expired_cache_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.worldmonitor_signals.CACHE_DIR", tmp_path)
        df = pd.DataFrame({"x": [1.0]})
        _write_df_cache("expired.csv", df)
        old = _time_mod.time() - 7200  # 2 hours old
        os.utime(tmp_path / "expired.csv", (old, old))
        assert _read_df_cache("expired.csv", max_age_seconds=3600) is None

    def test_write_failure_is_silent(self, tmp_path, monkeypatch):
        """A full disk / readonly cache dir must not crash the overlay."""
        bad_dir = tmp_path / "does" / "not" / "exist"
        monkeypatch.setattr("backend.worldmonitor_signals.CACHE_DIR", bad_dir)
        # Should not raise.
        _write_df_cache("x.csv", pd.DataFrame({"a": [1]}))


class TestYfinanceFetchersUseCache:
    """The four yfinance-backed fetchers must read/write the CSV cache."""

    def test_fetch_vix_hits_cache_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.worldmonitor_signals.CACHE_DIR", tmp_path)
        call_count = {"n": 0}

        class _FakeYF:
            @staticmethod
            def download(*args, **kwargs):
                call_count["n"] += 1
                idx = pd.date_range("2026-01-01", periods=3)
                return pd.DataFrame({
                    "Close": [20.0, 21.0, 22.0],
                    "Open": [20.0, 21.0, 22.0],
                }, index=idx)

        monkeypatch.setattr("backend.worldmonitor_signals.yf", _FakeYF, raising=False)
        monkeypatch.setattr("backend.worldmonitor_signals.YF_OK", True, raising=False)

        from backend.worldmonitor_signals import fetch_vix
        a = fetch_vix(period="3mo")
        b = fetch_vix(period="3mo")
        assert not a.empty
        assert not b.empty
        assert call_count["n"] == 1, "second call should have hit disk cache"
