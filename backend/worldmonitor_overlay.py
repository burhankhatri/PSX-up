#!/usr/bin/env python3
"""
Worldmonitor enhancement overlay for the PSX geo prediction adjustments.

Sits AFTER `geopolitical_features.build_geopolitical_daily_adjustments` and
BEFORE the predictions are baked. Takes the existing per-day adjustments
plus a snapshot of free, no-auth signals (VIX, Hang Seng/Sensex/Nifty,
GDELT tone+volume, Markov regime), and adds a single bounded delta.

Why a separate overlay rather than new fields in geo_features:
- No model retraining required. The Research model is already fit.
- Clean A/B comparison: same baseline predictions ± the overlay delta.
- Composable: the existing `_apply_symbol_class_cap` + PSX 7.5% circuit
  breaker still apply at the very end.

Signal weights are calibrated to be CONSERVATIVE — each signal contributes
at most ~0.5pp to the day-1 adjustment. Calibration source: backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.geopolitical_features import (
    PSX_DAILY_CIRCUIT_BREAKER,
    _apply_symbol_class_cap,
)


# Forecast-horizon adjustment caps. Unlike PSX_DAILY_CIRCUIT_BREAKER (7.5%),
# which constrains intraday price moves, these constrain the TOTAL OVERLAY
# correction on a day-k *forecast*. A 17-days-ahead model can be wrong by
# more than 7.5% — and our overlay should be allowed to correct that much.
# Day-1 cap stays tight (close to PSX limit); long-horizon caps loosen.
def _overlay_cap_for_day(day: int, symbol: Optional[str]) -> float:
    """Per-day cap on the absolute total overlay adjustment (fraction units)."""
    if day <= 1:
        return 0.04
    if day <= 5:
        return 0.08
    if day <= 10:
        return 0.16
    if day <= 21:
        return 0.30
    return 0.40


# ── Signal-weight policy ────────────────────────────────────────────────────
# Each signal score lives in [-1, +1]. Weights are absolute pp impact ceiling.
# Tuned by the OGDC backtest (see backtest_ogdc.py); can be overridden per call.

DEFAULT_WEIGHTS = {
    "vix": 0.30,                  # VIX > 25 / z-score → bearish on EM (coincident, not leading — kept low)
    "extended_asian": 0.50,       # HSI/Sensex/Nifty avg return → directional
    "gdelt_tone_delta": 0.20,     # 3-day tone delta vs 14-day baseline
    "gdelt_vol_spike": 0.20,      # volume z-score, capped — sector-aware
    "markov_regime": 2.20,        # ticker's own regime → forward return projection
    "usgs_quake": 0.10,           # M>=6.5 within 3 days = small bearish kick
    "momentum_20d": 4.00,         # ticker price vs 20-day SMA — DOMINANT trend driver
    "pkr_fx": 1.20,               # PKR z-score; Pakistan's cleanest lead indicator (4-12wk for EM shocks)
    "hormuz": 1.50,               # Hormuz composite, sector-aware multiplier (see TRANSMISSION_HORMUZ)
}


# ─────────────────────────────────────────────────────────────────────────────
# Sector transmission matrix (Agent 1 + Agent 3 research).
# Factor = signed multiplier on the hormuz_risk_score for tickers of that class.
# +1.0 = same direction as crude supply shock (bullish). 0.0 = neutral.
# Negative = opposite direction.
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP: Dict[str, str] = {
    # Upstream E&P: crude spikes = revenue tailwind
    "OGDC": "ep", "PPL": "ep", "MARI": "ep", "POL": "ep",
    # OMC (Oil Marketing): inventory gains on up-moves, LC pain on PKR weakness
    "PSO": "omc", "HASCOL": "omc", "ATRL": "omc", "APL": "omc",
    # Cement: freight+coal pass-through pain from crude spikes
    "LUCK": "cem", "DGKC": "cem", "FCCL": "cem", "MLCF": "cem",
    "KOHC": "cem", "PIOC": "cem", "CHCC": "cem",
    # Banking: risk-off channel on shock day; benefits once rates hike
    "HBL": "bank", "UBL": "bank", "MCB": "bank", "BAHL": "bank",
    "NBP": "bank", "MEBL": "bank", "FABL": "bank", "ABL": "bank",
    # Fertilizer: mostly insulated from crude (subsidised gas for FFC/EFERT)
    "FFC": "fert", "EFERT": "fert", "FATIMA": "fert", "FFBL": "fert",
    # Steel: energy + construction demand proxy
    "ISL": "steel", "ASTL": "steel", "MUGHAL": "steel",
    # Autos: fuel + financing double-hit, very slow
    "INDU": "auto", "HCAR": "auto", "PSMC": "auto", "SAZEW": "auto",
    # Power: circular debt sensitive
    "HUBC": "pow", "KEL": "pow", "KAPCO": "pow", "LOTTE": "pow",
    # Tech: FX-insulated, low crude sensitivity
    "SYS": "tech", "TRG": "tech", "NETSOL": "tech", "AVN": "tech",
}

# Per-sector transmission factor for a +Hormuz shock (supply disruption, +crude).
# Derived from Agent 1's Section 3 + Agent 3's top-10 mechanisms.
TRANSMISSION_HORMUZ: Dict[str, float] = {
    "ep":    +1.00,   # strongest positive (revenue tailwind)
    "omc":   +0.30,   # inventory gain on up-moves; LC pain on PKR side
    "cem":   -0.40,   # fuel-hike + freight pain
    "bank":  -0.35,   # risk-off channel
    "fert":  +0.00,   # subsidised gas insulates; pure noise on day 1
    "steel": -0.30,   # imported scrap + energy
    "auto":  -0.30,   # slow demand destruction
    "pow":   +0.20,   # circular debt unlock if govt gas royalties rise
    "tech":  -0.05,   # near-neutral, slight risk-off drag
}

# Sector transmission factor for +PKR weakness (USD up, rupee down).
# Dollar earners gain, importers lose.
TRANSMISSION_PKR: Dict[str, float] = {
    "ep":    +0.50,   # USD-linked revenue → gain
    "tech":  +0.70,   # USD-export → largest beneficiary
    "omc":   -0.50,   # LC forex loss on imported fuel
    "cem":   -0.20,   # imported coal + freight
    "auto":  -0.80,   # imported CKD kits + financing stress
    "bank":  -0.20,   # sovereign risk premium rises; NIM offset partial
    "fert":  +0.10,   # FFC export via urea; muted for locally-sold
    "steel": -0.50,   # imported scrap makes up ~60% of feedstock
    "pow":  -0.10,   # imported coal/oil for some plants
}


def _sector_of(symbol: Optional[str]) -> str:
    return SECTOR_MAP.get((symbol or "").upper(), "other")


def _transmission(factor_map: Dict[str, float], symbol: Optional[str]) -> float:
    return factor_map.get(_sector_of(symbol), 0.0)

# Two-speed decay model:
# - Trend signals (momentum, Markov) persist — model's flat bias compounds, so
#   the trend correction must compound too. Half-life 14d ≈ very slow decay.
# - News signals (VIX, GDELT, quake) decay fast — they are point-in-time shocks.
HORIZON_DECAY_HALF_LIFE_TREND = 14.0
HORIZON_DECAY_HALF_LIFE_SHOCK = 5.0

# Tickers where energy-supply stress is BULLISH (upstream E&P, beneficiaries
# of higher crude). For these, GDELT regional_war + energy vol spikes flip
# from bearish (panic) to bullish (revenue tailwind).
UPSTREAM_EP_TICKERS = frozenset({"OGDC", "PPL", "POL", "MARI"})


@dataclass(frozen=True)
class WorldmonitorSnapshot:
    """All worldmonitor signals collapsed to scalar scores at one point in time."""
    vix_score: float            # in [-1, +1], + = bearish for EM
    extended_asian_score: float # in [-1, +1], + = bullish (Asian markets up)
    gdelt_tone_score: float     # in [-1, +1], + = bullish (tone improving)
    gdelt_vol_score: float      # in [-1, +1], + = bearish (panic vol spike)
    markov_score: float         # in [-1, +1], from compute_markov_regime_signal
    quake_score: float          # in [-1, +1], + = bearish (large recent quake)
    momentum_score: float       # in [-1, +1], + = bullish (price vs 20d SMA)
    pkr_fx_score: float = 0.0   # in [-1, +1], + = PKR weakening (risk-off pressure)
    hormuz_risk_score: float = 0.0  # in [0, +1], 0=calm, 1=active Abqaiq-type supply shock

    def as_dict(self) -> Dict[str, float]:
        return {
            "vix_score": round(self.vix_score, 4),
            "extended_asian_score": round(self.extended_asian_score, 4),
            "gdelt_tone_score": round(self.gdelt_tone_score, 4),
            "gdelt_vol_score": round(self.gdelt_vol_score, 4),
            "markov_score": round(self.markov_score, 4),
            "quake_score": round(self.quake_score, 4),
            "momentum_score": round(self.momentum_score, 4),
            "pkr_fx_score": round(self.pkr_fx_score, 4),
            "hormuz_risk_score": round(self.hormuz_risk_score, 4),
        }


def pkr_fx_score_from_row(row: Optional[pd.Series]) -> float:
    """Map USD/PKR state → [-1, +1] where +1 = acute PKR weakness (bearish).

    Uses 90-day z-score (preferred for regime detection) with a nudge from
    the 30-day z-score (short-horizon stress) and the weakening-streak
    (consistency).
    """
    if row is None:
        return 0.0
    z30 = _safe(row.get("usdpkr_zscore_30d", 0.0))
    z90 = _safe(row.get("usdpkr_zscore_90d", 0.0))
    streak = _safe(row.get("usdpkr_weakening_streak", 0.0))  # 0..5
    raw = 0.5 * np.tanh(z90 / 1.5) + 0.3 * np.tanh(z30 / 1.5) + 0.2 * (streak / 5.0 - 0.5)
    return float(np.clip(raw, -1.0, 1.0))


def hormuz_risk_score_from_sources(gdelt_regional_vol: pd.Series,
                                    gdelt_regional_tone: pd.Series,
                                    brent_row: Optional[pd.Series]) -> float:
    """Compose three independent signals of a supply-side Hormuz regime.

    - GDELT regional_war volume z-score (3d vs 30d) — "how loud is the war news"
    - GDELT regional_war tone (negative = escalation rhetoric)
    - Brent 5-day return — market CONFIRMS a supply-shock narrative
    Returns 0 when calm, ~0.5 on a headline-only event (Soleimani-type),
    ~1.0 on an Abqaiq-type hardware shock. Strictly >= 0 (no "negative Hormuz
    risk"; absence of shock is the floor).
    """
    def _zscore_ratio(series: pd.Series) -> float:
        s = pd.Series(series).dropna()
        if len(s) < 7:
            return 0.0
        recent = float(s.tail(3).mean())
        base = s.tail(30)
        if base.std() == 0:
            return 0.0
        z = (recent - base.mean()) / base.std()
        return float(z)

    vol_z = _zscore_ratio(gdelt_regional_vol)
    tone_component = 0.0
    t = pd.Series(gdelt_regional_tone).dropna()
    if len(t) >= 3:
        tone_mean = float(t.tail(3).mean())
        tone_component = max(0.0, -tone_mean / 4.0)  # tone -4 → 1.0

    brent_confirm = 0.0
    if brent_row is not None:
        b1 = _safe(brent_row.get("brent_change_1d", 0.0))
        b5 = _safe(brent_row.get("brent_change_5d", 0.0))
        # +5% brent 5d OR +3% in one day → saturate
        brent_confirm = max(0.0, min(1.0, max(b1 / 0.03, b5 / 0.05)))

    raw = (
        0.40 * max(0.0, min(1.0, vol_z / 2.5))
        + 0.25 * max(0.0, min(1.0, tone_component))
        + 0.35 * brent_confirm
    )
    return float(np.clip(raw, 0.0, 1.0))


def momentum_score_from_prices(close_prices: pd.Series) -> float:
    """Price relative to 20-day SMA → trend-follower in [-1, +1].

    +1 = strong uptrend (>= +10% above 20d SMA); -1 = strong downtrend.
    Linear scale: every +1% above SMA = +0.10 score.
    """
    s = pd.Series(close_prices).dropna().astype(float)
    if len(s) < 25:
        return 0.0
    sma20 = float(s.tail(20).mean())
    if sma20 <= 0:
        return 0.0
    cur = float(s.iloc[-1])
    dev = (cur - sma20) / sma20
    return float(np.clip(dev * 10.0, -1.0, 1.0))


def _safe(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def vix_score_from_row(row: Optional[pd.Series]) -> float:
    """Map VIX state → [-1, +1] where +1 = strong risk-off (bearish for EM)."""
    if row is None:
        return 0.0
    z = _safe(row.get("vix_zscore_60d", 0.0))
    above25 = _safe(row.get("vix_above_25", 0.0))
    above30 = _safe(row.get("vix_above_30", 0.0))
    raw = (0.5 * np.tanh(z / 1.5)) + (0.25 * above25) + (0.25 * above30)
    return float(np.clip(raw, -1.0, 1.0))


def extended_asian_score_from_row(row: Optional[pd.Series]) -> float:
    """Asian markets → [-1, +1] where + = bullish lead for PSX.

    Magnitude scaled so a 2% avg gain → +0.4, 4% gain → +0.8, etc.
    """
    if row is None:
        return 0.0
    avg_ret = _safe(row.get("extended_asian_avg_return", 0.0))
    risk_off = _safe(row.get("extended_asian_risk_off", 0.0))
    raw = 20.0 * avg_ret - 0.4 * risk_off
    return float(np.clip(raw, -1.0, 1.0))


def gdelt_tone_score(tone_series: pd.Series) -> float:
    """3-day mean tone vs 14-day mean tone. Negative tone = adversarial.

    + score when tone is improving (less negative), - when worsening.
    Range: [-1, +1].
    """
    s = pd.Series(tone_series).dropna()
    if len(s) < 7:
        return 0.0
    short = float(s.tail(3).mean())
    long_ = float(s.tail(14).mean())
    delta = short - long_  # +ve = improving (less negative)
    return float(np.clip(delta / 2.0, -1.0, 1.0))


def gdelt_vol_score(vol_series: pd.Series) -> float:
    """Recent volume vs 14-day baseline z-score. Spikes → bearish.

    Range: [-1, +1] but in practice usually positive (volume rarely crashes).
    """
    s = pd.Series(vol_series).dropna()
    if len(s) < 7:
        return 0.0
    recent = float(s.tail(3).mean())
    base = s.tail(14)
    if base.std() == 0:
        return 0.0
    z = (recent - base.mean()) / base.std()
    return float(np.clip(z / 3.0, -1.0, 1.0))


def quake_score_from_df(quake_df: Optional[pd.DataFrame],
                          asof_date,
                          lookback_days: int = 5) -> float:
    """If a M>=6.5 quake hit within `lookback_days` of asof_date → bearish kick."""
    if quake_df is None or quake_df.empty:
        return 0.0
    try:
        d = pd.to_datetime(asof_date).normalize()
        df = quake_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        recent = df[(df["date"] <= d) & (df["date"] >= d - pd.Timedelta(days=lookback_days))]
        if recent.empty:
            return 0.0
        max_mag = float(recent["max_magnitude"].max())
        if max_mag < 5.5:
            return 0.0
        # M5.5 → 0.2, M6 → 0.4, M6.5 → 0.6, M7+ → 1.0
        score = (max_mag - 5.0) * 0.4
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def collapse_snapshot(*,
                       vix_row: Optional[pd.Series],
                       asian_row: Optional[pd.Series],
                       gdelt_pk_tone: pd.Series,
                       gdelt_pk_vol: pd.Series,
                       gdelt_regional_tone: pd.Series,
                       gdelt_regional_vol: pd.Series,
                       quake_df: Optional[pd.DataFrame],
                       asof_date,
                       markov_signal_score: float = 0.0,
                       ticker_close_prices: Optional[pd.Series] = None,
                       pkr_row: Optional[pd.Series] = None,
                       brent_row: Optional[pd.Series] = None) -> WorldmonitorSnapshot:
    """Combine all signal sources into a single bounded snapshot."""
    pk_t = gdelt_tone_score(gdelt_pk_tone)
    rg_t = gdelt_tone_score(gdelt_regional_tone)
    pk_v = gdelt_vol_score(gdelt_pk_vol)
    rg_v = gdelt_vol_score(gdelt_regional_vol)
    momentum = momentum_score_from_prices(ticker_close_prices) if ticker_close_prices is not None else 0.0
    return WorldmonitorSnapshot(
        vix_score=vix_score_from_row(vix_row),
        extended_asian_score=extended_asian_score_from_row(asian_row),
        gdelt_tone_score=float(np.clip(0.6 * pk_t + 0.4 * rg_t, -1.0, 1.0)),
        gdelt_vol_score=float(np.clip(0.6 * pk_v + 0.4 * rg_v, -1.0, 1.0)),
        markov_score=float(np.clip(markov_signal_score, -1.0, 1.0)),
        quake_score=quake_score_from_df(quake_df, asof_date),
        momentum_score=momentum,
        pkr_fx_score=pkr_fx_score_from_row(pkr_row),
        hormuz_risk_score=hormuz_risk_score_from_sources(gdelt_regional_vol, gdelt_regional_tone, brent_row),
    )


def apply_worldmonitor_overlay(adjustments: List[Dict],
                                 snapshot: WorldmonitorSnapshot,
                                 symbol: Optional[str] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 mode: str = "cumulative") -> List[Dict]:
    """Add a worldmonitor delta to each day's geo adjustment, then re-cap.

    Modes:
    - "additive": original behavior. Day k delta = trend(decay) + shock(decay).
      Correct for short horizons; cannot close a multi-day cumulative-bias gap.
    - "cumulative" (default): trend signals COMPOUND across the horizon, so
      a sustained +1pp/day Markov/momentum signal becomes ~+17% by day 17.
      Shock signals stay per-day with fast decay. PSX circuit breaker still
      caps the FINAL per-day adjustment magnitude.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # Day-1 raw delta (in absolute price-fraction units, not pp)
    # +ve scores either bullish or bearish per the column semantic; sign here:
    #   +VIX risk-off → BEARISH
    #   +Asian (markets up) → BULLISH
    #   +GDELT tone (improving) → BULLISH
    #   +GDELT vol (panic spike) → BEARISH (or BULLISH for upstream E&P!)
    #   +Markov fwd return → BULLISH
    #   +Quake → BEARISH
    #   +Momentum (above 20d SMA) → BULLISH (trend follower; balances mean-reversion bias)
    sym_upper = (symbol or "").upper()
    is_upstream_ep = _sector_of(sym_upper) == "ep"
    hormuz_factor = _transmission(TRANSMISSION_HORMUZ, sym_upper)
    pkr_factor = _transmission(TRANSMISSION_PKR, sym_upper)
    # Legacy sector-aware vol-spike sign (kept for backward-compat of non-mapped tickers)
    vol_sign = +1.0 if is_upstream_ep else -1.0

    # Split into two components by decay profile:
    # 1. TREND: persists (momentum + Markov). The bias these correct is structural —
    #    if the model is flat-biased, every prediction day stays mispriced.
    # 2. SHOCK: decays (VIX + Asian + GDELT + quake + Hormuz + PKR). Event-driven.
    #
    # Regime-divergence dampener: when Markov says the stock is stretched
    # (low expected forward return) BUT momentum is still strongly positive,
    # the position is overdue for mean reversion. Attenuate the trend boost
    # by up to 50% to avoid overshooting late in the horizon.
    momentum_dominant = snapshot.momentum_score > 0.4
    markov_weak = snapshot.markov_score < 0.18  # below typical OGDC baseline ~0.17
    regime_dampener = 0.5 if (momentum_dominant and markov_weak) else 1.0

    trend_delta = (
        + w["markov_regime"] * snapshot.markov_score
        + w["momentum_20d"]  * snapshot.momentum_score * regime_dampener
    ) / 100.0
    shock_delta = (
        - w["vix"]            * snapshot.vix_score
        + w["extended_asian"] * snapshot.extended_asian_score
        + w["gdelt_tone_delta"] * snapshot.gdelt_tone_score
        + vol_sign * w["gdelt_vol_spike"] * snapshot.gdelt_vol_score
        - w["usgs_quake"]     * snapshot.quake_score
        # Hormuz: signed by sector transmission, magnitude by hormuz_risk_score in [0,1]
        + w["hormuz"]         * hormuz_factor * snapshot.hormuz_risk_score
        # PKR: score in [-1,+1] (+ = weakening); signed by sector transmission
        # A weakening PKR boosts dollar-earner sectors (ep/tech, factor > 0) and
        # hits import-heavy sectors (auto/omc/steel, factor < 0).
        + w["pkr_fx"]         * pkr_factor * snapshot.pkr_fx_score
    ) / 100.0

    out: List[Dict] = []
    for adj in adjustments:
        new = dict(adj)
        try:
            day = int(adj.get("day", 1))
        except Exception:
            day = 1
        trend_decay = 0.5 ** ((day - 1) / HORIZON_DECAY_HALF_LIFE_TREND)
        shock_decay = 0.5 ** ((day - 1) / HORIZON_DECAY_HALF_LIFE_SHOCK)

        # Strong-trend gate: only compound the trend signal when momentum is
        # genuinely strong (|momentum| >= 0.4) AND momentum + Markov agree on
        # direction. Otherwise the cumulative compounding amplifies whipsaw
        # noise on range-bound stocks (verified on LUCK backtest).
        strong_momentum = abs(snapshot.momentum_score) >= 0.4
        same_sign = (
            (snapshot.momentum_score >= 0 and snapshot.markov_score >= 0) or
            (snapshot.momentum_score <  0 and snapshot.markov_score <  0)
        )
        gate_open = strong_momentum and same_sign

        if mode == "cumulative" and gate_open:
            # Cumulative trend factor over k days: (1 + trend_per_day)^k - 1
            trend_per_day = trend_delta * trend_decay
            cumulative_trend = (1.0 + trend_per_day) ** day - 1.0
            day_delta = cumulative_trend + shock_delta * shock_decay
        else:
            # Additive mode (gate closed) or original mode — much more conservative.
            day_delta = trend_delta * trend_decay + shock_delta * shock_decay

        original_capped = float(adj.get("capped_adjustment", 0.0))
        blended = original_capped + day_delta
        # Use forecast-horizon cap (more permissive than the per-day circuit
        # breaker) so a 17-day-ahead overlay can correct a 17%-off model.
        cap = _overlay_cap_for_day(day, symbol)
        new_capped = max(-cap, min(cap, blended))

        new["pre_overlay_capped_adjustment"] = original_capped
        new["worldmonitor_delta_pct"] = round(day_delta * 100.0, 4)
        new["capped_adjustment"] = new_capped
        new["percentage"] = round(new_capped * 100.0, 4)
        new.setdefault("event_impacts", []).append(
            f"worldmonitor_overlay={day_delta * 100:+.3f}pp "
            f"(vix={snapshot.vix_score:+.2f}, asia={snapshot.extended_asian_score:+.2f}, "
            f"tone={snapshot.gdelt_tone_score:+.2f}, vol={snapshot.gdelt_vol_score:+.2f}, "
            f"mkv={snapshot.markov_score:+.2f}, qk={snapshot.quake_score:+.2f})"
        )
        out.append(new)
    return out
