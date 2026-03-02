"""
Typed runtime configuration – single source of truth for all feature flags.

Every env-driven knob lives here so callers import one object instead of
scattering os.getenv() across a dozen modules.

Usage:
    from backend.runtime_config import get_runtime_config
    cfg = get_runtime_config()
    if cfg.enable_geo_features:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Literal


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str, allowed: tuple[str, ...] | None = None) -> str:
    val = os.getenv(name, default).strip().lower()
    if allowed and val not in allowed:
        return default
    return val


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable snapshot of all runtime flags for one request lifecycle."""

    # --- Model variant (rollout) ---
    model_variant: str = "baseline"  # baseline | shadow | upgraded

    # --- Geopolitical features ---
    enable_geo_features: bool = False

    # --- Sentiment adjustment mode ---
    sentiment_adjust_mode: str = "legacy"  # legacy | date_aware

    # --- TradingView cache ---
    tradingview_cache_ttl_min: int = 60

    # --- News / index recall ---
    enable_index_news_recall: bool = True
    enable_index_recall_in_model: bool = False

    # --- Prediction tweaks ---
    prediction_tweaks_enabled: bool = False
    tweak_neutral_band_pct: float = 1.0
    tweak_min_confidence: float = 0.82
    tweak_williams_brake: bool = True
    tweak_max_upside_cap_pct: float = 6.0
    tweak_max_downside_cap_pct: float = -6.0
    tweak_bias_correction_pct: float = -1.5

    # --- Direction logging ---
    logged_direction_source: str = "stable"  # stable | raw

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_shadow(self) -> bool:
        return self.model_variant == "shadow"

    @property
    def is_upgraded(self) -> bool:
        return self.model_variant == "upgraded"

    @property
    def is_baseline(self) -> bool:
        return self.model_variant == "baseline"


def load_runtime_config() -> RuntimeConfig:
    """Build config from current environment variables."""
    return RuntimeConfig(
        model_variant=_env_str(
            "MODEL_VARIANT", "baseline", ("baseline", "shadow", "upgraded")
        ),
        enable_geo_features=_env_flag("ENABLE_GEO_FEATURES", False),
        sentiment_adjust_mode=_env_str(
            "SENTIMENT_ADJUST_MODE", "legacy", ("legacy", "date_aware")
        ),
        tradingview_cache_ttl_min=_env_int("TRADINGVIEW_CACHE_TTL_MIN", 60),
        enable_index_news_recall=_env_flag("ENABLE_INDEX_NEWS_RECALL", True),
        enable_index_recall_in_model=_env_flag("ENABLE_INDEX_RECALL_IN_MODEL", False),
        prediction_tweaks_enabled=_env_flag("PREDICTION_TWEAKS_ENABLED", False),
        tweak_neutral_band_pct=_env_float("PRED_TWEAK_NEUTRAL_BAND_PCT", 1.0),
        tweak_min_confidence=_env_float("PRED_TWEAK_MIN_CONFIDENCE", 0.82),
        tweak_williams_brake=_env_flag("PRED_TWEAK_WILLIAMS_BRAKE", True),
        tweak_max_upside_cap_pct=_env_float("PRED_TWEAK_MAX_UPSIDE_CAP_PCT", 6.0),
        tweak_max_downside_cap_pct=_env_float("PRED_TWEAK_MAX_DOWNSIDE_CAP_PCT", -6.0),
        tweak_bias_correction_pct=_env_float("PRED_TWEAK_BIAS_CORRECTION_PCT", -1.5),
        logged_direction_source=_env_str(
            "LOGGED_DIRECTION_SOURCE", "stable", ("stable", "raw")
        ),
    )


# Module-level singleton (re-reads env on import; cheap to re-call).
_cached_config: RuntimeConfig | None = None


def get_runtime_config(*, force_reload: bool = False) -> RuntimeConfig:
    """Return a cached RuntimeConfig. Pass force_reload=True to re-read env."""
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = load_runtime_config()
    return _cached_config
