"""Shared helpers for the post_process pipeline.

Keep this file tiny. It exists so that `direction_from_change_pct` has exactly
one authoritative definition inside the package. The legacy copies in
``backend.prediction_tuning`` and ``backend.prediction_reasoning`` remain for
now and delegate here will happen in a later migration step.
"""

from __future__ import annotations


def direction_from_change_pct(change_pct: float, neutral_band_pct: float = 0.0) -> str:
    if abs(change_pct) <= neutral_band_pct:
        return "NEUTRAL"
    return "BULLISH" if change_pct > 0 else "BEARISH"
