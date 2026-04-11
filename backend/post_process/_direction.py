"""Direction metadata computation.

Extracted verbatim from ``backend.stock_analyzer_fixed`` so the pivot /
path-shape / logged-direction derivation lives in one place and can be
unit-tested against fixtures. No algorithmic changes relative to the
inline block it replaces — any observable drift is a bug.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from backend.post_process._shared import direction_from_change_pct


NEUTRAL_META: Dict[str, Any] = {
    'raw_direction': 'NEUTRAL',
    'stable_direction': 'NEUTRAL',
    'logged_direction': 'NEUTRAL',
    'logged_direction_source': 'stable',
    'display_direction': 'NEUTRAL',
    'display_upside_pct': 0.0,
    'stability_note': '',
    'near_term_direction': 'NEUTRAL',
    'day7_direction': 'NEUTRAL',
    'path_shape': 'volatile_flat',
}


def compute_direction_meta(
    display_predictions: List[Dict[str, Any]],
    live_tweak_config: Any,
    runtime_cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """Derive pivot / stability / path-shape metadata for the websocket payload.

    Parameters
    ----------
    display_predictions:
        The prediction list that drives the dashboard view. Pass the
        geo-adjusted series if geo is enabled and applied, otherwise the
        baseline series.
    live_tweak_config:
        ``TweakConfig`` instance whose ``neutral_band_pct`` attribute
        governs the neutral band used for pivot and near-term direction
        classification.
    runtime_cfg:
        Optional runtime config object exposing ``logged_direction_source``
        ("stable" or "raw"). Falls back to the ``LOGGED_DIRECTION_SOURCE``
        environment variable when absent.
    """
    if not display_predictions:
        return dict(NEUTRAL_META)

    neutral_band = float(getattr(live_tweak_config, "neutral_band_pct", 0.0))

    pivot_idx = 6 if len(display_predictions) >= 7 else len(display_predictions) - 1
    pivot_pred = display_predictions[pivot_idx]
    display_upside_pct = float(pivot_pred.get('upside_potential', 0) or 0)
    display_direction = direction_from_change_pct(display_upside_pct, neutral_band_pct=neutral_band)
    raw_direction = direction_from_change_pct(
        float(pivot_pred.get('upside_potential', 0) or 0),
        neutral_band_pct=neutral_band,
    )
    stable_direction = pivot_pred.get('stable_direction', raw_direction)
    logged_direction_source = (
        runtime_cfg.logged_direction_source
        if runtime_cfg is not None
        else os.getenv('LOGGED_DIRECTION_SOURCE', 'stable').strip().lower()
    )
    if logged_direction_source not in {'stable', 'raw'}:
        logged_direction_source = 'stable'
    logged_direction = stable_direction if logged_direction_source == 'stable' else raw_direction
    stability_note = (
        f"Display direction follows adjusted day-7 upside of {display_upside_pct:+.2f}%, "
        f"while stability state remains {stable_direction} for continuity logging."
        if display_direction != stable_direction
        else ""
    )

    near_term_upsides = [
        float(p.get('upside_potential', 0) or 0)
        for p in display_predictions[:min(7, len(display_predictions))]
    ]
    near_term_avg = sum(near_term_upsides) / len(near_term_upsides) if near_term_upsides else 0.0
    near_term_direction = direction_from_change_pct(near_term_avg, neutral_band_pct=neutral_band)
    day7_direction = display_direction

    if len(display_predictions) >= 14:
        later_upsides = [
            float(p.get('upside_potential', 0) or 0)
            for p in display_predictions[7:14]
        ]
        later_avg = sum(later_upsides) / len(later_upsides) if later_upsides else 0.0
    else:
        later_avg = near_term_avg

    if near_term_avg < -1.0 and later_avg > 1.0:
        path_shape = "near_term_drop_then_recover"
    elif near_term_avg > 1.0 and later_avg < -1.0:
        path_shape = "near_term_rise_then_decline"
    elif near_term_avg < -1.0 and later_avg < -1.0:
        path_shape = "steady_decline"
    elif near_term_avg > 1.0 and later_avg > 1.0:
        path_shape = "steady_rise"
    else:
        path_shape = "volatile_flat"

    return {
        'raw_direction': raw_direction,
        'stable_direction': stable_direction,
        'logged_direction': logged_direction,
        'logged_direction_source': logged_direction_source,
        'display_direction': display_direction,
        'display_upside_pct': round(display_upside_pct, 2),
        'stability_note': stability_note,
        'near_term_direction': near_term_direction,
        'day7_direction': day7_direction,
        'path_shape': path_shape,
    }
