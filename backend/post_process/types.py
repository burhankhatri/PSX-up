"""Dataclasses for the post_process pipeline.

Nothing here is wired into the orchestrator yet — these types define the
target shape the migration will converge on. The ``ws_payload`` property on
``PredictionResult`` is the mechanical backwards-compatibility guarantee:
the websocket handler in ``backend.stock_analyzer_fixed`` forwards exactly
these keys today, and ``ws_payload`` will become the single place that
knowledge lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class RunOverrides:
    tweak_config: Any = None
    enable_tuning: bool = True
    enable_stability: bool = True
    enable_logging: bool = True
    enable_reasoning: bool = True


@dataclass(frozen=True)
class PredictionRequest:
    symbol: str
    raw_df: pd.DataFrame
    feature_df: pd.DataFrame
    current_price: float
    baseline_predictions: list[dict]
    geo_predictions: Optional[list[dict]]
    analysis_id: str
    generated_at: datetime
    horizon_label: str = "Day 7"
    overrides: Optional[RunOverrides] = None


@dataclass(frozen=True)
class PredictionResult:
    monthly_predictions: list[dict]
    daily_predictions: list[dict]
    daily_predictions_without_geo: list[dict]
    daily_predictions_with_geo: list[dict]
    direction_meta: dict[str, Any]
    near_term_direction: str
    day7_direction: str
    path_shape: str
    prediction_reasoning: dict[str, Any]
    tuning: dict[str, Any]

    @property
    def ws_payload(self) -> dict[str, Any]:
        """Exact set of keys the websocket handler forwards to the dashboard.

        Changing this shape is a breaking change for the frontend. Any update
        here must be paired with a corresponding change in the dashboard and
        a snapshot-test refresh.
        """
        return {
            "direction_meta": self.direction_meta,
            "near_term_direction": self.near_term_direction,
            "day7_direction": self.day7_direction,
            "path_shape": self.path_shape,
            "monthly_predictions": self.monthly_predictions,
            "daily_predictions": self.daily_predictions,
            "daily_predictions_without_geo": self.daily_predictions_without_geo,
            "daily_predictions_with_geo": self.daily_predictions_with_geo,
            "prediction_reasoning": self.prediction_reasoning,
            "tuning": self.tuning,
        }
