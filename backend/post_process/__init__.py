"""Deep module unifying prediction post-processing.

This package is the target of the refactor described in
``docs/plans/RFC_prediction_post_process.md`` (GitHub issue #3).

In its current state (step 1 of the migration) this package only contains
type definitions and a shared helper. No orchestrator code imports from
here yet. Subsequent migration steps will move the tuning, stability,
reasoning, logging, and direction-pivot logic into this package behind a
``PredictionPipeline`` entry point.
"""

from backend.post_process._shared import direction_from_change_pct
from backend.post_process.pipeline import PredictionPipeline
from backend.post_process.types import (
    PredictionRequest,
    PredictionResult,
    RunOverrides,
)

__all__ = [
    "PredictionPipeline",
    "PredictionRequest",
    "PredictionResult",
    "RunOverrides",
    "direction_from_change_pct",
]
