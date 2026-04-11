"""Re-export shim kept for backwards compatibility.

The real implementation lives in :mod:`backend.post_process._logging`.
This shim exists so existing imports (``from backend.prediction_logger
import get_prediction_logger``) keep working during the migration in
issue #3. New code should import from :mod:`backend.post_process` instead.
"""

from backend.post_process._logging import (  # noqa: F401
    LOG_DIR,
    PredictionLogger,
    get_prediction_logger,
    log_prediction_variants,
)
