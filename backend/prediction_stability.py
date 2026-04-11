"""Re-export shim kept for backwards compatibility.

The real implementation lives in :mod:`backend.post_process._stability`.
This shim exists so existing imports (``from backend.prediction_stability
import PredictionStabilizer``) keep working during the migration in
issue #3. New code should import from :mod:`backend.post_process` instead.
"""

from backend.post_process._stability import (  # noqa: F401
    STATE_FILE,
    PredictionStabilizer,
)
