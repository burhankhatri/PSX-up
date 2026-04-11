"""Re-export shim kept for backwards compatibility.

The real implementation lives in :mod:`backend.post_process._reasoning`.
This shim exists so existing imports (``from backend.prediction_reasoning
import generate_prediction_reasoning``) keep working during the migration
in issue #3. New code should import from :mod:`backend.post_process`
instead.
"""

from backend.post_process._reasoning import (  # noqa: F401
    format_reasoning_for_display,
    generate_prediction_reasoning,
)
