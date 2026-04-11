"""Re-export shim kept for backwards compatibility.

The real implementation lives in :mod:`backend.post_process._tuning`.
This shim exists so existing imports (``from backend.prediction_tuning
import apply_prediction_tweaks``) keep working during the migration in
issue #3. New code should import from :mod:`backend.post_process` instead.
"""

from backend.post_process._tuning import (  # noqa: F401
    DEFAULT_TWEAK_CONFIG,
    PREDICTION_LOG_PATH,
    TweakConfig,
    _clamp,
    _derive_base_price,
    _fetch_actual_on_or_after,
    apply_prediction_tweaks,
    compute_per_symbol_bias,
    direction_from_change_pct,
    drift_snapshot,
    evaluate_prediction_log,
    get_live_tweak_config,
    run_ab,
    write_ab_report,
)
