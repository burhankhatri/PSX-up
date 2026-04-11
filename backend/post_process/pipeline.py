"""``PredictionPipeline`` — the single entry point for post-processing.

This module introduces the target interface described in
``docs/plans/RFC_prediction_post_process.md`` (GitHub issue #3). In the
current migration state (step 7) the orchestrator in
``backend.stock_analyzer_fixed`` still runs tuning, direction, reasoning,
and logging inline. ``PredictionPipeline.run`` exists so new call sites
(tests, future refactors of the orchestrator, alternative entry points)
can drive the chain through a single function today.

The pipeline is a pass-through over the already-extracted stages:

1. ``_tuning.apply_prediction_tweaks`` — optional, gated by
   ``TweakConfig.enabled`` and ``RunOverrides.enable_tuning``.
2. ``_direction.compute_direction_meta`` — always run.
3. ``_reasoning.generate_prediction_reasoning`` — optional, gated by
   ``RunOverrides.enable_reasoning``. The ``apply_stability`` argument
   stays ``False`` to preserve current behavior; wiring the stabilizer
   is explicitly step 8 of the RFC and out of scope here.
4. ``_logging.log_prediction_variants`` + ``backfill_actuals`` —
   optional, gated by ``RunOverrides.enable_logging``.

The result is returned as a frozen :class:`PredictionResult` whose
``ws_payload`` property owns the exact set of keys the websocket
handler forwards to the dashboard.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.post_process._direction import compute_direction_meta
from backend.post_process._logging import (
    PredictionLogger,
    get_prediction_logger,
    log_prediction_variants,
)
from backend.post_process._reasoning import generate_prediction_reasoning
from backend.post_process._tuning import (
    TweakConfig,
    apply_prediction_tweaks,
    get_live_tweak_config,
)
from backend.post_process.types import (
    PredictionRequest,
    PredictionResult,
    RunOverrides,
)


class PredictionPipeline:
    """Orchestrate tuning, direction, reasoning, and logging behind one call."""

    def __init__(
        self,
        *,
        logger: PredictionLogger,
        config: Optional[TweakConfig] = None,
    ) -> None:
        self._logger = logger
        self._config = config

    @classmethod
    def default(cls) -> "PredictionPipeline":
        """Construct with real collaborators — env-driven config and JSON logger."""
        return cls(
            logger=get_prediction_logger(),
            config=get_live_tweak_config(),
        )

    def run(self, req: PredictionRequest) -> PredictionResult:
        overrides = req.overrides or RunOverrides()
        config = overrides.tweak_config or self._config or get_live_tweak_config()
        neutral_band = float(getattr(config, "neutral_band_pct", 0.0))

        baseline = list(req.baseline_predictions)
        geo = list(req.geo_predictions) if req.geo_predictions else None

        tuning_meta: dict[str, Any] = {"enabled": bool(getattr(config, "enabled", False))}
        if overrides.enable_tuning and tuning_meta["enabled"]:
            if baseline:
                baseline = apply_prediction_tweaks(baseline, config)
            if geo:
                geo = apply_prediction_tweaks(geo, config)
            tuning_meta.update(
                {
                    "neutral_band_pct": neutral_band,
                    "applied_to_days": len(baseline),
                }
            )

        display = geo if geo else baseline
        direction_meta = compute_direction_meta(display, config, runtime_cfg=None)

        reasoning: dict[str, Any] = {}
        if overrides.enable_reasoning:
            try:
                reasoning = generate_prediction_reasoning(
                    req.feature_df,
                    symbol=req.symbol,
                    predicted_upside=direction_meta.get("display_upside_pct", 0.0),
                    direction_override=direction_meta.get("display_direction", "NEUTRAL"),
                    apply_stability=False,
                    neutral_band_pct=neutral_band,
                    horizon_label=req.horizon_label,
                )
            except Exception as exc:  # non-fatal, matches orchestrator behavior
                reasoning = {"error": str(exc)}

        if overrides.enable_logging:
            try:
                log_prediction_variants(
                    self._logger,
                    symbol=req.symbol,
                    current_price=req.current_price,
                    baseline_predictions=baseline,
                    geo_predictions=geo,
                    analysis_id=req.analysis_id,
                    prediction_generated_at=req.generated_at,
                    neutral_band_pct=neutral_band,
                    include_geo_variant=geo is not None,
                    sentiment_x_factor=req.sentiment_x_factor,
                    geo_x_factor=req.geo_x_factor,
                )
                self._logger.backfill_actuals(symbol=req.symbol, limit=32)
            except Exception:
                # Matches orchestrator: logging failures must not sink an analysis.
                pass

        return PredictionResult(
            monthly_predictions=baseline[:12],
            daily_predictions=baseline,
            daily_predictions_without_geo=baseline,
            daily_predictions_with_geo=geo or [],
            direction_meta=direction_meta,
            near_term_direction=direction_meta.get("near_term_direction", "NEUTRAL"),
            day7_direction=direction_meta.get("day7_direction", "NEUTRAL"),
            path_shape=direction_meta.get("path_shape", "volatile_flat"),
            prediction_reasoning=reasoning,
            tuning=tuning_meta,
        )
