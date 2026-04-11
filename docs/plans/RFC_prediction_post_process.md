# RFC: Unify prediction post-processing into a single deep module

**Status:** Proposed
**Date:** 2026-04-11
**Scope:** Refactor only. No model logic, no algorithm changes, no numerical behavior change.

## Motivation

Four modules claim to be a "post-processing chain" but in practice they are four shallow leaves that the orchestrator stitches together with ~150 lines of implicit-contract glue. Reading one module doesn't tell you what the chain does; only `backend/stock_analyzer_fixed.py` lines ~1860–2460 does, and that file is already a 2480-LOC god object.

The current chain (by call order, not module name):

1. `backend/stock_analyzer_fixed.py:1873` — `apply_prediction_tweaks(predictions, live_tweak_config)` mutates the list.
2. `backend/stock_analyzer_fixed.py:2222–2288` — 80-line inline block computing `direction_meta` (raw/stable/display/logged directions, `near_term_direction`, `day7_direction`, `path_shape`, `stability_note`).
3. `backend/stock_analyzer_fixed.py:2332` — `generate_prediction_reasoning(reasoning_df, ..., apply_stability=False, ...)` — where `reasoning_df` is a *second* pass of `research_model.preprocess(df)` or `merge_external_features(df)`.
4. `backend/stock_analyzer_fixed.py:2445–2458` — `_log_prediction_variants(...)` helper (54 lines, also in the orchestrator file) then `logger.backfill_actuals(...)`.

### Leaky seams catalogued

1. **`PredictionStabilizer` is dead code in the real path.** It is constructed only inside `backend/prediction_reasoning.py:391`, which the orchestrator calls with `apply_stability=False`. `pivot_pred.get('stable_direction', raw_direction)` at `stock_analyzer_fixed.py:2235` always falls through because nothing ever writes `stable_direction`. The 266-line module plus `data/prediction_state.json` state file exist but do not run. This refactor forces the decision: wire it back in or delete it. **We will wire it back in** — that matches the original intent and the existing state file.

2. **`neutral_band_pct` is read from `live_tweak_config` in three places** (`stock_analyzer_fixed.py:1876, 2229/2233/2252, 2338/2455`). Any config mutation between reads is silent inconsistency.

3. **`direction_from_change_pct` is redefined/reimported in 5 places**: `prediction_tuning.py:42`, `prediction_reasoning.py:14` (plus its own `except` fallback at line 16), `stock_analyzer_fixed.py:56/72`, and twice inside helper utilities.

4. **Mutation contract is implicit.** `apply_prediction_tweaks` adds `raw_upside_potential`, `raw_predicted_price`, `tweaked_direction`, `adaptive_bias_correction` by side effect. Callers just hope the keys are there.

5. **Reasoning rebuilds a DataFrame.** `stock_analyzer_fixed.py:2322–2330` runs `research_model.preprocess(df)` a second time (or `merge_external_features` as fallback) just to feed reasoning. The model already did this work.

6. **Variant logging lives in the orchestrator.** The 54-line `_log_prediction_variants` helper (`stock_analyzer_fixed.py:450`) is conceptually a `PredictionLogger` method, but the 507-line logger module has no knowledge of "baseline" vs "geo" variants.

## Non-goals

- **No change to tuning math.** Bias correction, clamp bounds, confidence gate, Williams brake, neutral-band zeroing — byte-for-byte identical.
- **No change to stability math.** Adaptive alpha (0.5 / 0.7 / 0.9), hysteresis thresholds (±7% / ±5%) unchanged.
- **No change to reasoning signal logic.** Same ~15 indicator columns, same strength weights, same bullish/bearish/neutral bins, same news event scoring via `brecorder_scraper`.
- **No change to the logger file format.** `data/prediction_logs/prediction_log.json` schema unchanged. Same for `data/prediction_state.json`.
- **No change to the WebSocket payload.** Every key the dashboard reads — `direction_meta`, `near_term_direction`, `day7_direction`, `path_shape`, `daily_predictions`, `prediction_reasoning`, `tuning`, `monthly_predictions`, `daily_predictions_with_geo`, etc. — stays identical, and this is enforced mechanically via a `ws_payload` property that builds the shape in one place.
- **Out of scope:** offline A/B eval (`prediction_tuning.py::evaluate_prediction_log`, `run_ab`, `drift_snapshot`, `write_ab_report`). Those stay where they are as a sibling `prediction_eval.py` — they read `prediction_log.json` directly.
- **Out of scope:** the `standalone_model/` copy. This RFC only touches `backend/`. We'll re-sync standalone in a follow-up.

## Proposed design

Single deep module `backend/post_process/` with two public entry points. Hybrid of Design A (minimal) with two borrowings from Design C (common-case).

```python
# backend/post_process/__init__.py

@dataclass(frozen=True)
class RunOverrides:
    tweak_config: TweakConfig | None = None
    enable_tuning: bool = True
    enable_stability: bool = True
    enable_logging: bool = True
    enable_reasoning: bool = True

@dataclass(frozen=True)
class PredictionRequest:
    symbol: str
    raw_df: pd.DataFrame                         # history + indicators
    feature_df: pd.DataFrame                     # preprocessed, reused — no rebuild
    current_price: float
    baseline_predictions: list[dict]             # from the model, pre-tuning
    geo_predictions: list[dict] | None           # None => geo variant skipped
    analysis_id: str
    generated_at: datetime
    horizon_label: str = "Day 7"
    overrides: RunOverrides | None = None

@dataclass(frozen=True)
class PredictionResult:
    monthly_predictions: list[dict]
    daily_predictions: list[dict]                # baseline, tuned, with raw_* audit keys
    daily_predictions_without_geo: list[dict]
    daily_predictions_with_geo: list[dict]
    direction_meta: dict[str, Any]
    near_term_direction: str
    day7_direction: str
    path_shape: str
    prediction_reasoning: dict[str, Any]
    tuning: dict[str, Any]

    @property
    def ws_payload(self) -> dict:
        """The exact shape the websocket handler must forward. Backwards compat lives here."""
        return {
            "direction_meta":                self.direction_meta,
            "near_term_direction":           self.near_term_direction,
            "day7_direction":                self.day7_direction,
            "path_shape":                    self.path_shape,
            "monthly_predictions":           self.monthly_predictions,
            "daily_predictions":             self.daily_predictions,
            "daily_predictions_without_geo": self.daily_predictions_without_geo,
            "daily_predictions_with_geo":    self.daily_predictions_with_geo,
            "prediction_reasoning":          self.prediction_reasoning,
            "tuning":                        self.tuning,
        }

class PredictionPipeline:
    @classmethod
    def default(cls) -> "PredictionPipeline": ...
    @classmethod
    def for_test(cls, *, state_path, log_path, config) -> "PredictionPipeline": ...
    def run(self, req: PredictionRequest) -> PredictionResult: ...
```

### Module layout

```
backend/post_process/
  __init__.py              # re-exports PredictionPipeline, PredictionRequest, PredictionResult, RunOverrides
  pipeline.py              # PredictionPipeline orchestrating the 4 stages (~120 LOC)
  types.py                 # dataclasses + ws_payload property
  _tuning.py               # moved verbatim from prediction_tuning.py (hot-path functions only)
  _stability.py            # moved verbatim from prediction_stability.py
  _reasoning.py            # moved verbatim from prediction_reasoning.py, with internal stabilizer dep removed
  _logging.py              # moved from prediction_logger.py + the _log_prediction_variants helper
  _direction.py            # the 80-line pivot/path_shape block, extracted verbatim
  _shared.py               # single authoritative `direction_from_change_pct`
```

### Internal flow inside `pipeline.run()`

```
req -> config (read env ONCE via RunOverrides.tweak_config or get_live_tweak_config())
    -> _tuning.apply(req.baseline_predictions, config)            # returns NEW list (no mutation)
    -> _tuning.apply(req.geo_predictions, config) if geo_predictions is not None
    -> _stability.apply(symbol, pivot_upside, pivot_direction)    # writes stable_direction into pivot_pred
    -> _direction.compute_direction_meta(...)                     # reads stable_direction — no more fall-through
    -> _reasoning.generate(req.feature_df, ..., direction_override=direction_meta['display_direction'])
    -> _logging.log_variants(baseline, geo, ...) + backfill_actuals
    -> build PredictionResult
```

Exactly one env read. Exactly one `direction_from_change_pct`. Exactly one preprocess (the caller provides `feature_df`). Stabilizer runs for real. No implicit mutation of caller lists.

## Migration plan — tiny commits

1. **Create `backend/post_process/` package skeleton** + `types.py` + `_shared.py` (the shared `direction_from_change_pct`). No callers wired yet. No behavior change. ~80 LOC new, 0 LOC deleted.
2. **Move `prediction_tuning.py` hot-path functions** (`TweakConfig`, `get_live_tweak_config`, `apply_prediction_tweaks`, `compute_per_symbol_bias`, `direction_from_change_pct`) into `backend/post_process/_tuning.py`. Leave `prediction_tuning.py` as a re-export shim so `evaluate_prediction_log` / `run_ab` / `drift_snapshot` and the existing imports still work. Run tests.
3. **Move `prediction_stability.py`** into `_stability.py`, same re-export shim. Run tests.
4. **Move `prediction_reasoning.py`** into `_reasoning.py`. Delete the internal `PredictionStabilizer` import and the `apply_stability` flag — reasoning becomes purely "given a direction, explain signals." Run tests (update the one caller that passes `apply_stability=False` — it's a no-op today).
5. **Move `prediction_logger.py`** into `_logging.py` and move `_log_prediction_variants` from `stock_analyzer_fixed.py:450` into it as `_logging.log_variants`. Re-export shim for `get_prediction_logger`.
6. **Extract `_direction.compute_direction_meta`** from `stock_analyzer_fixed.py:2222–2288` verbatim. No behavior change — just cut-and-paste into a function. Orchestrator calls the function.
7. **Introduce `PredictionPipeline.run()`** that runs steps 2–6 in order. Orchestrator's 150-line post-processing block becomes ~8 lines. `ws_payload` property drives the websocket spread.
8. **Wire stabilizer for real.** This is the one observable behavior change: `stable_direction` now gets populated on the pivot prediction, which changes `logged_direction` output when `LOGGED_DIRECTION_SOURCE=stable` (the default). Gate behind `RunOverrides.enable_stability` which defaults `True`. Document in the commit.
9. **Delete shims** once nothing imports the old module paths. Run `ripgrep` to confirm. Delete `prediction_tuning.py`, `prediction_stability.py`, `prediction_reasoning.py`, `prediction_logger.py` from `backend/` root (or leave as 2-line re-export shims if `standalone_model/` imports them — TBD once we audit).

Each step is one commit, one green test run. If step 8 reveals that nobody actually wanted the stabilizer wired back in, revert it in isolation — every other step is pure code motion.

## Tests

- **Golden snapshot.** Before step 1, capture `result` from `run_research_analysis_websocket` for 3 symbols (e.g., UBL, OGDC, PSO) with a pinned fixture DataFrame. Assert byte-equal `ws_payload` through every step except step 8 (stabilizer wiring), which will change `direction_meta.logged_direction` for some cases. Update the snapshot once, explicitly, with the diff reviewed.
- **Pipeline unit test** using `PredictionPipeline.for_test(state_path=tmp, log_path=tmp, ...)` — proves test isolation without module-level singletons.
- **Contract test on `ws_payload`**: assert the set of keys matches what `backend/stock_analyzer_fixed.py` sends in the `complete` websocket frame.
- **No-mutation test**: pass a `baseline_predictions` list, assert it is not mutated by `run()`.

## Risks

- **Stabilizer wiring is a real behavior change.** Small — it only affects `logged_direction` when `LOGGED_DIRECTION_SOURCE=stable` (the default). Mitigation: land steps 1–7 first, run in shadow for a day, then land step 8 on its own commit with a clear rollback.
- **`standalone_model/` drift.** If standalone imports `prediction_tuning` etc. by path, the re-export shims keep it working. If we delete the shims in step 9, standalone breaks. Audit before step 9.
- **Hidden callers.** `grep -r 'from backend.prediction_' .` before each step. Any hit outside `backend/` or `tests/` needs attention.

## Success criteria

- `backend/stock_analyzer_fixed.py` shrinks by ~150 LOC (the post-processing block + `_log_prediction_variants` helper).
- `direction_from_change_pct` appears exactly once in the codebase.
- `live_tweak_config` is constructed exactly once per pipeline run.
- `PredictionStabilizer` is either used (writes `stable_direction`) or deleted — not both.
- Websocket payload is byte-equal to pre-refactor for 3 snapshot symbols except for documented stabilizer-wiring deltas in step 8.
- Each commit is independently revertable.
