# PSX Prediction App - Codebase Index

Generated: 2026-03-03 (Asia/Karachi)  
Scope: repository snapshot in `/Users/burhankhatri/Downloads/folder/temp/psx-prediction-app`

## 1) Repository Map

Top-level directories:

- `backend/` - FastAPI backend, prediction pipeline, feature engineering, sentiment, screening.
- `web/` - Single-page frontend (`stock_analyzer.html`) using Chart.js + WebSockets.
- `data/` - Runtime artifacts: historical caches, predictions, model binaries, sentiment caches, logs.
- `tests/` - Unit tests + live/manual probes for parsers and sentiment/news pipeline.
- `standalone_model/` - Portable subset of backend modules for standalone model execution.
- `venv/` - Local virtual environment.

Top-level files:

- `README.md` - architecture, quick start, and API docs (some sections appear older than current routes).
- `requirements.txt` - lightweight API/runtime deps.
- `extract_model.py` - helper script to extract/move model assets.

## 2) Runtime Entry Points

- `backend/main.py`
  - Creates FastAPI app.
  - Mounts static UI at `/analyzer`.
  - Registers API + WebSocket routes.
  - Serves history and prediction tuning report APIs.

- `backend/stock_analyzer_fixed.py`
  - Core async analysis job lifecycle.
  - Generates/refreshes historical data.
  - Trains model (research-first, SOTA fallback).
  - Applies sentiment adjustments, optional geopolitical overlays, tuning, and logging.
  - Streams progress and final result over WebSocket.

- `web/stock_analyzer.html`
  - Main client UI and dashboard.
  - Starts analysis jobs, opens WebSockets, renders charts/tables, history, batch runs, commodity views.

## 3) Main Execution Flow

1. Client calls `POST /api/analyze-stock` (or `/api/batch-analyze`) -> receives `job_id`.
2. Client connects `WS /ws/progress/{job_id}`.
3. Backend pipeline in `websocket_progress(...)`:
   - validates/fetches PSX historical data,
   - loads and trains model (`PSXResearchModel` by default),
   - produces daily predictions,
   - runs sentiment analysis,
   - applies adjustment math and optional geo/recovery overlays,
   - generates monthly forecast summary,
   - logs prediction for later tuning/evaluation,
   - persists analysis JSON files in `data/`,
   - returns final payload (`stage=complete`).

## 4) API Index (`backend/main.py`)

Always-on routes:

- `GET /` - redirect to analyzer UI.
- `GET /analyzer` - serve dashboard HTML.
- `GET /health` - health check.
- `POST /api/analyze` - unified sync endpoint:
  - `KSE100` -> runs `analyze_kse100(...)`,
  - symbol -> returns cached predictions if present.
- `GET /api/history` - list saved analyses + prediction log excerpts.
- `GET /api/history/{filename}` - load one prediction file + chart/support metadata.
- `GET /api/prediction-tuning-report` - A/B tuning report + drift snapshot.

Conditionally registered routes:

- If stock analyzer imports successfully:
  - `POST /api/check-data`
  - `POST /api/analyze-stock`
  - `POST /api/batch-analyze`
  - `WS /ws/progress/{job_id}`

- If screener/sentiment modules import:
  - `GET /api/screener`
  - `GET /api/trending`
  - `GET /api/sentiment/{symbol}`

- If commodity module imports:
  - `GET /api/commodity/list`
  - `GET /api/commodity/{symbol}`
  - `POST /api/commodity/analyze`
  - `WS /ws/commodity/{job_id}`
  - `GET /api/commodity/history/{commodity}`

## 5) Backend Module Index

### API and orchestration

- `backend/main.py` - app setup and route registration.
- `backend/stock_analyzer_fixed.py` - async job orchestration, training, prediction, post-processing.
- `backend/runtime_config.py` - typed env-driven feature flags (`RuntimeConfig`).

### Core modeling

- `backend/research_model.py` - `PSXResearchModel`, iterated forecaster, research-weighted ensemble.
- `backend/sota_model.py` - SOTA ensemble stack (wavelets, multi-horizon components).
- `backend/stacking_ensemble.py` - stacking/meta-learner utilities.
- `backend/sector_models.py` - sector-aware model and feature management.
- `backend/williams_r_classifier.py` - Williams %R trend classifier.
- `backend/prediction_stability.py` - stability smoothing/post-processing.
- `backend/validated_indicators.py` - validated indicator calculations and feature lists.
- `backend/feature_validation.py` - SHAP/VIF/multicollinearity reduction tools.

### Data and external feature ingestion

- `backend/external_features.py` - USD/PKR, KSE100, commodities, KIBOR proxy, merges.
- `backend/tradingview_scraper.py` - TradingView technical signal scraping + cache.
- `backend/kse100_analyzer.py` - KSE100 data fetch + forecast.

### News, sentiment, and adjustments

- `backend/enhanced_news_fetcher.py` - multi-source scraping, relevance scoring, dedupe/ranking.
- `backend/sentiment_analyzer.py` - end-to-end sentiment pipeline with Groq + fallback logic.
- `backend/sentiment_math.py` - numeric conversion of sentiment/events into prediction adjustments.
- `backend/article_scraper.py` - deep Business Recorder article parsing and relevance scoring.
- `backend/brecorder_scraper.py` - BR + PSX notices scraping/caching helpers.
- `backend/geopolitical_features.py` - geo-risk features, shock detection, daily geo adjustments.
- `backend/recovery_predictor.py` - post-shock recovery scenario generation.

### Forecast interpretation and logging

- `backend/monthly_forecast.py` - monthly view + event/driver summaries.
- `backend/prediction_reasoning.py` - "why bullish/bearish" explanation generation.
- `backend/prediction_logger.py` - structured prediction log persistence.
- `backend/prediction_tuning.py` - tweak configs, A/B evaluation, drift snapshots.
- `backend/prediction_regression_check.py` - freeze/compare CLI for regression drift control.

### Screening and discovery

- `backend/stock_screener.py` - quick symbol scanner.
- `backend/smart_screener.py` - model-output-based screener scoring.
- `backend/top_stocks_analyzer.py` - simple top-stock screener helper.
- `backend/hot_stocks.py` - trending stocks from market/news scraping.

### Commodity analysis

- `backend/commodity_predictor.py` - gold/silver forecasting and factor explanation.

### Misc

- `backend/test_weekend_predictions.py` - weekend handling tests for prediction dates.
- `backend/__init__.py` - package marker.

## 6) Frontend Index

File: `web/stock_analyzer.html`

Core client responsibilities:

- Starts analysis (`analyzeStock`, `quickAnalyze`, batch orchestration).
- Builds WebSocket URLs and consumes streamed progress.
- Renders:
  - price/prediction charts,
  - confidence and insight panels,
  - geo overlay and recovery panels,
  - history comparisons and tuning report views,
  - screener/trending/commodity sections.

Notable JS entry functions:

- `analyzeStock(...)`
- `startBatchAnalysis(...)`
- `connectBatchWebSocket(...)`
- `renderResults(...)`
- `loadHistoryList(...)`
- `loadPredictionTuningReport(...)`
- `analyzeCommodity(...)`

## 7) Data Artifact Index (`data/`)

Primary file patterns:

- `<SYMBOL>_historical_with_indicators.json` - cached OHLCV + indicators.
- `<SYMBOL>_research_predictions_2026.json` - daily predictions from research model.
- `<SYMBOL>_sota_predictions_2026.json` - daily/monthly SOTA outputs (fallback/legacy path).
- `<SYMBOL>_complete_analysis.json` - consolidated response payload cache.
- `<SYMBOL>_backtest_results.json` - backtest stats (where generated).

Subdirectories:

- `data/models/` - serialized model artifacts (`.pkl`, feature selections, etc.).
- `data/news_cache/` - cached news/sentiment payloads.
- `data/tradingview_cache/` - cached TradingView technicals.
- `data/commodity_cache/` - commodity analysis caches.
- `data/prediction_logs/` - prediction logs, tuning reports, drift/shadow comparisons.

## 8) Tests Index (`tests/`)

Unit/integration tests:

- `test_geopolitical_upgrade.py` - runtime config, sentiment modes, geo features, shadow/drift checks.
- `test_enhanced_news_fetcher.py` - multi-source parsing/relevance behavior.
- `test_article_scraper_relevance.py` - article relevance/filter logic.
- `test_parser_fixtures.py` - fixture-driven parser checks.
- `test_sentiment_math.py` - adjustment math behavior.
- `test_sentiment_model_flags.py` - env-flag behavior in sentiment/model flows.

Live/manual probes:

- `live_kse100_sentiment_smoke.py`
- `live_parser_iteration.py`
- `live_scrape_probe.py`

Fixtures:

- `tests/fixtures/*.html` - saved source pages for parser tests.

## 9) Standalone Model Subtree

`standalone_model/backend/` mirrors many backend modules for portable use.

Current relationship (quick diff snapshot):

- Common modules with `backend/`: 21
- Byte-identical: 12
- Diverged copies: 9 (`article_scraper.py`, `enhanced_news_fetcher.py`, `external_features.py`, `research_model.py`, `sector_models.py`, `sentiment_analyzer.py`, `sentiment_math.py`, `stacking_ensemble.py`, `stock_analyzer_fixed.py`)
- Backend-only modules (not in standalone): 12 (includes `main.py`, `runtime_config.py`, `geopolitical_features.py`, `prediction_tuning.py`, etc.)

## 10) Key Env Flags (centralized in `backend/runtime_config.py`)

- `MODEL_VARIANT` - `baseline|shadow|upgraded`
- `ENABLE_GEO_FEATURES`
- `ENABLE_RECOVERY_PREDICTOR`
- `SENTIMENT_ADJUST_MODE` - `legacy|date_aware`
- `TRADINGVIEW_CACHE_TTL_MIN`
- `ENABLE_INDEX_NEWS_RECALL`
- `ENABLE_INDEX_RECALL_IN_MODEL`
- `PREDICTION_TWEAKS_ENABLED`
- `PRED_TWEAK_NEUTRAL_BAND_PCT`
- `PRED_TWEAK_MIN_CONFIDENCE`
- `PRED_TWEAK_WILLIAMS_BRAKE`
- `PRED_TWEAK_MAX_UPSIDE_CAP_PCT`
- `PRED_TWEAK_MAX_DOWNSIDE_CAP_PCT`
- `PRED_TWEAK_BIAS_CORRECTION_PCT`
- `LOGGED_DIRECTION_SOURCE` - `stable|raw`

`GROQ_API_KEY` is required for full LLM sentiment path.

## 11) Quick Navigation Commands

```bash
# List routes
rg -n "@app\\.(get|post|websocket)\\(" backend/main.py

# Find core pipeline functions
rg -n "async def websocket_progress|def analyze_stock|class PSXResearchModel" backend

# Inspect tests
rg -n "^class Test|def test_" tests
```
