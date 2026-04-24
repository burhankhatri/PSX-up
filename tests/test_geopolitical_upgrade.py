"""
Comprehensive test suite for the geopolitical upgrade plan.

Covers:
  - Runtime config loading and defaults
  - Sentiment adjustment modes (legacy vs date_aware)
  - Direction consistency
  - Geopolitical feature computation
  - Prediction regression check (freeze/compare)
  - Shadow comparison mechanics
  - Enhanced news fetcher macro terms
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# 1. Runtime Config Tests
# ============================================================================

class TestRuntimeConfig(unittest.TestCase):
    """Verify typed config loads correctly from env with safe defaults."""

    def test_defaults(self):
        from backend.runtime_config import load_runtime_config
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_runtime_config()
        self.assertEqual(cfg.model_variant, "baseline")
        self.assertFalse(cfg.enable_geo_features)
        self.assertEqual(cfg.sentiment_adjust_mode, "legacy")
        self.assertEqual(cfg.tradingview_cache_ttl_min, 60)
        self.assertFalse(cfg.prediction_tweaks_enabled)
        self.assertEqual(cfg.logged_direction_source, "stable")

    def test_env_override(self):
        from backend.runtime_config import load_runtime_config
        env = {
            "MODEL_VARIANT": "shadow",
            "ENABLE_GEO_FEATURES": "true",
            "SENTIMENT_ADJUST_MODE": "date_aware",
            "TRADINGVIEW_CACHE_TTL_MIN": "30",
            "PREDICTION_TWEAKS_ENABLED": "1",
            "LOGGED_DIRECTION_SOURCE": "raw",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
        self.assertEqual(cfg.model_variant, "shadow")
        self.assertTrue(cfg.enable_geo_features)
        self.assertEqual(cfg.sentiment_adjust_mode, "date_aware")
        self.assertEqual(cfg.tradingview_cache_ttl_min, 30)
        self.assertTrue(cfg.prediction_tweaks_enabled)
        self.assertEqual(cfg.logged_direction_source, "raw")
        self.assertTrue(cfg.is_shadow)
        self.assertFalse(cfg.is_baseline)

    def test_invalid_variant_falls_back(self):
        from backend.runtime_config import load_runtime_config
        with patch.dict(os.environ, {"MODEL_VARIANT": "bogus"}, clear=True):
            cfg = load_runtime_config()
        self.assertEqual(cfg.model_variant, "baseline")

    def test_to_dict(self):
        from backend.runtime_config import load_runtime_config
        cfg = load_runtime_config()
        d = cfg.to_dict()
        self.assertIn("model_variant", d)
        self.assertIn("enable_geo_features", d)

    def test_singleton_caching(self):
        from backend.runtime_config import get_runtime_config
        cfg1 = get_runtime_config(force_reload=True)
        cfg2 = get_runtime_config()
        self.assertIs(cfg1, cfg2)


# ============================================================================
# 2. Sentiment Adjustment Mode Tests
# ============================================================================

class TestSentimentAdjustMode(unittest.TestCase):
    """Verify legacy vs date_aware adjustment modes."""

    def _sample_sentiment(self):
        return {
            "sentiment_score": 0.5,
            "confidence": 0.8,
            "news_items": [
                {"title": "Company announces dividend payout", "date": "2026-02-01", "source": "PSX"},
            ],
        }

    def test_legacy_monthly_returns_month_keys(self):
        from backend.sentiment_math import get_rigorous_adjustment
        with patch.dict(os.environ, {"SENTIMENT_ADJUST_MODE": "legacy"}, clear=False):
            result = get_rigorous_adjustment(
                self._sample_sentiment(), prediction_length=3, frequency="monthly"
            )
        self.assertEqual(result["summary"]["frequency"], "monthly")
        for adj in result["adjustments"]:
            self.assertIn("month", adj)

    def test_date_aware_monthly_still_returns_month_keys(self):
        from backend.sentiment_math import get_rigorous_adjustment
        with patch.dict(os.environ, {"SENTIMENT_ADJUST_MODE": "date_aware"}, clear=False):
            result = get_rigorous_adjustment(
                self._sample_sentiment(), prediction_length=3, frequency="monthly"
            )
        # date_aware re-keys day adjustments to month labels
        self.assertEqual(result["summary"]["frequency"], "monthly")
        for adj in result["adjustments"]:
            self.assertIn("month", adj)

    def test_daily_mode_produces_day_keys(self):
        from backend.sentiment_math import get_rigorous_adjustment
        result = get_rigorous_adjustment(
            self._sample_sentiment(), prediction_length=5, frequency="daily"
        )
        self.assertEqual(len(result["adjustments"]), 5)
        self.assertIn("day", result["adjustments"][0])

    def test_date_aware_decay_differs_from_legacy(self):
        """date_aware should compute day-accurate decay instead of month*30."""
        from backend.sentiment_math import get_rigorous_adjustment
        sentiment = self._sample_sentiment()

        with patch.dict(os.environ, {"SENTIMENT_ADJUST_MODE": "legacy"}, clear=False):
            legacy = get_rigorous_adjustment(sentiment, prediction_length=2, frequency="monthly")

        with patch.dict(os.environ, {"SENTIMENT_ADJUST_MODE": "date_aware"}, clear=False):
            aware = get_rigorous_adjustment(sentiment, prediction_length=2, frequency="monthly")

        # Both should have 2 adjustments; values may differ due to day-indexed decay
        self.assertEqual(len(legacy["adjustments"]), 2)
        self.assertEqual(len(aware["adjustments"]), 2)


# ============================================================================
# 3. Direction Consistency Tests
# ============================================================================

class TestDirectionConsistency(unittest.TestCase):
    """Logged direction source follows config; raw/stable both surfaced."""

    def test_direction_from_change_pct(self):
        from backend.prediction_tuning import direction_from_change_pct
        self.assertEqual(direction_from_change_pct(2.0), "BULLISH")
        self.assertEqual(direction_from_change_pct(-1.0), "BEARISH")
        self.assertEqual(direction_from_change_pct(0.5, neutral_band_pct=1.0), "NEUTRAL")
        self.assertEqual(direction_from_change_pct(0.0), "NEUTRAL")

    def test_neutral_band_edge(self):
        from backend.prediction_tuning import direction_from_change_pct
        self.assertEqual(direction_from_change_pct(1.0, neutral_band_pct=1.0), "NEUTRAL")
        self.assertEqual(direction_from_change_pct(1.01, neutral_band_pct=1.0), "BULLISH")


# ============================================================================
# 4. Geopolitical Features Tests
# ============================================================================

class TestGeopoliticalFeatures(unittest.TestCase):
    """Verify geopolitical risk signal computation."""

    def test_neutral_on_empty(self):
        from backend.geopolitical_features import get_geopolitical_features_from_news
        result = get_geopolitical_features_from_news([])
        self.assertEqual(result["geo_conflict_risk"], 0.0)
        self.assertEqual(result["geo_energy_supply_risk"], 0.0)
        self.assertEqual(result["geo_news_volume"], 0.0)

    def test_conflict_detection(self):
        from backend.geopolitical_features import get_geopolitical_features_from_news
        news = [
            {"title": "Military conflict escalation in the region"},
            {"title": "Border tension rises after strike"},
        ]
        result = get_geopolitical_features_from_news(news)
        self.assertGreater(result["geo_conflict_risk"], 0.0)

    def test_energy_supply_detection(self):
        from backend.geopolitical_features import get_geopolitical_features_from_news
        news = [
            {"title": "Oil prices surge as Strait of Hormuz shipping lane threatened"},
            {"title": "OPEC announces supply disruption"},
        ]
        result = get_geopolitical_features_from_news(news)
        self.assertGreater(result["geo_energy_supply_risk"], 0.0)

    def test_sector_amplifier(self):
        from backend.geopolitical_features import get_geopolitical_features_from_news
        news = [{"title": "Oil supply disruption in gulf region"}]
        neutral = get_geopolitical_features_from_news(news, symbol=None)
        energy = get_geopolitical_features_from_news(news, symbol="OGDC")
        # Energy symbol should amplify energy_supply_risk
        self.assertGreaterEqual(energy["geo_energy_supply_risk"], neutral["geo_energy_supply_risk"])

    def test_volume_scaling(self):
        from backend.geopolitical_features import get_geopolitical_features_from_news
        small = get_geopolitical_features_from_news([{"title": "one"}])
        large = get_geopolitical_features_from_news([{"title": f"news {i}"} for i in range(20)])
        self.assertLess(small["geo_news_volume"], large["geo_news_volume"])

    def test_neutral_fallback_on_symbol_fetch_failure(self):
        from backend.geopolitical_features import neutral_geopolitical_features
        result = neutral_geopolitical_features()
        self.assertEqual(len(result), 6)
        self.assertTrue(all(v == 0.0 for v in result.values()))

    def test_build_geo_adjustments_neutral(self):
        from backend.geopolitical_features import build_geopolitical_daily_adjustments
        data = build_geopolitical_daily_adjustments({}, prediction_length=5, symbol="LUCK")
        self.assertEqual(len(data["adjustments"]), 5)
        self.assertTrue(all(abs(a["capped_adjustment"]) < 1e-12 for a in data["adjustments"]))
        self.assertAlmostEqual(data["summary"]["max_abs_adjustment_pct"], 0.0, places=6)

    def test_build_geo_adjustments_decay(self):
        from backend.geopolitical_features import build_geopolitical_daily_adjustments
        features = {
            "geo_conflict_risk": 1.0,
            "geo_energy_supply_risk": 0.2,
            "geo_regional_tension": 0.9,
            "geo_global_risk_off": 1.0,
            "geo_news_volume": 1.0,
        }
        data = build_geopolitical_daily_adjustments(features, prediction_length=21, symbol="LUCK")
        self.assertLess(data["adjustments"][0]["capped_adjustment"], 0.0)
        # Magnitude should decay over horizon
        self.assertGreater(
            abs(data["adjustments"][0]["capped_adjustment"]),
            abs(data["adjustments"][20]["capped_adjustment"]),
        )

    def test_build_geo_adjustments_energy_symbol_dampens_energy_risk(self):
        from backend.geopolitical_features import build_geopolitical_daily_adjustments
        features = {
            "geo_conflict_risk": 0.4,
            "geo_energy_supply_risk": 1.0,
            "geo_regional_tension": 0.3,
            "geo_global_risk_off": 0.6,
            "geo_news_volume": 0.8,
        }
        non_energy = build_geopolitical_daily_adjustments(features, prediction_length=1, symbol="LUCK")
        energy = build_geopolitical_daily_adjustments(features, prediction_length=1, symbol="OGDC")
        # Energy symbol should have less-negative (or more-positive) day-1 adjustment
        self.assertGreaterEqual(
            energy["adjustments"][0]["capped_adjustment"],
            non_energy["adjustments"][0]["capped_adjustment"],
        )

    def test_build_geo_adjustments_blends_ai_trajectory_with_conflict(self):
        from backend.geopolitical_features import build_geopolitical_daily_adjustments
        features = {
            "geo_conflict_risk": 1.0,
            "geo_energy_supply_risk": 0.7,
            "geo_regional_tension": 0.9,
            "geo_global_risk_off": 0.3,
            "geo_news_volume": 1.0,
        }
        shock_data = {
            "shock_detected": True,
            "emergency_multiplier": 2.5,
            "shock_half_life": 7,
            "trajectory": {
                "trajectory_score": 1.6,
                "confidence": 0.9,
                "llm_assessment": {
                    "market_impact_pct": 2.5,
                    "ceasefire_probability": 0.4,
                },
            },
        }

        data = build_geopolitical_daily_adjustments(
            features,
            prediction_length=21,
            symbol="KSE100",
            shock_data=shock_data,
            enable_ai_blend=True,
        )
        summary = data["summary"]
        self.assertIn("deterministic_day1_pct", summary)
        self.assertIn("ai_trajectory_day1_pct", summary)
        self.assertIn("blended_day1_pct", summary)
        self.assertLess(summary["deterministic_day1_pct"], 0)
        self.assertGreater(summary["ai_trajectory_day1_pct"], 0)
        self.assertLess(summary["blended_day1_pct"], 0)
        self.assertTrue(summary["direction_conflict"])
        self.assertEqual(summary["dominant_driver"], "deterministic")

    def test_build_geo_adjustments_mild_context_can_flip_with_ai_blend(self):
        from backend.geopolitical_features import build_geopolitical_daily_adjustments
        features = {
            "geo_conflict_risk": 0.05,
            "geo_energy_supply_risk": 0.0,
            "geo_regional_tension": 0.0,
            "geo_global_risk_off": 0.0,
            "geo_news_volume": 1.0,
        }
        shock_data = {
            "shock_detected": False,
            "emergency_multiplier": 1.0,
            "shock_half_life": 14,
            "trajectory": {
                "trajectory_score": 2.0,
                "confidence": 0.9,
                "llm_assessment": {
                    "market_impact_pct": 3.0,
                    "ceasefire_probability": 0.7,
                },
            },
        }
        # For a regular (non-index) ticker: a mild bearish deterministic
        # signal can be flipped positive by a strong AI trajectory blend.
        data = build_geopolitical_daily_adjustments(
            features,
            prediction_length=7,
            symbol="LUCK",
            shock_data=shock_data,
            enable_ai_blend=True,
        )
        summary = data["summary"]
        self.assertLess(summary["deterministic_day1_pct"], 0)
        self.assertGreater(summary["ai_trajectory_day1_pct"], 0)
        self.assertGreater(summary["blended_day1_pct"], 0)

        # For a benchmark index (KSE100), the A7 sign-flip guard blocks
        # AI from flipping the deterministic sign on days 1-3 — see
        # geopolitical_features.py:2186. AI blend is suppressed on day 1
        # but still contributes by day 7 when the near-term guard lifts.
        data_idx = build_geopolitical_daily_adjustments(
            features,
            prediction_length=7,
            symbol="KSE100",
            shock_data=shock_data,
            enable_ai_blend=True,
        )
        idx_summary = data_idx["summary"]
        self.assertLess(idx_summary["deterministic_day1_pct"], 0)
        self.assertEqual(idx_summary["ai_trajectory_day1_pct"], 0.0)
        self.assertGreater(idx_summary["ai_trajectory_day7_pct"], 0)

    def test_build_geo_adjustments_missing_ai_signal_falls_back_to_deterministic(self):
        from backend.geopolitical_features import build_geopolitical_daily_adjustments
        features = {
            "geo_conflict_risk": 0.8,
            "geo_energy_supply_risk": 0.6,
            "geo_regional_tension": 0.7,
            "geo_global_risk_off": 0.2,
            "geo_news_volume": 0.9,
        }
        shock_data = {
            "shock_detected": True,
            "emergency_multiplier": 2.0,
            "shock_half_life": 7,
            "trajectory": {},  # no llm_assessment / market_impact_pct
        }
        data = build_geopolitical_daily_adjustments(
            features,
            prediction_length=7,
            symbol="KSE100",
            shock_data=shock_data,
            enable_ai_blend=True,
        )
        summary = data["summary"]
        self.assertAlmostEqual(summary["ai_trajectory_day1_pct"], 0.0, places=6)
        self.assertAlmostEqual(summary["blended_day1_pct"], summary["deterministic_day1_pct"], places=6)

    def test_llm_prompt_includes_upstream_sector_override(self):
        from backend.geopolitical_features import _build_llm_trajectory_system_prompt

        prompt = _build_llm_trajectory_system_prompt(
            symbol="OGDC",
            fuel_hike_detected=True,
            circular_relief_detected=True,
            energy_supply_detected=True,
        )

        self.assertIn("upstream exploration & production stock", prompt)
        self.assertIn("BULLISH revenue catalysts", prompt)
        self.assertIn("Circular-debt plans, payment releases", prompt)
        self.assertIn("NEXT-DAY impact for this ticker", prompt)

    def test_llm_prompt_includes_downstream_headwind_context(self):
        from backend.geopolitical_features import _build_llm_trajectory_system_prompt

        prompt = _build_llm_trajectory_system_prompt(
            symbol="SAZEW",
            fuel_hike_detected=True,
            circular_relief_detected=False,
            energy_supply_detected=True,
        )

        self.assertIn("downstream autos stock", prompt)
        self.assertIn("BEARISH margin/demand headwinds", prompt)

    def test_assess_conflict_trajectory_passes_symbol_to_llm(self):
        from backend.geopolitical_features import assess_conflict_trajectory

        news = [{"title": "Middle East war pushes energy prices higher"}]
        with patch("backend.geopolitical_features.llm_assess_trajectory", return_value=None) as mock_llm:
            assess_conflict_trajectory(news, symbol="OGDC")

        mock_llm.assert_called_once()
        self.assertEqual(mock_llm.call_args.kwargs.get("symbol"), "OGDC")

    def test_build_geo_adjustments_upstream_flag_turns_overlay_positive(self):
        from backend.geopolitical_features import (
            build_geo_interpretation,
            build_geopolitical_daily_adjustments,
        )

        features = {
            "geo_conflict_risk": 0.64,
            "geo_energy_supply_risk": 0.72,
            "geo_regional_tension": 0.48,
            "geo_global_risk_off": 0.30,
            "geo_news_volume": 0.80,
            "geo_energy_war_overlap": 0.60,
        }
        news = [
            {
                "title": "Middle East war pushes energy prices higher after Strait of Hormuz threat",
                "description": "Govt hikes petrol price by Rs55 per litre",
            },
            {
                "title": "Circular debt payment release expected for E&P firms",
                "description": "Receivables cleared after fuel hike",
            },
        ]
        shock_data = {
            "shock_detected": True,
            "emergency_multiplier": 2.5,
            "shock_half_life": 7,
            "shock_reason": "Matched shock patterns: blockade",
        }
        # Pass crude_data so build_geo_interpretation's bullish_for_upstream
        # gate (oil_change_pct >= 2.0 OR oil_trend_pct >= 4.0) is satisfied;
        # without market crude evidence the gate keeps the interpretation
        # generic regardless of news.
        crude_data = {"oil_change_pct": 3.5, "oil_trend_pct": 5.0}
        interpretation = build_geo_interpretation(
            news_items=news,
            geo_features=features,
            shock_data=shock_data,
            symbol="OGDC",
            crude_data=crude_data,
        )

        result = build_geopolitical_daily_adjustments(
            features,
            prediction_length=7,
            symbol="OGDC",
            shock_data=shock_data,
            interpretation=interpretation,
        )

        self.assertTrue(interpretation["bullish_for_upstream"])
        self.assertEqual(result["summary"]["overlay_mode"], "upstream_tailwind")
        self.assertTrue(result["summary"]["bullish_for_upstream"])
        self.assertGreater(result["adjustments"][0]["capped_adjustment"], 0.0)

    def test_detect_geopolitical_shocks_extended_energy_patterns(self):
        from backend.geopolitical_features import detect_geopolitical_shocks
        news = [
            {"title": "Middle East war pushes energy prices higher after blockade threat"},
            {"title": "Strait of Hormuz closed as shipping disruption worsens"},
        ]
        shock = detect_geopolitical_shocks(news, symbol="OGDC")
        self.assertTrue(shock["shock_detected"])
        self.assertIn("shock_reason", shock)
        self.assertGreater(len(shock.get("matched_patterns", [])), 0)

    def test_energy_war_overlap_boosts_conflict_for_ogdc(self):
        from backend.geopolitical_features import get_geopolitical_features_from_news
        news = [
            {"title": "Middle East war pushes energy prices higher after Strait of Hormuz threat"},
            {"title": "Iran conflict sparks shipping disruption for oil cargoes"},
        ]
        result = get_geopolitical_features_from_news(news, symbol="OGDC")
        self.assertGreater(result["geo_energy_supply_risk"], 0.0)
        self.assertGreater(result["geo_conflict_risk"], 0.0)
        self.assertGreater(result["geo_energy_war_overlap"], 0.0)


# ============================================================================
# 5. Prediction Regression Check Tests
# ============================================================================

class TestPredictionRegressionCheck(unittest.TestCase):
    """Verify freeze/compare baseline drift checker."""

    def test_snapshot_from_prediction_file(self):
        from backend.prediction_regression_check import _normalize_point, _direction_from_pred

        pred = {
            "date": "2026-03-01",
            "predicted_price": 105.0,
            "upside_potential": 5.0,
            "confidence": 0.85,
        }
        norm = _normalize_point(pred)
        self.assertEqual(norm["predicted_price"], 105.0)
        self.assertEqual(norm["raw_direction"], "BULLISH")
        self.assertEqual(norm["confidence"], 0.85)

    def test_direction_from_pred(self):
        from backend.prediction_regression_check import _direction_from_pred
        self.assertEqual(_direction_from_pred({"upside_potential": 3.0}), "BULLISH")
        self.assertEqual(_direction_from_pred({"upside_potential": -2.0}), "BEARISH")
        self.assertEqual(_direction_from_pred({"upside_potential": 0}), "NEUTRAL")

    def test_pct_diff(self):
        from backend.prediction_regression_check import _pct_diff
        self.assertAlmostEqual(_pct_diff(100.0, 100.5), 0.5, places=2)
        self.assertAlmostEqual(_pct_diff(100.0, 100.0), 0.0, places=2)
        self.assertAlmostEqual(_pct_diff(0.0, 0.0), 0.0, places=2)

    def test_select_day(self):
        from backend.prediction_regression_check import _select_day
        preds = [
            {"day": 1, "predicted_price": 100},
            {"day": 7, "predicted_price": 107},
            {"day": 21, "predicted_price": 121},
        ]
        self.assertEqual(_select_day(preds, 7)["predicted_price"], 107)
        self.assertEqual(_select_day(preds, 1)["predicted_price"], 100)
        # Fallback to index if day key not found
        self.assertEqual(_select_day([], 1), {})

    def test_freeze_and_compare_roundtrip(self):
        """Integration: freeze a mock snapshot, then compare to itself (zero drift)."""
        from backend.prediction_regression_check import BASELINE_FILE, DRIFT_FILE

        # Create temporary prediction files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch DATA_DIR temporarily
            import backend.prediction_regression_check as rc_mod
            orig_data_dir = rc_mod.DATA_DIR
            orig_log_dir = rc_mod.LOG_DIR
            orig_baseline = rc_mod.BASELINE_FILE
            orig_drift = rc_mod.DRIFT_FILE

            tmp_path = Path(tmpdir)
            rc_mod.DATA_DIR = tmp_path
            rc_mod.LOG_DIR = tmp_path / "logs"
            rc_mod.LOG_DIR.mkdir()
            rc_mod.BASELINE_FILE = rc_mod.LOG_DIR / "baseline_snapshot.json"
            rc_mod.DRIFT_FILE = rc_mod.LOG_DIR / "drift_report.json"

            try:
                # Write a fake prediction file
                pred_data = {
                    "symbol": "TEST",
                    "generated_at": "2026-03-01T00:00:00",
                    "model": "test",
                    "current_price": 100.0,
                    "daily_predictions": [
                        {"day": i, "predicted_price": 100 + i * 0.5, "upside_potential": i * 0.5, "confidence": 0.9}
                        for i in range(1, 22)
                    ],
                }
                pred_file = tmp_path / "TEST_research_predictions_2026.json"
                with open(pred_file, "w") as f:
                    json.dump(pred_data, f)

                # Freeze
                exit_code = rc_mod.freeze(["TEST"])
                self.assertEqual(exit_code, 0)
                self.assertTrue(rc_mod.BASELINE_FILE.exists())

                # Compare to itself (should be zero drift)
                exit_code = rc_mod.compare(["TEST"], strict=True)
                self.assertEqual(exit_code, 0)

                with open(rc_mod.DRIFT_FILE) as f:
                    report = json.load(f)
                self.assertEqual(len(report["failures"]), 0)
                for point in report["symbols"]["TEST"]["points"].values():
                    self.assertAlmostEqual(point["drift_pct"], 0.0, places=4)
            finally:
                rc_mod.DATA_DIR = orig_data_dir
                rc_mod.LOG_DIR = orig_log_dir
                rc_mod.BASELINE_FILE = orig_baseline
                rc_mod.DRIFT_FILE = orig_drift


# ============================================================================
# 6. Enhanced News Fetcher Macro Terms Tests
# ============================================================================

class TestEnhancedNewsFetcherTerms(unittest.TestCase):
    """Verify expanded macro categories and relevance terms."""

    def test_macro_categories_include_conflict_terms(self):
        from backend.enhanced_news_fetcher import MACRO_CATEGORIES
        macro_str = " ".join(MACRO_CATEGORIES).lower()
        self.assertIn("conflict", macro_str)
        self.assertIn("recession", macro_str)
        self.assertIn("red sea", macro_str)

    def test_index_relevance_terms_include_geo(self):
        from backend.enhanced_news_fetcher import INDEX_RELEVANCE_TERMS
        macro_terms = " ".join(INDEX_RELEVANCE_TERMS["macro"]).lower()
        self.assertIn("opec", macro_terms)
        self.assertIn("recession", macro_terms)
        self.assertIn("middle east", macro_terms)

    def test_score_index_relevance_returns_positive_for_macro(self):
        from backend.enhanced_news_fetcher import score_index_relevance
        score = score_index_relevance("KSE-100 index drops amid global recession fears")
        self.assertGreater(score, 0.0)


# ============================================================================
# 7. Shadow Comparison Tests
# ============================================================================

class TestShadowComparison(unittest.TestCase):
    """Verify shadow comparison drift computation."""

    def test_shadow_comparison_computes_drift(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available in test environment")
        from backend.stock_analyzer_fixed import _run_shadow_comparison

        baseline = [
            {"day": 1, "predicted_price": 100.0, "upside_potential": 0.0},
            {"day": 2, "predicted_price": 100.5, "upside_potential": 0.5},
            {"day": 3, "predicted_price": 101.0, "upside_potential": 1.0},
            {"day": 4, "predicted_price": 101.5, "upside_potential": 1.5},
            {"day": 5, "predicted_price": 102.0, "upside_potential": 2.0},
            {"day": 6, "predicted_price": 102.5, "upside_potential": 2.5},
            {"day": 7, "predicted_price": 103.0, "upside_potential": 3.0},
        ]
        df = pd.DataFrame({"Close": [100.0] * 10, "Date": pd.date_range("2026-01-01", periods=10)})
        sentiment = {
            "sentiment_score": 0.3,
            "confidence": 0.7,
            "news_items": [{"title": "Company earnings beat expectations"}],
        }
        result = _run_shadow_comparison("TEST", baseline, df, sentiment)
        self.assertTrue(result["enabled"])
        self.assertIn("drift", result)
        self.assertIn("summary", result)
        self.assertIn("geo_features", result)
        self.assertIn("day_1", result["drift"])
        self.assertIn("day_7", result["drift"])

    def test_shadow_comparison_reuses_precomputed_upgraded(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available in test environment")
        from backend.stock_analyzer_fixed import _run_shadow_comparison

        baseline = [
            {"day": 1, "predicted_price": 100.0, "upside_potential": 0.0},
            {"day": 7, "predicted_price": 107.0, "upside_potential": 7.0},
            {"day": 21, "predicted_price": 121.0, "upside_potential": 21.0},
        ]
        upgraded = [
            {"day": 1, "predicted_price": 99.0, "upside_potential": -1.0},
            {"day": 7, "predicted_price": 105.0, "upside_potential": 5.0},
            {"day": 21, "predicted_price": 118.0, "upside_potential": 18.0},
        ]
        df = pd.DataFrame({"Close": [100.0] * 10, "Date": pd.date_range("2026-01-01", periods=10)})
        geo_features = {
            "geo_conflict_risk": 0.8,
            "geo_energy_supply_risk": 0.2,
            "geo_regional_tension": 0.7,
            "geo_global_risk_off": 0.6,
            "geo_news_volume": 0.5,
        }

        result = _run_shadow_comparison(
            "TEST",
            baseline,
            df,
            sentiment_result={},
            upgraded_predictions=upgraded,
            geo_features=geo_features,
        )
        self.assertEqual(result["geo_features"], geo_features)
        self.assertEqual(result["upgraded_count"], len(upgraded))
        self.assertAlmostEqual(result["drift"]["day_1"]["upgraded_price"], 99.0, places=2)


# ============================================================================
# 8. Prediction Tuning Integration Tests
# ============================================================================

class TestPredictionTuningIntegration(unittest.TestCase):
    """Verify tweaks preserve raw values and respect config."""

    def test_tweaks_disabled_returns_unchanged(self):
        from backend.prediction_tuning import apply_prediction_tweaks, TweakConfig
        preds = [{"predicted_price": 100.0, "upside_potential": 3.0, "confidence": 0.9}]
        cfg = TweakConfig(enabled=False)
        result = apply_prediction_tweaks(preds, cfg)
        self.assertEqual(result[0]["predicted_price"], 100.0)

    def test_tweaks_enabled_stores_raw(self):
        from backend.prediction_tuning import apply_prediction_tweaks, TweakConfig
        preds = [
            {"predicted_price": 100.0, "upside_potential": 3.0, "confidence": 0.9, "current_price": 100.0}
        ]
        cfg = TweakConfig(enabled=True, bias_correction_pct=0.0, neutral_band_pct=0.0)
        result = apply_prediction_tweaks(preds, cfg)
        self.assertIn("raw_upside_potential", result[0])
        self.assertEqual(result[0]["raw_upside_potential"], 3.0)

    def test_williams_brake_flattens_conflict(self):
        from backend.prediction_tuning import apply_prediction_tweaks, TweakConfig
        # Current documented behavior: when Williams confidence > 0.60, the
        # brake DAMPENS the conflicting prediction by 50% (not zeroes — see
        # backend/post_process/_tuning.py:160-167). Low-confidence Williams
        # signals are deliberately ignored so a single noisy bar can't kill
        # a high-confidence model call.
        cfg = TweakConfig(
            enabled=True,
            use_williams_conflict_brake=True,
            bias_correction_pct=0.0,
            neutral_band_pct=0.0,
        )

        # Case 1: high-confidence Williams DOWN dampens BULLISH prediction.
        preds_high = [{
            "predicted_price": 110.0, "upside_potential": 5.0,
            "confidence": 0.95, "williams_signal": "DOWN",
            "williams_confidence": 0.8,
            "current_price": 100.0,
        }]
        result_high = apply_prediction_tweaks(preds_high, cfg)
        self.assertAlmostEqual(result_high[0]["upside_potential"], 2.5, places=2)

        # Case 2: low-confidence Williams signal is ignored — original kept.
        preds_low = [{
            "predicted_price": 110.0, "upside_potential": 5.0,
            "confidence": 0.95, "williams_signal": "DOWN",
            "williams_confidence": 0.3,
            "current_price": 100.0,
        }]
        result_low = apply_prediction_tweaks(preds_low, cfg)
        self.assertEqual(result_low[0]["upside_potential"], 5.0)


if __name__ == "__main__":
    unittest.main()
