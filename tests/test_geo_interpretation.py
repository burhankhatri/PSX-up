import copy
import unittest
from pathlib import Path

from backend.geopolitical_features import (
    build_geo_interpretation,
    build_geopolitical_daily_adjustments,
)


class GeoInterpretationTests(unittest.TestCase):
    def test_upstream_energy_shock_returns_bullish_interpretation(self):
        news_items = [
            {
                "title": "Middle East war pushes energy prices higher as Strait of Hormuz shipping disruption worsens",
                "description": "Govt hikes petrol price by Rs55 per litre amid oil surge",
            },
            {
                "title": "Circular debt payment release expected for E&P firms",
                "description": "Receivables cleared after fuel hike",
            },
        ]
        geo_features = {
            "geo_conflict_risk": 0.64,
            "geo_energy_supply_risk": 0.72,
            "geo_regional_tension": 0.48,
            "geo_global_risk_off": 0.30,
            "geo_news_volume": 0.80,
            "geo_energy_war_overlap": 0.60,
        }
        shock_data = {"shock_detected": True}

        # build_geo_interpretation requires crude_data to confirm the
        # bullish_for_upstream gate (added to prevent generic "war news"
        # alone from flipping interpretation — see geopolitical_features.py
        # line ~1793: crude_confirmed = oil_change_pct >= 2.0 OR
        # oil_trend_pct >= 4.0).
        crude_data = {"oil_change_pct": 3.5, "oil_trend_pct": 5.0}

        result = build_geo_interpretation(
            news_items=news_items,
            geo_features=geo_features,
            shock_data=shock_data,
            symbol="OGDC",
            crude_data=crude_data,
        )

        self.assertEqual(result["sector_interpretation"], "upstream_energy_tailwind")
        self.assertEqual(result["polarity"], "bullish_for_upstream")
        self.assertTrue(result["bullish_for_upstream"])
        self.assertGreater(result["tailwind_score"], 0.50)
        self.assertGreater(result["cashflow_support_score"], 0.50)
        self.assertEqual(result["fuel_hike_magnitude_rs"], 55.0)
        self.assertGreater(result["shock_severity_score"], 0.0)
        self.assertGreater(result["fuel_hike_score"], 0.80)
        self.assertEqual(result["circular_debt_relief_score"], 1.0)

    def test_non_upstream_keeps_generic_interpretation(self):
        news_items = [
            {
                "title": "Middle East war pushes energy prices higher",
                "description": "Govt hikes petrol price by Rs55 per litre",
            }
        ]
        geo_features = {
            "geo_conflict_risk": 0.64,
            "geo_energy_supply_risk": 0.72,
            "geo_regional_tension": 0.48,
            "geo_global_risk_off": 0.30,
            "geo_news_volume": 0.80,
            "geo_energy_war_overlap": 0.60,
        }
        shock_data = {"shock_detected": True}

        result = build_geo_interpretation(
            news_items=news_items,
            geo_features=geo_features,
            shock_data=shock_data,
            symbol="PSO",
        )

        self.assertEqual(result["sector_interpretation"], "generic")
        self.assertFalse(result["bullish_for_upstream"])
        self.assertNotEqual(result["polarity"], "bullish_for_upstream")

    def test_interpretation_builder_is_additive_only(self):
        news_items = [
            {
                "title": "Middle East war pushes energy prices higher",
                "description": "Govt hikes petrol price by Rs55 per litre",
            }
        ]
        geo_features = {
            "geo_conflict_risk": 0.64,
            "geo_energy_supply_risk": 0.72,
            "geo_regional_tension": 0.48,
            "geo_global_risk_off": 0.30,
            "geo_news_volume": 0.80,
            "geo_energy_war_overlap": 0.60,
        }
        shock_data = {"shock_detected": True, "shock_reason": "Matched shock patterns: blockade"}
        before_adjustments = build_geopolitical_daily_adjustments(
            copy.deepcopy(geo_features),
            prediction_length=7,
            symbol="OGDC",
            shock_data=copy.deepcopy(shock_data),
        )

        geo_features_before = copy.deepcopy(geo_features)
        shock_data_before = copy.deepcopy(shock_data)
        news_before = copy.deepcopy(news_items)
        _ = build_geo_interpretation(
            news_items=news_items,
            geo_features=geo_features,
            shock_data=shock_data,
            symbol="OGDC",
        )
        after_adjustments = build_geopolitical_daily_adjustments(
            copy.deepcopy(geo_features),
            prediction_length=7,
            symbol="OGDC",
            shock_data=copy.deepcopy(shock_data),
        )

        self.assertEqual(before_adjustments, after_adjustments)
        self.assertEqual(geo_features, geo_features_before)
        self.assertEqual(shock_data, shock_data_before)
        self.assertEqual(news_items, news_before)

        results = {
            "daily_predictions_without_geo": [{"day": 7, "predicted_price": 280.94}],
            "daily_predictions_with_geo": [{"day": 7, "predicted_price": 278.32}],
            "direction_meta": {"display_direction": "BEARISH"},
            "geo_comparison": {"enabled": True},
        }
        baseline_before = copy.deepcopy(results["daily_predictions_without_geo"])
        geo_before = copy.deepcopy(results["daily_predictions_with_geo"])
        direction_before = copy.deepcopy(results["direction_meta"])

        results["geo_comparison"]["interpretation"] = build_geo_interpretation(
            news_items=news_items,
            geo_features=geo_features,
            shock_data=shock_data,
            symbol="OGDC",
        )

        self.assertEqual(results["daily_predictions_without_geo"], baseline_before)
        self.assertEqual(results["daily_predictions_with_geo"], geo_before)
        self.assertEqual(results["direction_meta"], direction_before)
        self.assertEqual(
            results["daily_predictions_with_geo"][0]["predicted_price"],
            geo_before[0]["predicted_price"],
        )

    def test_geo_overlay_ui_references_interpretation_fields(self):
        html = Path("web/stock_analyzer.html").read_text()
        self.assertIn("geoMeta.interpretation", html)
        self.assertIn("Catalyst Breakdown", html)
        self.assertIn("Shock Severity", html)
        self.assertIn("Fuel Hike Catalyst", html)
        self.assertIn("Circular Debt Relief", html)
        self.assertIn("Net Tailwind Strength", html)
        self.assertIn("downstream_headwind", html)
        self.assertIn("Margin Compression", html)
        self.assertIn("Interest Rate Squeeze", html)
        self.assertIn("Demand Destruction", html)
        self.assertIn("Projected Inflation Spike", html)
        self.assertIn("Net Headwind Strength", html)
        self.assertIn("geo-catalyst-fill-bearish", html)
        self.assertIn("renderGeoCatalystBreakdown", html)

    def test_major_blockade_and_rs55_hike_produce_larger_tailwind_than_mild_case(self):
        major_news = [
            {
                "title": "Middle East war pushes energy prices higher as Strait of Hormuz shipping disruption worsens",
                "description": "Govt hikes petrol price by Rs55 per litre amid blockade fears",
            },
            {
                "title": "Circular debt payment release expected for E&P firms",
                "description": "Receivables cleared after fuel hike",
            },
        ]
        mild_news = [
            {
                "title": "Oil prices rise on regional tensions",
                "description": "Energy market watches Middle East developments",
            }
        ]

        major_features = {
            "geo_conflict_risk": 0.64,
            "geo_energy_supply_risk": 0.72,
            "geo_regional_tension": 0.48,
            "geo_global_risk_off": 0.30,
            "geo_news_volume": 0.80,
            "geo_energy_war_overlap": 0.60,
        }
        mild_features = {
            "geo_conflict_risk": 0.25,
            "geo_energy_supply_risk": 0.52,
            "geo_regional_tension": 0.18,
            "geo_global_risk_off": 0.12,
            "geo_news_volume": 0.40,
            "geo_energy_war_overlap": 0.10,
        }
        major_shock = {
            "shock_detected": True,
            "max_severity": 3.0,
            "emergency_multiplier": 2.5,
            "shock_half_life": 7,
        }
        mild_shock = {
            "shock_detected": True,
            "max_severity": 1.5,
            "emergency_multiplier": 1.3,
            "shock_half_life": 10,
        }

        # Both interpretations require crude_data to satisfy the
        # bullish_for_upstream gate. The MAJOR case sees confirmed oil
        # rally; the MILD case shows oil drifting upward but not crossing
        # the gate's 2%/4% thresholds — keeping the comparative ordering.
        major_crude = {"oil_change_pct": 4.0, "oil_trend_pct": 6.0}
        mild_crude = {"oil_change_pct": 2.5, "oil_trend_pct": 4.5}

        major_interp = build_geo_interpretation(
            news_items=major_news,
            geo_features=major_features,
            shock_data=major_shock,
            symbol="OGDC",
            crude_data=major_crude,
        )
        mild_interp = build_geo_interpretation(
            news_items=mild_news,
            geo_features=mild_features,
            shock_data=mild_shock,
            symbol="OGDC",
            crude_data=mild_crude,
        )

        major_adj = build_geopolitical_daily_adjustments(
            major_features,
            prediction_length=7,
            symbol="OGDC",
            shock_data=major_shock,
            interpretation=major_interp,
        )
        mild_adj = build_geopolitical_daily_adjustments(
            mild_features,
            prediction_length=7,
            symbol="OGDC",
            shock_data=mild_shock,
            interpretation=mild_interp,
        )

        self.assertGreater(
            major_adj["adjustments"][0]["capped_adjustment"],
            mild_adj["adjustments"][0]["capped_adjustment"],
        )
        self.assertGreater(
            major_adj["summary"]["tailwind_strength"],
            mild_adj["summary"]["tailwind_strength"],
        )


if __name__ == "__main__":
    unittest.main()
