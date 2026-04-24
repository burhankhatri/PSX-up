import unittest

from backend.geopolitical_features import (
    build_geo_interpretation,
    build_geopolitical_daily_adjustments,
)


SEVERE_DOWNSTREAM_NEWS = [
    {
        "title": "Govt hikes petrol price by Rs55 per litre",
        "description": "Domestic fuel shock ripples across Pakistan",
        "source": "Dawn",
    },
    {
        "title": "Middle East war pushes energy prices higher after Strait of Hormuz blockade",
        "description": "Shipping disruption lifts crude and regional inflation fears",
        "source": "Business Recorder",
    },
    {
        "title": "SBP policy rate may stay high as inflation surge hits auto financing",
        "description": "Consumer financing slowdown raises car demand concerns",
        "source": "Geo News",
    },
]

SEVERE_GEO_FEATURES = {
    "geo_conflict_risk": 0.64,
    "geo_energy_supply_risk": 0.78,
    "geo_regional_tension": 0.48,
    "geo_global_risk_off": 0.35,
    "geo_news_volume": 0.80,
    "geo_energy_war_overlap": 0.55,
}

SEVERE_SHOCK = {
    "shock_detected": True,
    "max_severity": 3.0,
    "emergency_multiplier": 2.5,
    "shock_half_life": 7,
    "matched_patterns": ["blockade", "middle east war"],
}


class DownstreamGeoPenaltyTests(unittest.TestCase):
    def test_sazew_rs55_shock_enters_downstream_headwind_mode(self):
        interpretation = build_geo_interpretation(
            news_items=SEVERE_DOWNSTREAM_NEWS,
            geo_features=SEVERE_GEO_FEATURES,
            shock_data=SEVERE_SHOCK,
            symbol="SAZEW",
        )
        generic = build_geopolitical_daily_adjustments(
            SEVERE_GEO_FEATURES,
            prediction_length=7,
            symbol="SAZEW",
            shock_data=SEVERE_SHOCK,
        )
        downstream = build_geopolitical_daily_adjustments(
            SEVERE_GEO_FEATURES,
            prediction_length=7,
            symbol="SAZEW",
            shock_data=SEVERE_SHOCK,
            interpretation=interpretation,
        )

        self.assertTrue(interpretation["bearish_for_downstream"])
        self.assertEqual(interpretation["evidence_quality"], "good")
        self.assertGreater(interpretation["interest_rate_squeeze_score"], 0.70)
        self.assertEqual(downstream["summary"]["overlay_mode"], "downstream_headwind")
        self.assertLess(
            downstream["adjustments"][0]["capped_adjustment"],
            generic["adjustments"][0]["capped_adjustment"],
        )

    def test_tech_name_stays_mild_due_to_fx_offset(self):
        interpretation = build_geo_interpretation(
            news_items=SEVERE_DOWNSTREAM_NEWS,
            geo_features=SEVERE_GEO_FEATURES,
            shock_data=SEVERE_SHOCK,
            symbol="SYS",
        )

        self.assertEqual(interpretation["vulnerability_profile"]["sector"], "technology")
        self.assertLess(interpretation["headwind_strength"], 0.10)

    def test_cement_names_show_high_margin_compression(self):
        # LUCK / CHCC are cement manufacturers; a downstream macro shock
        # (petrol hike + regional war + high SBP policy rate) squeezes
        # their margins (fuel input cost + coal freight + interest drag).
        # Threshold is 0.40 after the MDPI-study recalibration of
        # VULNERABILITY_PROFILE (LUCK energy_input_drag 0.85 -> 0.50,
        # DGKC 0.84 -> 0.70) that brought cement sensitivities in line
        # with empirical estimates.
        for symbol in ("LUCK", "CHCC"):
            with self.subTest(symbol=symbol):
                interpretation = build_geo_interpretation(
                    news_items=SEVERE_DOWNSTREAM_NEWS,
                    geo_features=SEVERE_GEO_FEATURES,
                    shock_data=SEVERE_SHOCK,
                    symbol=symbol,
                )
                self.assertGreater(interpretation["margin_compression_score"], 0.40)

    def test_auto_names_show_high_interest_rate_squeeze(self):
        for symbol in ("INDU", "HCAR", "SAZEW"):
            with self.subTest(symbol=symbol):
                interpretation = build_geo_interpretation(
                    news_items=SEVERE_DOWNSTREAM_NEWS,
                    geo_features=SEVERE_GEO_FEATURES,
                    shock_data=SEVERE_SHOCK,
                    symbol=symbol,
                )
                self.assertGreater(interpretation["interest_rate_squeeze_score"], 0.75)

    def test_weak_news_does_not_trigger_downstream_override(self):
        weak_news = [
            {
                "title": "Govt hikes petrol price by Rs55 per litre",
                "description": "Consumers brace for higher fuel bills",
                "source": "Minute Mirror",
            }
        ]

        interpretation = build_geo_interpretation(
            news_items=weak_news,
            geo_features=SEVERE_GEO_FEATURES,
            shock_data=SEVERE_SHOCK,
            symbol="SAZEW",
        )

        self.assertFalse(interpretation["bearish_for_downstream"])
        self.assertEqual(interpretation["evidence_quality"], "weak")


if __name__ == "__main__":
    unittest.main()
