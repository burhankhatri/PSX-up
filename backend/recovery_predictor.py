#!/usr/bin/env python3
"""
Post-crash recovery predictor for PSX.

Uses historical crash-recovery patterns from Pakistan Stock Exchange to generate
three recovery scenarios (bull, base, bear) when a geopolitical shock is detected.

Key precedents:
  - 2025 India-Pakistan war: -12.6% crash → +15.76% in 3 days after ceasefire
  - 2020 COVID crash: -30% → gradual recovery over 4 months
  - 2022 political crisis: -8% → recovery in 2-3 weeks
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Historical crash-recovery patterns (PSX documented)
# ---------------------------------------------------------------------------

HISTORICAL_CRASH_PATTERNS: Dict[str, Dict] = {
    "india_pak_2025": {
        "trigger": "India-Pakistan military conflict (Pahalgam attack response)",
        "crash_pct": -12.6,
        "crash_type": "military",
        "recovery_trigger": "US-brokered ceasefire announcement",
        "recovery_day1_pct": 9.0,
        "recovery_3day_pct": 15.76,
        "full_recovery_days": 5,
        "additional_catalyst": "IMF funding approval coincided",
        "year": 2025,
    },
    "covid_2020": {
        "trigger": "Global pandemic lockdown + oil price war",
        "crash_pct": -30.0,
        "crash_type": "pandemic",
        "recovery_trigger": "Stimulus packages + lockdown easing",
        "recovery_day1_pct": 3.5,
        "recovery_3day_pct": 8.0,
        "full_recovery_days": 120,
        "additional_catalyst": "SBP rate cuts, fiscal stimulus",
        "year": 2020,
    },
    "political_crisis_2022": {
        "trigger": "Political instability (no-confidence vote, regime change)",
        "crash_pct": -8.0,
        "crash_type": "political",
        "recovery_trigger": "New government formation + policy continuity signals",
        "recovery_day1_pct": 2.0,
        "recovery_3day_pct": 5.5,
        "full_recovery_days": 18,
        "additional_catalyst": "IMF programme continuation confirmed",
        "year": 2022,
    },
    "balance_of_payments_2022": {
        "trigger": "PKR freefall + foreign reserves crisis",
        "crash_pct": -15.0,
        "crash_type": "economic",
        "recovery_trigger": "IMF bailout tranche + Saudi deposit",
        "recovery_day1_pct": 2.5,
        "recovery_3day_pct": 6.0,
        "full_recovery_days": 45,
        "additional_catalyst": "Friendly country deposits, SBP intervention",
        "year": 2022,
    },
    "red_sea_2024": {
        "trigger": "Red Sea shipping disruption / Houthi attacks",
        "crash_pct": -5.0,
        "crash_type": "energy",
        "recovery_trigger": "Shipping rerouting + oil price stabilisation",
        "recovery_day1_pct": 1.5,
        "recovery_3day_pct": 3.0,
        "full_recovery_days": 14,
        "additional_catalyst": "Pakistan not a direct party",
        "year": 2024,
    },
}


# Crash type classification from shock categories
SHOCK_CATEGORY_TO_CRASH_TYPE = {
    "conflict": "military",
    "energy": "energy",
    "market": "military",  # market halts usually from conflict
    "risk_off": "economic",
}


def _find_best_precedent(
    crash_pct: float, crash_type: str
) -> Dict:
    """Find the historical pattern most similar to the current crash."""
    best = None
    best_score = float("inf")

    for key, pattern in HISTORICAL_CRASH_PATTERNS.items():
        # Score by type match + magnitude similarity
        type_bonus = 0.0 if pattern["crash_type"] == crash_type else 5.0
        magnitude_diff = abs(abs(crash_pct) - abs(pattern["crash_pct"]))
        score = type_bonus + magnitude_diff

        if score < best_score:
            best_score = score
            best = {**pattern, "precedent_key": key}

    return best or list(HISTORICAL_CRASH_PATTERNS.values())[0]


def _generate_recovery_curve(
    current_price: float,
    day1_pct: float,
    day3_pct: float,
    full_recovery_days: int,
    crash_pct: float,
) -> List[Dict]:
    """Generate a day-by-day recovery curve using exponential approach."""
    pre_crash_price = current_price / (1 + crash_pct / 100.0)
    targets = []

    for day in range(1, min(full_recovery_days + 1, 31)):  # Cap at 30 days
        if day == 1:
            recovery_pct = day1_pct
        elif day <= 3:
            # Interpolate between day1 and day3
            t = (day - 1) / 2.0
            recovery_pct = day1_pct + t * (day3_pct - day1_pct)
        else:
            # Exponential approach: recovery slows over time
            total_recovery_needed = abs(crash_pct)
            remaining_after_3d = total_recovery_needed - day3_pct
            if remaining_after_3d <= 0:
                recovery_pct = total_recovery_needed
            else:
                days_remaining = max(1, full_recovery_days - 3)
                progress = 1.0 - math.exp(-2.0 * (day - 3) / days_remaining)
                recovery_pct = day3_pct + remaining_after_3d * progress

        target_price = current_price * (1 + recovery_pct / 100.0)
        targets.append({
            "day": day,
            "recovery_pct": round(recovery_pct, 2),
            "target_price": round(target_price, 2),
            "distance_from_pre_crash_pct": round(
                (target_price / pre_crash_price - 1) * 100, 2
            ),
        })

    return targets


def predict_recovery(
    current_crash_pct: float,
    crash_type: str = "military",
    has_ceasefire: bool = False,
) -> Dict[str, Dict]:
    """
    Generate 3 recovery scenarios based on historical precedents.

    Args:
        current_crash_pct: Negative percentage (e.g., -9.57)
        crash_type: "military", "economic", "energy", "pandemic", "political"
        has_ceasefire: Whether a ceasefire/resolution has been announced

    Returns:
        dict with "bull", "base", "bear" scenarios
    """
    precedent = _find_best_precedent(current_crash_pct, crash_type)
    crash_magnitude = abs(current_crash_pct)

    # Scale recovery proportionally to crash magnitude vs precedent
    scale = crash_magnitude / max(abs(precedent["crash_pct"]), 1.0)

    scenarios = {}

    # BULL CASE: Ceasefire/resolution → V-shaped (mirrors 2025 precedent)
    bull_day1 = precedent["recovery_day1_pct"] * scale
    bull_day3 = precedent["recovery_3day_pct"] * scale
    if has_ceasefire:
        bull_day1 *= 1.2  # Boost if ceasefire confirmed
        bull_day3 *= 1.2
    scenarios["bull"] = {
        "label": "V-Shaped Recovery (Ceasefire/Resolution)",
        "condition": "Ceasefire or diplomatic resolution announced",
        "day1_pct": round(bull_day1, 2),
        "day3_pct": round(bull_day3, 2),
        "day7_pct": round(bull_day3 * 1.15, 2),  # slight overshoot
        "full_recovery_days": max(3, int(precedent["full_recovery_days"] * 0.8)),
        "confidence": 0.75 if has_ceasefire else 0.30,
        "precedent": precedent["trigger"],
        "precedent_recovery": f"+{precedent['recovery_3day_pct']}% in 3 days ({precedent['year']})",
    }

    # BASE CASE: Gradual de-escalation → U-shaped
    base_day1 = bull_day1 * 0.35
    base_day3 = bull_day3 * 0.40
    base_day7 = bull_day3 * 0.60
    scenarios["base"] = {
        "label": "Gradual Recovery (De-escalation)",
        "condition": "Tensions ease gradually, no formal resolution",
        "day1_pct": round(base_day1, 2),
        "day3_pct": round(base_day3, 2),
        "day7_pct": round(base_day7, 2),
        "full_recovery_days": int(precedent["full_recovery_days"] * 2.5),
        "confidence": 0.45,
        "precedent": precedent["trigger"],
        "precedent_recovery": f"Scaled from {precedent['year']} pattern",
    }

    # BEAR CASE: Escalation continues → further downside then slow recovery
    bear_day1 = -(crash_magnitude * 0.15)  # another 15% of crash magnitude down
    bear_day3 = -(crash_magnitude * 0.25)
    bear_day7 = -(crash_magnitude * 0.10)  # starts stabilising
    scenarios["bear"] = {
        "label": "Extended Decline (Escalation Continues)",
        "condition": "Conflict escalates further, no resolution in sight",
        "day1_pct": round(bear_day1, 2),
        "day3_pct": round(bear_day3, 2),
        "day7_pct": round(bear_day7, 2),
        "full_recovery_days": int(precedent["full_recovery_days"] * 5),
        "confidence": 0.25,
        "precedent": "Worst-case extrapolation",
        "precedent_recovery": "Extended conflict scenario",
    }

    return scenarios


def get_recovery_analysis(
    symbol: str,
    current_price: float,
    recent_high: float,
    geo_shock_data: Dict,
) -> Dict:
    """
    Full recovery analysis when a geopolitical shock is detected.

    Args:
        symbol: Stock symbol (e.g., "KSE100")
        current_price: Current/last close price
        recent_high: Recent peak price (for crash % calculation)
        geo_shock_data: Output from detect_geopolitical_shocks()

    Returns:
        Structured recovery analysis with 3 scenarios
    """
    if not geo_shock_data.get("shock_detected"):
        return {"enabled": False, "reason": "No geopolitical shock detected"}

    crash_pct = ((current_price - recent_high) / recent_high) * 100.0 if recent_high > 0 else 0.0

    # Classify crash type from shock events
    shock_categories = [e.get("category", "conflict") for e in geo_shock_data.get("shock_events", [])]
    primary_category = max(set(shock_categories), key=shock_categories.count) if shock_categories else "conflict"
    crash_type = SHOCK_CATEGORY_TO_CRASH_TYPE.get(primary_category, "military")

    # Extract trajectory from shock data (set by detect_geopolitical_shocks)
    trajectory = geo_shock_data.get("trajectory", {})
    has_ceasefire = trajectory.get("has_ceasefire", False)
    traj_label = trajectory.get("trajectory", "stalemate")
    has_escalation = trajectory.get("has_escalation", False)

    scenarios = predict_recovery(
        current_crash_pct=crash_pct,
        crash_type=crash_type,
        has_ceasefire=has_ceasefire,
    )

    # Dynamically adjust scenario confidence based on trajectory
    if traj_label == "ceasefire":
        scenarios["bull"]["confidence"] = min(0.85, scenarios["bull"]["confidence"] + 0.30)
        scenarios["base"]["confidence"] = max(0.10, scenarios["base"]["confidence"] - 0.15)
        scenarios["bear"]["confidence"] = max(0.05, scenarios["bear"]["confidence"] - 0.20)
        scenarios["bull"]["condition"] = "Ceasefire/resolution DETECTED in news"
    elif traj_label == "de_escalating":
        scenarios["bull"]["confidence"] = min(0.70, scenarios["bull"]["confidence"] + 0.15)
        scenarios["base"]["confidence"] = min(0.60, scenarios["base"]["confidence"] + 0.10)
        scenarios["bear"]["confidence"] = max(0.10, scenarios["bear"]["confidence"] - 0.10)
        scenarios["base"]["condition"] = "De-escalation signals detected in news"
    elif traj_label == "escalating":
        scenarios["bull"]["confidence"] = max(0.10, scenarios["bull"]["confidence"] - 0.15)
        scenarios["base"]["confidence"] = max(0.15, scenarios["base"]["confidence"] - 0.10)
        scenarios["bear"]["confidence"] = min(0.55, scenarios["bear"]["confidence"] + 0.20)
        scenarios["bear"]["condition"] = "ACTIVE ESCALATION detected - conflict widening"
    # stalemate: keep defaults

    # If LLM provided a market impact estimate, use it to refine the base scenario
    llm_data = trajectory.get("llm_assessment")
    if llm_data and llm_data.get("market_impact_pct") is not None:
        llm_impact = llm_data["market_impact_pct"]
        # Blend LLM estimate into the base case day1 (50/50 with keyword-based)
        old_day1 = scenarios["base"]["day1_pct"]
        scenarios["base"]["day1_pct"] = round((old_day1 + llm_impact) / 2, 2)
        scenarios["base"]["llm_refined"] = True

    # Generate price targets for each scenario
    for key, scenario in scenarios.items():
        scenario["price_targets"] = {
            "day1": round(current_price * (1 + scenario["day1_pct"] / 100), 2),
            "day3": round(current_price * (1 + scenario["day3_pct"] / 100), 2),
            "day7": round(current_price * (1 + scenario["day7_pct"] / 100), 2),
        }

    return {
        "enabled": True,
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "recent_high": round(recent_high, 2),
        "crash_pct": round(crash_pct, 2),
        "crash_type": crash_type,
        "shock_severity": geo_shock_data.get("max_severity", 0),
        "trajectory": {
            "status": traj_label,
            "score": trajectory.get("trajectory_score", 0.0),
            "has_ceasefire": has_ceasefire,
            "has_escalation": has_escalation,
            "resolution_signals": trajectory.get("resolution_signals", []),
            "escalation_signals": trajectory.get("escalation_signals", []),
            "llm_assessment": trajectory.get("llm_assessment"),
            "assessment_method": trajectory.get("assessment_method", "keyword_only"),
        },
        "scenarios": scenarios,
        "precedent_used": _find_best_precedent(crash_pct, crash_type).get("precedent_key", ""),
        "disclaimer": (
            "Recovery predictions are based on historical PSX patterns and should not "
            "be treated as financial advice. Actual recovery depends on geopolitical "
            "developments, policy responses, and market sentiment."
        ),
        "generated_at": datetime.now().isoformat(),
    }
