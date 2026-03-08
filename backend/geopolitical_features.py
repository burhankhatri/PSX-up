#!/usr/bin/env python3
"""
Geopolitical feature pipeline for flagged shadow rollouts.

Computes date-indexed risk signals from the existing news/sentiment pipeline.
All features normalise to [0, 1] and default to neutral (0.0) on any failure.

Feature set:
  geo_conflict_risk      – armed conflict / military escalation
  geo_energy_supply_risk – oil / gas / shipping disruption
  geo_regional_tension   – Pakistan-specific & South-Asia neighbours
  geo_global_risk_off    – global safe-haven / sell-off signals
  geo_news_volume        – normalised geopolitical news volume
"""

from __future__ import annotations

import math
import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from backend.energy_shock_features import (
        ENERGY_SHOCK_SYMBOLS,
        parse_local_fuel_price_delta,
        score_circular_debt_signal,
        has_regional_war_terms,
        is_energy_supply_shock_text,
    )
except ImportError:
    from energy_shock_features import (  # type: ignore
        ENERGY_SHOCK_SYMBOLS,
        parse_local_fuel_price_delta,
        score_circular_debt_signal,
        has_regional_war_terms,
        is_energy_supply_shock_text,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq / LLM availability (reuse same client as sentiment_analyzer)
# ---------------------------------------------------------------------------
try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


def _get_groq_client():
    """Get a Groq client (shared pattern with sentiment_analyzer)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Term dictionaries (broader coverage than v1)
# ---------------------------------------------------------------------------

GEO_TERMS: Dict[str, List[str]] = {
    "conflict": [
        "war", "conflict", "attack", "strike", "escalation", "border tension",
        "military operation", "airstrikes", "missile", "drone strike", "ceasefire",
        "armed forces", "military buildup", "shelling", "invasion",
        "nuclear", "defence pact", "skirmish", "insurgency",
    ],
    "energy_supply": [
        "oil", "gas", "lng", "shipping lane", "strait of hormuz", "supply disruption",
        "pipeline", "opec", "energy crisis", "fuel shortage", "refinery",
        "brent crude", "wti", "petrochemical", "energy embargo",
        "red sea", "suez canal",
    ],
    "regional": [
        "iran", "israel", "gulf", "middle east", "afghanistan", "pakistan border",
        "india pakistan", "kashmir", "balochistan", "loc", "ttp",
        "cpec", "china belt road", "south asia", "saudi arabia",
        "taliban", "terrorism", "militant",
    ],
    "risk_off": [
        "sanctions", "safe haven", "global risk", "sell-off", "volatility spike",
        "recession", "default", "credit downgrade", "imf bailout",
        "capital flight", "currency crisis", "debt crisis",
        "trade war", "tariff", "embargo", "geopolitical risk",
    ],
}

# Sector-specific amplifiers: if a symbol is in a sensitive sector,
# certain risk categories get a small multiplier.
SECTOR_AMPLIFIERS: Dict[str, Dict[str, float]] = {
    "energy": {"energy_supply": 1.5, "conflict": 1.2},
    "cement": {"regional": 1.2},
    "banking": {"risk_off": 1.3},
    "fertilizer": {"energy_supply": 1.3},
    "technology": {"risk_off": 1.1},
}

# Symbol-to-sector mapping (common PSX tickers).
SYMBOL_SECTOR: Dict[str, str] = {
    "OGDC": "energy", "PPL": "energy", "PSO": "energy", "POL": "energy",
    "MARI": "energy", "ATRL": "energy",
    "LUCK": "cement", "DGKC": "cement", "MLCF": "cement", "PIOC": "cement",
    "FCCL": "cement", "CHCC": "cement", "KOHC": "cement",
    "HBL": "banking", "UBL": "banking", "MCB": "banking", "BAHL": "banking",
    "NBP": "banking", "ABL": "banking", "MEBL": "banking",
    "FFC": "fertilizer", "EFERT": "fertilizer", "FFBL": "fertilizer",
    "FATIMA": "fertilizer",
    "SYS": "technology", "TRG": "technology", "NETSOL": "technology",
}

CACHE_DIR = Path(__file__).parent.parent / "data" / "news_cache"
UPSTREAM_EP_SYMBOLS = frozenset({"OGDC", "PPL", "POL", "MARI"})

POSITIVE_OIL_CONTEXT_TERMS = (
    "oil prices rise",
    "oil prices rising",
    "oil prices higher",
    "oil prices surge",
    "oil prices jump",
    "crude rises",
    "crude gains",
    "crude jumps",
    "brent rises",
    "brent jumps",
    "brent surges",
    "wti rises",
    "wti jumps",
    "wti surges",
)

UPSTREAM_PAYMENT_RELIEF_TERMS = (
    "payment release",
    "payment released",
    "payments released",
    "payment to ogdc",
    "payment to ppl",
    "payment to mari",
    "payment to pol",
    "receivables cleared",
    "cash release",
    "receives rs",
    "received rs",
    "debt resolution",
)

# ---------------------------------------------------------------------------
# Shock event patterns – severity tiers for emergency detection
# ---------------------------------------------------------------------------

SHOCK_EVENT_PATTERNS: Dict[str, Dict] = {
    # CRITICAL – war declarations, active military strikes, market halts
    "declares war": {"severity": 3.0, "category": "conflict"},
    "open war": {"severity": 3.0, "category": "conflict"},
    "bombs": {"severity": 3.0, "category": "conflict"},
    "bombing": {"severity": 3.0, "category": "conflict"},
    "bombed": {"severity": 3.0, "category": "conflict"},
    "invades": {"severity": 3.0, "category": "conflict"},
    "invasion": {"severity": 3.0, "category": "conflict"},
    "martial law": {"severity": 3.0, "category": "conflict"},
    "nuclear strike": {"severity": 3.0, "category": "conflict"},
    "trading halted": {"severity": 3.0, "category": "market"},
    "market halted": {"severity": 3.0, "category": "market"},
    "circuit breaker": {"severity": 3.0, "category": "market"},
    "combat operations": {"severity": 3.0, "category": "conflict"},
    "airstrikes on": {"severity": 3.0, "category": "conflict"},
    "strait of hormuz closed": {"severity": 3.0, "category": "energy"},
    "strait of hormuz shut": {"severity": 3.0, "category": "energy"},
    "hormuz closure": {"severity": 3.0, "category": "energy"},
    "hormuz closure threat": {"severity": 3.0, "category": "energy"},
    "blockade": {"severity": 2.5, "category": "energy"},
    "blocked shipping lane": {"severity": 2.5, "category": "energy"},
    "middle east war pushes energy prices higher": {"severity": 2.5, "category": "energy"},
    # CRITICAL – PSX-specific market crash language
    "record single-day": {"severity": 3.0, "category": "market"},
    "record single day": {"severity": 3.0, "category": "market"},
    "record-day fall": {"severity": 3.0, "category": "market"},
    "panic selling": {"severity": 3.0, "category": "market"},
    "market panic": {"severity": 3.0, "category": "market"},
    # CRITICAL – war language as used by Pakistani/regional media
    "middle east war": {"severity": 3.0, "category": "conflict"},
    "us-iran conflict": {"severity": 3.0, "category": "conflict"},
    "us iran conflict": {"severity": 3.0, "category": "conflict"},
    "israel conflict": {"severity": 3.0, "category": "conflict"},
    "strikes on iran": {"severity": 3.0, "category": "conflict"},
    "strikes against iran": {"severity": 3.0, "category": "conflict"},
    "afghan war": {"severity": 3.0, "category": "conflict"},
    "pak afghan": {"severity": 2.5, "category": "conflict"},
    "pakistan afghan": {"severity": 2.5, "category": "conflict"},
    # SEVERE – major escalations, emergencies
    "ceasefire collapsed": {"severity": 2.0, "category": "conflict"},
    "ceasefire violated": {"severity": 2.0, "category": "conflict"},
    "border clashes": {"severity": 2.0, "category": "conflict"},
    "emergency declared": {"severity": 2.0, "category": "conflict"},
    "state of emergency": {"severity": 2.0, "category": "conflict"},
    "troops mobilized": {"severity": 2.0, "category": "conflict"},
    "military buildup": {"severity": 2.0, "category": "conflict"},
    "strait closed": {"severity": 2.0, "category": "energy"},
    "shipping disrupted": {"severity": 2.0, "category": "energy"},
    "shipping disruption": {"severity": 2.0, "category": "energy"},
    "oil embargo": {"severity": 2.0, "category": "energy"},
    "record crash": {"severity": 2.0, "category": "market"},
    "record drop": {"severity": 2.0, "category": "market"},
    "bloodbath": {"severity": 2.0, "category": "market"},
    "plunges": {"severity": 2.0, "category": "market"},
    "tumbles over": {"severity": 2.0, "category": "market"},
    "crashes": {"severity": 2.0, "category": "market"},
    "geopolitical tensions": {"severity": 2.0, "category": "conflict"},
    "sparks market": {"severity": 2.0, "category": "market"},
    # ELEVATED – sanctions, diplomatic breaks, mobilisation
    "sanctions imposed": {"severity": 1.5, "category": "risk_off"},
    "diplomatic ties severed": {"severity": 1.5, "category": "conflict"},
    "ambassador recalled": {"severity": 1.5, "category": "conflict"},
    "military operation": {"severity": 1.5, "category": "conflict"},
    "retaliatory strikes": {"severity": 1.5, "category": "conflict"},
    "crude surges": {"severity": 1.5, "category": "energy"},
    "oil surges": {"severity": 1.5, "category": "energy"},
    "credit downgrade": {"severity": 1.5, "category": "risk_off"},
    "default risk": {"severity": 1.5, "category": "risk_off"},
    "freight uncertainty": {"severity": 1.5, "category": "energy"},
    "rising middle east": {"severity": 1.5, "category": "conflict"},
}

# ---------------------------------------------------------------------------
# Resolution / de-escalation signals (positive)
# ---------------------------------------------------------------------------

RESOLUTION_PATTERNS: Dict[str, Dict] = {
    # STRONG resolution – ceasefire, peace deal, war ends
    "ceasefire announced": {"strength": 3.0, "type": "ceasefire"},
    "ceasefire agreed": {"strength": 3.0, "type": "ceasefire"},
    "ceasefire deal": {"strength": 3.0, "type": "ceasefire"},
    "peace deal": {"strength": 3.0, "type": "ceasefire"},
    "peace agreement": {"strength": 3.0, "type": "ceasefire"},
    "war ends": {"strength": 3.0, "type": "ceasefire"},
    "hostilities cease": {"strength": 3.0, "type": "ceasefire"},
    "truce agreed": {"strength": 3.0, "type": "ceasefire"},
    "truce announced": {"strength": 3.0, "type": "ceasefire"},
    "conflict ends": {"strength": 3.0, "type": "ceasefire"},
    # MODERATE de-escalation – talks, withdrawals, diplomacy
    "peace talks": {"strength": 2.0, "type": "de_escalation"},
    "ceasefire talks": {"strength": 2.0, "type": "de_escalation"},
    "diplomatic talks": {"strength": 2.0, "type": "de_escalation"},
    "negotiations begin": {"strength": 2.0, "type": "de_escalation"},
    "troop withdrawal": {"strength": 2.0, "type": "de_escalation"},
    "troops withdraw": {"strength": 2.0, "type": "de_escalation"},
    "de-escalation": {"strength": 2.0, "type": "de_escalation"},
    "tensions ease": {"strength": 2.0, "type": "de_escalation"},
    "diplomatic channels": {"strength": 1.5, "type": "de_escalation"},
    "un mediation": {"strength": 2.0, "type": "de_escalation"},
    "broker peace": {"strength": 2.0, "type": "de_escalation"},
    # MILD positive signals – IMF, stimulus, confidence
    "imf approves": {"strength": 1.5, "type": "catalyst"},
    "imf tranche": {"strength": 1.5, "type": "catalyst"},
    "stimulus package": {"strength": 1.5, "type": "catalyst"},
    "rate cut": {"strength": 1.0, "type": "catalyst"},
    "market recovers": {"strength": 1.0, "type": "catalyst"},
    "bull run": {"strength": 1.0, "type": "catalyst"},
    "investor confidence": {"strength": 1.0, "type": "catalyst"},
    "sanctions lifted": {"strength": 2.0, "type": "de_escalation"},
    "embargo lifted": {"strength": 2.0, "type": "de_escalation"},
    # PSX-specific recovery language
    "extends rally": {"strength": 1.5, "type": "catalyst"},
    "surges past": {"strength": 1.0, "type": "catalyst"},
    "surges over": {"strength": 1.0, "type": "catalyst"},
    "positive market": {"strength": 1.0, "type": "catalyst"},
    "record high": {"strength": 1.0, "type": "catalyst"},
    "awaits imf": {"strength": 1.0, "type": "catalyst"},
    "imf hopes": {"strength": 1.0, "type": "catalyst"},
}

# Escalation signals – war is getting worse (distinct from initial shock)
ESCALATION_PATTERNS: Dict[str, Dict] = {
    "war spreads": {"intensity": 3.0, "type": "escalation"},
    "conflict widens": {"intensity": 3.0, "type": "escalation"},
    "second front": {"intensity": 3.0, "type": "escalation"},
    "multiple fronts": {"intensity": 3.0, "type": "escalation"},
    "ground invasion": {"intensity": 3.0, "type": "escalation"},
    "ground troops": {"intensity": 2.5, "type": "escalation"},
    "full-scale war": {"intensity": 3.0, "type": "escalation"},
    "total war": {"intensity": 3.0, "type": "escalation"},
    "nuclear threat": {"intensity": 3.0, "type": "escalation"},
    "nuclear option": {"intensity": 3.0, "type": "escalation"},
    "rejects ceasefire": {"intensity": 2.5, "type": "rejection"},
    "ceasefire rejected": {"intensity": 2.5, "type": "rejection"},
    "talks collapse": {"intensity": 2.5, "type": "rejection"},
    "talks fail": {"intensity": 2.5, "type": "rejection"},
    "further strikes": {"intensity": 2.0, "type": "escalation"},
    "intensifies attack": {"intensity": 2.0, "type": "escalation"},
    "expands operation": {"intensity": 2.0, "type": "escalation"},
    "retaliates": {"intensity": 2.0, "type": "escalation"},
    "vows retaliation": {"intensity": 2.0, "type": "escalation"},
    "all-out offensive": {"intensity": 2.5, "type": "escalation"},
    "civilian casualties mount": {"intensity": 2.0, "type": "humanitarian"},
    "refugee crisis": {"intensity": 1.5, "type": "humanitarian"},
    "humanitarian crisis": {"intensity": 1.5, "type": "humanitarian"},
    # PSX-specific escalation language from real news
    "rising middle east conflict": {"intensity": 2.0, "type": "escalation"},
    "middle east war triggers": {"intensity": 2.5, "type": "escalation"},
    "conflict sparks": {"intensity": 2.0, "type": "escalation"},
    "sparks energy": {"intensity": 1.5, "type": "escalation"},
}


def llm_assess_trajectory(
    news_items: List[Dict],
) -> Optional[Dict]:
    """
    Use Groq LLM (llama-3.3-70b) to assess geopolitical trajectory from news.

    The LLM reads raw headlines and produces a structured assessment that
    captures nuance keyword matching cannot (e.g. sarcasm, context-dependent
    phrases, multi-factor situations).

    Returns None on any failure (API down, no key, parse error) so the caller
    can fall back to keyword-only assessment.

    Expected LLM output schema:
        {
            "trajectory": "escalating" | "de_escalating" | "ceasefire" | "stalemate",
            "severity": 1-10,
            "ceasefire_probability": 0.0-1.0,
            "market_impact_pct": float (estimated single-day PSX impact),
            "key_events": [{"event": str, "significance": "critical"|"major"|"minor"}],
            "reasoning": str
        }
    """
    if not _GROQ_AVAILABLE or not news_items:
        return None

    client = _get_groq_client()
    if not client:
        return None

    current_date = datetime.now().strftime("%Y-%m-%d")

    # Format news headlines with dates
    headlines = []
    for item in news_items[:20]:
        date = item.get("date") or item.get("published") or "unknown"
        source = item.get("source_name") or item.get("source") or "Unknown"
        title = item.get("title") or ""
        desc = (item.get("description") or item.get("summary") or "")[:200]
        headlines.append(f"[{date}] [{source}] {title}")
        if desc:
            headlines.append(f"   → {desc}")

    news_block = "\n".join(headlines)

    prompt = f"""You are a geopolitical risk analyst specialising in Pakistan Stock Exchange (PSX / KSE-100).

Today is {current_date}. Analyse these recent news headlines and assess the geopolitical situation
affecting Pakistan's stock market.

NEWS HEADLINES:
{news_block}

TASK: Produce a JSON assessment with these fields:

1. "trajectory" (string): One of:
   - "ceasefire" — a ceasefire, peace deal, or definitive de-escalation has been announced/confirmed
   - "de_escalating" — tensions are easing, diplomatic channels active, but no formal resolution
   - "escalating" — conflict is intensifying, new fronts opening, or major military operations underway
   - "stalemate" — situation is unclear, mixed signals, or no clear direction

2. "severity" (integer 1-10): How severe is the geopolitical situation for PSX?
   - 1-3: Minor tensions, limited market impact
   - 4-6: Significant crisis, expect 3-7% market movement
   - 7-9: Major crisis (war, severe sanctions), expect 7-15% market movement
   - 10: Existential threat (nuclear, full-scale invasion)

3. "ceasefire_probability" (float 0.0-1.0): Likelihood of ceasefire/resolution within 48 hours

4. "market_impact_pct" (float): Estimated NEXT-DAY PSX percentage move (negative = decline, positive = recovery).
   Be specific. Consider: Is the crash already priced in? Is recovery starting?

5. "key_events" (array): Up to 5 most important geopolitical events identified, each with:
   - "event" (string): Brief description
   - "significance" (string): "critical", "major", or "minor"

6. "reasoning" (string): 2-3 sentence explanation of your assessment.

IMPORTANT:
- Focus on events DIRECTLY affecting Pakistan: wars involving Pakistan, India-Pakistan tensions,
  regional conflicts (Afghanistan, Iran, Middle East), sanctions, IMF/economic crises.
- Consider historical PSX patterns: 2025 India-Pak war crashed 12.6% then recovered 15.76% in 3 days
  after ceasefire. COVID crashed 30% with 4-month recovery.
- If news shows the crash has ALREADY happened, assess whether recovery is likely.
- Be calibrated: a distant conflict (e.g., Russia-Ukraine) is severity 2-3 for PSX,
  a direct Pakistan war is severity 8-10.

Respond with ONLY valid JSON, no markdown or explanation outside the JSON."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800,
        )
        raw = completion.choices[0].message.content.strip()
        result = json.loads(raw)

        # Validate required fields
        valid_trajectories = {"escalating", "de_escalating", "ceasefire", "stalemate"}
        if result.get("trajectory") not in valid_trajectories:
            result["trajectory"] = "stalemate"

        result["severity"] = max(1, min(10, int(result.get("severity", 5))))
        result["ceasefire_probability"] = max(0.0, min(1.0, float(result.get("ceasefire_probability", 0.0))))
        result["market_impact_pct"] = float(result.get("market_impact_pct", 0.0))
        result["key_events"] = result.get("key_events", [])[:5]
        result["reasoning"] = str(result.get("reasoning", ""))[:500]
        result["llm_source"] = "groq/llama-3.3-70b"

        logger.info(f"LLM trajectory assessment: {result['trajectory']} (severity={result['severity']}, "
                     f"ceasefire_prob={result['ceasefire_probability']:.2f})")
        return result

    except Exception as e:
        logger.warning(f"LLM trajectory assessment failed: {e}")
        return None


def assess_conflict_trajectory(
    news_items: List[Dict],
    use_llm: bool = True,
) -> Dict:
    """
    Hybrid assessment: combines keyword matching with LLM analysis.

    When LLM is available and use_llm=True, both approaches run and are merged:
      - Keyword matching: fast, deterministic, always available
      - LLM analysis: nuanced, context-aware, may fail

    Merge strategy (when both available):
      - trajectory: LLM wins if it has high severity (>= 5), otherwise keyword wins
      - trajectory_score: weighted average (LLM 60%, keyword 40%)
      - has_ceasefire / has_escalation: OR of both
      - confidence: max of both, boosted by 0.1 when both agree

    Returns:
        {
            "trajectory": "escalating" | "de_escalating" | "ceasefire" | "stalemate",
            "trajectory_score": float (-3 to +3),
            "has_ceasefire": bool,
            "has_escalation": bool,
            "resolution_signals": [...],
            "escalation_signals": [...],
            "confidence": float (0-1),
            "llm_assessment": {...} | None,
            "assessment_method": "hybrid" | "keyword_only",
        }
    """
    if not news_items:
        return {
            "trajectory": "stalemate",
            "trajectory_score": 0.0,
            "has_ceasefire": False,
            "has_escalation": False,
            "resolution_signals": [],
            "escalation_signals": [],
            "confidence": 0.0,
            "llm_assessment": None,
            "assessment_method": "keyword_only",
        }

    # ----- Step 1: Keyword-based assessment (always runs) -----
    resolution_signals: List[Dict] = []
    escalation_signals: List[Dict] = []
    seen_resolution: set = set()
    seen_escalation: set = set()

    for item in news_items[:40]:
        title = (item.get("title") or "").lower()
        desc = (item.get("description") or item.get("summary") or "").lower()
        text = f"{title} {desc}"

        for pattern, info in RESOLUTION_PATTERNS.items():
            if pattern in text and pattern not in seen_resolution:
                seen_resolution.add(pattern)
                resolution_signals.append({
                    "pattern": pattern,
                    "strength": info["strength"],
                    "type": info["type"],
                    "headline": (item.get("title") or "")[:120],
                })

        for pattern, info in ESCALATION_PATTERNS.items():
            if pattern in text and pattern not in seen_escalation:
                seen_escalation.add(pattern)
                escalation_signals.append({
                    "pattern": pattern,
                    "intensity": info["intensity"],
                    "type": info["type"],
                    "headline": (item.get("title") or "")[:120],
                })

    resolution_score = sum(s["strength"] for s in resolution_signals)
    escalation_score = sum(s["intensity"] for s in escalation_signals)

    kw_has_ceasefire = any(s["type"] == "ceasefire" for s in resolution_signals)
    kw_has_strong_escalation = any(s["intensity"] >= 2.5 for s in escalation_signals)

    net_score = resolution_score - escalation_score
    kw_trajectory_score = max(-3.0, min(3.0, net_score))

    if kw_has_ceasefire and resolution_score > escalation_score:
        kw_trajectory = "ceasefire"
    elif resolution_score > escalation_score * 1.5:
        kw_trajectory = "de_escalating"
    elif escalation_score > resolution_score * 1.5:
        kw_trajectory = "escalating"
    else:
        kw_trajectory = "stalemate"

    total_signals = len(resolution_signals) + len(escalation_signals)
    kw_confidence = min(1.0, total_signals / 5.0)

    # ----- Step 2: LLM assessment (if enabled and available) -----
    llm_result = None
    if use_llm:
        llm_result = llm_assess_trajectory(news_items)

    # ----- Step 3: Merge keyword + LLM results -----
    if llm_result is not None:
        # Convert LLM severity (1-10) to trajectory score (-3 to +3)
        llm_traj = llm_result["trajectory"]
        llm_severity = llm_result["severity"]
        if llm_traj in ("escalating",):
            llm_score = -(llm_severity / 10.0) * 3.0  # severity 10 → -3.0
        elif llm_traj == "ceasefire":
            llm_score = (llm_result["ceasefire_probability"]) * 3.0
        elif llm_traj == "de_escalating":
            llm_score = (1.0 - llm_severity / 10.0) * 3.0 * 0.6
        else:  # stalemate
            llm_score = 0.0

        # Weighted merge: LLM 60%, keyword 40%
        merged_score = llm_score * 0.6 + kw_trajectory_score * 0.4
        merged_score = max(-3.0, min(3.0, merged_score))

        # LLM ceasefire detection
        llm_has_ceasefire = (llm_traj == "ceasefire" or llm_result["ceasefire_probability"] > 0.6)
        llm_has_escalation = (llm_traj == "escalating" and llm_severity >= 7)

        has_ceasefire = kw_has_ceasefire or llm_has_ceasefire
        has_escalation = kw_has_strong_escalation or llm_has_escalation

        # Trajectory: LLM wins for high severity, keyword wins for low
        if llm_severity >= 5:
            trajectory = llm_traj
        else:
            trajectory = kw_trajectory

        # Override: if both agree, high confidence
        if kw_trajectory == llm_traj:
            trajectory = kw_trajectory
            confidence = min(1.0, max(kw_confidence, 0.7) + 0.1)
        else:
            confidence = max(kw_confidence, 0.5)

        # Add LLM market impact estimate to the assessment
        llm_result_clean = {
            "trajectory": llm_result["trajectory"],
            "severity": llm_result["severity"],
            "ceasefire_probability": llm_result["ceasefire_probability"],
            "market_impact_pct": llm_result["market_impact_pct"],
            "key_events": llm_result["key_events"],
            "reasoning": llm_result["reasoning"],
            "llm_source": llm_result.get("llm_source", "groq/llama-3.3-70b"),
        }

        return {
            "trajectory": trajectory,
            "trajectory_score": round(merged_score, 2),
            "has_ceasefire": has_ceasefire,
            "has_escalation": has_escalation,
            "resolution_signals": sorted(resolution_signals, key=lambda s: -s["strength"]),
            "escalation_signals": sorted(escalation_signals, key=lambda s: -s["intensity"]),
            "confidence": round(confidence, 2),
            "llm_assessment": llm_result_clean,
            "assessment_method": "hybrid",
        }

    # ----- Keyword-only fallback -----
    return {
        "trajectory": kw_trajectory,
        "trajectory_score": round(kw_trajectory_score, 2),
        "has_ceasefire": kw_has_ceasefire,
        "has_escalation": kw_has_strong_escalation,
        "resolution_signals": sorted(resolution_signals, key=lambda s: -s["strength"]),
        "escalation_signals": sorted(escalation_signals, key=lambda s: -s["intensity"]),
        "confidence": round(kw_confidence, 2),
        "llm_assessment": None,
        "assessment_method": "keyword_only",
    }


def detect_geopolitical_shocks(
    news_items: List[Dict],
    symbol: Optional[str] = None,
) -> Dict:
    """
    Scan news for shock-level geopolitical events.

    Returns:
        {
            "shock_detected": bool,
            "max_severity": float,
            "shock_events": [{"pattern", "severity", "category", "headline"}],
            "emergency_multiplier": float,  # 1.0 = no shock
            "shock_half_life": int,         # days for decay
        }
    """
    if not news_items:
        return {
            "shock_detected": False,
            "max_severity": 0.0,
            "shock_events": [],
            "emergency_multiplier": 1.0,
            "shock_half_life": 14,
            "matched_patterns": [],
            "shock_reason": "No geopolitical shock detected",
        }

    shock_events: List[Dict] = []
    seen_patterns: set = set()

    for item in news_items[:40]:
        title = (item.get("title") or "").lower()
        desc = (item.get("description") or item.get("summary") or "").lower()
        text = f"{title} {desc}"

        for pattern, info in SHOCK_EVENT_PATTERNS.items():
            if pattern in text and pattern not in seen_patterns:
                seen_patterns.add(pattern)
                shock_events.append({
                    "pattern": pattern,
                    "severity": info["severity"],
                    "category": info["category"],
                    "headline": (item.get("title") or "")[:120],
                })

    if not shock_events:
        return {
            "shock_detected": False,
            "max_severity": 0.0,
            "shock_events": [],
            "emergency_multiplier": 1.0,
            "shock_half_life": 14,
            "matched_patterns": [],
            "shock_reason": "No geopolitical shock detected",
        }

    max_severity = max(e["severity"] for e in shock_events)
    num_shocks = len(shock_events)

    # Assess trajectory: is the situation getting better or worse?
    trajectory = assess_conflict_trajectory(news_items)

    # Emergency multiplier: base from max severity, compounds with more shocks
    if max_severity >= 3.0:
        base_multiplier = 2.5
    elif max_severity >= 2.0:
        base_multiplier = 1.8
    else:
        base_multiplier = 1.3

    # Compound: each additional shock adds 30% of base
    emergency_multiplier = base_multiplier * (1.0 + 0.3 * max(0, num_shocks - 1))

    # Modulate by trajectory:
    #   ceasefire   → halve the multiplier (recovery incoming)
    #   de-escalating → reduce by 30%
    #   escalating → boost by 20%
    #   stalemate  → no change
    traj = trajectory["trajectory"]
    if traj == "ceasefire":
        emergency_multiplier *= 0.5
    elif traj == "de_escalating":
        emergency_multiplier *= 0.7
    elif traj == "escalating":
        emergency_multiplier *= 1.2

    emergency_multiplier = min(emergency_multiplier, 4.0)  # hard cap

    # Shock events decay faster than normal mode.
    # Ceasefire/de-escalation uses the shortest half-life (panic fades quickest).
    if traj in ("ceasefire", "de_escalating"):
        shock_half_life = 5  # fast recovery expected
    elif max_severity >= 2.0:
        shock_half_life = 7
    else:
        shock_half_life = 10

    return {
        "shock_detected": True,
        "max_severity": max_severity,
        "shock_events": sorted(shock_events, key=lambda e: -e["severity"]),
        "emergency_multiplier": round(emergency_multiplier, 2),
        "shock_half_life": shock_half_life,
        "trajectory": trajectory,
        "matched_patterns": sorted({e["pattern"] for e in shock_events}),
        "shock_reason": "Matched shock patterns: " + ", ".join(
            sorted({e["pattern"] for e in shock_events})[:3]
        ),
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _term_score(text: str, terms: List[str], amplifier: float = 1.0) -> float:
    """Score text against a term list.  Normalise by adjusted denominator."""
    if not text:
        return 0.0
    hits = sum(1 for t in terms if t in text)
    raw = hits / max(1, len(terms) * 0.30)
    return _clamp01(raw * amplifier)


def _compute_energy_war_overlap(news_items: List[Dict]) -> float:
    has_energy = False
    has_regional = False

    for item in news_items[:40]:
        title = (item.get("title") or "").lower()
        desc = (item.get("description") or item.get("summary") or "").lower()
        text = f"{title} {desc}"
        energy_hit = is_energy_supply_shock_text(text) or any(term in text for term in GEO_TERMS["energy_supply"])
        regional_hit = has_regional_war_terms(text) or any(term in text for term in GEO_TERMS["regional"])
        if energy_hit and regional_hit:
            return 1.0
        has_energy = has_energy or energy_hit
        has_regional = has_regional or regional_hit

    return 1.0 if has_energy and has_regional else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def neutral_geopolitical_features() -> Dict[str, float]:
    """Safe neutral values; used as fallback on any error."""
    return {
        "geo_conflict_risk": 0.0,
        "geo_energy_supply_risk": 0.0,
        "geo_regional_tension": 0.0,
        "geo_global_risk_off": 0.0,
        "geo_news_volume": 0.0,
        "geo_energy_war_overlap": 0.0,
    }


def get_geopolitical_features_from_news(
    news_items: List[Dict],
    symbol: Optional[str] = None,
) -> Dict[str, float]:
    """Compute geo-risk features from a list of news dicts (must have 'title')."""
    if not news_items:
        return neutral_geopolitical_features()

    text = " ".join((item.get("title") or "").lower() for item in news_items[:40])
    volume = _clamp01(len(news_items) / 20.0)

    sector = SYMBOL_SECTOR.get((symbol or "").upper(), "")
    amplifiers = SECTOR_AMPLIFIERS.get(sector, {})

    features = {
        "geo_conflict_risk": _term_score(
            text, GEO_TERMS["conflict"], amplifiers.get("conflict", 1.0)
        ),
        "geo_energy_supply_risk": _term_score(
            text, GEO_TERMS["energy_supply"], amplifiers.get("energy_supply", 1.0)
        ),
        "geo_regional_tension": _term_score(
            text, GEO_TERMS["regional"], amplifiers.get("regional", 1.0)
        ),
        "geo_global_risk_off": _term_score(
            text, GEO_TERMS["risk_off"], amplifiers.get("risk_off", 1.0)
        ),
        "geo_news_volume": volume,
        "geo_energy_war_overlap": _compute_energy_war_overlap(news_items),
    }

    symbol_upper = (symbol or "").upper()
    if symbol_upper in ENERGY_SHOCK_SYMBOLS and features["geo_energy_war_overlap"] > 0:
        features["geo_conflict_risk"] = max(
            features["geo_conflict_risk"],
            min(
                1.0,
                0.25
                + (0.55 * features["geo_energy_supply_risk"])
                + (0.35 * features["geo_regional_tension"]),
            ),
        )
    return features


def get_geopolitical_features_for_symbol(
    symbol: str, use_cache: bool = True
) -> Dict[str, float]:
    """
    Pull geopolitical signals from existing sentiment/news pipeline.
    Returns neutral values on any failure – by design.
    """
    # Fast path: try reading from the news cache file directly (avoids
    # triggering a full sentiment fetch when we already have cached news).
    if use_cache:
        cache_file = CACHE_DIR / f"{symbol.upper()}_news.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                news_items = cached.get("news_items", [])
                if news_items:
                    return get_geopolitical_features_from_news(news_items, symbol)
            except Exception:
                pass

    # Slow path: call sentiment pipeline.
    try:
        from backend.sentiment_analyzer import get_stock_sentiment
    except Exception:
        return neutral_geopolitical_features()

    try:
        sentiment = get_stock_sentiment(symbol, use_cache=use_cache)
        news_items = sentiment.get("news_items", [])
        return get_geopolitical_features_from_news(news_items, symbol)
    except Exception:
        return neutral_geopolitical_features()


def default_geo_interpretation(enabled: bool = True) -> Dict[str, object]:
    return {
        "sector_interpretation": "generic",
        "polarity": "neutral" if enabled else "disabled",
        "bullish_for_upstream": False,
        "tailwind_score": 0.0,
        "cashflow_support_score": 0.0,
        "reason": (
            "No sector-specific geopolitical interpretation applied."
            if enabled
            else "Geo overlay disabled for this run."
        ),
    }


def _has_positive_oil_context(text: str) -> bool:
    text = (text or "").lower()
    return any(term in text for term in POSITIVE_OIL_CONTEXT_TERMS)


def _has_payment_relief_context(text: str) -> bool:
    text = (text or "").lower()
    return any(term in text for term in UPSTREAM_PAYMENT_RELIEF_TERMS)


def build_geo_interpretation(
    news_items: List[Dict],
    geo_features: Optional[Dict[str, float]] = None,
    shock_data: Optional[Dict] = None,
    symbol: Optional[str] = None,
) -> Dict[str, object]:
    """Build sector interpretation metadata shared by UI and geo overlay math."""
    interpretation = default_geo_interpretation(enabled=True)
    symbol_upper = (symbol or "").upper()
    geo = geo_features or {}
    shock = shock_data or {}

    conflict = _clamp01(geo.get("geo_conflict_risk", 0.0))
    energy = _clamp01(geo.get("geo_energy_supply_risk", 0.0))
    regional = _clamp01(geo.get("geo_regional_tension", 0.0))
    risk_off = _clamp01(geo.get("geo_global_risk_off", 0.0))
    overlap = _clamp01(geo.get("geo_energy_war_overlap", 0.0))
    shock_detected = bool(shock.get("shock_detected", False))

    fuel_hike_detected = False
    circular_relief_detected = False
    energy_supply_detected = False
    positive_oil_context = False

    for item in news_items or []:
        text = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("description", "")),
                str(item.get("summary", "")),
            ]
        ).lower()
        fuel_delta = parse_local_fuel_price_delta(text)
        if fuel_delta is not None and fuel_delta > 0:
            fuel_hike_detected = True
        if score_circular_debt_signal(text) > 0 or _has_payment_relief_context(text):
            circular_relief_detected = True
        if is_energy_supply_shock_text(text):
            energy_supply_detected = True
        if _has_positive_oil_context(text):
            positive_oil_context = True

    bullish_for_upstream = (
        symbol_upper in UPSTREAM_EP_SYMBOLS
        and energy >= 0.50
        and (
            shock_detected
            or overlap >= 0.25
            or positive_oil_context
            or fuel_hike_detected
        )
    )

    tailwind_score = _clamp01(
        (0.60 * energy)
        + (0.20 * overlap)
        + (0.10 if shock_detected or energy_supply_detected else 0.0)
        + (0.10 if positive_oil_context else 0.0)
    )
    cashflow_support_score = _clamp01(
        (0.65 if fuel_hike_detected else 0.0)
        + (0.35 if circular_relief_detected else 0.0)
    )

    if bullish_for_upstream:
        reasons: List[str] = []
        if shock_detected or overlap >= 0.25 or energy_supply_detected or positive_oil_context:
            reasons.append("Middle East energy shock is a sector tailwind for upstream E&P")
        if fuel_hike_detected or circular_relief_detected:
            reasons.append("Domestic fuel-price hike may support circular-debt cash recovery")
        if not reasons:
            reasons.append("Energy supply stress is being interpreted as an upstream sector tailwind")

        return {
            "sector_interpretation": "upstream_energy_tailwind",
            "polarity": "bullish_for_upstream",
            "bullish_for_upstream": True,
            "tailwind_score": round(tailwind_score, 2),
            "cashflow_support_score": round(cashflow_support_score, 2),
            "reason": ". ".join(reasons),
        }

    risk_pressure = _clamp01((0.45 * conflict) + (0.35 * risk_off) + (0.20 * regional))
    interpretation["polarity"] = "risk_off" if risk_pressure >= 0.25 else "neutral"
    interpretation["bullish_for_upstream"] = False
    interpretation["tailwind_score"] = (
        round(tailwind_score, 2) if symbol_upper in UPSTREAM_EP_SYMBOLS else 0.0
    )
    interpretation["cashflow_support_score"] = (
        round(cashflow_support_score, 2) if symbol_upper in UPSTREAM_EP_SYMBOLS else 0.0
    )
    interpretation["reason"] = (
        "Standard geo interpretation: conflict and supply stress are treated as broad market risk."
    )
    return interpretation


def build_geopolitical_daily_adjustments(
    geo_features: Dict[str, float],
    prediction_length: int,
    symbol: Optional[str] = None,
    shock_data: Optional[Dict] = None,
    interpretation: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build day-indexed geo adjustments compatible with apply_adjustments_to_predictions.

    Formula:
      risk = 0.35*conflict + 0.25*regional + 0.25*risk_off + 0.15*energy
      if energy-sector symbol: risk -= 0.20*energy
      mult = 0.5 + 0.5*geo_news_volume
      decay(day) = 0.5 ** ((day-1)/half_life)
      adj_factor = clamp(-0.04 * risk * mult * decay * emergency_mult, lower_cap, upper_cap)

    When shock_data is provided, caps are widened and decay is faster.
    """
    if prediction_length <= 0:
        return {
            "adjustments": [],
            "summary": {
                "max_abs_adjustment_pct": 0.0,
                "avg_adjustment_pct": 0.0,
                "risk_score": 0.0,
                "volume_multiplier": 0.5,
                "shock_detected": False,
                "overlay_mode": "risk_off",
                "bullish_for_upstream": False,
                "matched_patterns": [],
                "shock_reason": "No geopolitical shock detected",
                "methodology": "Deterministic geo risk post-processing (empty horizon)",
            },
        }

    geo = geo_features or {}
    conflict = _clamp01(geo.get("geo_conflict_risk", 0.0))
    energy = _clamp01(geo.get("geo_energy_supply_risk", 0.0))
    regional = _clamp01(geo.get("geo_regional_tension", 0.0))
    risk_off = _clamp01(geo.get("geo_global_risk_off", 0.0))
    volume = _clamp01(geo.get("geo_news_volume", 0.0))
    energy_war_overlap = _clamp01(geo.get("geo_energy_war_overlap", 0.0))

    risk_score = (
        (0.35 * conflict)
        + (0.25 * regional)
        + (0.25 * risk_off)
        + (0.15 * energy)
    )

    # Shock-adjusted parameters
    shock = shock_data or {}
    shock_detected = shock.get("shock_detected", False)
    emergency_multiplier = shock.get("emergency_multiplier", 1.0)
    half_life = float(shock.get("shock_half_life", 14))
    interp = interpretation or {}
    bullish_for_upstream = bool(interp.get("bullish_for_upstream", False))
    tailwind_score = _clamp01(float(interp.get("tailwind_score", 0.0) or 0.0))
    cashflow_support_score = _clamp01(float(interp.get("cashflow_support_score", 0.0) or 0.0))

    sector = SYMBOL_SECTOR.get((symbol or "").upper(), "")
    energy_shock_mode = bool(shock_detected) or energy_war_overlap > 0 or (
        sector == "energy" and conflict > 0.0 and energy > 0.3
    )
    if sector == "energy" and not energy_shock_mode:
        risk_score -= (0.20 * energy)

    risk_score = max(-1.0, min(1.0, risk_score))
    volume_multiplier = 0.5 + (0.5 * volume)

    # Widen caps during shocks: e.g. -0.05 * 2.5 = -0.125 (12.5% max single-day drop)
    if bullish_for_upstream:
        lower_cap = -0.02 * emergency_multiplier
        upper_cap = 0.05 * emergency_multiplier
        tailwind_strength = _clamp01((0.75 * tailwind_score) + (0.25 * cashflow_support_score))
        overlay_mode = "upstream_tailwind"
    else:
        lower_cap = -0.05 * emergency_multiplier
        upper_cap = 0.02 * emergency_multiplier
        tailwind_strength = 0.0
        overlay_mode = "risk_off"

    adjustments: List[Dict] = []
    for day in range(1, prediction_length + 1):
        decay = 0.5 ** ((day - 1) / half_life)
        if bullish_for_upstream:
            raw_adjustment = 0.025 * tailwind_strength * volume_multiplier * decay * emergency_multiplier
        else:
            raw_adjustment = -0.04 * risk_score * volume_multiplier * decay * emergency_multiplier
        capped_adjustment = max(lower_cap, min(upper_cap, raw_adjustment))
        pct = capped_adjustment * 100.0
        event_impacts = [
            f"geo_risk_score={risk_score:.3f}",
            f"geo_decay_day_{day}={decay:.4f}",
        ]
        if bullish_for_upstream:
            event_impacts.append(f"upstream_tailwind={tailwind_strength:.3f}")
            event_impacts.append(f"cashflow_support={cashflow_support_score:.3f}")
        if shock_detected:
            event_impacts.append(f"emergency_multiplier={emergency_multiplier}")
            event_impacts.append(f"shock_half_life={half_life}")
        adjustments.append(
            {
                "day": day,
                "raw_adjustment": raw_adjustment,
                "capped_adjustment": capped_adjustment,
                "percentage": round(pct, 4),
                "event_impacts": event_impacts,
            }
        )

    pct_values = [float(a["percentage"]) for a in adjustments]
    max_abs = max((abs(v) for v in pct_values), default=0.0)
    avg = sum(pct_values) / len(pct_values) if pct_values else 0.0

    methodology = (
        "Deterministic geo risk post-processing with weighted risk, "
        f"news-volume multiplier, and {half_life:.0f}-day half-life decay"
    )
    if shock_detected:
        methodology += f" [SHOCK MODE: {emergency_multiplier}x emergency multiplier]"
    if bullish_for_upstream:
        methodology += " [UPSTREAM TAILWIND MODE]"

    return {
        "adjustments": adjustments,
        "summary": {
            "max_abs_adjustment_pct": round(max_abs, 4),
            "avg_adjustment_pct": round(avg, 4),
            "risk_score": round(risk_score, 6),
            "volume_multiplier": round(volume_multiplier, 6),
            "shock_detected": shock_detected,
            "overlay_mode": overlay_mode,
            "bullish_for_upstream": bullish_for_upstream,
            "emergency_multiplier": emergency_multiplier if shock_detected else 1.0,
            "shock_events": shock.get("shock_events", []) if shock_detected else [],
            "matched_patterns": shock.get("matched_patterns", []),
            "shock_reason": shock.get("shock_reason", "No geopolitical shock detected"),
            "methodology": methodology,
        },
    }
