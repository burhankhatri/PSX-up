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

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    }
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


def build_geopolitical_daily_adjustments(
    geo_features: Dict[str, float],
    prediction_length: int,
    symbol: Optional[str] = None,
) -> Dict[str, object]:
    """
    Build day-indexed geo adjustments compatible with apply_adjustments_to_predictions.

    Formula:
      risk = 0.35*conflict + 0.25*regional + 0.25*risk_off + 0.15*energy
      if energy-sector symbol: risk -= 0.20*energy
      mult = 0.5 + 0.5*geo_news_volume
      decay(day) = 0.5 ** ((day-1)/14)
      adj_factor = clamp(-0.04 * risk * mult * decay, -0.05, 0.02)
    """
    if prediction_length <= 0:
        return {
            "adjustments": [],
            "summary": {
                "max_abs_adjustment_pct": 0.0,
                "avg_adjustment_pct": 0.0,
                "risk_score": 0.0,
                "volume_multiplier": 0.5,
                "methodology": "Deterministic geo risk post-processing (empty horizon)",
            },
        }

    geo = geo_features or {}
    conflict = _clamp01(geo.get("geo_conflict_risk", 0.0))
    energy = _clamp01(geo.get("geo_energy_supply_risk", 0.0))
    regional = _clamp01(geo.get("geo_regional_tension", 0.0))
    risk_off = _clamp01(geo.get("geo_global_risk_off", 0.0))
    volume = _clamp01(geo.get("geo_news_volume", 0.0))

    risk_score = (
        (0.35 * conflict)
        + (0.25 * regional)
        + (0.25 * risk_off)
        + (0.15 * energy)
    )

    sector = SYMBOL_SECTOR.get((symbol or "").upper(), "")
    if sector == "energy":
        risk_score -= (0.20 * energy)

    risk_score = max(-1.0, min(1.0, risk_score))
    volume_multiplier = 0.5 + (0.5 * volume)

    adjustments: List[Dict] = []
    for day in range(1, prediction_length + 1):
        decay = 0.5 ** ((day - 1) / 14.0)
        raw_adjustment = -0.04 * risk_score * volume_multiplier * decay
        capped_adjustment = max(-0.05, min(0.02, raw_adjustment))
        pct = capped_adjustment * 100.0
        adjustments.append(
            {
                "day": day,
                "raw_adjustment": raw_adjustment,
                "capped_adjustment": capped_adjustment,
                "percentage": round(pct, 4),
                "event_impacts": [
                    f"geo_risk_score={risk_score:.3f}",
                    f"geo_decay_day_{day}={decay:.4f}",
                ],
            }
        )

    pct_values = [float(a["percentage"]) for a in adjustments]
    max_abs = max((abs(v) for v in pct_values), default=0.0)
    avg = sum(pct_values) / len(pct_values) if pct_values else 0.0

    return {
        "adjustments": adjustments,
        "summary": {
            "max_abs_adjustment_pct": round(max_abs, 4),
            "avg_adjustment_pct": round(avg, 4),
            "risk_score": round(risk_score, 6),
            "volume_multiplier": round(volume_multiplier, 6),
            "methodology": (
                "Deterministic geo risk post-processing with weighted risk, "
                "news-volume multiplier, and 14-day half-life decay"
            ),
        },
    }
