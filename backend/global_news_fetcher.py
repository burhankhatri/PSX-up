#!/usr/bin/env python3
"""
🌍 GLOBAL GEOPOLITICAL NEWS FETCHER
Fetches international news about oil, war, energy supply disruptions, and
geopolitical events from major global sources via RSS feeds and web scraping.

This fills the critical gap where the existing news pipeline only pulls from
Pakistani domestic sources, missing massive global events like:
- Iran war / Strait of Hormuz blockade
- OPEC decisions
- Oil price surges
- Middle East escalation
- Russia-Ukraine energy impacts

These global events directly impact PSX stocks, especially energy sector
(OGDC, PPL, PSO, POL, MARI) and import-sensitive sectors.
"""

import json
import logging
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ============================================================================
# CACHE CONFIG
# ============================================================================

CACHE_DIR = Path(__file__).parent.parent / "data" / "external_cache"
CACHE_FILE = CACHE_DIR / "global_geo_news.json"
CACHE_TTL_HOURS = 6  # Refresh every 6 hours

# ============================================================================
# RSS FEEDS — Major international news sources
# ============================================================================

RSS_FEEDS = {
    # Al Jazeera — best for Middle East coverage
    "Al Jazeera": [
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    # Reuters — global wire service
    "Reuters": [
        "https://news.google.com/rss/search?q=oil+prices+OR+iran+war+OR+strait+of+hormuz+OR+opec+OR+middle+east+conflict&hl=en&gl=US&ceid=US:en",
    ],
    # CNBC — energy/markets
    "CNBC": [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",  # Energy
    ],
    # BBC — world news
    "BBC": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
    ],
}

# Google News RSS for targeted queries
GOOGLE_NEWS_QUERIES = [
    "oil prices today",
    "Iran war latest",
    "Strait of Hormuz shipping",
    "OPEC production",
    "Middle East conflict oil",
    "Russia Ukraine oil",
    "energy crisis global",
    "oil supply disruption",
    "Pakistan oil imports",
    "LNG prices surge",
    "Brent crude",
]

# ============================================================================
# RELEVANCE KEYWORDS — filter for geopolitically relevant articles
# ============================================================================

GEO_RELEVANCE_KEYWORDS: Dict[str, Set[str]] = {
    "oil_energy": {
        "oil", "crude", "brent", "wti", "petroleum", "opec", "lng", "gas prices",
        "energy prices", "fuel", "refinery", "petrochemical", "oil prices",
        "barrel", "gasoline", "diesel", "natural gas", "energy crisis",
        "oil surge", "oil spike", "oil supply",
    },
    "war_conflict": {
        "war", "strike", "missile", "drone", "attack", "bombing", "escalation",
        "military", "airstrikes", "invasion", "conflict", "ceasefire",
        "retaliation", "nuclear", "sanctions", "troops", "casualties",
        "killed", "wounded", "defense", "defence",
    },
    "shipping_trade": {
        "strait of hormuz", "hormuz", "shipping", "blockade", "maritime",
        "supply chain", "trade route", "red sea", "suez", "tanker",
        "shipping lane", "port", "embargo", "trade war", "tariff",
    },
    "regions": {
        "iran", "israel", "middle east", "gulf", "saudi", "qatar", "uae",
        "russia", "ukraine", "china", "pakistan", "india", "opec",
        "lebanon", "hezbollah", "houthi", "yemen", "iraq", "syria",
        "afghanistan", "turkey", "bahrain", "kuwait",
    },
    "markets": {
        "recession", "inflation", "interest rate", "fed", "central bank",
        "stock market", "crash", "sell-off", "risk-off", "safe haven",
        "gold", "dollar", "currency", "emerging markets", "imf",
    },
}

# Minimum relevance score to include an article
MIN_RELEVANCE_SCORE = 2

# ============================================================================
# SECTOR IMPACT MAPPING — which news categories matter for which PSX sectors
# ============================================================================

SECTOR_NEWS_WEIGHT = {
    "exploration_production": {"oil_energy": 2.0, "war_conflict": 1.5, "shipping_trade": 1.8, "regions": 1.3},
    "oil_marketing": {"oil_energy": 2.0, "war_conflict": 1.3, "shipping_trade": 1.5, "regions": 1.0},
    "power": {"oil_energy": 1.5, "war_conflict": 1.0, "shipping_trade": 1.2, "regions": 0.8},
    "cement": {"oil_energy": 1.2, "war_conflict": 0.8, "shipping_trade": 1.0, "regions": 0.7},
    "fertilizer": {"oil_energy": 1.5, "war_conflict": 1.0, "shipping_trade": 1.3, "regions": 0.8},
    "banks": {"oil_energy": 0.8, "war_conflict": 1.0, "shipping_trade": 0.7, "markets": 1.5},
    "autos": {"oil_energy": 1.3, "war_conflict": 0.8, "shipping_trade": 1.0, "regions": 0.7},
    "technology": {"oil_energy": 0.5, "war_conflict": 0.7, "shipping_trade": 0.5, "markets": 1.2},
    "steel": {"oil_energy": 1.2, "war_conflict": 0.8, "shipping_trade": 1.0, "regions": 0.7},
}


# ============================================================================
# FETCHING FUNCTIONS
# ============================================================================

def _fetch_url(url: str, timeout: int = 12) -> str:
    """Fetch URL content using curl (same pattern as enhanced_news_fetcher)."""
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', str(timeout),
             '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
             '-H', 'Accept: application/rss+xml, application/xml, text/xml, */*',
             url],
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Failed to fetch {url}: {e}")
    return ""


def _parse_rss(xml_text: str, source: str) -> List[Dict]:
    """Parse RSS/Atom XML into news items."""
    items = []
    if not xml_text or not xml_text.strip():
        return items

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Try cleaning common issues
        xml_text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', xml_text)
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return items

    # Handle RSS 2.0
    for item in root.iter('item'):
        title_el = item.find('title')
        link_el = item.find('link')
        desc_el = item.find('description')
        pub_date_el = item.find('pubDate')

        title = (title_el.text or '').strip() if title_el is not None else ''
        link = (link_el.text or '').strip() if link_el is not None else ''
        description = (desc_el.text or '').strip() if desc_el is not None else ''
        pub_date = (pub_date_el.text or '').strip() if pub_date_el is not None else ''

        if not title or len(title) < 15:
            continue

        # Clean HTML from description
        description = re.sub(r'<[^>]+>', '', description)[:500]

        # Parse date
        date_str = _parse_rss_date(pub_date)

        items.append({
            'title': title[:300],
            'url': link,
            'source': source,
            'description': description,
            'date': date_str,
            'is_global_geo': True,
            'is_macro': True,
            'is_direct': False,
            'scope': 'global_geopolitical',
        })

    # Handle Atom feeds
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    for entry in root.iter('{http://www.w3.org/2005/Atom}entry'):
        title_el = entry.find('atom:title', ns)
        link_el = entry.find('atom:link', ns)
        summary_el = entry.find('atom:summary', ns)
        updated_el = entry.find('atom:updated', ns)

        title = (title_el.text or '').strip() if title_el is not None else ''
        link = link_el.get('href', '') if link_el is not None else ''
        description = (summary_el.text or '').strip() if summary_el is not None else ''
        pub_date = (updated_el.text or '').strip() if updated_el is not None else ''

        if not title or len(title) < 15:
            continue

        description = re.sub(r'<[^>]+>', '', description)[:500]
        date_str = _parse_rss_date(pub_date)

        items.append({
            'title': title[:300],
            'url': link,
            'source': source,
            'description': description,
            'date': date_str,
            'is_global_geo': True,
            'is_macro': True,
            'is_direct': False,
            'scope': 'global_geopolitical',
        })

    return items


def _parse_rss_date(date_str: str) -> str:
    """Parse various RSS date formats into YYYY-MM-DD."""
    if not date_str:
        return datetime.now().strftime('%Y-%m-%d')

    # RFC 822: "Mon, 18 Mar 2026 12:00:00 GMT"
    for fmt in [
        '%a, %d %b %Y %H:%M:%S %Z',
        '%a, %d %b %Y %H:%M:%S %z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d',
    ]:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue

    # Fallback: extract date-like pattern
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m:
        return m.group(0)

    return datetime.now().strftime('%Y-%m-%d')


def _score_relevance(title: str, description: str = "") -> Dict:
    """
    Score an article's geopolitical relevance.
    Returns {'score': float, 'categories': set of matched categories}.
    """
    text = f"{title} {description}".lower()
    score = 0
    matched_categories = set()

    for category, keywords in GEO_RELEVANCE_KEYWORDS.items():
        category_hits = 0
        for kw in keywords:
            if kw in text:
                category_hits += 1
        if category_hits > 0:
            matched_categories.add(category)
            score += min(category_hits, 4)  # Cap per category

    # Bonus for multi-category hits (e.g., oil + war = very relevant)
    if len(matched_categories) >= 2:
        score += len(matched_categories)

    return {'score': score, 'categories': matched_categories}


def _filter_by_recency(
    articles: List[Dict],
    max_age_hours: int = 48,
    fallback_days: int = 7,
) -> List[Dict]:
    """
    Filter articles by recency. Keeps only recent articles to prevent stale
    news from polluting geopolitical analysis.

    Strategy:
      1. Primary: articles within max_age_hours (48h = today + yesterday)
      2. Fallback: if <3 articles, expand to fallback_days (7 days)
      3. Safety: if still empty, return original list
    """
    if not articles:
        return articles

    now = datetime.now()
    total = len(articles)

    for article in articles:
        date_str = (article.get('date') or '')[:10]
        try:
            article_date = datetime.strptime(date_str, '%Y-%m-%d')
            article['age_hours'] = (now - article_date).total_seconds() / 3600
        except (ValueError, TypeError):
            article['age_hours'] = 999  # Unknown date → treat as old

    # Primary: last 48 hours
    recent = [a for a in articles if a.get('age_hours', 999) <= max_age_hours]
    if len(recent) >= 3:
        logger.info(f"📅 Date filter: {len(recent)}/{total} articles within {max_age_hours}h")
        print(f"📅 Date filter: {len(recent)}/{total} articles within {max_age_hours}h")
        return recent

    # Fallback: last 7 days
    fallback_hours = fallback_days * 24
    expanded = [a for a in articles if a.get('age_hours', 999) <= fallback_hours]
    if expanded:
        logger.info(f"📅 Date filter fallback: {len(expanded)}/{total} articles within {fallback_days}d")
        print(f"📅 Date filter fallback: {len(expanded)}/{total} articles within {fallback_days}d")
        return expanded

    # Safety: return all (never return empty when there IS data)
    logger.info(f"📅 Date filter: no recent articles found, using all {total}")
    return articles


def _fetch_google_news_rss(query: str) -> List[Dict]:
    """Fetch news from Google News RSS for a specific query."""
    encoded = query.replace(' ', '+')
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
    xml = _fetch_url(url, timeout=10)
    return _parse_rss(xml, "Google News")


# ============================================================================
# MAIN API
# ============================================================================

def fetch_global_geo_news(
    force_refresh: bool = False,
    max_articles: int = 50,
) -> List[Dict]:
    """
    Fetch global geopolitical/oil/war news from international sources.

    Returns a list of news items in the same format as enhanced_news_fetcher:
    [{'title': str, 'url': str, 'source': str, 'date': str, 'description': str,
      'is_global_geo': True, 'is_macro': True, 'scope': 'global_geopolitical',
      'relevance_score': float, 'geo_categories': list}]
    """
    # Check cache first
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            logger.info(f"🌍 Global geo news: {len(cached)} articles from cache")
            return cached

    logger.info("🌍 Fetching global geopolitical news from international sources...")
    all_articles: List[Dict] = []
    seen_titles: Set[str] = set()

    # 1. Fetch from RSS feeds
    for source, feeds in RSS_FEEDS.items():
        for feed_url in feeds:
            try:
                xml = _fetch_url(feed_url)
                articles = _parse_rss(xml, source)
                for article in articles:
                    title_key = article['title'].lower().strip()[:100]
                    if title_key not in seen_titles:
                        seen_titles.add(title_key)
                        all_articles.append(article)
            except Exception as e:
                logger.debug(f"RSS fetch failed for {source}: {e}")

    # 2. Fetch targeted Google News queries
    for query in GOOGLE_NEWS_QUERIES:
        try:
            articles = _fetch_google_news_rss(query)
            for article in articles:
                title_key = article['title'].lower().strip()[:100]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_articles.append(article)
        except Exception as e:
            logger.debug(f"Google News query failed for '{query}': {e}")

    # 3. Score relevance and filter
    scored_articles = []
    for article in all_articles:
        relevance = _score_relevance(
            article.get('title', ''),
            article.get('description', '')
        )
        if relevance['score'] >= MIN_RELEVANCE_SCORE:
            article['relevance_score'] = relevance['score']
            article['geo_categories'] = list(relevance['categories'])
            scored_articles.append(article)

    # 3.5. Date filter: keep only recent news (48h primary, 7d fallback)
    scored_articles = _filter_by_recency(scored_articles, max_age_hours=48, fallback_days=7)

    # 4. Sort by recency then relevance
    scored_articles.sort(
        key=lambda x: (
            x.get('age_hours', 999) <= 24,   # Last 24h first
            x.get('age_hours', 999) <= 48,   # Last 48h second
            x.get('relevance_score', 0),      # Then by relevance
        ),
        reverse=True
    )

    # 5. Limit output
    result = scored_articles[:max_articles]

    # 6. Cache
    _save_cache(result)

    logger.info(f"🌍 Global geo news: {len(result)} relevant articles fetched")
    print(f"🌍 Global geo news: {len(result)} relevant articles from {len(RSS_FEEDS) + len(GOOGLE_NEWS_QUERIES)} sources")

    return result


def get_global_news_for_symbol(
    symbol: str,
    sector: Optional[str] = None,
) -> List[Dict]:
    """
    Get global geopolitical news weighted for a specific PSX symbol/sector.

    Energy stocks (OGDC, PPL, PSO, etc.) get higher weight for oil/energy news.
    Banks get higher weight for market/recession news.
    Returns articles sorted by sector-weighted relevance.
    """
    articles = fetch_global_geo_news()
    if not articles:
        return []

    # Auto-detect sector
    if sector is None:
        try:
            from backend.enhanced_news_fetcher import COMPANY_ALIASES
            sector = COMPANY_ALIASES.get(symbol.upper(), {}).get('sector', 'unknown')
        except ImportError:
            sector = 'unknown'

    weights = SECTOR_NEWS_WEIGHT.get(sector, {})

    # Re-score articles with sector weighting
    weighted_articles = []
    for article in articles:
        base_score = article.get('relevance_score', 1)
        categories = article.get('geo_categories', [])

        sector_multiplier = 1.0
        for cat in categories:
            sector_multiplier = max(sector_multiplier, weights.get(cat, 1.0))

        article_copy = dict(article)
        article_copy['sector_weighted_score'] = base_score * sector_multiplier
        weighted_articles.append(article_copy)

    weighted_articles.sort(key=lambda x: x.get('sector_weighted_score', 0), reverse=True)

    return weighted_articles


def get_global_news_summary(articles: Optional[List[Dict]] = None) -> Dict:
    """
    Generate a structured summary of global geopolitical conditions
    for use in the sentiment/geo prompt context.
    """
    if articles is None:
        articles = fetch_global_geo_news()

    if not articles:
        return {
            "available": False,
            "headline_count": 0,
            "categories": {},
            "summary_text": "No global news data available.",
        }

    # Count categories
    category_counts: Dict[str, int] = {}
    for article in articles:
        for cat in article.get('geo_categories', []):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Build summary text from top headlines
    top_headlines = [a['title'] for a in articles[:15]]
    summary_lines = []
    for i, headline in enumerate(top_headlines, 1):
        summary_lines.append(f"  {i}. {headline}")

    # Detect severity level
    oil_war_overlap = 0
    for article in articles:
        cats = set(article.get('geo_categories', []))
        if 'oil_energy' in cats and ('war_conflict' in cats or 'shipping_trade' in cats):
            oil_war_overlap += 1

    if oil_war_overlap >= 5:
        severity = "CRITICAL"
        severity_desc = "Major oil supply disruption from active conflict"
    elif oil_war_overlap >= 2:
        severity = "HIGH"
        severity_desc = "Significant geopolitical risk to energy markets"
    elif category_counts.get('war_conflict', 0) >= 3:
        severity = "ELEVATED"
        severity_desc = "Active conflict with potential energy market impact"
    elif category_counts.get('oil_energy', 0) >= 3:
        severity = "MODERATE"
        severity_desc = "Notable oil/energy market developments"
    else:
        severity = "LOW"
        severity_desc = "Normal geopolitical conditions"

    return {
        "available": True,
        "headline_count": len(articles),
        "categories": category_counts,
        "severity": severity,
        "severity_description": severity_desc,
        "oil_war_overlap_count": oil_war_overlap,
        "top_headlines": top_headlines,
        "summary_text": "\n".join(summary_lines),
        "fetched_at": datetime.now().isoformat(),
    }


# ============================================================================
# CACHE HELPERS
# ============================================================================

def _load_cache() -> Optional[List[Dict]]:
    """Load cached global news if still valid. Filters stale articles on load."""
    try:
        if not CACHE_FILE.exists():
            return None

        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)

        cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
        if datetime.now() - cached_at > timedelta(hours=CACHE_TTL_HOURS):
            return None

        articles = data.get('articles', [])
        # Filter stale cached articles — prevents old news from polluting analysis
        articles = _filter_by_recency(articles, max_age_hours=48, fallback_days=7)
        return articles
    except Exception:
        return None


def _save_cache(articles: List[Dict]) -> None:
    """Save articles to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump({
                'cached_at': datetime.now().isoformat(),
                'article_count': len(articles),
                'articles': articles,
            }, f, indent=2, default=str)
    except Exception as e:
        logger.debug(f"Failed to save global news cache: {e}")


def clear_cache() -> None:
    """Force clear the global news cache."""
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
    except Exception:
        pass


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌍 Testing Global Geopolitical News Fetcher")
    print("=" * 60)

    articles = fetch_global_geo_news(force_refresh=True)
    print(f"\nTotal articles found: {len(articles)}")

    summary = get_global_news_summary(articles)
    print(f"\nSeverity: {summary['severity']} — {summary['severity_description']}")
    print(f"Categories: {summary['categories']}")
    print(f"Oil-War overlap: {summary['oil_war_overlap_count']}")

    print(f"\nTop headlines:")
    for headline in summary.get('top_headlines', [])[:10]:
        print(f"  • {headline}")

    # Test sector weighting for OGDC
    print(f"\n\n--- OGDC (Energy) weighted news ---")
    ogdc_news = get_global_news_for_symbol('OGDC')
    for item in ogdc_news[:5]:
        print(f"  [{item.get('sector_weighted_score', 0):.1f}] {item['title'][:80]}")
