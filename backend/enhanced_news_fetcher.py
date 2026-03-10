"""
🔍 ENHANCED NEWS FETCHER
Multi-source news aggregation with company alias expansion, parent company detection,
and sector-wide news capture. Designed to catch major market-moving events like
UAE investments, IMF loans, policy changes that affect stock groups.
"""

import re
import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
from urllib.parse import urlparse
import math

try:
    from backend.energy_shock_features import (
        ENERGY_SHOCK_SECTORS,
        ENERGY_SHOCK_SYMBOLS,
    )
except ImportError:
    from energy_shock_features import (  # type: ignore
        ENERGY_SHOCK_SECTORS,
        ENERGY_SHOCK_SYMBOLS,
    )

# ============================================================================
# COMPANY ALIASES & PARENT COMPANIES
# Maps stock symbols to all related search terms
# ============================================================================

COMPANY_ALIASES = {
    # Fauji Group - CRITICAL (UAE $1B news missed without this)
    'FFC': {
        'names': ['Fauji Fertilizer', 'FFC', 'Fauji Fertilizer Company'],
        'parent': 'Fauji Foundation',
        'sector': 'fertilizer',
        'sector_peers': ['FFBL', 'EFERT', 'FATIMA'],
    },
    'FFBL': {
        'names': ['Fauji Fertilizer Bin Qasim', 'FFBL', 'Fauji Bin Qasim'],
        'parent': 'Fauji Foundation',
        'sector': 'fertilizer',
        'sector_peers': ['FFC', 'EFERT', 'FATIMA'],
    },
    'FCCL': {
        'names': ['Fauji Cement', 'FCCL', 'Fauji Cement Company'],
        'parent': 'Fauji Foundation',
        'sector': 'cement',
        'sector_peers': ['LUCK', 'DGKC', 'MLCF', 'PIOC'],
    },
    
    # Oil & Gas
    'PSO': {
        'names': ['Pakistan State Oil', 'PSO', 'State Oil'],
        'parent': None,
        'sector': 'oil_marketing',
        'sector_peers': ['SHEL', 'APL', 'HASCOL'],
    },
    'PPL': {
        'names': ['Pakistan Petroleum', 'PPL', 'Pakistan Petroleum Limited'],
        'parent': None,
        'sector': 'exploration_production',
        'sector_peers': ['OGDC', 'POL', 'MARI'],
    },
    'OGDC': {
        'names': ['OGDC', 'Oil and Gas Development Company', 'Oil Gas Development'],
        'parent': None,
        'sector': 'exploration_production',
        'sector_peers': ['PPL', 'POL', 'MARI'],
    },
    'POL': {
        'names': ['Pakistan Oilfields', 'POL', 'Pakistan Oilfields Limited'],
        'parent': 'Attock Group',
        'sector': 'exploration_production',
        'sector_peers': ['OGDC', 'PPL', 'MARI'],
    },
    'MARI': {
        'names': ['Mari Petroleum', 'MARI', 'Mari Gas'],
        'parent': None,
        'sector': 'exploration_production',
        'sector_peers': ['OGDC', 'PPL', 'POL'],
    },

    # Autos
    'SAZEW': {
        'names': ['Sazgar Engineering', 'SAZEW', 'Sazgar Engineering Works', 'Haval Pakistan'],
        'parent': None,
        'sector': 'autos',
        'sector_peers': ['INDU', 'HCAR', 'PSMC', 'MTL'],
    },
    'INDU': {
        'names': ['Indus Motor', 'INDU', 'Toyota Indus', 'Indus Motor Company'],
        'parent': None,
        'sector': 'autos',
        'sector_peers': ['SAZEW', 'HCAR', 'PSMC', 'MTL'],
    },
    'HCAR': {
        'names': ['Honda Atlas Cars', 'HCAR', 'Honda Atlas', 'Honda Cars Pakistan'],
        'parent': 'Atlas Group',
        'sector': 'autos',
        'sector_peers': ['SAZEW', 'INDU', 'PSMC', 'MTL'],
    },
    'PSMC': {
        'names': ['Pak Suzuki', 'PSMC', 'Pak Suzuki Motor', 'Suzuki Pakistan'],
        'parent': 'Suzuki Motor',
        'sector': 'autos',
        'sector_peers': ['SAZEW', 'INDU', 'HCAR', 'MTL'],
    },
    'MTL': {
        'names': ['Millat Tractors', 'MTL', 'Millat'],
        'parent': 'Millat Group',
        'sector': 'autos',
        'sector_peers': ['SAZEW', 'INDU', 'HCAR', 'PSMC'],
    },

    # Cement
    'LUCK': {
        'names': ['Lucky Cement', 'LUCK', 'Lucky Cement Limited'],
        'parent': 'Lucky Group',
        'sector': 'cement',
        'sector_peers': ['DGKC', 'MLCF', 'PIOC', 'FCCL', 'CHCC', 'KOHC'],
    },
    'DGKC': {
        'names': ['DG Khan Cement', 'DGKC', 'DG Cement'],
        'parent': 'Nishat Group',
        'sector': 'cement',
        'sector_peers': ['LUCK', 'MLCF', 'PIOC', 'FCCL'],
    },
    'MLCF': {
        'names': ['Maple Leaf Cement', 'MLCF', 'Maple Leaf'],
        'parent': None,
        'sector': 'cement',
        'sector_peers': ['LUCK', 'DGKC', 'FCCL', 'CHCC', 'KOHC'],
    },
    'CHCC': {
        'names': ['Cherat Cement', 'CHCC', 'Cherat Cement Company'],
        'parent': None,
        'sector': 'cement',
        'sector_peers': ['LUCK', 'DGKC', 'FCCL', 'MLCF', 'KOHC'],
    },
    'KOHC': {
        'names': ['Kohat Cement', 'KOHC', 'Kohat Cement Company'],
        'parent': None,
        'sector': 'cement',
        'sector_peers': ['LUCK', 'DGKC', 'FCCL', 'MLCF', 'CHCC'],
    },

    # Steel
    'ISL': {
        'names': ['International Steels', 'ISL', 'International Steels Limited'],
        'parent': 'International Industries',
        'sector': 'steel',
        'sector_peers': ['ASTL', 'MUGHAL'],
    },
    'ASTL': {
        'names': ['Amreli Steels', 'ASTL', 'Amreli Steel'],
        'parent': None,
        'sector': 'steel',
        'sector_peers': ['ISL', 'MUGHAL'],
    },
    'MUGHAL': {
        'names': ['Mughal Iron and Steel', 'MUGHAL', 'Mughal Steel'],
        'parent': None,
        'sector': 'steel',
        'sector_peers': ['ISL', 'ASTL'],
    },
    
    # Banks
    'HBL': {
        'names': ['Habib Bank', 'HBL', 'Habib Bank Limited'],
        'parent': 'Aga Khan Fund',
        'sector': 'banks',
        'sector_peers': ['UBL', 'MCB', 'NBP', 'ABL', 'BAFL', 'MEBL'],
    },
    'UBL': {
        'names': ['United Bank', 'UBL', 'United Bank Limited'],
        'parent': 'Bestway Group',
        'sector': 'banks',
        'sector_peers': ['HBL', 'MCB', 'NBP', 'ABL', 'BAFL'],
    },
    'MEBL': {
        'names': ['Meezan Bank', 'MEBL', 'Meezan'],
        'parent': None,
        'sector': 'islamic_banks',
        'sector_peers': ['HBL', 'UBL', 'MCB', 'BAHL'],
    },
    
    # Tech
    'SYS': {
        'names': ['Systems Limited', 'SYS', 'Systems Ltd'],
        'parent': None,
        'sector': 'technology',
        'sector_peers': ['TRG', 'NETSOL'],
    },
    'TRG': {
        'names': ['TRG Pakistan', 'TRG', 'The Resource Group'],
        'parent': None,
        'sector': 'technology',
        'sector_peers': ['SYS', 'NETSOL', 'AVN'],
    },
    'NETSOL': {
        'names': ['NetSol Technologies', 'NETSOL', 'NetSol'],
        'parent': None,
        'sector': 'technology',
        'sector_peers': ['SYS', 'TRG', 'AVN'],
    },
    'AVN': {
        'names': ['Avanceon', 'AVN', 'Avanceon Limited'],
        'parent': None,
        'sector': 'technology',
        'sector_peers': ['SYS', 'TRG', 'NETSOL'],
    },
    
    # Fertilizer
    'EFERT': {
        'names': ['Engro Fertilizers', 'EFERT', 'Engro Fert'],
        'parent': 'Engro Corporation',
        'sector': 'fertilizer',
        'sector_peers': ['FFC', 'FFBL', 'FATIMA'],
    },
    'FATIMA': {
        'names': ['Fatima Fertilizer', 'FATIMA', 'Fatima Group'],
        'parent': None,
        'sector': 'fertilizer',
        'sector_peers': ['FFC', 'FFBL', 'EFERT'],
    },
    
    # Engro Group
    'ENGRO': {
        'names': ['Engro Corporation', 'ENGRO', 'Engro Corp'],
        'parent': None,
        'sector': 'conglomerate',
        'sector_peers': ['EFERT', 'EPCL', 'EFOOD'],
    },
    
    # Power
    'HUBC': {
        'names': ['Hub Power', 'HUBCO', 'Hub Power Company'],
        'parent': None,
        'sector': 'power',
        'sector_peers': ['KAPCO', 'NCPL', 'KEL', 'NPL'],
    },
    'KEL': {
        'names': ['K-Electric', 'KEL', 'Karachi Electric'],
        'parent': None,
        'sector': 'power',
        'sector_peers': ['HUBC', 'KAPCO', 'NCPL'],
    },
}

# Sector keywords for broad searches
SECTOR_KEYWORDS = {
    'fertilizer': ['fertilizer Pakistan', 'urea prices', 'DAP prices', 'fertilizer subsidy'],
    'autos': [
        'auto sector Pakistan',
        'car financing Pakistan',
        'auto sales Pakistan',
        'SBP auto financing',
        'consumer financing Pakistan',
        'petrol prices Pakistan',
        'PKR car demand',
    ],
    'cement': [
        'cement sector Pakistan',
        'cement exports',
        'construction Pakistan',
        'CPEC projects',
        'coal prices Pakistan',
        'power tariff Pakistan',
        'construction slowdown Pakistan',
        'interest rates Pakistan',
    ],
    'oil_marketing': ['OMC Pakistan', 'fuel prices Pakistan', 'petroleum levy', 'oil imports'],
    'exploration_production': ['oil gas Pakistan', 'petroleum exploration', 'OGRA', 'oil discovery'],
    'banks': ['banking sector Pakistan', 'SBP policy rate', 'KIBOR', 'monetary policy', 'ADR ratio'],
    'islamic_banks': ['islamic banking Pakistan', 'sukuk', 'sharia compliant'],
    'steel': [
        'steel sector Pakistan',
        'power tariff Pakistan steel',
        'construction slowdown Pakistan',
        'imported scrap Pakistan',
    ],
    'technology': [
        'IT exports Pakistan',
        'software exports Pakistan',
        'outsourcing Pakistan',
        'USD PKR software exports',
        'Pakistan tech exports',
    ],
    'power': ['power sector Pakistan', 'circular debt', 'electricity tariff', 'NEPRA'],
    'conglomerate': [],
}

ENERGY_SCOPE_QUERIES = {
    'oil_marketing': [
        'fuel prices Pakistan',
        'petroleum levy',
        'circular debt',
        'petrol prices Pakistan',
        'diesel prices Pakistan',
    ],
    'exploration_production': [
        'fuel prices Pakistan',
        'circular debt',
        'OGRA Pakistan',
        'petrol prices Pakistan',
        'diesel prices Pakistan',
    ],
}

ENERGY_GEO_DOMESTIC_QUERIES = [
    'petrol price hike Pakistan',
    'diesel price hike Pakistan',
    'fuel price hike Pakistan',
    'circular debt plan Pakistan',
    'circular debt payment release Pakistan',
    'receivables cleared Pakistan energy',
]

ENERGY_MACRO_QUERIES = [
    'Middle East war pushes energy prices higher',
    'Strait of Hormuz blockade',
    'blocked shipping lane oil',
    'shipping disruption oil',
    'Pakistan petrol diesel hiked',
]

ENERGY_SCOPE_HINTS = frozenset([
    'fuel prices',
    'petrol',
    'diesel',
    'price hike',
    'petrol price hike',
    'diesel price hike',
    'petroleum levy',
    'circular debt',
    'circular debt plan',
    'payment release',
    'receivables cleared',
    'ogra',
    'oil discovery',
    'oil and gas',
    'exploration',
])

ENERGY_MACRO_HINTS = frozenset([
    'middle east war',
    'energy prices higher',
    'strait of hormuz',
    'hormuz',
    'shipping disruption',
    'blocked shipping lane',
    'blockade',
    'petrol',
    'diesel',
    'fuel prices',
    'petroleum levy',
    'circular debt',
    'circular debt plan',
    'payment release',
    'receivables cleared',
    'oil prices',
    'iran',
    'israel',
    'gulf',
])

DOWNSTREAM_GEO_SYMBOLS = frozenset({
    'SAZEW', 'INDU', 'HCAR', 'PSMC', 'MTL',
    'LUCK', 'CHCC', 'DGKC', 'MLCF', 'FCCL', 'KOHC',
    'ISL', 'ASTL', 'MUGHAL',
    'SYS', 'TRG', 'NETSOL', 'AVN',
})

DOWNSTREAM_MACRO_QUERIES = [
    'petrol price hike Pakistan',
    'diesel price hike Pakistan',
    'SBP policy rate Pakistan',
    'inflation surge Pakistan',
    'auto financing Pakistan',
    'electricity tariff Pakistan',
    'coal prices Pakistan',
]

# Macro news categories that affect all stocks
MACRO_CATEGORIES = [
    'IMF Pakistan',
    'SBP policy rate',
    'KIBOR rate',
    'USD PKR exchange',
    'Pakistan forex reserves',
    'UAE Pakistan investment',
    'Saudi Arabia Pakistan investment',
    'China CPEC Pakistan',
    'Pakistan budget',
    'KSE-100 index',
    'PSX market today',
    # Geopolitical / conflict terms (Phase 3 expansion)
    'Pakistan India tensions',
    'Middle East conflict oil',
    'Iran Israel escalation',
    'Red Sea shipping disruption',
    'Pakistan Afghanistan border',
    'OPEC production cut',
    'global recession risk',
    'Pakistan credit rating',
]

# Index symbols that require market-wide retrieval strategy
INDEX_SYMBOLS = {'KSE100', 'KSE-100', 'PSX'}

# Expanded query pack for index/news-mode recall
INDEX_QUERY_PACK = [
    'PSX market today',
    'KSE-100 index',
    'Pakistan stocks',
    'PSX trading session',
    'market capitalization PSX',
    'Pakistan stock exchange',
    'PSX top gainers',
    'PSX top losers',
]

# Terms used to score index-level relevance
INDEX_RELEVANCE_TERMS = {
    'core': [
        'psx', 'kse', 'kse-100', 'kse100', 'stock exchange',
        'pakistan stocks', 'market capitalization'
    ],
    'macro': [
        'sbp', 'policy rate', 'kibor', 'imf', 'usd pkr',
        'forex reserves', 'budget', 'inflation', 'geopolitical',
        'conflict', 'war', 'sanctions', 'oil prices',
        'opec', 'recession', 'credit rating', 'trade war',
        'tariff', 'energy crisis', 'red sea', 'middle east',
        'border tension', 'military', 'defence',
    ],
    'business_hints': ['business', 'markets', 'stocks', 'economy'],
}


# ============================================================================
# MULTI-SOURCE NEWS SCRAPING
# ============================================================================

NEWS_SOURCES = {
    'business_recorder': {
        'search_url': 'https://www.brecorder.com/?s={}',
        'fallback_url': 'https://www.brecorder.com/search?query={}',
        'priority': 1,
    },
    'dawn': {
        'search_url': 'https://www.dawn.com/search?q={}',
        'fallback_url': 'https://www.dawn.com/news/business',
        'priority': 1,
    },
    'pakistan_today': {
        'search_url': 'https://www.pakistantoday.com.pk/?s={}',
        'fallback_url': 'https://www.pakistantoday.com.pk/category/business/',
        'priority': 1,
    },
    'express_tribune': {
        'search_url': 'https://tribune.com.pk/?s={}',
        'fallback_url': 'https://tribune.com.pk/business/psx',
        'priority': 2,
    },
    'geo_news': {
        'search_url': 'https://www.geo.tv/search/{}',
        'fallback_url': 'https://www.geo.tv/category/business',
        'priority': 2,
    },
    'minute_mirror': {
        'search_url': 'https://minutemirror.com.pk/?s={}',
        'fallback_url': 'https://minutemirror.com.pk/category/business/',
        'priority': 3,
    },
}

SOURCE_BASE_URLS = {
    'business recorder': 'https://www.brecorder.com',
    'dawn': 'https://www.dawn.com',
    'pakistan today': 'https://www.pakistantoday.com.pk',
    'express tribune': 'https://tribune.com.pk',
    'geo news': 'https://www.geo.tv',
    'minute mirror': 'https://minutemirror.com.pk',
    'sbp': 'https://www.sbp.org.pk',
}


def _env_flag(name: str, default: bool) -> bool:
    """Read boolean environment flag safely."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {'1', 'true', 'yes', 'on'}


def determine_retrieval_mode(symbol: str, retrieval_mode: str = 'auto') -> str:
    """
    Decide whether to run strict symbol-mode or broader index-mode retrieval.
    """
    mode = (retrieval_mode or 'auto').strip().lower()
    if mode in {'symbol_mode', 'index_mode'}:
        return mode
    return 'index_mode' if symbol.upper() in INDEX_SYMBOLS else 'symbol_mode'


def normalize_news_key(item: Dict) -> str:
    """Build a stable dedupe key from normalized title + date + URL path suffix."""
    title = re.sub(r'\s+', ' ', item.get('title', '').strip().lower())
    url = (item.get('url') or '').strip().lower()
    date_part = str(item.get('date') or '')[:10].strip().lower()
    if url.startswith('http'):
        url = re.sub(r'^https?://', '', url).split('?', 1)[0].rstrip('/')
    return f"{title[:180]}::{date_part}::{url[-120:]}"


def _safe_date_score(date_str: str) -> float:
    """
    Convert article date into a 0..1 recency score with a 14-day half-life.
    """
    try:
        d = datetime.strptime((date_str or '')[:10], '%Y-%m-%d')
        days_old = max(0, (datetime.now() - d).days)
    except Exception:
        days_old = 7
    return float(math.exp(-days_old / 14))


def score_index_relevance(title: str, url: str = '') -> float:
    """
    Weighted relevance score for index-level market articles.
    """
    text = f"{title} {url}".lower()
    score = 0.0

    for term in INDEX_RELEVANCE_TERMS['core']:
        if term in text:
            score += 1.0

    for term in INDEX_RELEVANCE_TERMS['macro']:
        if term in text:
            score += 0.6

    for hint in INDEX_RELEVANCE_TERMS['business_hints']:
        if hint in text:
            score += 0.3

    # Penalize clearly irrelevant verticals
    if any(x in text for x in ['sports', 'entertainment', 'lifestyle', 'opinion']):
        score -= 0.8

    return max(0.0, round(score, 3))


def _contains_market_term(title: str, url: str = '') -> bool:
    text = f"{title} {url}".lower()
    return any(
        x in text for x in [
            'psx', 'kse-100', 'kse100', 'stock exchange', 'market today', 'pakistan stocks'
        ]
    )


def dedupe_and_rank_news(news_items: List[Dict], retrieval_mode: str) -> List[Dict]:
    """
    Dedupe and rank results by relevance + source credibility + recency.
    """
    unique: Dict[str, Dict] = {}
    for item in news_items:
        key = normalize_news_key(item)
        existing = unique.get(key)
        if existing is None:
            unique[key] = item
            continue
        # Keep the richer/better-scoring record
        cur_score = float(item.get('relevance_score', 0))
        old_score = float(existing.get('relevance_score', 0))
        cur_scope = item.get('scope', 'macro')
        old_scope = existing.get('scope', 'macro')
        scope_weight = {'symbol': 3, 'sector': 2, 'macro': 1}
        if cur_score > old_score or (
            cur_score == old_score and scope_weight.get(cur_scope, 0) > scope_weight.get(old_scope, 0)
        ):
            unique[key] = item

    ranked = []
    for item in unique.values():
        source = (item.get('source') or '').lower()
        source_cred = SOURCE_CREDIBILITY.get(source, 0.6) if 'SOURCE_CREDIBILITY' in globals() else 0.6
        recency = _safe_date_score(item.get('date', ''))
        if retrieval_mode == 'index_mode':
            relevance = float(item.get('relevance_score', 0.0))
            rank_score = 0.5 * relevance + 0.3 * source_cred + 0.2 * recency
        else:
            scope_bonus = {
                'symbol': 1.0,
                'sector': 0.75,
                'macro': 0.6,
            }.get(item.get('scope', 'macro'), 0.5)
            rank_score = 0.5 * scope_bonus + 0.3 * source_cred + 0.2 * recency
        item['rank_score'] = round(rank_score, 4)
        ranked.append(item)

    ranked.sort(key=lambda x: (x.get('rank_score', 0), x.get('date', '')), reverse=True)
    return ranked


def fetch_news_curl_with_status(url: str, timeout: int = 10) -> Dict:
    """
    Fetch URL content with an explicit status for diagnostics.
    """
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', str(timeout),
             '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
             url],
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode != 0:
            return {'html': '', 'status': 'blocked'}
        if not result.stdout:
            return {'html': '', 'status': 'empty'}
        return {'html': result.stdout, 'status': 'ok'}
    except subprocess.TimeoutExpired:
        return {'html': '', 'status': 'timeout'}
    except Exception:
        return {'html': '', 'status': 'error'}


def fetch_news_curl(url: str, timeout: int = 10) -> str:
    """Fetch URL content using curl"""
    return fetch_news_curl_with_status(url, timeout=timeout).get('html', '')


def _normalize_href(source: str, href: str) -> str:
    """Normalize relative links to absolute for better scoring/dedupe."""
    href = (href or '').strip()
    if href.startswith('http://') or href.startswith('https://'):
        return href
    if href.startswith('//'):
        return f"https:{href}"
    base = SOURCE_BASE_URLS.get(source.lower(), '')
    if href.startswith('/') and base:
        return f"{base}{href}"
    return href


def _clean_anchor_text(text: str) -> str:
    text = re.sub(r'<script.*?</script>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<style.*?</style>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;|&#160;', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'&amp;', '&', text, flags=re.IGNORECASE)
    text = re.sub(r'&quot;', '"', text, flags=re.IGNORECASE)
    text = re.sub(r'&#39;', "'", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _is_likely_article_link(source: str, href: str, title: str) -> bool:
    """Source-aware guardrails to avoid nav/widgets while keeping article links."""
    href_l = (href or '').lower()
    title_l = (title or '').lower()
    path = urlparse(href_l).path

    if not href or href.startswith('#'):
        return False
    if any(x in href_l for x in ['javascript:', 'mailto:']):
        return False
    if any(x in href_l for x in ['/contact', '/privacy', '/about', '/profile', '/advertise']):
        return False
    if any(
        x in title_l for x in [
            'ramazan calendar', 'weather forecast', 'satellite parameters',
            'live tv', 'newsletter', 'careers', 'obituaries'
        ]
    ):
        return False

    source_l = source.lower()
    if source_l == 'dawn':
        return '/news/' in path
    if source_l == 'express tribune':
        return '/story/' in path or '/business/' in path
    if source_l == 'geo news':
        return '/latest/' in path or '/category/business' in path
    if source_l == 'business recorder':
        return '/news/' in path or '/markets/' in path or '/trends/psx' in path
    if source_l == 'minute mirror':
        return bool(re.search(r'-\d+/?$', path)) or '/business/' in path
    if source_l == 'pakistan today':
        return bool(re.search(r'/\d{4}/\d{2}/\d{2}/', path)) or '/category/business' in path

    # Generic fallback
    return len(path) > 8


def extract_articles_from_html(html: str, source: str) -> List[Dict]:
    """Extract article titles and links from HTML"""
    articles = []

    seen = set()
    # Parse complete anchor tags so we can use title=/alt= fallbacks when inner text is empty.
    for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>.*?</a>', html, re.IGNORECASE | re.DOTALL):
        whole = m.group(0)
        href = _normalize_href(source, m.group(1))

        # Primary title source: inner text
        inner = re.sub(r'^<a[^>]*>|</a>$', '', whole, flags=re.IGNORECASE | re.DOTALL)
        title = _clean_anchor_text(inner)

        # Fallbacks for image-based cards
        if not title:
            m_title = re.search(r'\btitle="([^"]+)"', whole, flags=re.IGNORECASE | re.DOTALL)
            if m_title:
                title = _clean_anchor_text(m_title.group(1))
        if not title:
            m_alt = re.search(r'\balt="([^"]+)"', whole, flags=re.IGNORECASE | re.DOTALL)
            if m_alt:
                title = _clean_anchor_text(m_alt.group(1))

        title_l = title.lower()
        if len(title) < 25 or len(title) > 220:
            continue
        if title_l in seen:
            continue
        if not _is_likely_article_link(source, href, title):
            continue

        seen.add(title_l)
        articles.append({
            'title': title[:300],
            'url': href,
            'source': source,
            'date': datetime.now().strftime('%Y-%m-%d'),
        })

        if len(articles) >= 20:
            break
    
    return articles


def _symbol_alias_terms(symbol: str) -> Set[str]:
    info = COMPANY_ALIASES.get(symbol.upper(), {})
    terms = {symbol.lower()}
    for name in info.get('names', []):
        terms.add(name.lower())
    if info.get('parent'):
        terms.add(str(info['parent']).lower())
    return {t for t in terms if t}


def _sector_relevance_terms(symbol: str) -> Set[str]:
    symbol = symbol.upper()
    info = COMPANY_ALIASES.get(symbol, {})
    sector = info.get('sector', '')
    terms = {term.lower() for term in SECTOR_KEYWORDS.get(sector, [])}
    for peer in info.get('sector_peers', []):
        peer_info = COMPANY_ALIASES.get(peer, {})
        peer_names = peer_info.get('names', [peer])
        terms.update(name.lower() for name in peer_names[:1])
    if sector in ENERGY_SHOCK_SECTORS or symbol in ENERGY_SHOCK_SYMBOLS:
        terms.update(ENERGY_SCOPE_HINTS)
    return {t for t in terms if t}


def _macro_relevance_terms(symbol: str) -> Set[str]:
    symbol = symbol.upper()
    terms = {term.lower() for term in MACRO_CATEGORIES}
    if symbol in ENERGY_SHOCK_SYMBOLS:
        terms.update(ENERGY_MACRO_HINTS)
    if symbol in DOWNSTREAM_GEO_SYMBOLS:
        terms.update(term.lower() for term in DOWNSTREAM_MACRO_QUERIES)
        sector = COMPANY_ALIASES.get(symbol, {}).get('sector', '')
        terms.update(term.lower() for term in SECTOR_KEYWORDS.get(sector, []))
    return {t for t in terms if t}


def get_search_query_specs(
    symbol: str,
    retrieval_mode: str = 'auto',
    geo_mode: bool = False,
) -> List[Dict[str, str]]:
    """Get scoped search queries tailored for symbol mode vs index mode."""
    mode = determine_retrieval_mode(symbol, retrieval_mode=retrieval_mode)
    symbol = symbol.upper()
    specs: List[Dict[str, str]] = []

    if mode == 'index_mode':
        for query in INDEX_QUERY_PACK + MACRO_CATEGORIES:
            specs.append({'query': query, 'scope': 'macro'})
        unique = []
        seen = set()
        for spec in specs:
            key = (spec['query'], spec['scope'])
            if key not in seen:
                seen.add(key)
                unique.append(spec)
        return unique

    if symbol in COMPANY_ALIASES:
        info = COMPANY_ALIASES[symbol]

        for name in info['names'][:2]:
            specs.append({'query': f"{name} PSX", 'scope': 'symbol'})

        if info['names']:
            specs.append({'query': info['names'][0], 'scope': 'symbol'})

        if info.get('parent'):
            specs.append({'query': info['parent'], 'scope': 'sector'})

        sector = info.get('sector')
        if symbol in ENERGY_SHOCK_SYMBOLS and sector in ENERGY_SCOPE_QUERIES:
            for query in ENERGY_SCOPE_QUERIES[sector]:
                specs.append({'query': query, 'scope': 'sector'})
            for query in ENERGY_MACRO_QUERIES:
                specs.append({'query': query, 'scope': 'macro'})
        if geo_mode and symbol in ENERGY_SHOCK_SYMBOLS:
            for query in ENERGY_GEO_DOMESTIC_QUERIES:
                specs.append({'query': query, 'scope': 'sector'})
        if geo_mode and symbol in DOWNSTREAM_GEO_SYMBOLS:
            for query in SECTOR_KEYWORDS.get(sector, []):
                specs.append({'query': query, 'scope': 'sector'})
            for query in DOWNSTREAM_MACRO_QUERIES:
                specs.append({'query': query, 'scope': 'macro'})
    else:
        specs.append({'query': f"{symbol} PSX", 'scope': 'symbol'})
        specs.append({'query': f"{symbol} Pakistan stock", 'scope': 'symbol'})

    unique = []
    seen = set()
    for spec in specs:
        key = (spec['query'], spec['scope'])
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def get_search_queries(symbol: str, retrieval_mode: str = 'auto', geo_mode: bool = False) -> List[str]:
    return [
        spec['query']
        for spec in get_search_query_specs(symbol, retrieval_mode=retrieval_mode, geo_mode=geo_mode)
    ]


def _is_symbol_relevant(title: str, queries: List[str]) -> bool:
    title_lower = title.lower()
    return any(q.lower() in title_lower for q in queries)


def _is_scope_relevant(article: Dict, symbol: str, scope: str, query: str) -> bool:
    text = f"{article.get('title', '')} {article.get('url', '')}".lower()
    if scope == 'symbol':
        alias_terms = _symbol_alias_terms(symbol)
        if any(term in text for term in alias_terms):
            return True
        return query.lower() in text
    if scope == 'sector':
        return any(term in text for term in _sector_relevance_terms(symbol))
    if scope == 'macro':
        return any(term in text for term in _macro_relevance_terms(symbol))
    return False


_GEO_CONFLICT_HINTS = frozenset([
    'conflict', 'war', 'strike', 'tension', 'sanctions', 'oil',
    'opec', 'red sea', 'recession', 'tariff', 'credit rating',
    'middle east', 'iran', 'shipping', 'energy crisis',
])


def _is_index_relevant(article: Dict) -> bool:
    score = score_index_relevance(article.get('title', ''), article.get('url', ''))
    article['relevance_score'] = score
    # Keep market-direct titles even if macro score is modest.
    if _contains_market_term(article.get('title', ''), article.get('url', '')):
        return score >= 0.6
    # Relax threshold for geopolitical/conflict articles (they affect PSX even
    # without an explicit "PSX" or "KSE" mention in the headline).
    title_lower = (article.get('title') or '').lower()
    if any(hint in title_lower for hint in _GEO_CONFLICT_HINTS):
        return score >= 0.6
    return score >= 1.0


def fetch_business_fallback(source_name: str, source_config: Dict, retrieval_mode: str,
                            symbol: str, query_specs: List[Dict[str, str]], max_items: int = 3) -> List[Dict]:
    """
    Fetch from source business pages when search endpoints are sparse/noisy.
    """
    fallback_url = source_config.get('fallback_url')
    if not fallback_url:
        return []
    if '{}' in fallback_url:
        query = query_specs[0]['query'] if query_specs else 'PSX market today'
        q = query.replace(' ', '+')
        fallback_url = fallback_url.format(q)

    result = fetch_news_curl_with_status(fallback_url, timeout=6)
    html = result.get('html', '')
    if not html:
        return []

    articles = extract_articles_from_html(html, source_name.replace('_', ' ').title())
    kept: List[Dict] = []
    for article in articles:
        if retrieval_mode == 'index_mode':
            if _is_index_relevant(article):
                article['is_direct'] = False
                article['is_macro'] = True
                article['scope'] = 'macro'
                kept.append(article)
        else:
            matched_scope = None
            for spec in query_specs:
                if _is_scope_relevant(article, symbol, spec['scope'], spec['query']):
                    matched_scope = spec['scope']
                    break
            if matched_scope:
                article['is_direct'] = matched_scope == 'symbol'
                article['is_macro'] = matched_scope == 'macro'
                article['scope'] = matched_scope
                kept.append(article)
        if len(kept) >= max_items:
            break
    return kept


def fetch_psx_notice_fallback(max_items: int = 5) -> List[Dict]:
    """Fetch index-relevant items from Business Recorder PSX notices page."""
    url = "https://www.brecorder.com/trends/psx-notice"
    result = fetch_news_curl_with_status(url, timeout=8)
    html = result.get('html', '')
    if not html:
        return []

    articles = extract_articles_from_html(html, 'Business Recorder')
    kept: List[Dict] = []
    for article in articles:
        article['is_direct'] = False
        article['is_macro'] = True
        article['scope'] = 'macro'
        article['relevance_score'] = score_index_relevance(article.get('title', ''), article.get('url', ''))
        if article['relevance_score'] >= 0.8:
            kept.append(article)
        if len(kept) >= max_items:
            break
    return kept


def fetch_sbp_monetary_policy(max_items: int = 5) -> List[Dict]:
    """Fetch monetary policy statements directly from SBP website.

    SBP (State Bank of Pakistan) publishes MPC decisions and Monetary Policy
    Statements at sbp.org.pk/m_policy/mon.asp.  This fills the gap where
    general news search only weakly detects 'monetary policy unchanged' or
    rate change announcements.
    """
    sbp_urls = [
        # Primary: Monetary Policy Statements page (clean table of PDF links)
        'https://www.sbp.org.pk/m_policy/mon.asp',
        # Fallback: press releases
        'https://www.sbp.org.pk/press/releases.asp',
    ]

    policy_keywords = [
        'monetary policy statement', 'monetary policy decision',
        'policy rate', 'interest rate', 'discount rate',
        'rate unchanged', 'rate cut', 'rate hike', 'basis points',
        'mpc decision', 'monetary policy committee', 'rate decision',
        'kept unchanged', 'raised', 'lowered', 'tightening', 'easing',
        'monetary policy information',
    ]

    articles: List[Dict] = []
    seen: Set[str] = set()

    for url in sbp_urls:
        try:
            result = fetch_news_curl_with_status(url, timeout=10)
            html = result.get('html', '')
            if not html:
                continue

            for m in re.finditer(
                r'<a[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
                html, re.IGNORECASE | re.DOTALL
            ):
                href = m.group(1).strip()
                title = _clean_anchor_text(m.group(2))
                if len(title) < 15 or len(title) > 300:
                    continue

                title_lower = title.lower()
                if title_lower in seen:
                    continue

                # Only keep monetary-policy-relevant items
                if not any(kw in title_lower for kw in policy_keywords):
                    continue

                # Skip Urdu versions to avoid duplicates
                if '(urdu)' in title_lower:
                    continue

                # Normalize SBP relative links
                if href and not href.startswith('http'):
                    if href.startswith('/'):
                        href = f"https://www.sbp.org.pk{href}"
                    else:
                        href = f"https://www.sbp.org.pk/m_policy/{href}"

                # Extract date from title like "Mar 09, 2026"
                date_match = re.search(
                    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}',
                    title, re.IGNORECASE
                )
                if date_match:
                    try:
                        parsed_date = datetime.strptime(
                            date_match.group(0).replace(',', ''), '%b %d %Y'
                        )
                        article_date = parsed_date.strftime('%Y-%m-%d')
                    except ValueError:
                        article_date = datetime.now().strftime('%Y-%m-%d')
                else:
                    article_date = datetime.now().strftime('%Y-%m-%d')

                seen.add(title_lower)
                articles.append({
                    'title': title[:300],
                    'url': href,
                    'source': 'SBP',
                    'date': article_date,
                    'is_macro': True,
                    'is_direct': True,
                    'scope': 'macro',
                    'relevance_score': 1.0,
                })
                if len(articles) >= max_items:
                    break
            if len(articles) >= max_items:
                break
        except Exception:
            continue

    return articles


def fetch_multi_source_news(
    symbol: str,
    max_per_source: int = 5,
    retrieval_mode: str = 'auto',
    include_diagnostics: bool = False,
    geo_mode: bool = False,
) -> Union[Dict, List[Dict]]:
    """
    Fetch news from multiple sources with mode-aware relevance scoring and diagnostics.
    """
    mode = determine_retrieval_mode(symbol, retrieval_mode=retrieval_mode)
    query_specs = get_search_query_specs(symbol, retrieval_mode=mode, geo_mode=geo_mode)
    queries = [spec['query'] for spec in query_specs]
    diagnostics = {
        'retrieval_mode': mode,
        'geo_mode': geo_mode,
        'per_source': {},
        'fallback_chain': [],
        'filtered_count': 0,
        'kept_count': 0,
    }

    all_news: List[Dict] = []

    print(f"\n📰 ENHANCED NEWS FETCH: {symbol}")
    print(f"   Mode: {mode} | Queries: {queries[:5]}...")

    # 1) Primary multi-source query pass
    for source_name, source_config in NEWS_SOURCES.items():
        source_stats = {
            'fetched': 0,
            'parsed': 0,
            'filtered_out': 0,
            'kept': 0,
            'status': 'ok',
        }
        try:
            source_news: List[Dict] = []
            for spec in query_specs:
                query = spec['query']
                scope = spec['scope']
                search_url = source_config['search_url'].format(query.replace(' ', '+'))
                fetched = fetch_news_curl_with_status(search_url)
                status = fetched.get('status', 'error')
                if status != 'ok':
                    source_stats['status'] = status
                    continue

                source_stats['fetched'] += 1
                articles = extract_articles_from_html(
                    fetched.get('html', ''),
                    source_name.replace('_', ' ').title()
                )
                source_stats['parsed'] += len(articles)

                for article in articles:
                    is_relevant = (
                        _is_index_relevant(article)
                        if mode == 'index_mode'
                        else _is_scope_relevant(article, symbol, scope, query)
                    )
                    if not is_relevant:
                        source_stats['filtered_out'] += 1
                        continue

                    article['scope'] = 'macro' if mode == 'index_mode' else scope
                    article['is_direct'] = article['scope'] == 'symbol'
                    article['is_macro'] = article['scope'] == 'macro'
                    article['matched_query'] = query
                    source_news.append(article)
                    source_stats['kept'] += 1

                if len(source_news) >= max_per_source:
                    break

            if source_news:
                print(f"   ✅ {source_name}: {len(source_news)} relevant articles")
                all_news.extend(source_news[:max_per_source])
            diagnostics['per_source'][source_name] = source_stats
        except Exception as e:
            source_stats['status'] = 'error'
            diagnostics['per_source'][source_name] = source_stats
            print(f"   ⚠️ {source_name}: Error - {str(e)[:40]}")

    # 2) Fallback: business section pages
    # For index mode, always run this pass (search pages can be noisy/empty).
    if len(all_news) < 5 or mode == 'index_mode':
        for source_name, source_config in NEWS_SOURCES.items():
            fallback_items = fetch_business_fallback(
                source_name=source_name,
                source_config=source_config,
                retrieval_mode=mode,
                symbol=symbol,
                query_specs=query_specs,
                max_items=3
            )
            if fallback_items:
                all_news.extend(fallback_items)
                diagnostics['fallback_chain'].append(
                    {'stage': 'business_fallback', 'source': source_name, 'added': len(fallback_items)}
                )

    # 3) Fallback: PSX notices (index mode only)
    if mode == 'index_mode' and len(all_news) < 5:
        notice_items = fetch_psx_notice_fallback(max_items=5)
        if notice_items:
            all_news.extend(notice_items)
            diagnostics['fallback_chain'].append(
                {'stage': 'psx_notice_fallback', 'source': 'business_recorder', 'added': len(notice_items)}
            )

    # 4) Broad macro fallback (always available; deeper for index mode)
    macro_news = fetch_macro_news(
        symbol=symbol,
        retrieval_mode=mode,
        max_topics=8 if mode == 'index_mode' else 3,
        geo_mode=geo_mode,
    )
    if macro_news:
        all_news.extend(macro_news)
        diagnostics['fallback_chain'].append(
            {'stage': 'macro_fallback', 'source': 'multi', 'added': len(macro_news)}
        )

    # 5) SBP monetary policy — direct source for rate decisions
    try:
        sbp_articles = fetch_sbp_monetary_policy(max_items=3)
        if sbp_articles:
            all_news.extend(sbp_articles)
            diagnostics['fallback_chain'].append(
                {'stage': 'sbp_monetary_policy', 'source': 'sbp', 'added': len(sbp_articles)}
            )
            print(f"   🏦 SBP: {len(sbp_articles)} monetary policy articles")
    except Exception:
        pass

    # Final dedupe/rank
    ranked_news = dedupe_and_rank_news(all_news, retrieval_mode=mode)
    diagnostics['filtered_count'] = sum(v.get('filtered_out', 0) for v in diagnostics['per_source'].values())
    diagnostics['kept_count'] = len(ranked_news)
    diagnostics['sources_attempted'] = len(NEWS_SOURCES)
    diagnostics['sources_successful'] = sum(1 for v in diagnostics['per_source'].values() if v.get('kept', 0) > 0)

    print(f"   📊 Total: {len(ranked_news)} relevant articles")

    response = {
        'news_items': ranked_news,
        'queries_used': queries,
        'retrieval_mode': mode,
        'news_fetch_diagnostics': diagnostics,
        'sources_attempted': diagnostics['sources_attempted'],
        'sources_successful': diagnostics['sources_successful'],
        'filtered_count': diagnostics['filtered_count'],
    }
    if include_diagnostics:
        return response
    return ranked_news


# Sources ordered by reliability for financial content (Minute Mirror works
# best with curl for PSX-related queries; others often return homepage/JS).
_MACRO_SEARCH_SOURCES = [
    ('Minute Mirror', 'https://minutemirror.com.pk/?s={}'),
    ('Pakistan Today', 'https://www.pakistantoday.com.pk/?s={}'),
    ('Dawn', 'https://www.dawn.com/search?q={}'),
    ('Business Recorder', 'https://www.brecorder.com/?s={}'),
]

# Prioritise PSX-specific macro queries (these return financial content);
# abstract geopolitical queries ("Iran Israel escalation") rarely yield
# stock-relevant results from Pakistani news sources via curl search.
_MACRO_PRIORITY_QUERIES = [
    'PSX market today',
    'KSE-100 index',
    'Pakistan stocks',
    'SBP policy rate Pakistan',
    'IMF Pakistan economy',
    'USD PKR exchange rate',
    'Pakistan budget economy',
    'oil prices Pakistan economy',
]


def fetch_macro_news(
    symbol: Optional[str] = None,
    retrieval_mode: str = 'symbol_mode',
    max_topics: int = 3,
    geo_mode: bool = False,
) -> List[Dict]:
    """Fetch macro economic news that affects PSX; broader in index mode.

    Uses prioritised PSX-specific queries across multiple sources to improve
    recall.  The geopolitical/conflict terms in MACRO_CATEGORIES are still
    used for *scoring* fetched articles, even when they can't be used as
    effective search queries (most Pakistani news search endpoints are noisy
    or broken with curl).
    """
    macro_articles: List[Dict] = []
    seen: Set[str] = set()

    if retrieval_mode == 'index_mode':
        topics = list(dict.fromkeys(
            _MACRO_PRIORITY_QUERIES + INDEX_QUERY_PACK + MACRO_CATEGORIES
        ))[:max_topics]
    else:
        symbol_upper = (symbol or '').upper()
        if geo_mode and symbol_upper in ENERGY_SHOCK_SYMBOLS:
            topics = list(dict.fromkeys(
                ENERGY_GEO_DOMESTIC_QUERIES + ENERGY_MACRO_QUERIES + MACRO_CATEGORIES
            ))[:max_topics + 6]
        elif symbol_upper in ENERGY_SHOCK_SYMBOLS:
            topics = list(dict.fromkeys(
                ENERGY_MACRO_QUERIES + MACRO_CATEGORIES
            ))[:max_topics + 3]
        elif geo_mode and symbol_upper in DOWNSTREAM_GEO_SYMBOLS:
            sector = COMPANY_ALIASES.get(symbol_upper, {}).get('sector', '')
            topics = list(dict.fromkeys(
                DOWNSTREAM_MACRO_QUERIES + SECTOR_KEYWORDS.get(sector, []) + MACRO_CATEGORIES
            ))[:max_topics + 4]
        else:
            topics = MACRO_CATEGORIES[:max_topics]

    target = 8 if retrieval_mode == 'index_mode' else 5

    for topic in topics:
        if len(macro_articles) >= target:
            break
        for source_name, url_template in _MACRO_SEARCH_SOURCES:
            if len(macro_articles) >= target:
                break
            try:
                search_url = url_template.format(topic.replace(' ', '+'))
                fetched = fetch_news_curl_with_status(search_url, timeout=8)
                if fetched.get('status') != 'ok':
                    continue

                articles = extract_articles_from_html(
                    fetched.get('html', ''), source_name
                )
                added_from_source = 0
                # Scan more articles (relevant ones may not be first)
                for article in articles[:10]:
                    key = normalize_news_key(article)
                    if key in seen:
                        continue
                    if retrieval_mode == 'index_mode':
                        if not _is_index_relevant(article):
                            continue
                    seen.add(key)
                    article['is_macro'] = True
                    article['is_direct'] = False
                    article['scope'] = 'macro'
                    macro_articles.append(article)
                    added_from_source += 1
                    if added_from_source >= 3:
                        break

                # If this source yielded results for this topic, move to next topic
                if added_from_source > 0:
                    break
            except Exception:
                continue

    return macro_articles[:target]


# ============================================================================
# SENTIMENT SCORING WITH TIME DECAY
# ============================================================================

SOURCE_CREDIBILITY = {
    'business recorder': 1.0,
    'dawn': 1.0,
    'pakistan today': 0.9,
    'express tribune': 0.9,
    'psx': 1.0,
    'geo news': 0.8,
    'minute mirror': 0.7,
    'sbp': 1.0,
}


def calculate_news_bias(
    news_items: List[Dict],
    sentiment_scores: List[float]  # From LLM analysis
) -> Dict:
    """
    Calculate time-weighted news bias.
    
    Returns:
        {
            'bias': float (-1 to +1),
            'confidence': float (0 to 1),
            'signals': list of signal descriptions
        }
    """
    if not news_items or not sentiment_scores:
        return {'bias': 0, 'confidence': 0, 'signals': []}
    
    total_weight = 0
    weighted_sentiment = 0
    signals = []
    
    for i, (item, sentiment) in enumerate(zip(news_items, sentiment_scores)):
        # Time decay (7-day half-life)
        try:
            pub_date = datetime.strptime(item.get('date', ''), '%Y-%m-%d')
            days_old = (datetime.now() - pub_date).days
        except:
            days_old = 0
        
        decay = math.exp(-days_old / 7)
        
        # Source credibility
        source = item.get('source', '').lower()
        cred = SOURCE_CREDIBILITY.get(source, 0.5)
        
        # Direct vs sector relevance
        relevance = 1.0 if item.get('is_direct') else 0.5
        
        # Macro importance
        if item.get('is_macro'):
            relevance = 0.7  # Medium importance
        
        weight = decay * cred * relevance
        weighted_sentiment += sentiment * weight
        total_weight += weight
        
        # Track significant signals
        if abs(sentiment) > 0.3:
            direction = "🟢 Bullish" if sentiment > 0 else "🔴 Bearish"
            signals.append({
                'title': item['title'][:80],
                'source': item.get('source', 'Unknown'),
                'direction': direction,
                'strength': abs(sentiment)
            })
    
    bias = weighted_sentiment / total_weight if total_weight > 0 else 0
    confidence = min(total_weight / 5, 1.0)  # Max confidence at 5 weighted items
    
    return {
        'bias': round(bias, 3),
        'confidence': round(confidence, 3),
        'signals': signals[:5]  # Top 5 signals
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_enhanced_news_for_symbol(
    symbol: str,
    retrieval_mode: str = 'auto',
    geo_mode: bool = False,
) -> Dict:
    """
    Main function: Get comprehensive news for a symbol.
    
    Returns:
        {
            'news_items': list of articles,
            'queries_used': list of search terms,
            'sources_checked': list of sources,
            'parent_company': str or None,
            'sector': str,
            'sector_peers': list
        }
    """
    symbol = symbol.upper()
    
    fetch_result = fetch_multi_source_news(
        symbol=symbol,
        retrieval_mode=retrieval_mode,
        include_diagnostics=True,
        geo_mode=geo_mode,
    )
    news_items = fetch_result.get('news_items', [])
    queries = fetch_result.get('queries_used', [])
    
    company_info = COMPANY_ALIASES.get(symbol, {})
    
    return {
        'news_items': news_items,
        'queries_used': queries,
        'retrieval_mode': fetch_result.get('retrieval_mode', determine_retrieval_mode(symbol)),
        'news_fetch_diagnostics': fetch_result.get('news_fetch_diagnostics', {}),
        'sources_attempted': fetch_result.get('sources_attempted', len(NEWS_SOURCES)),
        'sources_successful': fetch_result.get('sources_successful', 0),
        'filtered_count': fetch_result.get('filtered_count', 0),
        'sources_checked': list(NEWS_SOURCES.keys()),
        'parent_company': company_info.get('parent'),
        'sector': company_info.get('sector', 'unknown'),
        'sector_peers': company_info.get('sector_peers', []),
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test with FFC (should catch Fauji Foundation news)
    print("\n" + "=" * 60)
    print("Testing Enhanced News Fetcher")
    print("=" * 60)
    
    result = get_enhanced_news_for_symbol('FFC')
    
    print(f"\nQueries used: {result['queries_used']}")
    print(f"Parent company: {result['parent_company']}")
    print(f"Sector: {result['sector']}")
    print(f"\nNews found ({len(result['news_items'])} articles):")
    
    for item in result['news_items'][:5]:
        direct = "✓" if item.get('is_direct') else " "
        print(f"  [{direct}] {item['source']}: {item['title'][:60]}...")
