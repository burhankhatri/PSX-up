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
        'sector_peers': ['SYS', 'NETSOL'],
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
    'cement': ['cement sector Pakistan', 'cement exports', 'construction Pakistan', 'CPEC projects'],
    'oil_marketing': ['OMC Pakistan', 'fuel prices Pakistan', 'petroleum levy', 'oil imports'],
    'exploration_production': ['oil gas Pakistan', 'petroleum exploration', 'OGRA', 'oil discovery'],
    'banks': ['banking sector Pakistan', 'SBP policy rate', 'KIBOR', 'monetary policy', 'ADR ratio'],
    'islamic_banks': ['islamic banking Pakistan', 'sukuk', 'sharia compliant'],
    'technology': ['IT exports Pakistan', 'software exports', 'tech Pakistan'],
    'power': ['power sector Pakistan', 'circular debt', 'electricity tariff', 'NEPRA'],
    'conglomerate': [],
}

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
    """Build a stable dedupe key from normalized title + URL path suffix."""
    title = re.sub(r'\s+', ' ', item.get('title', '').strip().lower())
    url = (item.get('url') or '').strip().lower()
    if url.startswith('http'):
        url = re.sub(r'^https?://', '', url).split('?', 1)[0].rstrip('/')
    return f"{title[:180]}::{url[-120:]}"


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
        if cur_score > old_score:
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
            direct_bonus = 1.0 if item.get('is_direct') else 0.5
            rank_score = 0.5 * direct_bonus + 0.3 * source_cred + 0.2 * recency
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


def get_search_queries(symbol: str, retrieval_mode: str = 'auto') -> List[str]:
    """Get search queries tailored for symbol mode vs index mode."""
    mode = determine_retrieval_mode(symbol, retrieval_mode=retrieval_mode)
    symbol = symbol.upper()
    queries: List[str] = []

    if mode == 'index_mode':
        queries.extend(INDEX_QUERY_PACK)
        queries.extend(MACRO_CATEGORIES)
        return list(dict.fromkeys(queries))

    if symbol in COMPANY_ALIASES:
        info = COMPANY_ALIASES[symbol]

        for name in info['names'][:2]:
            queries.append(f"{name} PSX")

        if info['names']:
            queries.append(info['names'][0])

        if info.get('parent'):
            queries.append(info['parent'])
    else:
        queries.append(f"{symbol} PSX")
        queries.append(f"{symbol} Pakistan stock")

    return list(dict.fromkeys(queries))


def _is_symbol_relevant(title: str, queries: List[str]) -> bool:
    title_lower = title.lower()
    return any(q.lower() in title_lower for q in queries)


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
                            queries: List[str], max_items: int = 3) -> List[Dict]:
    """
    Fetch from source business pages when search endpoints are sparse/noisy.
    """
    fallback_url = source_config.get('fallback_url')
    if not fallback_url:
        return []
    if '{}' in fallback_url:
        q = (queries[0] if queries else 'PSX market today').replace(' ', '+')
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
                kept.append(article)
        else:
            if _is_symbol_relevant(article.get('title', ''), queries):
                article['is_direct'] = True
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
        article['relevance_score'] = score_index_relevance(article.get('title', ''), article.get('url', ''))
        if article['relevance_score'] >= 0.8:
            kept.append(article)
        if len(kept) >= max_items:
            break
    return kept


def fetch_multi_source_news(
    symbol: str,
    max_per_source: int = 5,
    retrieval_mode: str = 'auto',
    include_diagnostics: bool = False
) -> Union[Dict, List[Dict]]:
    """
    Fetch news from multiple sources with mode-aware relevance scoring and diagnostics.
    """
    mode = determine_retrieval_mode(symbol, retrieval_mode=retrieval_mode)
    queries = get_search_queries(symbol, retrieval_mode=mode)
    diagnostics = {
        'retrieval_mode': mode,
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
            for query in queries[:4]:
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
                        else _is_symbol_relevant(article.get('title', ''), queries)
                    )
                    if not is_relevant:
                        source_stats['filtered_out'] += 1
                        continue

                    article['is_direct'] = (mode == 'symbol_mode')
                    article['is_macro'] = (mode == 'index_mode')
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
                queries=queries,
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
        retrieval_mode=mode,
        max_topics=8 if mode == 'index_mode' else 3
    )
    if macro_news:
        all_news.extend(macro_news)
        diagnostics['fallback_chain'].append(
            {'stage': 'macro_fallback', 'source': 'multi', 'added': len(macro_news)}
        )

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


def fetch_macro_news(retrieval_mode: str = 'symbol_mode', max_topics: int = 3) -> List[Dict]:
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

def get_enhanced_news_for_symbol(symbol: str, retrieval_mode: str = 'auto') -> Dict:
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
        include_diagnostics=True
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
