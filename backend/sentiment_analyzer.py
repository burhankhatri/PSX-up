#!/usr/bin/env python3
"""
🔮 PRODUCTION-GRADE AI SENTIMENT ANALYZER
Uses Premium News Fetcher (10+ sources) + Groq LLM for intelligent analysis.
Designed to be a "fortune teller" for PSX stocks.
"""

import os
import json
import subprocess
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

# Import premium news fetcher
try:
    from backend.premium_news_fetcher import fetch_premium_news, get_news_for_sentiment_analysis
    PREMIUM_FETCHER_AVAILABLE = True
except ImportError:
    try:
        from premium_news_fetcher import fetch_premium_news, get_news_for_sentiment_analysis
        PREMIUM_FETCHER_AVAILABLE = True
    except ImportError:
        PREMIUM_FETCHER_AVAILABLE = False
        print("⚠️  Premium news fetcher not available, using fallback")

# Import enhanced news fetcher (multi-source with company aliases)
try:
    from backend.enhanced_news_fetcher import (
        get_enhanced_news_for_symbol, 
        fetch_multi_source_news,
        COMPANY_ALIASES,
        determine_retrieval_mode
    )
    ENHANCED_FETCHER_AVAILABLE = True
except ImportError:
    try:
        from enhanced_news_fetcher import (
            get_enhanced_news_for_symbol,
            fetch_multi_source_news,
            COMPANY_ALIASES,
            determine_retrieval_mode
        )
        ENHANCED_FETCHER_AVAILABLE = True
    except ImportError:
        ENHANCED_FETCHER_AVAILABLE = False
        COMPANY_ALIASES = {}
        determine_retrieval_mode = None
        print("⚠️  Enhanced news fetcher not available")

# Import article scraper for enriched data
try:
    from backend.article_scraper import get_enriched_stock_data, fetch_live_fundamentals
    ARTICLE_SCRAPER_AVAILABLE = True
except ImportError:
    try:
        from article_scraper import get_enriched_stock_data, fetch_live_fundamentals
        ARTICLE_SCRAPER_AVAILABLE = True
    except ImportError:
        ARTICLE_SCRAPER_AVAILABLE = False
        print("⚠️  Article scraper not available")

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed")

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / "data" / "news_cache"
CACHE_DURATION_HOURS = 4  # Cache news for 4 hours

INDEX_SYMBOLS = {'KSE100', 'KSE-100', 'PSX'}


def _env_flag(name: str, default: bool) -> bool:
    """Read boolean env var with a safe default."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {'1', 'true', 'yes', 'on'}

# Stock symbol to company name mapping
STOCK_COMPANIES = {
    'LUCK': ('Lucky Cement', 'Lucky Cement Limited', 'LUCK'),
    'HBL': ('Habib Bank', 'Habib Bank Limited', 'HBL'),
    'UBL': ('United Bank', 'United Bank Limited', 'UBL'),
    'MCB': ('MCB Bank', 'MCB Bank Limited', 'Muslim Commercial Bank'),
    'OGDC': ('OGDC', 'Oil and Gas Development Company', 'Oil Gas Development'),
    'PPL': ('Pakistan Petroleum', 'PPL', 'Pakistan Petroleum Limited'),
    'PSO': ('Pakistan State Oil', 'PSO', 'State Oil'),
    'ENGRO': ('Engro', 'Engro Corporation', 'Engro Corp'),
    'FFC': ('Fauji Fertilizer', 'FFC', 'Fauji Fertilizer Company'),
    'FATIMA': ('Fatima Fertilizer', 'Fatima', 'Fatima Group'),
    'HUBC': ('Hub Power', 'HUBCO', 'Hub Power Company'),
    'SYS': ('Systems Limited', 'SYS', 'Systems Ltd'),
    'TRG': ('TRG Pakistan', 'TRG', 'The Resource Group'),
    'NETSOL': ('NetSol Technologies', 'NETSOL', 'NetSol'),
    'AVN': ('Avanceon', 'AVN', 'Avanceon Limited'),
    'NESTLE': ('Nestle Pakistan', 'Nestle', 'NESTLE'),
    'MARI': ('Mari Petroleum', 'MARI', 'Mari Gas'),
    'ISL': ('International Steels', 'ISL', 'International Steels Limited'),
    'ASTL': ('Amreli Steels', 'ASTL', 'Amreli Steel'),
    'MUGHAL': ('Mughal Iron and Steel', 'MUGHAL', 'Mughal Steel'),
    'KAPCO': ('Kot Addu Power', 'KAPCO', 'Kot Addu'),
    'NCPL': ('Nishat Chunian Power', 'NCPL', 'Nishat Power'),
    'MEBL': ('Meezan Bank', 'MEBL', 'Meezan'),
    'SEARL': ('Searle Pakistan', 'SEARL', 'Searle'),
    'PIOC': ('Pioneer Cement', 'PIOC', 'Pioneer'),
    'DGKC': ('DG Khan Cement', 'DGKC', 'DG Cement'),
    'MLCF': ('Maple Leaf Cement', 'MLCF', 'Maple Leaf'),
    'KOHC': ('Kohat Cement', 'KOHC', 'Kohat'),
    'KEL': ('K-Electric', 'KEL', 'Karachi Electric'),
    'NBP': ('National Bank', 'NBP', 'National Bank Pakistan'),
    'ABL': ('Allied Bank', 'ABL', 'Allied Bank Limited'),
    'BAFL': ('Bank Alfalah', 'BAFL', 'Alfalah'),
    'BAHL': ('Bank Al Habib', 'BAHL', 'Al Habib'),
    'POL': ('Pakistan Oilfields', 'POL', 'Pakistan Oilfields Limited'),
    'ATRL': ('Attock Refinery', 'ATRL', 'Attock'),
    'EFERT': ('Engro Fertilizers', 'EFERT', 'Engro Fert'),
    'CHCC': ('Cherat Cement', 'CHCC', 'Cherat'),
    'FCCL': ('Fauji Cement', 'FCCL', 'Fauji Cement Company'),
    'PTC': ('Pakistan Tobacco', 'PTC', 'Pak Tobacco'),
    'GLAXO': ('GlaxoSmithKline', 'GSK Pakistan', 'Glaxo'),
    'SAZEW': ('Sazgar Engineering', 'SAZEW', 'Sazgar Engineering Works'),
    'INDU': ('Indus Motor', 'INDU', 'Toyota Indus'),
    'HCAR': ('Honda Atlas Cars', 'HCAR', 'Honda Cars'),
    'PSMC': ('Pak Suzuki', 'PSMC', 'Pak Suzuki Motor'),
    'MTL': ('Millat Tractors', 'MTL', 'Millat'),
}

# News source configurations
NEWS_SOURCES = {
    'business_recorder': {
        'search_url': 'https://www.brecorder.com/?s={}',
        'selectors': {
            'articles': '.story-title a, .entry-title a, article h2 a, .td-module-title a',
            'dates': '.entry-date, .td-post-date, time, .post-date',
        }
    },
    'dawn': {
        'search_url': 'https://www.dawn.com/search?q={}',
        'selectors': {
            'articles': 'h2 a, .story__title a, article h3 a',
            'dates': '.timestamp, time, .story__time',
        }
    },
    'tribune': {
        'search_url': 'https://tribune.com.pk/?s={}',
        'selectors': {
            'articles': '.entry-title a, h2.title a, article h2 a',
            'dates': '.entry-date, time, .post-date',
        }
    },
    'arynews': {
        'search_url': 'https://arynews.tv/?s={}',
        'selectors': {
            'articles': 'h3 a, .entry-title a, article h2 a',
            'dates': '.entry-date, time',
        }
    }
}


# ============================================================================
# SELENIUM DRIVER MANAGEMENT
# ============================================================================

_driver_lock = threading.Lock()
_driver = None

def get_selenium_driver():
    """Get or create a headless Chrome driver (singleton)"""
    global _driver
    
    if not SELENIUM_AVAILABLE:
        return None
    
    with _driver_lock:
        if _driver is None:
            try:
                chrome_options = Options()
                chrome_options.add_argument('--headless=new')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-infobars')
                chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                
                service = Service(ChromeDriverManager().install())
                _driver = webdriver.Chrome(service=service, options=chrome_options)
                _driver.set_page_load_timeout(20)
                print("✅ Chrome WebDriver initialized")
            except Exception as e:
                print(f"❌ Failed to initialize WebDriver: {e}")
                return None
        
        return _driver


def close_driver():
    """Close the Selenium driver"""
    global _driver
    with _driver_lock:
        if _driver:
            _driver.quit()
            _driver = None


# ============================================================================
# CACHING SYSTEM
# ============================================================================

def get_cache_path(symbol: str) -> Path:
    """Get cache file path for a symbol"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol.upper()}_news.json"


def load_cached_news(symbol: str) -> Optional[Dict]:
    """Load cached news if not expired"""
    cache_path = get_cache_path(symbol)
    
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            cached_time = datetime.fromisoformat(cached.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_time < timedelta(hours=CACHE_DURATION_HOURS):
                return cached
        except:
            pass
    
    return None


def save_news_to_cache(symbol: str, news_data: Dict):
    """Save news to cache"""
    cache_path = get_cache_path(symbol)
    news_data['cached_at'] = datetime.now().isoformat()
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(news_data, f, indent=2)
    except:
        pass


# ============================================================================
# NEWS SCRAPING WITH SELENIUM
# ============================================================================

def scrape_news_selenium(driver, source_name: str, search_url: str, selectors: Dict, search_term: str) -> List[Dict]:
    """Scrape news from a source using Selenium"""
    news_items = []
    
    try:
        url = search_url.format(search_term.replace(' ', '+'))
        driver.get(url)
        
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find articles
        articles = []
        for selector in selectors['articles'].split(', '):
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                articles.extend(elements)
            except:
                continue
        
        # Extract article info
        seen_titles = set()
        for article in articles[:15]:
            try:
                title = article.text.strip()
                href = article.get_attribute('href') or ''
                
                if title and len(title) > 15 and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    
                    # Try to find associated date
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    news_items.append({
                        'title': title[:300],
                        'url': href,
                        'date': date_str,
                        'source': source_name.replace('_', ' ').title()
                    })
            except:
                continue
        
    except Exception as e:
        print(f"  ⚠️ Error scraping {source_name}: {str(e)[:50]}")
    
    return news_items


def scrape_psx_announcements(symbol: str) -> List[Dict]:
    """Scrape PSX company announcements page"""
    news_items = []
    
    try:
        import subprocess
        url = f"https://dps.psx.com.pk/company/{symbol.upper()}"
        
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '15', url],
            capture_output=True, text=True, timeout=20
        )
        
        if result.returncode == 0 and result.stdout:
            html = result.stdout
            
            # Find announcement table rows
            rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL | re.IGNORECASE)
            
            for row in rows:
                # Extract date and title
                cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
                if len(cells) >= 2:
                    date_text = cells[0].strip()
                    title = ' '.join(cells[1:3]).strip()
                    
                    if len(title) > 10:
                        news_items.append({
                            'title': title[:300],
                            'date': date_text,
                            'source': 'PSX',
                            'url': url
                        })
    except:
        pass
    
    return news_items[:10]


def fetch_all_news_with_meta(symbol: str, company_names: Tuple[str, ...]) -> Dict[str, Any]:
    """
    Fetch news from all sources with diagnostics and retrieval metadata.
    Priority: Enhanced fetcher -> Selenium fallback -> curl fallback.
    """
    all_news: List[Dict] = []
    symbol = symbol.upper()
    retrieval_mode = 'symbol_mode'
    enable_index_news_recall = _env_flag('ENABLE_INDEX_NEWS_RECALL', True)
    if determine_retrieval_mode:
        retrieval_mode = determine_retrieval_mode(symbol, retrieval_mode='auto')
    if not enable_index_news_recall and symbol in INDEX_SYMBOLS:
        retrieval_mode = 'symbol_mode'

    diagnostics: Dict[str, Any] = {
        'retrieval_mode': retrieval_mode,
        'news_fetch_diagnostics': {},
        'sources_attempted': 0,
        'sources_successful': 0,
        'filtered_count': 0,
        'fallback_path': [],
    }

    # First check PSX announcements (no Selenium needed)
    print(f"  📰 Checking PSX announcements...")
    psx_news = scrape_psx_announcements(symbol)
    all_news.extend(psx_news)
    print(f"     Found {len(psx_news)} PSX items")

    # Enhanced multi-source fetcher (with mode-aware retrieval)
    if ENHANCED_FETCHER_AVAILABLE:
        print("  🔍 Using enhanced multi-source fetcher with company aliases...")
        try:
            enhanced_result = get_enhanced_news_for_symbol(symbol, retrieval_mode=retrieval_mode)
            enhanced_news = enhanced_result.get('news_items', [])

            diagnostics.update({
                'retrieval_mode': enhanced_result.get('retrieval_mode', retrieval_mode),
                'news_fetch_diagnostics': enhanced_result.get('news_fetch_diagnostics', {}),
                'sources_attempted': enhanced_result.get('sources_attempted', 0),
                'sources_successful': enhanced_result.get('sources_successful', 0),
                'filtered_count': enhanced_result.get('filtered_count', 0),
            })

            if enhanced_news:
                all_news.extend(enhanced_news)
                print(f"     ✅ Found {len(enhanced_news)} articles via enhanced fetcher")
                queries = enhanced_result.get('queries_used', [])
                parent = enhanced_result.get('parent_company')
                if parent:
                    print(f"     📊 Parent company: {parent}")
                if len(queries) > 1:
                    print(f"     🔎 Also searched: {', '.join(queries[1:3])}")
            else:
                diagnostics['fallback_path'].append('enhanced_fetcher_empty')
        except Exception as e:
            diagnostics['fallback_path'].append('enhanced_fetcher_error')
            print(f"     ⚠️ Enhanced fetcher error: {str(e)[:50]}")

    # Selenium fallback (if still sparse)
    if len(all_news) < 5:
        driver = get_selenium_driver()

        if driver:
            search_terms = [symbol] + list(company_names[:2])

            for source_name, config in NEWS_SOURCES.items():
                print(f"  🌐 Searching {source_name.replace('_', ' ').title()}...")

                for search_term in search_terms:
                    try:
                        items = scrape_news_selenium(
                            driver,
                            source_name,
                            config['search_url'],
                            config['selectors'],
                            search_term
                        )

                        relevant_items: List[Dict] = []
                        for item in items:
                            title_lower = item['title'].lower()
                            if retrieval_mode == 'index_mode':
                                # For index mode, accept market-level terms instead of exact symbol mentions
                                if any(
                                    key in title_lower for key in
                                    ['psx', 'kse', 'stock exchange', 'market', 'pakistan stocks', 'kse-100']
                                ):
                                    relevant_items.append(item)
                            elif any(term.lower() in title_lower for term in [symbol] + list(company_names)):
                                relevant_items.append(item)

                        all_news.extend(relevant_items)

                        if relevant_items:
                            print(f"     Found {len(relevant_items)} relevant items for '{search_term}'")
                            diagnostics['fallback_path'].append(f"selenium:{source_name}")
                            break
                    except Exception as e:
                        print(f"     ⚠️ Error: {str(e)[:40]}")
                        continue
        else:
            print("  ⚠️ Selenium not available, using curl fallback...")
            for name in company_names[:2]:
                all_news.extend(fetch_news_curl(name))
            diagnostics['fallback_path'].append('curl_fallback')

    # Deduplicate
    seen = set()
    unique_news: List[Dict] = []
    for item in all_news:
        key = item.get('title', '').lower()[:120]
        if key and key not in seen:
            seen.add(key)
            unique_news.append(item)

    return {
        'news_items': unique_news[:25],
        **diagnostics,
    }


def fetch_all_news(symbol: str, company_names: Tuple[str, ...]) -> List[Dict]:
    """Backward-compatible wrapper returning only news list."""
    return fetch_all_news_with_meta(symbol, company_names).get('news_items', [])


def fetch_news_curl(search_term: str) -> List[Dict]:
    """Fallback: fetch news using curl"""
    news = []
    
    try:
        url = f"https://www.brecorder.com/?s={search_term.replace(' ', '+')}"
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '10', url],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0:
            # Extract links with titles
            matches = re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]{20,150})</a>', result.stdout)
            for href, title in matches[:5]:
                if search_term.lower() in title.lower():
                    news.append({
                        'title': title.strip(),
                        'url': href,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'Business Recorder'
                    })
    except:
        pass
    
    return news


# ============================================================================
# GROQ INTEGRATION
# ============================================================================

def get_groq_client():
    """Get Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


def analyze_with_ai(symbol: str, company_name: str, news_items: List[Dict], enriched_data: Dict = None) -> Dict:
    """Use Groq (Llama 3.3 70B) for intelligent sentiment analysis with anti-hallucination guardrails.
    
    Now enhanced with:
    - Full article content from BR Research
    - Live fundamental data (P/E, dividend yield)
    - Quality score for trend dampening guidance
    """
    
    if not GROQ_AVAILABLE:
        return fallback_analysis(news_items)
    
    # Get current date for context
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Format news with explicit dates
    if news_items:
        news_text = "\n".join([
            f"• [{item.get('date', 'unknown date')}] [{item.get('source_name', item.get('source', 'Unknown'))}] {item['title']}"
            for item in news_items[:15]
        ])
    else:
        news_text = "No recent news found."
    
    # Build enriched context from article scraper
    enriched_context = ""
    fundamental_summary = ""
    quality_guidance = ""
    
    if enriched_data:
        # Add fundamental data
        fundamentals = enriched_data.get('fundamentals', {})
        if fundamentals:
            fundamental_parts = []
            if fundamentals.get('pe_ratio'):
                pe = fundamentals['pe_ratio']
                pe_assessment = "undervalued" if pe < 10 else "fairly valued" if pe < 18 else "premium valued"
                fundamental_parts.append(f"P/E Ratio: {pe:.1f} ({pe_assessment})")
            if fundamentals.get('dividend_yield'):
                dy = fundamentals['dividend_yield']
                dy_assessment = "high yield" if dy > 6 else "good yield" if dy > 3 else "low yield"
                fundamental_parts.append(f"Dividend Yield: {dy:.1f}% ({dy_assessment})")
            if fundamentals.get('price'):
                fundamental_parts.append(f"Current Price: PKR {fundamentals['price']:.2f}")
            
            if fundamental_parts:
                fundamental_summary = "LIVE FUNDAMENTAL DATA: " + " | ".join(fundamental_parts)
        
        # Add quality guidance for trend dampening
        quality_score = enriched_data.get('quality_score', 0.5)
        if quality_score > 0.65:
            quality_guidance = "\n⚠️ QUALITY STOCK ALERT: This stock has strong fundamentals. Be cautious about overly bearish predictions based on short-term price movements. Consider mean reversion toward fair value."
        elif quality_score > 0.55:
            quality_guidance = "\n📊 This stock has above-average fundamentals. Balance short-term trends with fundamental value."
        
        # Add full article content (this is the key enhancement!)
        articles = enriched_data.get('articles', [])
        if articles:
            enriched_context = "\n\n📰 FULL ARTICLE CONTENT (from Business Recorder Research):\n"
            for i, article in enumerate(articles[:2], 1):
                content = article.get('content', '')[:2000]  # Limit content length
                metrics = article.get('financial_metrics', {})
                enriched_context += f"\n--- Article {i}: {article.get('title', 'Unknown')} ---\n"
                enriched_context += f"Date: {article.get('date', 'Unknown')}\n"
                enriched_context += f"Content: {content}\n"
                
                if metrics:
                    enriched_context += f"\nExtracted Metrics:\n"
                    if metrics.get('revenue_mentioned'):
                        enriched_context += f"  • {metrics['revenue_mentioned']}\n"
                    if metrics.get('profit_mentioned'):
                        enriched_context += f"  • {metrics['profit_mentioned']}\n"
                    if metrics.get('margin_mentioned'):
                        enriched_context += f"  • {metrics['margin_mentioned']}\n"
                    if metrics.get('sentiment_bias'):
                        enriched_context += f"  • Content Sentiment: {metrics['sentiment_bias']}\n"
    
    prompt = f"""You are a BALANCED Pakistani stock market analyst. Today's date is {current_date}.

🔮 FORTUNE TELLER ANALYSIS MODE - Enhanced with Fundamentals

CRITICAL RULES:
1. WEIGH FUNDAMENTALS HEAVILY - P/E ratios, dividend yields, and growth metrics are key indicators
2. If fundamentals are STRONG (low P/E, good dividend, revenue/profit growth), DO NOT be overly bearish
3. Short-term price dips in quality stocks often present buying opportunities
4. Only cite facts from the provided news AND article content
5. When fundamentals conflict with short-term price trend, FAVOR FUNDAMENTALS
6. A stock with P/E < 12 and dividend yield > 4% is typically undervalued
{quality_guidance}

{fundamental_summary}

Analyze the following news about {symbol} ({company_name}):

{news_text}
{enriched_context}

Based on the news AND fundamental data above, provide a BALANCED analysis. 

RESPOND IN JSON FORMAT ONLY:
{{
    "sentiment_score": <float from -1.0 to +1.0, use 0 if unclear>,
    "signal": "<BUY|HOLD|SELL>",
    "confidence": <float 0.0-1.0, lower if news is sparse or old>,
    "verified_events": ["ONLY list events that appear verbatim in headlines above"],
    "price_impact": {{
        "estimate": "<use 'unclear' unless there is very specific financial data>",
        "timeframe": "<unclear if not specified in news>",
        "reasoning": "<brief reasoning based ONLY on provided headlines>"
    }},
    "risks": ["only risks mentioned or implied in the headlines"],
    "catalysts": ["ONLY catalysts explicitly mentioned in headlines - do NOT invent any"],
    "data_quality": "<good|limited|poor> - based on how much actionable news we have",
    "summary": "<2-3 factual sentences ONLY referencing the actual headlines, acknowledge uncertainty>"
}}

ANTI-HALLUCINATION CHECKLIST before responding:
- Did I only cite facts from the headlines above? 
- Did I avoid making up specific % predictions?
- Did I avoid inventing acquisitions/deals not in the headlines?
- Am I being appropriately uncertain given sparse data?

Return ONLY valid JSON."""

    try:
        client = get_groq_client()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.05  # Lower temperature = more conservative
        )
        
        response_text = completion.choices[0].message.content.strip()
        result = json.loads(response_text)
        
        result['model'] = 'llama-3.3-70b-versatile'
        result['analyzed_at'] = datetime.now().isoformat()
        
        # Rename verified_events to key_events for compatibility
        if 'verified_events' in result:
            result['key_events'] = result.pop('verified_events')
        
        # Map to simpler signal for UI
        signal_map = {
            'STRONG_BUY': 'BULLISH',
            'BUY': 'BULLISH',
            'HOLD': 'NEUTRAL',
            'SELL': 'BEARISH',
            'STRONG_SELL': 'BEARISH'
        }
        result['signal_simple'] = signal_map.get(result.get('signal', 'HOLD'), 'NEUTRAL')
        
        return result
        
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return fallback_analysis(news_items)


def fallback_analysis(news_items: List[Dict]) -> Dict:
    """Fallback when Claude is unavailable"""
    text = ' '.join([item.get('title', '') for item in news_items]).lower()
    
    bullish = sum(1 for w in ['profit', 'growth', 'dividend', 'acquire', 'expansion', 'record', 'surge'] if w in text)
    bearish = sum(1 for w in ['loss', 'decline', 'drop', 'fraud', 'investigation', 'shutdown'] if w in text)
    
    if bullish > bearish:
        score, signal = min(0.5, bullish * 0.15), 'BULLISH'
    elif bearish > bullish:
        score, signal = max(-0.5, -bearish * 0.15), 'BEARISH'
    else:
        score, signal = 0, 'NEUTRAL'
    
    return {
        'sentiment_score': score,
        'signal': signal,
        'signal_simple': signal,
        'confidence': 0.3,
        'key_events': [],
        'price_impact': {'estimate': 'unclear', 'timeframe': 'unclear', 'reasoning': 'Keyword analysis only'},
        'summary': 'Analysis based on keyword matching (AI unavailable)',
        'model': 'fallback',
        'analyzed_at': datetime.now().isoformat()
    }


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def get_stock_sentiment(symbol: str, use_cache: bool = True) -> Dict:
    """
    🔮 Main function: Get comprehensive AI-powered sentiment for a stock.
    This is the "fortune teller" function.
    
    ENHANCED with:
    - Full article content from BR Research
    - Live fundamental data (P/E, dividend yield)
    - Quality score for balanced analysis
    """
    symbol = symbol.upper()
    
    # Get company names
    company_names = STOCK_COMPANIES.get(symbol, (symbol,))
    company_name = company_names[0] if company_names else symbol
    
    print(f"\n🔮 SENTIMENT ANALYSIS: {symbol} ({company_name})")
    print("=" * 50)
    
    # Check cache
    if use_cache:
        cached = load_cached_news(symbol)
        if cached:
            print("📦 Using cached analysis (less than 4 hours old)")
            return cached
    
    # 🆕 FETCH ENRICHED DATA (articles + fundamentals)
    # For index symbols (e.g., KSE100), company-level BR research scraping is often
    # sparse/blocked and not representative. Use multi-source market news instead.
    enriched_data = None
    if ARTICLE_SCRAPER_AVAILABLE and symbol not in INDEX_SYMBOLS:
        print("📚 Fetching enriched data (articles + fundamentals)...")
        try:
            enriched_data = get_enriched_stock_data(symbol)
            if enriched_data.get('has_rich_data'):
                print(f"   ✅ Found {enriched_data.get('article_count', 0)} articles, quality score: {enriched_data.get('quality_score', 0):.2f}")
            else:
                print("   ⚠️ Limited enriched data available")
        except Exception as e:
            print(f"   ⚠️ Error fetching enriched data: {e}")
    elif symbol in INDEX_SYMBOLS:
        print("📚 Skipping company research scraper for index symbol; using market-wide news sources.")
    
    news_meta: Dict[str, Any] = {
        'retrieval_mode': 'symbol_mode',
        'news_fetch_diagnostics': {},
        'sources_attempted': 0,
        'sources_successful': 0,
        'filtered_count': 0,
        'fallback_path': [],
    }

    # Fetch news using PREMIUM fetcher (10+ sources!)
    if PREMIUM_FETCHER_AVAILABLE:
        print("🚀 Using PREMIUM NEWS FETCHER (10+ sources)...")
        premium_result = fetch_premium_news(symbol, use_cache=False, verbose=True)
        news_items = premium_result.get('news_items', [])
        sources_searched = premium_result.get('sources_searched', [])
        news_meta.update({
            'retrieval_mode': 'symbol_mode',
            'sources_attempted': len(sources_searched),
            'sources_successful': len([s for s in sources_searched if s]),
            'filtered_count': 0,
        })
        print(f"📊 Total news items found: {len(news_items)}")
    else:
        # Fallback to basic fetcher
        print("📰 Fetching news from multiple sources...")
        fetched = fetch_all_news_with_meta(symbol, company_names)
        news_items = fetched.get('news_items', [])
        sources_searched = list(NEWS_SOURCES.keys()) + ['PSX']
        news_meta.update({
            'retrieval_mode': fetched.get('retrieval_mode', 'symbol_mode'),
            'news_fetch_diagnostics': fetched.get('news_fetch_diagnostics', {}),
            'sources_attempted': fetched.get('sources_attempted', 0),
            'sources_successful': fetched.get('sources_successful', 0),
            'filtered_count': fetched.get('filtered_count', 0),
            'fallback_path': fetched.get('fallback_path', []),
        })
        print(f"📊 Total news items found: {len(news_items)}")
    
    # Display found news
    if news_items:
        print("\n📋 Headlines found:")
        for item in news_items[:5]:
            source = item.get('source_name', item.get('source', 'Unknown'))
            print(f"   • [{source}] {item['title'][:70]}...")
    
    # Analyze with AI (Groq) - NOW WITH ENRICHED DATA!
    print("\n🤖 Analyzing with Groq (Llama 3.3) + Enriched Context...")
    analysis = analyze_with_ai(symbol, company_name, news_items, enriched_data=enriched_data)
    
    # Build complete result
    result = {
        'symbol': symbol,
        'company': company_name,
        'news_count': len(news_items),
        'news_items': news_items[:15],  # More news items now!
        'sources_searched': sources_searched,
        'retrieval_mode': news_meta.get('retrieval_mode', 'symbol_mode'),
        'news_fetch_diagnostics': news_meta.get('news_fetch_diagnostics', {}),
        'sources_attempted': news_meta.get('sources_attempted', 0),
        'sources_successful': news_meta.get('sources_successful', 0),
        'filtered_count': news_meta.get('filtered_count', 0),
        'fallback_path': news_meta.get('fallback_path', []),
        'enriched_data_available': enriched_data is not None and enriched_data.get('has_rich_data', False),
        'quality_score': enriched_data.get('quality_score', 0.5) if enriched_data else 0.5,
        'fundamentals': enriched_data.get('fundamentals', {}) if enriched_data else {},
        **analysis
    }
    
    # Generate emoji
    signal = result.get('signal_simple', result.get('signal', 'NEUTRAL'))
    if signal in ['BULLISH', 'STRONG_BUY', 'BUY']:
        result['signal_emoji'] = '🟢'
    elif signal in ['BEARISH', 'STRONG_SELL', 'SELL']:
        result['signal_emoji'] = '🔴'
    else:
        result['signal_emoji'] = '🟡'
    
    # Cache result
    save_news_to_cache(symbol, result)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"🎯 RESULT: {result.get('signal_emoji', '🟡')} {result.get('signal', 'NEUTRAL')}")
    print(f"📈 Sentiment Score: {result.get('sentiment_score', 0):.2f}")
    print(f"🎯 Confidence: {result.get('confidence', 0):.0%}")
    if result.get('price_impact'):
        pi = result['price_impact']
        print(f"💰 Expected Impact: {pi.get('estimate', 'unclear')} over {pi.get('timeframe', 'unclear')}")
    print(f"📝 Summary: {result.get('summary', 'N/A')}")
    
    return result


# Backward compatibility aliases
def analyze_news_for_stock(symbol: str, news_items: List[Dict] = None) -> Dict:
    return get_stock_sentiment(symbol)


async def get_market_sentiment() -> Dict:
    """Get overall market sentiment"""
    stocks = ['OGDC', 'HBL', 'LUCK']
    total, count = 0, 0
    
    for symbol in stocks:
        try:
            result = get_stock_sentiment(symbol, use_cache=True)
            total += result.get('sentiment_score', 0)
            count += 1
        except:
            continue
    
    avg = total / count if count > 0 else 0
    
    return {
        'market_sentiment': round(avg, 3),
        'signal': 'BULLISH' if avg > 0.15 else 'BEARISH' if avg < -0.15 else 'NEUTRAL',
        'emoji': '🟢' if avg > 0.15 else '🔴' if avg < -0.15 else '🟡',
        'stocks_analyzed': count,
        'analyzed_at': datetime.now().isoformat()
    }


# ============================================================================
# ML MODEL INTEGRATION - Numeric sentiment features for predictions
# ============================================================================

def get_sentiment_score_for_model(symbol: str, use_cache: bool = True) -> Dict:
    """
    Get sentiment features suitable for ML model integration.
    
    Returns dict with numeric features that can be added to model DataFrame:
    - news_bias: -1 to +1 (bearish to bullish)
    - news_volume: 0-1 normalized (0=no news, 1=many articles)
    - news_recency: 0-1 (0=old/no news, 1=very recent)
    - available: bool (True if we have actual data)
    
    Falls back to neutral values if anything fails.
    """
    NEUTRAL_RESULT = {
        'news_bias': 0.0,
        'news_volume': 0.5,
        'news_recency': 0.5,
        'available': False
    }

    symbol_upper = symbol.upper()
    allow_index_recall_in_model = _env_flag('ENABLE_INDEX_RECALL_IN_MODEL', False)
    if symbol_upper in INDEX_SYMBOLS and not allow_index_recall_in_model:
        # Keep model input behavior stable until explicit rollout.
        return NEUTRAL_RESULT
    
    try:
        # Get sentiment (uses cache if available)
        result = get_stock_sentiment(symbol_upper, use_cache=use_cache)
        
        if result.get('error'):
            return NEUTRAL_RESULT
        
        # Extract sentiment score (-1 to 1)
        sentiment = result.get('sentiment_score', 0)
        
        # Map signal to bias if sentiment score is exactly 0
        signal = result.get('signal', 'NEUTRAL')
        if sentiment == 0:
            if signal in ['BUY', 'BULLISH', 'STRONG_BUY']:
                sentiment = 0.3
            elif signal in ['SELL', 'BEARISH', 'STRONG_SELL']:
                sentiment = -0.3
        
        # Calculate news volume (normalize 0-10 articles to 0-1)
        news_items = result.get('news_items', [])
        news_volume = min(len(news_items) / 10.0, 1.0)
        
        # Calculate recency (days since most recent news)
        news_recency = 0.5  # Default to medium
        if news_items:
            try:
                dates = []
                for item in news_items:
                    date_str = item.get('date', '')
                    if date_str:
                        try:
                            d = datetime.strptime(date_str[:10], '%Y-%m-%d')
                            dates.append(d)
                        except:
                            pass
                
                if dates:
                    most_recent = max(dates)
                    days_old = (datetime.now() - most_recent).days
                    # Map: 0 days = 1.0, 7+ days = 0.0
                    news_recency = max(0, 1.0 - (days_old / 7.0))
            except:
                pass
        
        return {
            'news_bias': round(sentiment, 4),
            'news_volume': round(news_volume, 4),
            'news_recency': round(news_recency, 4),
            'available': True
        }
        
    except Exception as e:
        print(f"⚠️ Sentiment score error for {symbol}: {str(e)[:50]}")
        return NEUTRAL_RESULT


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🔮 PSX FORTUNE TELLER - AI-Powered Sentiment Analysis")
    print("="*70)
    print()
    print(f"✅ Groq: {'Available' if GROQ_AVAILABLE else 'Not Available'}")
    print(f"✅ API Key: {'Loaded' if os.getenv('GROQ_API_KEY') else 'Missing'}")
    print()
    
    # Test with a stock
    test_symbols = ['LUCK', 'SYS']
    
    for symbol in test_symbols:
        result = get_stock_sentiment(symbol, use_cache=False)
        print()
    
    # Cleanup
    close_driver()
    print("\n✅ Done!")
