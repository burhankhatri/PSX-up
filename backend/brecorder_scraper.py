"""
🔍 BUSINESS RECORDER SCRAPER
Targeted scraper for Business Recorder company pages and PSX notices.
Uses Selenium for Cloudflare bypass, falls back to cached data.

Key URLs:
- Company-specific: https://www.brecorder.com/company/{symbol}
- PSX Notices: https://www.brecorder.com/trends/psx-notice
"""

import re
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import threading

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "news_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Selenium imports (optional)
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

# Symbol to URL mapping (for companies with different URL slugs)
SYMBOL_TO_SLUG = {
    'LUCK': 'luck',
    'FFC': 'ffc',
    'FFBL': 'ffbl',
    'FCCL': 'fccl',
    'PSO': 'pso',
    'PPL': 'ppl',
    'OGDC': 'ogdc',
    'ENGRO': 'engro',
    'EFERT': 'efert',
    'HBL': 'hbl',
    'UBL': 'ubl',
    'MCB': 'mcb',
    'NBP': 'nbp',
    'MEBL': 'mebl',
    'SYS': 'sys',
    'TRG': 'trg',
    'HUBC': 'hubpower',
    'KEL': 'ke',
    'DGKC': 'dgkhan',
    'MLCF': 'mapleleaf',
    'PIOC': 'pioneer',
    'KOHC': 'kohat',
    'CHCC': 'cherat',
    'POL': 'pol',
    'ATRL': 'attock-refinery',
    'MARI': 'mari',
    'KAPCO': 'kapco',
    'FATIMA': 'fatima',
    'NESTLE': 'nestle',
    'INDU': 'indus-motor',
    'HCAR': 'honda-atlas',
}

# News impact categories
# IMPORTANT: Ordered longest-first because the matching loop breaks on first hit.
# Context-aware keywords prevent misclassification (e.g. "profit down" != positive).
NEWS_IMPACT = [
    # Negative profit/revenue (must come before bare 'profit')
    ('profit decline', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('profit drop', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('profit down', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('profit falls', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('profit fell', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('profit slump', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('revenue decline', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('revenue drop', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    # Positive profit/revenue
    ('profit growth', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('profit increase', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('profit surge', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('profit up', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('record profit', {'impact': 0.4, 'duration_days': 30, 'type': 'positive'}),
    ('profit rise', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('revenue increase', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('revenue growth', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    # Debt context-aware (must come before bare 'debt')
    ('circular debt plan', {'impact': 0.2, 'duration_days': 30, 'type': 'positive'}),
    ('debt relief', {'impact': 0.2, 'duration_days': 30, 'type': 'positive'}),
    ('debt settlement', {'impact': 0.2, 'duration_days': 30, 'type': 'positive'}),
    ('debt payment', {'impact': 0.2, 'duration_days': 30, 'type': 'positive'}),
    ('receives', {'impact': 0.2, 'duration_days': 14, 'type': 'positive'}),
    ('debt crisis', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('debt burden', {'impact': -0.2, 'duration_days': 30, 'type': 'negative'}),
    ('debt pile', {'impact': -0.2, 'duration_days': 30, 'type': 'negative'}),
    # Standard keywords
    ('dividend', {'impact': 0.3, 'duration_days': 30, 'type': 'positive'}),
    ('merger', {'impact': 0.5, 'duration_days': 90, 'type': 'positive'}),
    ('acquisition', {'impact': 0.4, 'duration_days': 60, 'type': 'positive'}),
    ('stock split', {'impact': 0.2, 'duration_days': 14, 'type': 'positive'}),
    ('investment', {'impact': 0.4, 'duration_days': 60, 'type': 'positive'}),
    ('expansion', {'impact': 0.3, 'duration_days': 45, 'type': 'positive'}),
    ('privatization', {'impact': 0.4, 'duration_days': 90, 'type': 'neutral'}),
    ('loss', {'impact': -0.3, 'duration_days': 30, 'type': 'negative'}),
    ('debt', {'impact': -0.2, 'duration_days': 30, 'type': 'negative'}),
    ('lawsuit', {'impact': -0.3, 'duration_days': 60, 'type': 'negative'}),
    ('penalty', {'impact': -0.2, 'duration_days': 14, 'type': 'negative'}),
    ('regulatory', {'impact': -0.1, 'duration_days': 30, 'type': 'negative'}),
]

# Driver lock for thread safety
_driver_lock = threading.Lock()
_driver = None


def get_driver():
    """Get or create Selenium driver"""
    global _driver
    if not SELENIUM_AVAILABLE:
        return None
    
    with _driver_lock:
        if _driver is None:
            try:
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option('excludeSwitches', ['enable-automation'])
                options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
                
                # Use simple Chrome() - works better than ChromeDriverManager
                _driver = webdriver.Chrome(options=options)
                _driver.set_page_load_timeout(30)
            except Exception as e:
                print(f"❌ Failed to init driver: {e}")
                return None
        return _driver


def close_driver():
    """Close Selenium driver"""
    global _driver
    with _driver_lock:
        if _driver:
            _driver.quit()
            _driver = None


def scrape_brecorder_company_page(symbol: str) -> List[Dict]:
    """
    Scrape company-specific page from Business Recorder.
    URL: https://www.brecorder.com/company/{slug}
    """
    slug = SYMBOL_TO_SLUG.get(symbol.upper(), symbol.lower())
    url = f"https://www.brecorder.com/company/{slug}"
    articles = []
    
    driver = get_driver()
    if not driver:
        print(f"   ⚠️ No Selenium driver, checking cache for {symbol}")
        return load_from_cache(symbol, 'brecorder_company')
    
    try:
        print(f"   🌐 Fetching: {url}")
        driver.get(url)
        
        # Wait for page load (Cloudflare challenge)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )
        
        # Extract headlines using selector: a.story__link or article headlines
        headlines = driver.find_elements(By.CSS_SELECTOR, "a.story__link, article h2 a, .story-title a")
        
        seen = set()
        for elem in headlines[:15]:
            try:
                title = elem.text.strip()
                href = elem.get_attribute('href') or ''
                
                if title and len(title) > 20 and title.lower() not in seen:
                    seen.add(title.lower())
                    
                    # Extract date if visible
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    articles.append({
                        'title': title,
                        'url': href,
                        'source': 'Business Recorder',
                        'date': date_str,
                        'symbol': symbol.upper(),
                        'is_direct': True
                    })
            except:
                continue
        
        # Cache results
        save_to_cache(symbol, 'brecorder_company', articles)
        print(f"   ✅ Found {len(articles)} articles from BR")
        
    except Exception as e:
        print(f"   ⚠️ BR scrape failed: {str(e)[:50]}")
        articles = load_from_cache(symbol, 'brecorder_company')
    
    return articles


def scrape_psx_notices() -> List[Dict]:
    """
    Scrape PSX notices from Business Recorder.
    URL: https://www.brecorder.com/trends/psx-notice
    """
    url = "https://www.brecorder.com/trends/psx-notice"
    notices = []
    
    driver = get_driver()
    if not driver:
        print("   ⚠️ No Selenium driver for PSX notices")
        return load_from_cache('_psx_notices', 'psx_notices')
    
    try:
        print(f"   🌐 Fetching PSX notices...")
        driver.get(url)
        
        # Wait for load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )
        
        # Extract notices
        headlines = driver.find_elements(By.CSS_SELECTOR, "a.story__link, article h2 a")
        
        seen = set()
        for elem in headlines[:20]:
            try:
                title = elem.text.strip()
                href = elem.get_attribute('href') or ''
                
                if title and len(title) > 20 and title.lower() not in seen:
                    seen.add(title.lower())
                    notices.append({
                        'title': title,
                        'url': href,
                        'source': 'PSX Notice',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'is_notice': True
                    })
            except:
                continue
        
        save_to_cache('_psx_notices', 'psx_notices', notices)
        print(f"   ✅ Found {len(notices)} PSX notices")
        
    except Exception as e:
        print(f"   ⚠️ PSX notices scrape failed: {str(e)[:50]}")
        notices = load_from_cache('_psx_notices', 'psx_notices')
    
    return notices


def calculate_news_impact_score(articles: List[Dict]) -> Dict:
    """
    Calculate mathematical news impact for model integration.
    
    Returns:
        {
            'news_bias': float (-1 to +1),
            'impact_magnitude': float (0 to 1),
            'avg_impact_duration': int (days),
            'key_events': list of categorized events,
            'recommendation': str
        }
    """
    if not articles:
        return {
            'news_bias': 0.0,
            'impact_magnitude': 0.0,
            'avg_impact_duration': 0,
            'key_events': [],
            'recommendation': 'NEUTRAL - No recent news'
        }
    
    total_impact = 0.0
    total_duration = 0
    key_events = []
    
    for article in articles:
        title_lower = article['title'].lower()
        
        # Check for impact keywords
        for keyword, config in NEWS_IMPACT:
            if keyword in title_lower:
                impact = config['impact']
                duration = config['duration_days']
                event_type = config['type']
                
                total_impact += impact
                total_duration += duration
                
                key_events.append({
                    'headline': article['title'][:80],
                    'category': keyword,
                    'impact': impact,
                    'duration_days': duration,
                    'type': event_type
                })
                break
    
    # Normalize
    num_events = len(key_events) if key_events else 1
    news_bias = max(-1, min(1, total_impact / num_events)) if key_events else 0
    avg_duration = total_duration // num_events if key_events else 0
    
    # Generate recommendation
    if news_bias > 0.3:
        recommendation = f"BULLISH - {num_events} positive event(s) detected"
    elif news_bias < -0.3:
        recommendation = f"BEARISH - {num_events} negative event(s) detected"
    else:
        recommendation = f"NEUTRAL - Mixed or no significant events"
    
    return {
        'news_bias': round(news_bias, 3),
        'impact_magnitude': round(abs(news_bias), 3),
        'avg_impact_duration': avg_duration,
        'key_events': key_events[:5],  # Top 5
        'recommendation': recommendation
    }


def save_to_cache(symbol: str, cache_type: str, data: List[Dict]):
    """Save data to cache"""
    cache_file = CACHE_DIR / f"{symbol.upper()}_{cache_type}.json"
    try:
        cache_data = {
            'symbol': symbol,
            'type': cache_type,
            'cached_at': datetime.now().isoformat(),
            'articles': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except:
        pass


def load_from_cache(symbol: str, cache_type: str, max_age_hours: int = 6) -> List[Dict]:
    """Load data from cache if not too old"""
    cache_file = CACHE_DIR / f"{symbol.upper()}_{cache_type}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get('articles', [])
        except:
            pass
    
    return []


def get_brecorder_news_for_symbol(symbol: str) -> Dict:
    """
    Main function: Get comprehensive news from Business Recorder.
    
    Returns complete news analysis with impact scoring.
    """
    symbol = symbol.upper()
    
    print(f"\n📰 BUSINESS RECORDER SCRAPE: {symbol}")
    
    # 1. Get company-specific news
    company_news = scrape_brecorder_company_page(symbol)
    
    # 2. Get PSX notices (filter for this symbol)
    psx_notices = scrape_psx_notices()
    relevant_notices = [n for n in psx_notices if symbol in n['title'].upper()]
    
    # 3. Combine all articles
    all_articles = company_news + relevant_notices
    
    # 4. Calculate news impact
    impact = calculate_news_impact_score(all_articles)
    
    print(f"   📊 News Bias: {impact['news_bias']:.2f} | {impact['recommendation']}")
    
    return {
        'symbol': symbol,
        'articles': all_articles,
        'company_news_count': len(company_news),
        'psx_notices_count': len(relevant_notices),
        'impact': impact,
        'fetched_at': datetime.now().isoformat()
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Business Recorder Scraper")
    print("=" * 60)
    
    # Test LUCK
    result = get_brecorder_news_for_symbol('LUCK')
    
    print(f"\nArticles found: {len(result['articles'])}")
    for article in result['articles'][:5]:
        print(f"  • {article['title'][:60]}...")
    
    print(f"\nImpact Analysis:")
    print(f"  Bias: {result['impact']['news_bias']}")
    print(f"  Recommendation: {result['impact']['recommendation']}")
    
    if result['impact']['key_events']:
        print(f"  Key Events:")
        for event in result['impact']['key_events']:
            print(f"    - {event['category']}: {event['headline'][:50]}...")
    
    close_driver()
    print("\n✅ Done!")
