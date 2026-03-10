#!/usr/bin/env python3
"""
📊 EXTERNAL FEATURES MODULE
Research-backed external data for PSX prediction model.

Per peer-reviewed research (2020-2025):
- USD/PKR is the MOST CRITICAL external predictor for PSX
- KSE-100 beta explains most stock movement
- Oil prices affect energy sector (OGDC, PPL, PSO)
- Gold correlates with PKR weakness

Data Sources:
- USD/PKR: Yahoo Finance (PKR=X) - 5+ years available
- KSE-100: PSX DPS API (dps.psx.com.pk)
- Oil/Gold: Yahoo Finance (CL=F, GC=F)
- Nikkei 225 / KOSPI: Yahoo Finance (^N225, ^KS11) - Asian leading indicators
- KIBOR: Hardcoded (SBP updates manually)
"""

import numpy as np
import pandas as pd
import subprocess
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# TradingView scraper
try:
    from backend.tradingview_scraper import get_tradingview_indicators
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False

try:
    from backend.energy_shock_features import build_energy_event_feature_frame
    ENERGY_SHOCK_FEATURES_AVAILABLE = True
except ImportError:
    try:
        from energy_shock_features import build_energy_event_feature_frame
        ENERGY_SHOCK_FEATURES_AVAILABLE = True
    except ImportError:
        ENERGY_SHOCK_FEATURES_AVAILABLE = False

try:
    from backend.enhanced_news_fetcher import get_enhanced_news_for_symbol
    ENHANCED_NEWS_AVAILABLE = True
except ImportError:
    try:
        from enhanced_news_fetcher import get_enhanced_news_for_symbol
        ENHANCED_NEWS_AVAILABLE = True
    except ImportError:
        ENHANCED_NEWS_AVAILABLE = False

# Try yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Run: pip install yfinance")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / "data" / "external_cache"
NEWS_CACHE_DIR = Path(__file__).parent.parent / "data" / "news_cache"

# Current KIBOR/SBP Policy Rate (update periodically from SBP website)
# Source: https://www.sbp.org.pk/ecodata/kibor_index.asp
KIBOR_RATE = 0.13  # 13% as of Dec 2024 (policy rate cut from 15%)

ENERGY_SHOCK_FEATURE_COLUMNS = [
    'local_fuel_price_delta_rs',
    'local_fuel_price_shock',
    'circular_debt_signal',
    'geo_shock_signal',
    'energy_shock_regime',
    'kse_oil_interaction',
    'kse_energy_shock_interaction',
]


def _load_energy_news_items(symbol: Optional[str]) -> List[Dict]:
    symbol_upper = (symbol or '').upper()
    if not symbol_upper:
        return []

    cache_file = NEWS_CACHE_DIR / f"{symbol_upper}_news.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            items = cached.get('news_items', [])
            if items:
                return items
        except Exception:
            pass

    if ENHANCED_NEWS_AVAILABLE:
        try:
            fetched = get_enhanced_news_for_symbol(symbol_upper, retrieval_mode='symbol_mode')
            return fetched.get('news_items', [])
        except Exception:
            return []
    return []


def _ensure_energy_shock_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ENERGY_SHOCK_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in {'local_fuel_price_delta_rs', 'circular_debt_signal', 'kse_oil_interaction', 'kse_energy_shock_interaction'} else 0
    return df


# ============================================================================
# USD/PKR EXCHANGE RATE
# ============================================================================

def fetch_usd_pkr(start_date: str = None, end_date: str = None, 
                  period: str = "5y") -> pd.DataFrame:
    """
    Fetch USD/PKR exchange rate from Yahoo Finance.
    
    Research: USD/PKR is the #1 external predictor for PSX.
    Shows ~0.1 correlation with KSE-100, but high predictive power
    for emerging market stocks.
    
    Args:
        start_date: Start date (YYYY-MM-DD) or None for period-based
        end_date: End date (YYYY-MM-DD) or None for period-based  
        period: Period string if dates not specified (1mo, 3mo, 1y, 5y)
    
    Returns:
        DataFrame with columns: date, usdpkr_close, usdpkr_change, 
        usdpkr_volatility, usdpkr_trend
    """
    if not YFINANCE_AVAILABLE:
        print("⚠️ yfinance not available, returning empty DataFrame")
        return pd.DataFrame()
    
    try:
        if start_date and end_date:
            data = yf.download('PKR=X', start=start_date, end=end_date, progress=False)
        else:
            data = yf.download('PKR=X', period=period, progress=False)
        
        if data.empty:
            print("⚠️ No USD/PKR data returned")
            return pd.DataFrame()
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = pd.DataFrame({
            'date': data.index,
            'usdpkr_close': data['Close'].values,
            'usdpkr_change': data['Close'].pct_change().values,
            'usdpkr_volatility': data['Close'].pct_change().rolling(20).std().values,
            # PKR weakening trend (bad for stocks)
            'usdpkr_trend': (data['Close'] / data['Close'].shift(20) - 1).values,
            # Is PKR strengthening? (good signal)
            'usdpkr_strengthening': (data['Close'] < data['Close'].shift(5)).astype(int).values
        })
        
        df = df.reset_index(drop=True)
        print(f"✅ Fetched {len(df)} USD/PKR data points")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching USD/PKR: {e}")
        return pd.DataFrame()


# ============================================================================
# KSE-100 INDEX (From PSX DPS API)
# ============================================================================

def fetch_kse100_month(month: int, year: int) -> List[Dict]:
    """Fetch KSE-100 data for a specific month from PSX."""
    url = "https://dps.psx.com.pk/historical"
    post_data = f"month={month}&year={year}&symbol=KSE100"
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST', url, '-d', post_data],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode != 0:
            return []
        
        # Parse HTML table
        rows = re.findall(r'<tr>.*?</tr>', result.stdout, re.DOTALL)
        data = []
        
        for row in rows:
            cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
            if len(cells) >= 6:
                try:
                    date_str = cells[0].strip()
                    date_obj = datetime.strptime(date_str, "%b %d, %Y")
                    
                    data.append({
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'kse100_open': float(cells[1].replace(',', '')),
                        'kse100_high': float(cells[2].replace(',', '')),
                        'kse100_low': float(cells[3].replace(',', '')),
                        'kse100_close': float(cells[4].replace(',', '')),
                        'kse100_volume': float(cells[5].replace(',', ''))
                    })
                except:
                    continue
        
        return data
    except Exception as e:
        print(f"⚠️ Error fetching KSE100 {month}/{year}: {e}")
        return []


def fetch_kse100(start_year: int = 2020, end_date: str = None) -> pd.DataFrame:
    """
    Fetch KSE-100 index data from PSX DPS API.
    
    Research: Market beta (vs KSE-100) explains most stock movement.
    This is the #1 signal for individual stock prediction.
    
    Args:
        start_year: Year to start fetching from
        end_date: End date (defaults to today)
    
    Returns:
        DataFrame with KSE-100 OHLCV data
    """
    current_date = datetime.now()
    end_year = current_date.year
    end_month = current_date.month
    
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        end_year = end_dt.year
        end_month = end_dt.month
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        start_m = 1 if year > start_year else 1
        end_m = end_month if year == end_year else 12
        
        for month in range(start_m, end_m + 1):
            month_data = fetch_kse100_month(month, year)
            all_data.extend(month_data)
    
    if not all_data:
        print("⚠️ No KSE-100 data fetched")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add derived features
    df['kse100_return'] = df['kse100_close'].pct_change()
    df['kse100_volatility'] = df['kse100_return'].rolling(20).std()
    df['kse100_trend'] = df['kse100_close'] / df['kse100_close'].shift(20) - 1
    df['kse100_above_sma50'] = (df['kse100_close'] > df['kse100_close'].rolling(50).mean()).astype(int)
    df['kse100_above_sma200'] = (df['kse100_close'] > df['kse100_close'].rolling(200).mean()).astype(int)
    
    print(f"✅ Fetched {len(df)} KSE-100 data points ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


# ============================================================================
# OIL & COMMODITIES
# ============================================================================

def fetch_commodities(start_date: str = None, end_date: str = None,
                      period: str = "5y") -> pd.DataFrame:
    """
    Fetch oil and gold prices from Yahoo Finance.
    
    Research: Oil prices affect energy sector (OGDC, PPL, PSO).
    Gold often correlates with PKR weakness (flight to safety).
    
    Returns:
        DataFrame with oil and gold prices
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        # Fetch both commodities
        if start_date and end_date:
            oil = yf.download('CL=F', start=start_date, end=end_date, progress=False)
            gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
        else:
            oil = yf.download('CL=F', period=period, progress=False)
            gold = yf.download('GC=F', period=period, progress=False)
        
        # Handle multi-level columns
        for df in [oil, gold]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        # Merge on date
        result = pd.DataFrame({'date': oil.index})
        
        if not oil.empty:
            result['oil_close'] = oil['Close'].values
            result['oil_change'] = oil['Close'].pct_change().values
            result['oil_trend'] = (oil['Close'] / oil['Close'].shift(20) - 1).values
        
        if not gold.empty:
            # Align gold to oil dates
            gold_aligned = gold.reindex(oil.index, method='ffill')
            result['gold_close'] = gold_aligned['Close'].values
            result['gold_change'] = gold_aligned['Close'].pct_change().values
            result['gold_trend'] = (gold_aligned['Close'] / gold_aligned['Close'].shift(20) - 1).values
        
        result = result.reset_index(drop=True)
        print(f"✅ Fetched {len(result)} commodity data points")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching commodities: {e}")
        return pd.DataFrame()


# ============================================================================
# ASIAN MARKETS (Leading Indicators for PSX)
# ============================================================================
# TSE opens 09:00 JST (04:00 PKT), closes 15:00 JST (10:00 PKT)
# KRX opens 09:00 KST (05:00 PKT), closes 15:30 KST (11:30 PKT)
# PSX opens 09:30 PKT — Asian markets give 4-5 hours of lead time

ASIAN_MARKET_TICKERS = {
    'nikkei': '^N225',
    'kospi': '^KS11',
}

ASIAN_CACHE_FILE = CACHE_DIR / "asian_markets.json"


def fetch_asian_markets(start_date: str = None, end_date: str = None,
                        period: str = "5y") -> pd.DataFrame:
    """
    Fetch Nikkei 225 and KOSPI from Yahoo Finance.

    These are leading indicators for PSX — Asian markets open 4-5 hours
    before PSX. When both crash at open (e.g., March 9 2026), PSX follows.

    Returns:
        DataFrame with columns: date, nikkei_close, nikkei_change,
        nikkei_open_gap, kospi_close, kospi_change, kospi_open_gap,
        asian_avg_return, asian_risk_off_signal, etc.
    """
    if not YFINANCE_AVAILABLE:
        print("⚠️ yfinance not available, returning empty DataFrame")
        return pd.DataFrame()

    try:
        dl_kwargs = dict(progress=False)
        if start_date and end_date:
            dl_kwargs.update(start=start_date, end=end_date)
        else:
            dl_kwargs['period'] = period

        nikkei = yf.download('^N225', **dl_kwargs)
        kospi = yf.download('^KS11', **dl_kwargs)

        # Handle multi-level columns
        for df_raw in [nikkei, kospi]:
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)

        if nikkei.empty and kospi.empty:
            print("⚠️ No Asian market data returned")
            return pd.DataFrame()

        # Use nikkei as base timeline, align kospi
        base = nikkei if not nikkei.empty else kospi
        result = pd.DataFrame({'date': base.index})

        if not nikkei.empty:
            result['nikkei_close'] = nikkei['Close'].values
            result['nikkei_change'] = nikkei['Close'].pct_change().values
            result['nikkei_trend'] = (nikkei['Close'] / nikkei['Close'].shift(20) - 1).values
            # Open gap: how much market gapped at open vs previous close
            result['nikkei_open_gap'] = ((nikkei['Open'] - nikkei['Close'].shift(1)) / nikkei['Close'].shift(1)).values
            # Intraday momentum: did selling continue through the day?
            result['nikkei_intraday_momentum'] = ((nikkei['Close'] - nikkei['Open']) / nikkei['Open']).values

        if not kospi.empty:
            kospi_aligned = kospi.reindex(base.index, method='ffill')
            result['kospi_close'] = kospi_aligned['Close'].values
            result['kospi_change'] = kospi_aligned['Close'].pct_change().values
            result['kospi_trend'] = (kospi_aligned['Close'] / kospi_aligned['Close'].shift(20) - 1).values
            result['kospi_open_gap'] = ((kospi_aligned['Open'] - kospi_aligned['Close'].shift(1)) / kospi_aligned['Close'].shift(1)).values
            result['kospi_intraday_momentum'] = ((kospi_aligned['Close'] - kospi_aligned['Open']) / kospi_aligned['Open']).values

        # Composite features
        nk_change = result.get('nikkei_change', pd.Series(0.0, index=result.index))
        ks_change = result.get('kospi_change', pd.Series(0.0, index=result.index))
        result['asian_avg_return'] = (nk_change.fillna(0) + ks_change.fillna(0)) / 2

        # Risk-off signal: 1.0 if both drop >2%, 0.5 if one drops >2%, 0.0 otherwise
        nk_crash = nk_change.fillna(0) < -0.02
        ks_crash = ks_change.fillna(0) < -0.02
        result['asian_risk_off_signal'] = 0.0
        result.loc[nk_crash | ks_crash, 'asian_risk_off_signal'] = 0.5
        result.loc[nk_crash & ks_crash, 'asian_risk_off_signal'] = 1.0

        result = result.reset_index(drop=True)
        print(f"✅ Fetched {len(result)} Asian market data points (Nikkei + KOSPI)")
        return result

    except Exception as e:
        print(f"❌ Error fetching Asian markets: {e}")
        return pd.DataFrame()


def fetch_asian_market_realtime() -> Dict:
    """
    Fetch real-time Asian market status for crash detection.
    Uses intraday data to detect opening gaps before PSX opens.

    Returns:
        Dict with nikkei/kospi status, crash_warning, crash_severity, etc.
    """
    if not YFINANCE_AVAILABLE:
        return {"error": "yfinance not available", "crash_warning": False, "crash_severity": "none"}

    # Check cache first (5-minute TTL)
    if ASIAN_CACHE_FILE.exists():
        try:
            with open(ASIAN_CACHE_FILE, 'r') as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached.get('checked_at', '2000-01-01'))
            if (datetime.now() - cached_at).total_seconds() < 300:
                return cached
        except Exception:
            pass

    result = {
        "nikkei": {"current": None, "prev_close": None, "change_pct": None, "open_gap_pct": None, "status": "unknown"},
        "kospi": {"current": None, "prev_close": None, "change_pct": None, "open_gap_pct": None, "status": "unknown"},
        "asian_risk_off_signal": 0.0,
        "crash_warning": False,
        "crash_severity": "none",
        "warning_message": "",
        "checked_at": datetime.now().isoformat(),
    }

    for name, ticker in ASIAN_MARKET_TICKERS.items():
        try:
            # Fetch last 5 days to get previous close + today's data
            data = yf.download(ticker, period="5d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if data.empty or len(data) < 2:
                result[name]["status"] = "no_data"
                continue

            prev_close = float(data['Close'].iloc[-2])
            current = float(data['Close'].iloc[-1])
            today_open = float(data['Open'].iloc[-1])

            change_pct = ((current - prev_close) / prev_close) * 100
            open_gap_pct = ((today_open - prev_close) / prev_close) * 100

            result[name] = {
                "current": round(current, 2),
                "prev_close": round(prev_close, 2),
                "change_pct": round(change_pct, 2),
                "open_gap_pct": round(open_gap_pct, 2),
                "status": "open" if data.index[-1].date() == datetime.now().date() else "closed",
            }
        except Exception as e:
            result[name]["status"] = f"error: {str(e)[:100]}"

    # Calculate crash severity
    nk_drop = abs(min(result["nikkei"].get("change_pct") or 0, 0))
    ks_drop = abs(min(result["kospi"].get("change_pct") or 0, 0))

    if nk_drop >= 2 and ks_drop >= 2:
        result["crash_severity"] = "severe"
        result["asian_risk_off_signal"] = 1.0
        result["crash_warning"] = True
        result["warning_message"] = f"Both Nikkei (-{nk_drop:.1f}%) & KOSPI (-{ks_drop:.1f}%) crashed. Regional sell-off risk elevated for PSX."
    elif nk_drop >= 2 or ks_drop >= 2:
        result["crash_severity"] = "moderate"
        result["asian_risk_off_signal"] = 0.5
        result["crash_warning"] = True
        which = "Nikkei" if nk_drop >= 2 else "KOSPI"
        drop = nk_drop if nk_drop >= 2 else ks_drop
        result["warning_message"] = f"{which} dropped {drop:.1f}%. Regional sell-off risk for PSX."
    elif nk_drop >= 1.5 or ks_drop >= 1.5:
        result["crash_severity"] = "mild"
        result["asian_risk_off_signal"] = 0.25
        result["crash_warning"] = True
        result["warning_message"] = "Asian markets under pressure. Monitor PSX carefully."
    else:
        result["crash_severity"] = "none"
        result["asian_risk_off_signal"] = 0.0
        result["crash_warning"] = False
        result["warning_message"] = ""

    # Cache result
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(ASIAN_CACHE_FILE, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    except Exception:
        pass

    return result


# ============================================================================
# STOCK BETA CALCULATION
# ============================================================================

def calculate_stock_beta(stock_returns: np.ndarray, 
                         market_returns: np.ndarray, 
                         window: int = 63) -> np.ndarray:
    """
    Calculate rolling beta of stock vs KSE-100.
    
    Research: Market beta explains most stock movement.
    Beta > 1: More volatile than market
    Beta < 1: Less volatile than market
    
    Args:
        stock_returns: Array of stock daily returns
        market_returns: Array of KSE-100 daily returns
        window: Rolling window (63 days = ~3 months)
    
    Returns:
        Array of rolling beta values
    """
    betas = np.full(len(stock_returns), np.nan)
    
    for t in range(window, len(stock_returns)):
        stock_window = stock_returns[t-window:t]
        market_window = market_returns[t-window:t]
        
        # Remove NaN
        mask = ~(np.isnan(stock_window) | np.isnan(market_window))
        if mask.sum() < window // 2:
            continue
        
        # Beta = Cov(stock, market) / Var(market)
        cov = np.cov(stock_window[mask], market_window[mask])[0, 1]
        var = np.var(market_window[mask])
        betas[t] = cov / var if var > 1e-8 else 1.0
    
    return betas


def calculate_correlation(series1: np.ndarray, series2: np.ndarray, 
                          window: int = 63) -> np.ndarray:
    """
    Calculate rolling correlation between two series.
    Useful for USD/PKR vs stock correlation.
    """
    corrs = np.full(len(series1), np.nan)
    
    for t in range(window, len(series1)):
        s1 = series1[t-window:t]
        s2 = series2[t-window:t]
        
        mask = ~(np.isnan(s1) | np.isnan(s2))
        if mask.sum() < window // 2:
            continue
        
        corrs[t] = np.corrcoef(s1[mask], s2[mask])[0, 1]
    
    return corrs


# ============================================================================
# KIBOR / INTEREST RATE
# ============================================================================

def get_kibor_rate() -> float:
    """
    Get current KIBOR/SBP policy rate.
    
    Note: SBP only provides PDFs, not an API.
    This is manually updated from: https://www.sbp.org.pk/ecodata/kibor_index.asp
    
    Returns:
        Current KIBOR rate as decimal (e.g., 0.13 for 13%)
    """
    return KIBOR_RATE


def get_kibor_features(length: int) -> pd.DataFrame:
    """
    Generate KIBOR-based features.
    Since rate changes infrequently, we provide:
    - Current rate (static)
    - High rate indicator (affects bank stocks)
    """
    rate = get_kibor_rate()
    
    return pd.DataFrame({
        'kibor_rate': [rate] * length,
        'kibor_high': [1 if rate > 0.10 else 0] * length,  # High if > 10%
        'kibor_very_high': [1 if rate > 0.15 else 0] * length  # Very high if > 15%
    })


# ============================================================================
# MERGE EXTERNAL FEATURES WITH STOCK DATA
# ============================================================================

def merge_external_features(stock_df: pd.DataFrame, 
                            symbol: str = None,
                            cache: bool = True) -> pd.DataFrame:
    """
    Merge all external features with stock DataFrame.
    
    This is the main API for the SOTA model.
    
    Args:
        stock_df: DataFrame with stock data (must have 'Date' column)
        symbol: Stock symbol (for sector-specific features)
        cache: Whether to cache external data
    
    Returns:
        DataFrame with both stock and external features
    """
    if 'Date' not in stock_df.columns:
        print("⚠️ stock_df must have 'Date' column")
        return stock_df
    
    df = stock_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get date range
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')
    start_year = df['Date'].min().year
    
    print(f"\n📊 MERGING EXTERNAL FEATURES")
    print(f"   Date range: {start_date} to {end_date}")
    print("=" * 50)
    
    # 1. USD/PKR
    print("\n1. Fetching USD/PKR...")
    usdpkr = fetch_usd_pkr(start_date=start_date, end_date=end_date)
    if not usdpkr.empty:
        usdpkr['date'] = pd.to_datetime(usdpkr['date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            usdpkr.sort_values('date'),
            left_on='Date',
            right_on='date',
            direction='backward'
        )
        df = df.drop(columns=['date'], errors='ignore')
        print(f"   ✅ Added {len([c for c in df.columns if 'usdpkr' in c])} USD/PKR features")
    
    # 2. KSE-100 features
    if symbol and symbol.upper() != 'KSE100':
        # For regular stocks: fetch KSE100 as external feature
        print("\n2. Fetching KSE-100...")
        kse100 = fetch_kse100(start_year=start_year, end_date=end_date)
        if not kse100.empty:
            df = pd.merge_asof(
                df.sort_values('Date'),
                kse100.sort_values('date'),
                left_on='Date',
                right_on='date',
                direction='backward'
            )
            df = df.drop(columns=['date'], errors='ignore')
            
            # Calculate beta vs KSE-100
            if 'Close' in df.columns and 'kse100_return' in df.columns:
                stock_returns = df['Close'].pct_change().fillna(0).values
                df['stock_beta'] = calculate_stock_beta(stock_returns, df['kse100_return'].fillna(0).values)
                
                # Correlation with KSE-100
                df['kse100_correlation'] = calculate_correlation(
                    df['Close'].pct_change().fillna(0).values,
                    df['kse100_return'].fillna(0).values
                )
            
            print(f"   ✅ Added {len([c for c in df.columns if 'kse100' in c.lower() or 'beta' in c.lower()])} KSE-100 features")
    else:
        # For KSE100 itself: use its own data as features (real features, not zeros!)
        print("\n2. Adding KSE-100 features from own data...")
        
        # Use KSE100's own OHLCV data as features
        df['kse100_open'] = df['Open'].values
        df['kse100_high'] = df['High'].values
        df['kse100_low'] = df['Low'].values
        df['kse100_close'] = df['Close'].values
        df['kse100_volume'] = df['Volume'].values
        
        # Calculate returns and momentum from KSE100's own data
        df['kse100_return'] = df['Close'].pct_change().fillna(0)
        df['kse100_volatility'] = df['kse100_return'].rolling(20).std().fillna(0)
        df['kse100_trend'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0)
        
        # Moving average signals
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        df['kse100_above_sma50'] = (df['Close'] > sma50).astype(int).fillna(0)
        df['kse100_above_sma200'] = (df['Close'] > sma200).astype(int).fillna(0)
        
        # Beta and correlation with itself (perfect correlation)
        df['stock_beta'] = 1.0  # KSE100 has beta of 1.0 with itself
        df['kse100_correlation'] = 1.0  # Perfect correlation with itself
        
        # Additional momentum features from KSE100's own data
        df['kse100_momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1).fillna(0)
        df['kse100_momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
        df['kse100_momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0)
        
        print(f"   ✅ Added {len([c for c in df.columns if 'kse100' in c.lower() or 'beta' in c.lower()])} real KSE-100 features from own data")
    
    # 3. Commodities (Oil, Gold)
    print("\n3. Fetching Commodities...")
    commodities = fetch_commodities(start_date=start_date, end_date=end_date)
    if not commodities.empty:
        commodities['date'] = pd.to_datetime(commodities['date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            commodities.sort_values('date'),
            left_on='Date',
            right_on='date',
            direction='backward'
        )
        df = df.drop(columns=['date'], errors='ignore')
        
        # Sector-specific: Energy stocks correlate with oil
        energy_symbols = ['OGDC', 'PPL', 'PSO', 'POL', 'MARI', 'ATRL']
        if symbol and symbol.upper() in energy_symbols:
            if 'oil_change' in df.columns and 'Close' in df.columns:
                df['oil_correlation'] = calculate_correlation(
                    df['Close'].pct_change().values,
                    df['oil_change'].values
                )
            print(f"   ✅ Added oil correlation for energy stock {symbol}")
        
        print(f"   ✅ Added {len([c for c in df.columns if 'oil' in c.lower() or 'gold' in c.lower()])} commodity features")
    
    # 4. Asian Markets (Leading Indicators)
    print("\n4. Fetching Asian Markets (Nikkei, KOSPI)...")
    asian = fetch_asian_markets(start_date=start_date, end_date=end_date)
    if not asian.empty:
        asian['date'] = pd.to_datetime(asian['date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            asian.sort_values('date'),
            left_on='Date',
            right_on='date',
            direction='backward'
        )
        df = df.drop(columns=['date'], errors='ignore')

        # Asian market correlation with stock
        if 'asian_avg_return' in df.columns and 'Close' in df.columns:
            df['asian_correlation'] = calculate_correlation(
                df['Close'].pct_change().fillna(0).values,
                df['asian_avg_return'].fillna(0).values
            )

        asian_cols = [c for c in df.columns if 'nikkei' in c.lower() or 'kospi' in c.lower() or 'asian' in c.lower()]
        print(f"   ✅ Added {len(asian_cols)} Asian market features")
    else:
        print("   ⚠️ No Asian market data available")

    # 5. Energy-shock features
    print("\n5. Adding energy-shock features...")
    df = _ensure_energy_shock_columns(df)
    if ENERGY_SHOCK_FEATURES_AVAILABLE:
        try:
            news_items = _load_energy_news_items(symbol)
            energy_features = build_energy_event_feature_frame(df, news_items, symbol=symbol)
            if not energy_features.empty:
                for col in ENERGY_SHOCK_FEATURE_COLUMNS:
                    if col in energy_features.columns:
                        df[col] = energy_features[col].values
                print(
                    "   ✅ Added energy-shock features: "
                    f"fuel_delta={float(pd.to_numeric(df['local_fuel_price_delta_rs'], errors='coerce').fillna(0).iloc[-1]):+.2f}, "
                    f"circular_debt={float(pd.to_numeric(df['circular_debt_signal'], errors='coerce').fillna(0).iloc[-1]):+.1f}, "
                    f"regime={int(pd.to_numeric(df['energy_shock_regime'], errors='coerce').fillna(0).iloc[-1])}"
                )
            else:
                print("   ⚠️ No energy-shock feature frame available, using defaults")
        except Exception as e:
            print(f"   ⚠️ Energy-shock feature build failed: {e}")

    kse_return_series = pd.to_numeric(
        df['kse100_return'] if 'kse100_return' in df.columns else pd.Series(0.0, index=df.index),
        errors='coerce'
    ).fillna(0.0)
    oil_change_series = pd.to_numeric(
        df['oil_change'] if 'oil_change' in df.columns else pd.Series(0.0, index=df.index),
        errors='coerce'
    ).fillna(0.0)
    geo_shock_series = pd.to_numeric(
        df['geo_shock_signal'] if 'geo_shock_signal' in df.columns else pd.Series(0, index=df.index),
        errors='coerce'
    ).fillna(0)
    fuel_shock_series = pd.to_numeric(
        df['local_fuel_price_shock'] if 'local_fuel_price_shock' in df.columns else pd.Series(0, index=df.index),
        errors='coerce'
    ).fillna(0)

    if 'kse_oil_interaction' in df.columns:
        df['kse_oil_interaction'] = kse_return_series * oil_change_series
    if 'energy_shock_regime' in df.columns:
        oil_abs = oil_change_series.abs()
        df['energy_shock_regime'] = (
            (geo_shock_series > 0)
            | (fuel_shock_series > 0)
            | (oil_abs >= 0.05)
        ).astype(int)
        df['kse_energy_shock_interaction'] = kse_return_series * pd.to_numeric(
            df['energy_shock_regime'], errors='coerce'
        ).fillna(0.0)

    # 6. KIBOR
    print("\n6. Adding KIBOR features...")
    kibor_df = get_kibor_features(len(df))
    for col in kibor_df.columns:
        df[col] = kibor_df[col].values
    print(f"   ✅ Added {len(kibor_df.columns)} KIBOR features")

    # 7. TradingView Technical Indicators (uses scraper cache TTL; no forced cache eviction)
    if TRADINGVIEW_AVAILABLE and symbol:
        print(f"\n6. Fetching TradingView indicators for {symbol}...")

        # Fetch from TradingView (cache-aware in tradingview_scraper.py)
        tv_result = get_tradingview_indicators(symbol, fallback_local=None)
        tv_indicators = tv_result.get('indicators', {})
        
        if tv_indicators and tv_result['source'] == 'tradingview':
            # IMPORTANT: TradingView values are CURRENT real-time values
            # To prevent data leakage during backtesting, we ONLY apply them to the LAST row
            # This ensures historical rows don't have future information
            tv_count = 0
            last_idx = df.index[-1]

            # Initialize all TV columns with NaN (will be filled only for last row)
            # KEY OSCILLATORS (Most Important)
            if 'rsi_14' in tv_indicators:
                df['tv_rsi_14'] = np.nan
                df.loc[last_idx, 'tv_rsi_14'] = tv_indicators['rsi_14']
                print(f"   📊 TradingView RSI: {tv_indicators['rsi_14']:.2f}")
                tv_count += 1

            if 'macd_level' in tv_indicators:
                df['tv_macd'] = np.nan
                df.loc[last_idx, 'tv_macd'] = tv_indicators['macd_level']
                tv_count += 1

            if 'stochastic_k' in tv_indicators:
                df['tv_stochastic_k'] = np.nan
                df.loc[last_idx, 'tv_stochastic_k'] = tv_indicators['stochastic_k']
                tv_count += 1

            if 'adx' in tv_indicators:
                df['tv_adx'] = np.nan
                df.loc[last_idx, 'tv_adx'] = tv_indicators['adx']
                tv_count += 1

            if 'momentum_10' in tv_indicators:
                df['tv_momentum'] = np.nan
                df.loc[last_idx, 'tv_momentum'] = tv_indicators['momentum_10']
                tv_count += 1

            if 'williams_r' in tv_indicators:
                df['tv_williams_r'] = np.nan
                df.loc[last_idx, 'tv_williams_r'] = tv_indicators['williams_r']
                tv_count += 1

            if 'cci' in tv_indicators:
                df['tv_cci'] = np.nan
                df.loc[last_idx, 'tv_cci'] = tv_indicators['cci']
                tv_count += 1

            if 'awesome_oscillator' in tv_indicators:
                df['tv_awesome'] = np.nan
                df.loc[last_idx, 'tv_awesome'] = tv_indicators['awesome_oscillator']
                tv_count += 1

            if 'bull_bear_power' in tv_indicators:
                df['tv_bull_bear'] = np.nan
                df.loc[last_idx, 'tv_bull_bear'] = tv_indicators['bull_bear_power']
                tv_count += 1

            # MOVING AVERAGES (Important for trend)
            if 'ema_10' in tv_indicators:
                df['tv_ema_10'] = np.nan
                df.loc[last_idx, 'tv_ema_10'] = tv_indicators['ema_10']
                tv_count += 1

            if 'sma_10' in tv_indicators:
                df['tv_sma_10'] = np.nan
                df.loc[last_idx, 'tv_sma_10'] = tv_indicators['sma_10']
                tv_count += 1

            if 'ema_20' in tv_indicators:
                df['tv_ema_20'] = np.nan
                df.loc[last_idx, 'tv_ema_20'] = tv_indicators['ema_20']
                tv_count += 1

            if 'sma_20' in tv_indicators:
                df['tv_sma_20'] = np.nan
                df.loc[last_idx, 'tv_sma_20'] = tv_indicators['sma_20']
                tv_count += 1

            if 'ema_50' in tv_indicators:
                df['tv_ema_50'] = np.nan
                df.loc[last_idx, 'tv_ema_50'] = tv_indicators['ema_50']
                tv_count += 1

            if 'sma_50' in tv_indicators:
                df['tv_sma_50'] = np.nan
                df.loc[last_idx, 'tv_sma_50'] = tv_indicators['sma_50']
                tv_count += 1

            if 'ema_100' in tv_indicators:
                df['tv_ema_100'] = np.nan
                df.loc[last_idx, 'tv_ema_100'] = tv_indicators['ema_100']
                tv_count += 1

            if 'sma_100' in tv_indicators:
                df['tv_sma_100'] = np.nan
                df.loc[last_idx, 'tv_sma_100'] = tv_indicators['sma_100']
                tv_count += 1

            # COMPOSITE FEATURES
            # Price vs EMAs (important trend signals)
            current_price = df['Close'].iloc[-1]

            if 'ema_20' in tv_indicators and tv_indicators['ema_20'] > 0:
                df['tv_price_vs_ema20'] = np.nan
                df.loc[last_idx, 'tv_price_vs_ema20'] = (current_price / tv_indicators['ema_20'] - 1) * 100
                tv_count += 1

            if 'ema_50' in tv_indicators and tv_indicators['ema_50'] > 0:
                df['tv_price_vs_ema50'] = np.nan
                df.loc[last_idx, 'tv_price_vs_ema50'] = (current_price / tv_indicators['ema_50'] - 1) * 100
                tv_count += 1

            # Recommendation counts
            if 'recommendation_buy' in tv_indicators:
                df['tv_rec_buy'] = np.nan
                df['tv_rec_sell'] = np.nan
                df['tv_rec_neutral'] = np.nan
                df['tv_recommendation_score'] = np.nan

                df.loc[last_idx, 'tv_rec_buy'] = tv_indicators['recommendation_buy']
                df.loc[last_idx, 'tv_rec_sell'] = tv_indicators['recommendation_sell']
                df.loc[last_idx, 'tv_rec_neutral'] = tv_indicators['recommendation_neutral']

                # Calculate recommendation score (-1 to +1)
                total = (tv_indicators['recommendation_buy'] +
                        tv_indicators['recommendation_sell'] +
                        tv_indicators['recommendation_neutral'])
                if total > 0:
                    df.loc[last_idx, 'tv_recommendation_score'] = (
                        (tv_indicators['recommendation_buy'] - tv_indicators['recommendation_sell']) / total
                    )
                tv_count += 4

            print(f"   ✅ Added {tv_count} TradingView indicators (LAST ROW ONLY - no data leakage)")
        elif tv_indicators and tv_result['source'] == 'local_fallback':
            print(f"   ℹ️ Using local indicators (TradingView unavailable)")
        else:
            print(f"   ⚠️ TradingView unavailable, using local indicators")
    
    # 8. USD/PKR correlation with stock
    if 'usdpkr_change' in df.columns and 'Close' in df.columns:
        df['usdpkr_correlation'] = calculate_correlation(
            df['Close'].pct_change().values,
            df['usdpkr_change'].values
        )
    
    # Sort back to original order
    df = df.sort_values('Date').reset_index(drop=True)
    
    print("\n" + "=" * 50)
    print(f"✅ EXTERNAL FEATURES COMPLETE")
    print(f"   Total features added: {len([c for c in df.columns if c not in stock_df.columns])}")
    print(f"   Final DataFrame: {len(df)} rows x {len(df.columns)} columns")
    
    return df


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📊 EXTERNAL FEATURES MODULE - TEST")
    print("=" * 70)
    
    # Test USD/PKR
    print("\n1. Testing USD/PKR fetch...")
    usdpkr = fetch_usd_pkr(period="1mo")
    if not usdpkr.empty:
        print(f"   Latest: {usdpkr['usdpkr_close'].iloc[-1]:.2f} PKR/USD")
    
    # Test KSE-100
    print("\n2. Testing KSE-100 fetch...")
    kse100 = fetch_kse100(start_year=2024)
    if not kse100.empty:
        print(f"   Latest: {kse100['kse100_close'].iloc[-1]:,.2f}")
    
    # Test Commodities
    print("\n3. Testing commodities fetch...")
    commodities = fetch_commodities(period="1mo")
    if not commodities.empty:
        print(f"   Oil: ${commodities['oil_close'].iloc[-1]:.2f}")
        print(f"   Gold: ${commodities['gold_close'].iloc[-1]:.2f}")
    
    # Test Asian Markets
    print("\n4. Testing Asian markets fetch...")
    asian = fetch_asian_markets(period="1mo")
    if not asian.empty:
        print(f"   Nikkei: {asian['nikkei_close'].iloc[-1]:,.2f}")
        if 'kospi_close' in asian.columns:
            print(f"   KOSPI: {asian['kospi_close'].iloc[-1]:,.2f}")
        print(f"   Risk-off signal: {asian['asian_risk_off_signal'].iloc[-1]}")

    # Test real-time crash detection
    print("\n5. Testing Asian market real-time status...")
    rt = fetch_asian_market_realtime()
    print(f"   Crash warning: {rt.get('crash_warning')}, Severity: {rt.get('crash_severity')}")

    # Test merge with dummy stock data
    print("\n6. Testing merge with stock data...")
    dummy_stock = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(100000, 1000000, 100)
    })
    
    merged = merge_external_features(dummy_stock, symbol='OGDC')
    print(f"   New columns: {[c for c in merged.columns if c not in dummy_stock.columns]}")
    
    print("\n✅ All tests complete!")
