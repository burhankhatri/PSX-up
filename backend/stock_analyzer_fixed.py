#!/usr/bin/env python3
"""
Backend API for Stock Analysis - Complete & Fixed
Handles data fetching, model training, and progress updates via WebSocket
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import os
import subprocess
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import asyncio
from typing import Dict, List, Union, Optional

logger = logging.getLogger(__name__)
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Note: This module exports functions, not a FastAPI app instance
# The routes are defined here but will be added to the main app in main.py

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from backend.runtime_config import get_runtime_config, RuntimeConfig
except Exception:
    get_runtime_config = None
    RuntimeConfig = None

try:
    from backend.prediction_tuning import (
        apply_prediction_tweaks,
        get_live_tweak_config,
        direction_from_change_pct,
    )
except Exception:
    # Safe fallbacks if tuning module is unavailable
    def apply_prediction_tweaks(predictions, config):
        return predictions

    def get_live_tweak_config():
        class _Cfg:
            enabled = False
            neutral_band_pct = 0.0
        return _Cfg()

    def direction_from_change_pct(change_pct, neutral_band_pct=0.0):
        if abs(change_pct) <= neutral_band_pct:
            return "NEUTRAL"
        return "BULLISH" if change_pct > 0 else "BEARISH"

class StockRequest(BaseModel):
    symbol: str
    horizon: Union[int, str] = 21  # Default to 21 days (research-validated), can be int or 'full'
    enable_geo_features: Optional[bool] = None

progress_data = {}
INDEX_SYMBOLS = {'KSE100', 'KSE-100', 'PSX'}


async def safe_send(websocket: WebSocket, data: dict) -> bool:
    """Safely send data through websocket, checking if connection is still open."""
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
    except Exception:
        pass
    return False


def _merge_geo_overlay_news(symbol: str, sentiment_result: Optional[dict]) -> tuple[list[dict], dict]:
    """Geo-only news expansion so the overlay can use sector/macro evidence without moving baseline sentiment.

    Now includes global geopolitical news from international sources (Reuters, Al Jazeera,
    BBC, CNBC, Google News) to capture major events like oil price surges, wars,
    Strait of Hormuz blockades, OPEC decisions, etc. that the Pakistani-only news
    sources miss entirely.
    """
    base_news = [
        item for item in ((sentiment_result or {}).get("news_items", []) or [])
        if isinstance(item, dict)
    ]
    merged = list(base_news)
    diagnostics = {}

    try:
        from backend.enhanced_news_fetcher import (
            determine_retrieval_mode,
            fetch_multi_source_news,
            normalize_news_key,
        )
    except Exception:
        return merged, diagnostics

    try:
        fetch_result = fetch_multi_source_news(
            symbol=symbol,
            retrieval_mode=determine_retrieval_mode(symbol, retrieval_mode="auto"),
            include_diagnostics=True,
            geo_mode=True,
        )
    except Exception:
        return merged, diagnostics

    diagnostics = fetch_result.get("news_fetch_diagnostics", {})

    # ── Inject global geopolitical news (international sources) ──
    global_news = []
    try:
        from backend.global_news_fetcher import get_global_news_for_symbol, get_global_news_summary
        global_news = get_global_news_for_symbol(symbol)
        if global_news:
            summary = get_global_news_summary(global_news)
            diagnostics["global_geo_news"] = {
                "count": len(global_news),
                "severity": summary.get("severity", "UNKNOWN"),
                "categories": summary.get("categories", {}),
                "oil_war_overlap": summary.get("oil_war_overlap_count", 0),
            }
    except Exception as e:
        logger.warning(f"Global geo news fetch failed: {e}")

    seen = set()
    unique_news = []
    # Process: base news + local geo news + global geo news
    all_items = merged + list(fetch_result.get("news_items", []) or []) + global_news
    for item in all_items:
        if not isinstance(item, dict):
            continue
        try:
            key = normalize_news_key(item)
        except Exception:
            key = f"{item.get('title', '')[:180]}::{str(item.get('date', ''))[:10]}::{item.get('url', '')[-120:]}"
        if key in seen:
            continue
        seen.add(key)
        unique_news.append(item)

    return unique_news, diagnostics


def _build_geo_macro_prompt_context() -> dict:
    """Collect request-scoped Nikkei/KOSPI + crude + global geopolitical context for geo sentiment prompts."""
    context = {
        "available": False,
        "nikkei": {},
        "kospi": {},
        "crude": {},
        "global_geo": {},
    }
    try:
        from backend.external_features import fetch_asian_market_realtime, fetch_commodities, fetch_pakistan_fuel_prices
    except Exception:
        return context

    try:
        asian = fetch_asian_market_realtime() or {}
        context["nikkei"] = asian.get("nikkei", {}) or {}
        context["kospi"] = asian.get("kospi", {}) or {}
    except Exception:
        pass

    try:
        commodities = fetch_commodities(period="1mo")
        if commodities is not None and not commodities.empty:
            def _last_val(col):
                if col in commodities.columns:
                    v = pd.to_numeric(commodities[col], errors="coerce").iloc[-1]
                    return round(float(v), 4) if pd.notna(v) else None
                return None

            context["crude"] = {
                "oil_close": _last_val("oil_close"),
                "oil_change_pct": round(_last_val("oil_change") * 100, 2) if _last_val("oil_change") is not None else None,
                "oil_trend_pct": round(_last_val("oil_trend") * 100, 2) if _last_val("oil_trend") is not None else None,
                "brent_close": _last_val("brent_close"),
                "brent_change_pct": round(_last_val("brent_change") * 100, 2) if _last_val("brent_change") is not None else None,
                "brent_trend_pct": round(_last_val("brent_trend") * 100, 2) if _last_val("brent_trend") is not None else None,
                "natgas_close": _last_val("natgas_close"),
                "natgas_change_pct": round(_last_val("natgas_change") * 100, 2) if _last_val("natgas_change") is not None else None,
                "natgas_trend_pct": round(_last_val("natgas_trend") * 100, 2) if _last_val("natgas_trend") is not None else None,
                "gold_close": _last_val("gold_close"),
                "gold_change_pct": round(_last_val("gold_change") * 100, 2) if _last_val("gold_change") is not None else None,
                "gold_trend_pct": round(_last_val("gold_trend") * 100, 2) if _last_val("gold_trend") is not None else None,
            }
    except Exception:
        pass

    # Pakistan local fuel prices (OGRA-regulated)
    try:
        context["pk_fuel"] = fetch_pakistan_fuel_prices()
    except Exception:
        context["pk_fuel"] = {}

    # ── Global geopolitical news summary ──
    try:
        from backend.global_news_fetcher import get_global_news_summary
        global_summary = get_global_news_summary()
        if global_summary.get("available"):
            context["global_geo"] = {
                "severity": global_summary.get("severity", "UNKNOWN"),
                "severity_description": global_summary.get("severity_description", ""),
                "headline_count": global_summary.get("headline_count", 0),
                "oil_war_overlap": global_summary.get("oil_war_overlap_count", 0),
                "categories": global_summary.get("categories", {}),
                "top_headlines": global_summary.get("top_headlines", [])[:10],
            }
    except Exception as e:
        logger.debug(f"Global geo context unavailable: {e}")

    context["available"] = bool(context["nikkei"] or context["kospi"] or context["crude"] or context["global_geo"])
    return context


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _normalize_prediction_date(date_value) -> Optional[str]:
    if date_value in (None, ""):
        return None
    try:
        return pd.to_datetime(date_value).strftime('%Y-%m-%d')
    except Exception:
        return None


def _resolve_prediction_target_date(
    prediction: Optional[Dict],
    prediction_generated_at: Optional[Union[str, datetime]],
    horizon_days: int,
) -> Optional[str]:
    if isinstance(prediction, dict):
        direct_date = _normalize_prediction_date(prediction.get('date') or prediction.get('target_date'))
        if direct_date:
            return direct_date

    if prediction_generated_at:
        try:
            base_dt = pd.to_datetime(prediction_generated_at)
            return (base_dt + pd.Timedelta(days=horizon_days)).strftime('%Y-%m-%d')
        except Exception:
            pass

    return (datetime.now() + timedelta(days=horizon_days)).strftime('%Y-%m-%d')


def _prediction_direction(prediction: Optional[Dict], neutral_band_pct: float = 0.0) -> str:
    if not isinstance(prediction, dict):
        return 'NEUTRAL'
    stable = str(prediction.get('stable_direction', '') or '').upper()
    if stable == 'UP':
        return 'BULLISH'
    if stable == 'DOWN':
        return 'BEARISH'
    if stable in {'BULLISH', 'BEARISH', 'NEUTRAL'}:
        return stable
    return direction_from_change_pct(
        _safe_float(prediction.get('upside_potential'), 0.0),
        neutral_band_pct=neutral_band_pct,
    )


def _price_error_pct(predicted_price: Optional[float], actual_price: Optional[float]) -> Optional[float]:
    pred = _safe_float(predicted_price, 0.0)
    actual = _safe_float(actual_price, 0.0)
    if pred <= 0 or actual <= 0:
        return None
    return round(((pred - actual) / actual) * 100.0, 4)


def _infer_postmortem_root_cause(
    symbol: str,
    geo_comparison: Optional[Dict],
    postmortem: Dict,
) -> tuple[Optional[str], List[str]]:
    geo = geo_comparison or {}
    interp = (geo.get('interpretation') or {}) if isinstance(geo, dict) else {}
    summary = (geo.get('adjustment_summary') or {}) if isinstance(geo, dict) else {}
    shock = (geo.get('shock_data') or {}) if isinstance(geo, dict) else {}
    macro = (geo.get('macro_confirmation') or {}) if isinstance(geo, dict) else {}
    crude = (macro.get('crude') or {}) if isinstance(macro, dict) else {}
    stock_health = (geo.get('stock_health') or {}) if isinstance(geo, dict) else {}

    helped_flags = [
        postmortem[h].get('geo_helped')
        for h in ('day1', 'day7')
        if isinstance(postmortem.get(h), dict) and postmortem[h].get('geo_helped') is not None
    ]
    if not helped_flags:
        return None, []

    tags: List[str] = []
    symbol_upper = str(symbol or '').upper()
    oil_change_pct = _safe_float(crude.get('oil_change_pct'), 0.0)
    oil_trend_pct = _safe_float(crude.get('oil_trend_pct'), 0.0)
    momentum_20d = _safe_float(stock_health.get('momentum_20d'), 0.0)

    geo_worsened = any(flag is False for flag in helped_flags)
    geo_helped = any(flag is True for flag in helped_flags)

    dominant_miss_driver: Optional[str]
    if geo_worsened:
        if interp.get('sector_interpretation') == 'upstream_energy_tailwind':
            dominant_miss_driver = 'upstream_tailwind_overestimate'
            if momentum_20d < 0:
                tags.append('momentum_ignored')
            if oil_change_pct < 2.0 and oil_trend_pct < 4.0:
                tags.append('no_crude_confirmation')
            if str(interp.get('evidence_quality', 'weak')).lower() in {'weak', 'limited'}:
                tags.append('weak_upstream_evidence')
        elif bool(summary.get('shock_detected')) or bool(shock.get('shock_detected')):
            dominant_miss_driver = 'shock_overpenalty'
            if _safe_float(summary.get('emergency_multiplier'), 1.0) >= 2.0:
                tags.append('emergency_multiplier_aggressive')
            if symbol_upper in INDEX_SYMBOLS:
                tags.append('index_overlay_excess')
            if bool(summary.get('direction_conflict')):
                tags.append('ai_deterministic_conflict')
        else:
            dominant_miss_driver = 'geo_overlay_miscalibration'
    elif geo_helped:
        dominant_miss_driver = 'geo_overlay_helped'
        if bool(summary.get('shock_detected')) or bool(shock.get('shock_detected')):
            tags.append('shock_signal_confirmed')
    else:
        dominant_miss_driver = None

    return dominant_miss_driver, sorted(set(tags))


def build_forecast_postmortem(
    symbol: str,
    current_price: float,
    baseline_predictions: List[Dict],
    geo_predictions: Optional[List[Dict]] = None,
    geo_comparison: Optional[Dict] = None,
    prediction_generated_at: Optional[Union[str, datetime]] = None,
) -> Optional[Dict]:
    """Compare baseline and geo forecast checkpoints against realized closes when available."""
    if not baseline_predictions:
        return None

    try:
        from backend.prediction_tuning import _fetch_actual_on_or_after
    except Exception:
        return None

    cache: Dict = {}
    payload: Dict[str, Dict] = {}
    geo_predictions = geo_predictions or []
    horizons = (
        ('day1', 'day_1', 0, 1),
        ('day7', 'day_7', 6, 7),
    )

    for payload_key, _, idx, horizon_days in horizons:
        baseline_pred = baseline_predictions[idx] if idx < len(baseline_predictions) else None
        geo_pred = geo_predictions[idx] if idx < len(geo_predictions) else None
        if baseline_pred is None and geo_pred is None:
            continue

        target_date = _resolve_prediction_target_date(
            baseline_pred or geo_pred,
            prediction_generated_at,
            horizon_days,
        )
        if not target_date:
            continue

        try:
            actual_price, actual_date = _fetch_actual_on_or_after(symbol, target_date, cache)
        except Exception:
            continue
        if actual_price is None:
            continue

        baseline_price = _safe_float((baseline_pred or {}).get('predicted_price'), 0.0)
        geo_price = _safe_float((geo_pred or {}).get('predicted_price'), 0.0) if geo_pred else None
        baseline_error_pct = _price_error_pct(baseline_price, actual_price)
        geo_error_pct = _price_error_pct(geo_price, actual_price) if geo_price else None
        baseline_abs = abs(baseline_error_pct) if baseline_error_pct is not None else None
        geo_abs = abs(geo_error_pct) if geo_error_pct is not None else None
        geo_helped = None
        if baseline_abs is not None and geo_abs is not None:
            if abs(geo_abs - baseline_abs) <= 1e-9:
                geo_helped = None
            else:
                geo_helped = geo_abs < baseline_abs

        geo_adjustment_pct = None
        if geo_price is not None and baseline_price > 0:
            geo_adjustment_pct = round(((geo_price - baseline_price) / baseline_price) * 100.0, 4)

        payload[payload_key] = {
            'target_date': target_date,
            'actual_date_used': actual_date,
            'baseline_predicted_price': round(baseline_price, 2) if baseline_price > 0 else None,
            'geo_predicted_price': round(geo_price, 2) if geo_price else None,
            'actual_price': round(float(actual_price), 2),
            'actual_change_pct': round(((float(actual_price) - current_price) / current_price) * 100.0, 4)
            if current_price > 0 else None,
            'baseline_error_pct': baseline_error_pct,
            'geo_error_pct': geo_error_pct,
            'geo_adjustment_pct': geo_adjustment_pct,
            'geo_helped': geo_helped,
        }

    if not payload:
        return None

    dominant_miss_driver, root_cause_tags = _infer_postmortem_root_cause(symbol, geo_comparison, payload)
    payload['dominant_miss_driver'] = dominant_miss_driver
    payload['root_cause_tags'] = root_cause_tags
    return payload


from backend.post_process._logging import log_prediction_variants as _log_prediction_variants  # noqa: E402


def fetch_month_data(symbol: str, month: int, year: int):
    """Fetch historical data for a specific month"""
    url = "https://dps.psx.com.pk/historical"
    post_data = f"month={month}&year={year}&symbol={symbol}"
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST', url, '-d', post_data],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except:
        return None

def parse_html_table(html):
    """Parse HTML table to extract OHLCV data"""
    rows = re.findall(r'<tr>.*?</tr>', html, re.DOTALL)
    data = []
    
    for row in rows:
        cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
        
        if len(cells) >= 6:
            try:
                date_str = cells[0].strip()
                date_obj = datetime.strptime(date_str, "%b %d, %Y")
                
                open_price = float(cells[1].strip().replace(',', ''))
                high_price = float(cells[2].strip().replace(',', ''))
                low_price = float(cells[3].strip().replace(',', ''))
                close_price = float(cells[4].strip().replace(',', ''))
                volume = float(cells[5].strip().replace(',', ''))
                
                data.append({
                    'Date': date_obj.strftime('%Y-%m-%d'),
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
            except:
                continue
    
    return data

def calculate_basic_indicators(data):
    """Calculate basic technical indicators"""
    if not data or len(data) == 0:
        return []  # Return empty list for empty data
    
    df = pd.DataFrame(data)
    
    # Check if required columns exist
    if 'Date' not in df.columns:
        return []
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Price_Change'] = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    df['Volume_Change'] = df['Volume'].diff()
    
    for window in [20, 50, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    
    records = df.to_dict('records')
    for record in records:
        if 'Date' in record and pd.notna(record['Date']):
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
    return records

async def fetch_historical_data_async(symbol: str, progress_callback=None, existing_data=None):
    """
    Fetch historical data with progress updates.
    If existing_data is provided, only fetches new data from the last date forward (incremental update).
    """
    symbol = symbol.upper()
    all_data = []
    current_year = datetime.now().year
    current_month = datetime.now().month
    start_year = 2020
    
    # Determine what months to fetch
    if existing_data and len(existing_data) > 0:
        # Incremental update: find last date and fetch only new months
        existing_dates = [pd.to_datetime(d['Date']).date() for d in existing_data if 'Date' in d]
        if existing_dates:
            last_date = max(existing_dates)
            last_year = last_date.year
            last_month = last_date.month
            
            # Start from the SAME month as last date (to catch new days in that month)
            start_fetch_year = last_year
            start_fetch_month = last_month

            # Use existing data as base
            all_data = existing_data.copy()

            if progress_callback:
                await progress_callback({
                    'stage': 'fetching',
                    'progress': 10,
                    'message': f'📥 Found existing data up to {last_date}. Checking for new data from {start_fetch_year}-{start_fetch_month:02d}...'
                })
        else:
            # Existing data but no valid dates, fetch everything
            start_fetch_year = start_year
            start_fetch_month = 1
            if progress_callback:
                await progress_callback({
                    'stage': 'fetching',
                    'progress': 10,
                    'message': f'📡 Existing data invalid. Fetching all historical data...'
                })
    else:
        # Full fetch from scratch
        start_fetch_year = start_year
        start_fetch_month = 1
        if progress_callback:
            await progress_callback({
                'stage': 'fetching',
                'progress': 10,
                'message': f'📡 First-time analysis: Fetching all historical data from {start_year} to today...'
            })
    
    # Calculate total months to fetch
    if start_fetch_year < current_year or (start_fetch_year == current_year and start_fetch_month <= current_month):
        total_months = (current_year - start_fetch_year) * 12 + (current_month - start_fetch_month + 1)
    else:
        total_months = 0
    
    fetched = 0
    new_data = []
    
    # Only fetch if there are months to fetch
    if total_months > 0:
        for year in range(start_fetch_year, current_year + 1):
            start_month = start_fetch_month if year == start_fetch_year else 1
            end_month = current_month if year == current_year else 12
            
            for month in range(start_month, end_month + 1):
                fetched += 1
                progress = int((fetched / total_months) * 40) + 10  # 10-50%
                
                if progress_callback:
                    await progress_callback({
                        'stage': 'fetching',
                        'progress': progress,
                        'message': f'Fetching {year}-{month:02d}... ({fetched}/{total_months})'
                    })
                
                html = fetch_month_data(symbol, month, year)
                
                if html:
                    month_data = parse_html_table(html)
                    if month_data:
                        new_data.extend(month_data)
                
                await asyncio.sleep(0.05)
        
        # Merge new data with existing (if any) - deduplicate by date
        if new_data:
            # Add new data
            all_data.extend(new_data)

            # Deduplicate by date (keep latest version of each date)
            seen_dates = set()
            unique_data = []
            original_count = len(existing_data) if existing_data else 0

            # Process in reverse so newer records take precedence
            for record in reversed(all_data):
                date_key = record.get('Date', '')
                if date_key and date_key not in seen_dates:
                    seen_dates.add(date_key)
                    unique_data.append(record)

            unique_data.reverse()  # Restore chronological order
            unique_data.sort(key=lambda x: x.get('Date', ''))
            all_data = unique_data

            actual_new = len(all_data) - original_count
            if progress_callback:
                if actual_new > 0:
                    await progress_callback({
                        'stage': 'fetching',
                        'progress': 45,
                        'message': f'✅ Added {actual_new} new trading days. Total: {len(all_data)} records.'
                    })
                else:
                    await progress_callback({
                        'stage': 'fetching',
                        'progress': 45,
                        'message': f'✅ Cache is up-to-date. No new trading days from PSX.'
                    })
        elif existing_data:
            # No new data found, but we have existing data - ensure it's sorted and deduplicated
            seen_dates = set()
            unique_data = []
            for record in all_data:
                date_key = record.get('Date', '')
                if date_key and date_key not in seen_dates:
                    seen_dates.add(date_key)
                    unique_data.append(record)
            
            unique_data.sort(key=lambda x: x.get('Date', ''))
            
            if progress_callback:
                await progress_callback({
                    'stage': 'fetching',
                    'progress': 45,
                    'message': f'✅ No new data found. Using existing data ({len(unique_data)} records)...'
                })
            
            # Save the cleaned data
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            filename = data_dir / f"{symbol}_historical_with_indicators.json"
            with open(filename, 'w') as f:
                json.dump(unique_data, f, indent=2)
            
            return unique_data
    
    # Check if we got any data
    if not all_data or len(all_data) == 0:
        if progress_callback:
            await progress_callback({
                'stage': 'error',
                'progress': 0,
                'message': f'❌ No historical data found for {symbol}. This symbol may not exist on PSX or has no trading history.'
            })
        raise ValueError(f"No historical data found for symbol {symbol}")
    
    if progress_callback:
        await progress_callback({
            'stage': 'calculating',
            'progress': 50,
            'message': 'Calculating technical indicators...'
        })
    
    all_data = calculate_basic_indicators(all_data)
    
    if not all_data:
        if progress_callback:
            await progress_callback({
                'stage': 'error',
                'progress': 0,
                'message': f'❌ Failed to process data for {symbol}. Data format may be invalid.'
            })
        raise ValueError(f"Failed to process data for symbol {symbol}")
    
    # Remove duplicates by Date (keep the latest entry if duplicates exist)
    seen_dates = set()
    unique_data = []
    for record in all_data:
        date_key = record.get('Date', '')
        if date_key and date_key not in seen_dates:
            seen_dates.add(date_key)
            unique_data.append(record)
    
    all_data = unique_data
    all_data.sort(key=lambda x: x.get('Date', ''))
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    filename = data_dir / f"{symbol}_historical_with_indicators.json"
    with open(filename, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    if progress_callback:
        await progress_callback({
            'stage': 'fetch_complete',
            'progress': 55,
            'message': f'✅ Fetched {len(all_data)} records'
        })
    
    return all_data

def load_data(symbol):
    """Load historical data with indicators"""
    data_file = Path(__file__).parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
    
    if not data_file.exists():
        return None
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def calculate_advanced_features(df):
    """Calculate advanced features"""
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Range'] = abs(df['Open'] - df['Close']) / df['Close']
    
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['SMA20_vs_SMA50'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)  # Safe division - prevent crash when loss = 0
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma20
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    df['ATR_14'] = df['High_Low_Range'].rolling(window=14).mean()
    
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    return df

def prepare_training_data(df):
    """Prepare features and targets"""
    df['Target_Next_Day'] = df['Close'].shift(-1)
    
    feature_cols = [c for c in df.columns if c not in ['Date', 'Target_Next_Day', 'Target_Next_Week', 'Target_Next_Month']]
    
    feature_null_counts = df[feature_cols].isnull().sum(axis=1)
    max_allowed_nulls = len(feature_cols) * 0.2
    
    df_clean = df[(feature_null_counts <= max_allowed_nulls) & df['Target_Next_Day'].notna()].copy()
    df_clean[feature_cols] = df_clean[feature_cols].ffill().bfill()
    
    for col in feature_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    X = df_clean[feature_cols]
    y = df_clean['Target_Next_Day']
    
    return X, y, feature_cols, df_clean

def feature_selection(X, y, k=30):
    """Select best features"""
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.median())
    
    selector = SelectKBest(f_regression, k=min(k, X_clean.shape[1]))
    X_selected = selector.fit_transform(X_clean, y)
    
    selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
    
    return X_selected, selected_features, selector

def roll_forward_features(df, predicted_price, date_offset=1):
    """Roll forward features after a prediction"""
    new_row = df.iloc[-1:].copy()
    new_row['Close'] = predicted_price
    new_row['Date'] = new_row['Date'] + pd.Timedelta(days=date_offset)
    new_row['Open'] = predicted_price
    new_row['High'] = predicted_price
    new_row['Low'] = predicted_price
    new_row['Volume'] = df['Volume'].iloc[-20:].mean()
    
    df_extended = pd.concat([df, new_row], ignore_index=True)
    df_extended['High_Low_Range'] = (df_extended['High'] - df_extended['Low']) / df_extended['Close']
    df_extended['Open_Close_Range'] = abs(df_extended['Open'] - df_extended['Close']) / df_extended['Close']
    
    for window in [5, 10, 20, 50, 100, 200]:
        df_extended[f'SMA_{window}'] = df_extended['Close'].rolling(window=window).mean()
        df_extended[f'EMA_{window}'] = df_extended['Close'].ewm(span=window, adjust=False).mean()
    
    df_extended['Price_vs_SMA20'] = (df_extended['Close'] - df_extended['SMA_20']) / df_extended['SMA_20']
    df_extended['Price_vs_SMA50'] = (df_extended['Close'] - df_extended['SMA_50']) / df_extended['SMA_50']
    df_extended['SMA20_vs_SMA50'] = (df_extended['SMA_20'] - df_extended['SMA_50']) / df_extended['SMA_50']
    
    df_extended['Returns'] = df_extended['Close'].pct_change()
    df_extended['Log_Returns'] = np.log(df_extended['Close'] / df_extended['Close'].shift(1))
    
    delta = df_extended['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)  # Safe division - prevent crash when loss = 0
    df_extended['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df_extended['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_extended['Close'].ewm(span=26, adjust=False).mean()
    df_extended['MACD'] = ema12 - ema26
    df_extended['MACD_Signal'] = df_extended['MACD'].ewm(span=9, adjust=False).mean()
    df_extended['MACD_Histogram'] = df_extended['MACD'] - df_extended['MACD_Signal']
    
    sma20 = df_extended['Close'].rolling(window=20).mean()
    std20 = df_extended['Close'].rolling(window=20).std()
    df_extended['BB_Upper'] = sma20 + (std20 * 2)
    df_extended['BB_Lower'] = sma20 - (std20 * 2)
    df_extended['BB_Width'] = (df_extended['BB_Upper'] - df_extended['BB_Lower']) / sma20
    df_extended['BB_Position'] = (df_extended['Close'] - df_extended['BB_Lower']) / (df_extended['BB_Upper'] - df_extended['BB_Lower'])
    
    df_extended['Volume_SMA_20'] = df_extended['Volume'].rolling(window=20).mean()
    df_extended['Volume_Ratio'] = df_extended['Volume'] / df_extended['Volume_SMA_20']
    
    df_extended['Volatility_20'] = df_extended['Returns'].rolling(window=20).std() * np.sqrt(252)
    df_extended['ATR_14'] = df_extended['High_Low_Range'].rolling(window=14).mean()
    
    df_extended['Momentum_10'] = df_extended['Close'] / df_extended['Close'].shift(10) - 1
    df_extended['Momentum_20'] = df_extended['Close'] / df_extended['Close'].shift(20) - 1
    
    for lag in [1, 2, 3, 5, 10]:
        df_extended[f'Close_Lag_{lag}'] = df_extended['Close'].shift(lag)
    
    df_extended['Year'] = df_extended['Date'].dt.year
    df_extended['Month'] = df_extended['Date'].dt.month
    df_extended['DayOfWeek'] = df_extended['Date'].dt.dayofweek
    
    return df_extended

def generate_monthly_predictions_proper(models, scaler, df, feature_cols, selected_features, symbol, end_date='2026-12-31'):
    """Generate monthly predictions with proper feature roll-forward"""
    if selected_features:
        feature_cols = selected_features
    
    df_working = df.copy()
    current_date = df_working['Date'].iloc[-1]
    current_price = df_working['Close'].iloc[-1]
    
    monthly_predictions = []
    prediction_date = current_date
    end_date_obj = pd.to_datetime(end_date)
    trading_days_per_month = 21
    month_count = 0
    
    # Generate predictions through end_date (no arbitrary month limit for 2026 coverage)
    while prediction_date < end_date_obj:
        latest_features = df_working[feature_cols].iloc[-1:].copy()
        latest_features = latest_features.ffill().bfill()
        for col in latest_features.columns:
            if latest_features[col].isnull().any():
                latest_features[col].fillna(latest_features[col].median(), inplace=True)
        
        latest_features_scaled = scaler.transform(latest_features)
        
        predictions = {}
        for name, model in models.items():
            pred = model.predict(latest_features_scaled)[0]
            predictions[name] = float(pred)
        
        ensemble_preds = [pred for pred in predictions.values()]
        ensemble_pred = np.mean(ensemble_preds)
        upside = (ensemble_pred - current_price) / current_price * 100
        
        monthly_predictions.append({
            'month': prediction_date.strftime('%Y-%m'),
            'date': prediction_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predicted_price': float(ensemble_pred),
            'upside_potential': float(upside),
            'rf_prediction': predictions.get('rf', 0),
            'gb_prediction': predictions.get('gb', 0),
            'xgb_prediction': predictions.get('xgb', 0)
        })
        
        df_working = roll_forward_features(df_working, ensemble_pred, date_offset=trading_days_per_month)
        prediction_date = prediction_date + pd.DateOffset(months=1)
        month_count += 1
    
    predictions_file = Path(__file__).parent.parent / "data" / f"{symbol}_monthly_predictions_2026_fixed.json"
    with open(predictions_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'generated_at': datetime.now().isoformat(),
            'current_price': float(df['Close'].iloc[-1]),
            'current_date': current_date.strftime('%Y-%m-%d'),
            'monthly_predictions': monthly_predictions
        }, f, indent=2)
    
    return monthly_predictions

def backtest_trading_strategy(models, scaler, df, feature_cols, selected_features, symbol, initial_capital=100000, transaction_cost=0.001):
    """Backtest trading strategy"""
    if selected_features:
        feature_cols = selected_features
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    history_df = df.iloc[:split_idx].copy()
    
    capital = initial_capital
    shares = 0
    trades = []
    portfolio_values = []
    
    for i in range(len(test_df) - 1):
        current_row = test_df.iloc[i:i+1]
        current_price = current_row['Close'].iloc[0]
        
        hist_plus_current = pd.concat([history_df, current_row], ignore_index=True)
        hist_plus_current[feature_cols] = hist_plus_current[feature_cols].ffill().bfill()
        for col in feature_cols:
            if hist_plus_current[col].isnull().any():
                hist_plus_current[col].fillna(hist_plus_current[col].median(), inplace=True)
        
        scaler_local = StandardScaler()
        scaler_local.fit(hist_plus_current[feature_cols].iloc[:-1])
        features_scaled = scaler_local.transform(current_row[feature_cols])
        
        preds = []
        for name, model in models.items():
            pred = model.predict(features_scaled)[0]
            preds.append(pred)
        
        predicted_price = np.mean(preds)
        expected_return = (predicted_price - current_price) / current_price
        
        if expected_return > 0.01 and shares == 0:
            shares_to_buy = capital / (current_price * (1 + transaction_cost))
            cost = shares_to_buy * current_price * (1 + transaction_cost)
            if cost <= capital:
                capital -= cost
                shares += shares_to_buy
                trades.append({'date': str(current_row['Date'].iloc[0]), 'action': 'BUY', 'price': float(current_price), 'shares': float(shares_to_buy)})
        elif expected_return < -0.01 and shares > 0:
            proceeds = shares * current_price * (1 - transaction_cost)
            capital += proceeds
            shares_sold = shares
            shares = 0
            trades.append({'date': str(current_row['Date'].iloc[0]), 'action': 'SELL', 'price': float(current_price), 'shares': float(shares_sold)})
        
        portfolio_value = capital + (shares * current_price)
        portfolio_values.append({'date': str(current_row['Date'].iloc[0]), 'portfolio_value': float(portfolio_value)})
        history_df = pd.concat([history_df, current_row], ignore_index=True)
    
    final_price = test_df.iloc[-1]['Close']
    final_portfolio_value = capital + (shares * final_price)
    
    total_return = (final_portfolio_value - initial_capital) / initial_capital * 100
    buy_hold_return = (final_price - test_df.iloc[0]['Close']) / test_df.iloc[0]['Close'] * 100
    
    portfolio_series = pd.Series([p['portfolio_value'] for p in portfolio_values])
    sharpe_ratio = (portfolio_series.pct_change().mean() / portfolio_series.pct_change().std()) * np.sqrt(252) if portfolio_series.pct_change().std() > 0 else 0
    max_drawdown = ((portfolio_series - portfolio_series.expanding().max()) / portfolio_series.expanding().max()).min() * 100
    
    backtest_file = Path(__file__).parent.parent / "data" / f"{symbol}_backtest_results.json"
    with open(backtest_file, 'w') as f:
        json.dump({
            'initial_capital': initial_capital,
            'final_portfolio_value': float(final_portfolio_value),
            'total_return': float(total_return),
            'buy_hold_return': float(buy_hold_return),
            'excess_return': float(total_return - buy_hold_return),
            'num_trades': len(trades),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'trades': trades
        }, f, indent=2)
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades)
    }

async def train_with_progress(X, y, selected_features, symbol, websocket):
    """Train models with progress updates"""
    if selected_features:
        X = X[selected_features]
    
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    
    scaler = StandardScaler()
    all_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        await websocket.send_json({
            'stage': 'training',
            'progress': 60 + (fold_idx * 3),
            'message': f'Training fold {fold_idx + 1}/5 (walk-forward validation)...'
        })
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        fold_models = {}
        fold_results = {}
        
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        fold_models['rf'] = rf
        fold_results['rf'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred)
        }
        
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        fold_models['gb'] = gb
        fold_results['gb'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'r2': r2_score(y_test, gb_pred)
        }
        
        if XGBOOST_AVAILABLE:
            xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb.fit(X_train_scaled, y_train)
            xgb_pred = xgb.predict(X_test_scaled)
            fold_models['xgb'] = xgb
            fold_results['xgb'] = {
                'mae': mean_absolute_error(y_test, xgb_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'r2': r2_score(y_test, xgb_pred)
            }
        
        ensemble_preds = [fold_models[m].predict(X_test_scaled) for m in fold_models.keys()]
        ensemble_pred = np.mean(ensemble_preds, axis=0)
        fold_results['ensemble'] = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred)
        }
        
        all_results.append(fold_results)
    
    await websocket.send_json({
        'stage': 'training',
        'progress': 78,
        'message': 'Training final models on full dataset...'
    })
    
    X_scaled = scaler.fit_transform(X)
    final_models = {}
    
    rf_final = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf_final.fit(X_scaled, y)
    final_models['rf'] = rf_final
    
    gb_final = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb_final.fit(X_scaled, y)
    final_models['gb'] = gb_final
    
    if XGBOOST_AVAILABLE:
        xgb_final = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_final.fit(X_scaled, y)
        final_models['xgb'] = xgb_final
    
    avg_results = {}
    for metric in ['mae', 'rmse', 'r2']:
        avg_results[metric] = {}
        for model_name in ['rf', 'gb', 'xgb', 'ensemble']:
            values = [r[model_name][metric] for r in all_results if model_name in r]
            if values:
                avg_results[metric][model_name] = np.mean(values)
    
    models_dir = Path(__file__).parent.parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in final_models.items():
        joblib.dump(model, models_dir / f"{symbol}_{name}_fixed.pkl")
    joblib.dump(scaler, models_dir / f"{symbol}_scaler_fixed.pkl")
    
    if selected_features:
        with open(models_dir / f"{symbol}_selected_features_fixed.json", 'w') as f:
            json.dump(selected_features, f, indent=2)
    
    return final_models, avg_results, scaler, selected_features

# Route will be added in main.py
async def check_data(symbol: str):
    """Check if historical data exists for a symbol"""
    symbol = symbol.upper()
    data_file = Path(__file__).parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
    
    exists = data_file.exists()
    result = {
        'symbol': symbol,
        'exists': exists,
        'file_path': str(data_file) if exists else None
    }
    
    if exists:
        with open(data_file, 'r') as f:
            data = json.load(f)
            result['record_count'] = len(data)
            if data:
                result['date_range'] = {
                    'start': data[0]['Date'],
                    'end': data[-1]['Date']
                }
    
    return result

# Route will be added in main.py
async def analyze_stock(request: StockRequest):
    """Start stock analysis - returns job ID"""
    symbol = request.symbol.upper()

    # Handle horizon parameter ('full' → None for no limit, number → days)
    horizon = request.horizon
    if horizon == 'full':
        horizon_days = None  # No limit (will default to Dec 2026)
    elif isinstance(horizon, int):
        horizon_days = horizon
    else:
        horizon_days = 21  # Default: research-validated 21 days

    job_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    progress_data[job_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing...',
        'symbol': symbol,
        'horizon': horizon_days,  # Store for websocket handler
        'enable_geo_features': request.enable_geo_features,
    }
    return {'job_id': job_id, 'symbol': symbol}

def _run_shadow_comparison(
    symbol: str,
    baseline_predictions: list,
    df: pd.DataFrame,
    sentiment_result: dict | None,
    upgraded_predictions: Optional[list] = None,
    geo_features: Optional[dict] = None,
) -> dict:
    """Run upgraded pipeline in shadow and compare against baseline predictions.

    Baseline output remains user-facing; this function only computes drift
    metrics for the tuning report.
    """
    from backend.sentiment_math import get_rigorous_adjustment, apply_adjustments_to_predictions
    from backend.geopolitical_features import (
        get_geopolitical_features_from_news,
        build_geopolitical_daily_adjustments,
        build_geo_interpretation,
        detect_geopolitical_shocks,
    )

    comparison: dict = {"enabled": True, "status": "ok"}

    # 1. Compute/reuse geo features
    geo_news_items, geo_news_diagnostics = _merge_geo_overlay_news(symbol, sentiment_result)
    if geo_features is None:
        geo_features = get_geopolitical_features_from_news(geo_news_items, symbol)
    comparison["geo_features"] = geo_features or {}
    news_items = geo_news_items
    if geo_news_diagnostics:
        comparison["overlay_news_diagnostics"] = geo_news_diagnostics
    shock_data = detect_geopolitical_shocks(news_items, symbol, macro_context=None) if news_items else {}
    geo_interpretation = build_geo_interpretation(
        news_items=news_items,
        geo_features=geo_features or {},
        shock_data=shock_data,
        symbol=symbol,
        crude_data=None,       # shadow comparison: no crude gate needed
        stock_health=None,
    )
    comparison["interpretation"] = geo_interpretation

    # 2. Recompute upgraded series only if caller did not provide it
    if upgraded_predictions is None:
        upgraded_predictions = baseline_predictions
        if sentiment_result:
            upgraded_adj = get_rigorous_adjustment(
                sentiment_result,
                prediction_length=len(baseline_predictions),
                frequency="daily",
            )
            current_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0
            preds_with_price = [dict(p, current_price=current_close) for p in baseline_predictions]
            upgraded_predictions = apply_adjustments_to_predictions(
                preds_with_price, upgraded_adj["adjustments"]
            )
        geo_adj_data = build_geopolitical_daily_adjustments(
            geo_features or {},
            prediction_length=len(upgraded_predictions),
            symbol=symbol,
            shock_data=shock_data,
            interpretation=geo_interpretation,
        )
        if geo_adj_data.get("adjustments"):
            current_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0
            upgraded_with_price = [dict(p, current_price=current_close) for p in upgraded_predictions]
            upgraded_predictions = apply_adjustments_to_predictions(
                upgraded_with_price, geo_adj_data["adjustments"]
            )
        comparison["adjustment_summary"] = geo_adj_data.get("summary", {})
    comparison["upgraded_count"] = len(upgraded_predictions or [])

    # 3. Compute drift metrics at key horizons
    drift_points = {}
    for day_idx, label in [(0, "day_1"), (6, "day_7"), (20, "day_21")]:
        if day_idx >= len(baseline_predictions) or day_idx >= len(upgraded_predictions):
            continue
        bp = float(baseline_predictions[day_idx].get("predicted_price", 0) or 0)
        up = float(upgraded_predictions[day_idx].get("predicted_price", 0) or 0)
        drift_pct = ((up - bp) / bp * 100.0) if bp > 0 else 0.0
        b_dir = "BULLISH" if float(baseline_predictions[day_idx].get("upside_potential", 0) or 0) > 0 else "BEARISH"
        u_dir = "BULLISH" if float(upgraded_predictions[day_idx].get("upside_potential", 0) or 0) > 0 else "BEARISH"
        drift_points[label] = {
            "baseline_price": round(bp, 2),
            "upgraded_price": round(up, 2),
            "drift_pct": round(drift_pct, 4),
            "direction_match": b_dir == u_dir,
        }

    comparison["drift"] = drift_points

    # 4. Aggregate summary
    all_drifts = [abs(d["drift_pct"]) for d in drift_points.values()]
    direction_matches = [d["direction_match"] for d in drift_points.values()]
    comparison["summary"] = {
        "median_drift_pct": round(sorted(all_drifts)[len(all_drifts) // 2], 4) if all_drifts else 0.0,
        "max_drift_pct": round(max(all_drifts), 4) if all_drifts else 0.0,
        "direction_agreement_pct": round(
            sum(direction_matches) / len(direction_matches) * 100, 1
        ) if direction_matches else 100.0,
    }

    # 5. Persist to tuning report
    try:
        report_dir = Path(__file__).parent.parent / "data" / "prediction_logs"
        report_dir.mkdir(parents=True, exist_ok=True)
        shadow_file = report_dir / "shadow_comparison_latest.json"
        import json as _json
        with open(shadow_file, "w") as f:
            _json.dump(
                {
                    "symbol": symbol,
                    "compared_at": datetime.now().isoformat(),
                    **comparison,
                },
                f,
                indent=2,
            )
    except Exception:
        pass  # non-fatal

    return comparison


# Route will be added in main.py
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for progress updates - Now with SOTA Model!"""
    await websocket.accept()
    await asyncio.sleep(0)  # Yield so other batch connections can be accepted before heavy work

    try:
        # Check if job_id exists in progress_data
        if job_id not in progress_data:
            await websocket.send_json({
                'stage': 'error',
                'progress': 0,
                'message': f'Job ID {job_id} not found. Please start analysis first.'
            })
            await websocket.close()
            return
        
        symbol = progress_data[job_id]['symbol']
        horizon_days = progress_data[job_id].get('horizon', 21)  # Get horizon, default 21
        request_geo_toggle = progress_data[job_id].get('enable_geo_features')

        # Load runtime config once per request lifecycle
        _rcfg = get_runtime_config(force_reload=True) if get_runtime_config else None

        await websocket.send_json({
            'stage': 'checking',
            'progress': 5,
            'message': f'🔍 Checking existing data for {symbol}...'
        })
        
        data_file = Path(__file__).parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
        data_exists = data_file.exists()
        needs_refresh = False
        
        # Check if data exists and is fresh (up to today)
        if data_exists:
            df_temp = load_data(symbol)
            if df_temp is not None and len(df_temp) > 0:
                last_date = pd.to_datetime(df_temp['Date'].max()).date()
                today = datetime.now().date()
                # If data is at least 1 day old, check for new data
                days_old = (today - last_date).days
                if days_old >= 1:
                    needs_refresh = True
                    await websocket.send_json({
                        'stage': 'fetching',
                        'progress': 10,
                        'message': f'🔄 Data is {days_old} day(s) old. Checking for latest data...'
                    })
            else:
                needs_refresh = True
        
        if not data_exists or needs_refresh:
            if not needs_refresh:
                await websocket.send_json({
                    'stage': 'fetching',
                    'progress': 10,
                    'message': f'📡 First-time analysis: Fetching all historical data for {symbol} from 2020 to today...'
                })
            
            # Load existing data for incremental update
            existing_data = None
            if needs_refresh and data_exists:
                try:
                    with open(data_file, 'r') as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = None
            
            async def progress_callback(update):
                await safe_send(websocket, update)

            await fetch_historical_data_async(symbol, progress_callback, existing_data=existing_data)
        
        await websocket.send_json({
            'stage': 'loading',
            'progress': 50,
            'message': '📊 Loading and preparing data...'
        })
        
        df = load_data(symbol)
        if df is None:
            await websocket.send_json({
                'stage': 'error',
                'progress': 0,
                'message': f'Failed to load data for {symbol}'
            })
            try:
                del progress_data[job_id]
            except KeyError:
                pass
            return

        # CRITICAL: Check data freshness after fetch
        # Warn user if data is still stale (PSX website may not have updated)
        last_data_date = pd.to_datetime(df['Date'].max()).date()
        today = datetime.now().date()
        days_stale = (today - last_data_date).days

        # Calculate trading days stale (skip weekends)
        trading_days_stale = 0
        check_date = last_data_date
        while check_date < today:
            check_date += pd.Timedelta(days=1)
            if check_date.weekday() < 5:  # Monday=0 to Friday=4
                trading_days_stale += 1

        if trading_days_stale > 0:
            await websocket.send_json({
                'stage': 'loading',
                'progress': 52,
                'message': f'⚠️ Data last updated: {last_data_date} ({trading_days_stale} trading day(s) behind). Predictions will start from the next uncovered trading session.'
            })

        # Request-scoped geo toggle: single source of truth for this run.
        geo_enabled = (
            bool(request_geo_toggle)
            if request_geo_toggle is not None
            else (
                bool(_rcfg.enable_geo_features) if _rcfg else (
                    os.getenv("ENABLE_GEO_FEATURES", "false").strip().lower() in {"1", "true", "yes", "on"}
                )
            )
        )
        geo_prompt_context = _build_geo_macro_prompt_context() if geo_enabled else {}

        # Try to use RESEARCH MODEL (NEW: Based on peer-reviewed PSX studies)
        reasoning = None  # Will be recomputed from adjusted day-7 output
        research_model = None
        try:
            from backend.research_model import PSXResearchModel, get_realistic_benchmarks
            USE_RESEARCH_MODEL = True
        except ImportError:
            USE_RESEARCH_MODEL = False
            
        # Fallback to SOTA model if research model not available
        if not USE_RESEARCH_MODEL:
            try:
                from backend.sota_model import SOTAEnsemblePredictor, PYWT_AVAILABLE, train_sota_model_with_progress, get_quality_score_from_sentiment
            except ImportError:
                await websocket.send_json({
                    'stage': 'error',
                    'progress': 0,
                    'message': 'Neither research_model nor sota_model available'
                })
                try:
                    del progress_data[job_id]
                except KeyError:
                    pass
                return
        
        if USE_RESEARCH_MODEL:
            await websocket.send_json({
                'stage': 'preprocessing',
                'progress': 55,
                'message': '🔬 Using Research-Backed Model (SVM + MLP, 85% PSX accuracy)...'
            })
            
            # Initialize research model
            research_model = PSXResearchModel(
                use_wavelet=True,
                symbol=symbol,
                enable_geo_context=geo_enabled,
            )
            
            await websocket.send_json({
                'stage': 'training',
                'progress': 60,
                'message': '🔬 Training research ensemble (SVM 35% + MLP 35% + GB 15% + Ridge 15%)...'
            })
            
            # Train model (includes external features + validated indicators)
            metrics = research_model.fit(df, verbose=False)
            
            # Check if accuracy is realistic
            benchmarks = get_realistic_benchmarks()
            accuracy = metrics.get('ensemble_accuracy', 0)
            
            accuracy_msg = f'Trend Accuracy: {accuracy:.1%}'
            if accuracy > benchmarks['direction_accuracy']['likely_overfit']:
                accuracy_msg += ' ⚠️ (may be overfit)'
            elif accuracy > benchmarks['direction_accuracy']['realistic_good']:
                accuracy_msg += ' ✅ (realistic range)'
            
            await websocket.send_json({
                'stage': 'training',
                'progress': 80,
                'message': f'📊 Training complete! {accuracy_msg}'
            })
            
            # Save model
            models_dir = Path(__file__).parent.parent / "data" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            research_model.save(models_dir, symbol)
            
            await websocket.send_json({
                'stage': 'predicting',
                'progress': 85,
                'message': '🔮 Generating daily predictions with confidence decay through Dec 2026...'
            })
            
            # Progress callback for prediction loop
            def prediction_progress_sync(update):
                """Sync wrapper that safely sends updates"""
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(safe_send(websocket, update))
                except Exception:
                    pass  # Don't break if progress update fails
            
            # Generate daily predictions with iterated forecasting
            # Use horizon_days for 21-day predictions, or full year if None
            if horizon_days:
                daily_predictions = research_model.predict_daily(
                    df,
                    end_date='2026-12-31',
                    max_horizon=horizon_days,
                    progress_callback=prediction_progress_sync
                )
            else:
                # Full year predictions (no horizon limit)
                daily_predictions = research_model.predict_daily(
                    df,
                    end_date='2026-12-31',
                    progress_callback=prediction_progress_sync,
                    force_full_year=True
                )
            
            # Generate prediction reasoning (explains WHY bullish/bearish)
            try:
                from backend.prediction_reasoning import generate_prediction_reasoning
                # Use preprocessed data for reasoning
                df_processed = research_model.preprocess(df)
                # Get the final predicted upside from the last prediction
                final_upside = daily_predictions[-1]['upside_potential'] if daily_predictions else None
                reasoning = generate_prediction_reasoning(df_processed, symbol=symbol, predicted_upside=final_upside)
            except Exception as e:
                print(f"⚠️ Reasoning generation failed: {e}")
                reasoning = {'error': str(e)}
            
            # Save predictions with reasoning
            pred_file = Path(__file__).parent.parent / "data" / f"{symbol}_research_predictions_2026.json"
            with open(pred_file, 'w') as f:
                import json as json_module
                json_module.dump({
                    'symbol': symbol,
                    'generated_at': datetime.now().isoformat(),
                    'model': '🔬 Research Model (SVM + MLP + External Features)',
                    'model_weights': metrics.get('weights', {}),
                    'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items() if k != 'weights'},
                    'external_features_used': True,
                    'prediction_reasoning': reasoning,  # NEW: Shows why bullish/bearish
                    'daily_predictions': daily_predictions
                }, f, indent=2)
            
            predictions = daily_predictions
            
        else:
            # Legacy SOTA model path
            await websocket.send_json({
                'stage': 'preprocessing',
                'progress': 55,
                'message': '🔬 Applying wavelet denoising (db4 DWT)...'
            })
            
            # 🆕 Get quality score from sentiment analyzer (for trend dampening)
            quality_score = 0.5  # Default neutral
            try:
                from backend.sentiment_analyzer import get_stock_sentiment
                # Quick sentiment check for quality score
                sentiment_result = get_stock_sentiment(symbol, use_cache=True)
                quality_score = sentiment_result.get('quality_score', 0.5)
                if quality_score > 0.55:
                    await websocket.send_json({
                        'stage': 'preprocessing',
                        'progress': 57,
                        'message': f'📊 Quality stock detected (score: {quality_score:.2f}) - applying trend dampening'
                    })
            except Exception as e:
                print(f"Quality score fetch error (non-fatal): {e}")
            
            # Initialize SOTA model with quality score for trend dampening
            sota_model = SOTAEnsemblePredictor(
                lookback=150, 
                horizon=21, 
                use_wavelet=PYWT_AVAILABLE,
                quality_score=quality_score
            )
            
            await websocket.send_json({
                'stage': 'training',
                'progress': 60,
                'message': '🤖 Training 6-model SOTA ensemble (RF, ET, GB, XGBoost, LightGBM, Ridge)...'
            })
            
            # Train model
            metrics = sota_model.fit(df, verbose=False)
            
            await websocket.send_json({
                'stage': 'training',
                'progress': 80,
                'message': f'📊 Training complete! Trend Accuracy: {metrics["trend_accuracy"]:.1%}, R²: {metrics["r2"]:.4f}'
            })
            
            # Save model
            models_dir = Path(__file__).parent.parent / "data" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            sota_model.save(models_dir, symbol)
            
            await websocket.send_json({
                'stage': 'predicting',
                'progress': 85,
                'message': '🔮 FORTUNE TELLER: Generating daily predictions through Dec 2026...'
            })
            
            # Progress callback for SOTA prediction loop
            def sota_prediction_progress_sync(update):
                """Sync wrapper that safely sends updates"""
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(safe_send(websocket, update))
                except Exception:
                    pass  # Don't break if progress update fails
            
            # Generate DAILY predictions through 2026 (Fortune Teller feature!)
            # Use horizon_days for 21-day predictions, or full year if None
            if horizon_days:
                daily_predictions = sota_model.predict_daily(
                    df,
                    end_date='2026-12-31',
                    max_horizon=horizon_days,
                    progress_callback=sota_prediction_progress_sync
                )
            else:
                # Full year predictions (no horizon limit)
                daily_predictions = sota_model.predict_daily(
                    df,
                    end_date='2026-12-31',
                    progress_callback=sota_prediction_progress_sync,
                    force_full_year=True
                )
            
            # Also generate monthly summaries for backward compatibility
            monthly_predictions = sota_model.predict_future(df, months_ahead=24)
            
            # Save predictions
            pred_file = Path(__file__).parent.parent / "data" / f"{symbol}_sota_predictions_2026.json"
            with open(pred_file, 'w') as f:
                import json as json_module
                json_module.dump({
                    'symbol': symbol,
                    'generated_at': datetime.now().isoformat(),
                    'model': '🔮 Fortune Teller (N-BEATS + Wavelet + Multi-Horizon Ensemble)',
                    'metrics': {k: float(v) for k, v in metrics.items()},
                    'daily_predictions': daily_predictions,  # NEW: Daily predictions
                    'predictions': monthly_predictions  # Monthly for backward compatibility
                }, f, indent=2)
            
            # Use daily predictions for the UI
            predictions = daily_predictions
        
        # Common sentiment analysis for both models
        await websocket.send_json({
            'stage': 'backtesting',
            'progress': 88,
            'message': '🔮 Analyzing news sentiment with AI...'
        })
        
        # Get sentiment analysis and apply rigorous mathematical adjustments
        sentiment_result = None
        adjusted_predictions = predictions
        sentiment_summary = {}
        adjustment_data = {
            'adjustments': [],
            'summary': {
                'events_detected': 0,
                'event_types': [],
                'max_positive_adjustment': 0.0,
                'methodology': 'No sentiment adjustment applied'
            }
        }
        
        try:
            from backend.sentiment_analyzer import get_stock_sentiment
            from backend.sentiment_math import get_rigorous_adjustment, apply_adjustments_to_predictions
            
            # Fetch and analyze news with Groq (anti-hallucination prompt)
            sentiment_result = get_stock_sentiment(
                symbol,
                use_cache=False,
                geo_mode=geo_enabled,
                geo_prompt_context=geo_prompt_context if geo_enabled else None,
            )
            
            enable_index_recall_in_model = _rcfg.enable_index_recall_in_model if _rcfg else (
                os.getenv('ENABLE_INDEX_RECALL_IN_MODEL', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
            )
            apply_sentiment_to_predictions = not (symbol in INDEX_SYMBOLS and not enable_index_recall_in_model)

            # Calculate mathematically rigorous adjustments
            if apply_sentiment_to_predictions:
                adjustment_data = get_rigorous_adjustment(
                    sentiment_result,
                    prediction_length=len(predictions),
                    frequency='daily'
                )

                # Apply adjustments to predictions. Include current_price so upside_potential
                # stays synchronized with predicted_price after sentiment transforms.
                current_close = float(df['Close'].iloc[-1])
                predictions_with_current = []
                for p in predictions:
                    row = dict(p)
                    row['current_price'] = current_close
                    predictions_with_current.append(row)
                adjusted_predictions = apply_adjustments_to_predictions(
                    predictions_with_current,
                    adjustment_data['adjustments']
                )
            else:
                adjustment_data = {
                    'adjustments': [],
                    'summary': {
                        'events_detected': 0,
                        'max_positive_adjustment': 0.0,
                        'methodology': 'Index recall model-impact disabled by flag'
                    }
                }
                adjusted_predictions = predictions
            
            sentiment_summary = {
                'signal': sentiment_result.get('signal', 'NEUTRAL'),
                'signal_emoji': sentiment_result.get('signal_emoji', '🟡'),
                'sentiment_score': sentiment_result.get('sentiment_score', 0),
                'confidence': sentiment_result.get('confidence', 0),
                'news_count': sentiment_result.get('news_count', 0),
                'retrieval_mode': sentiment_result.get('retrieval_mode', 'symbol_mode'),
                'news_fetch_diagnostics': sentiment_result.get('news_fetch_diagnostics', {}),
                'sources_attempted': sentiment_result.get('sources_attempted', 0),
                'sources_successful': sentiment_result.get('sources_successful', 0),
                'filtered_count': sentiment_result.get('filtered_count', 0),
                'geo_prompt_context_used': sentiment_result.get('geo_prompt_context_used', False),
                'events_detected': adjustment_data['summary']['events_detected'],
                'detected_event_types': adjustment_data['summary'].get('event_types', []),
                'max_adjustment': adjustment_data['summary']['max_positive_adjustment'],
                'methodology': 'Research-backed event study with exponential decay',
                'model_impact_enabled': apply_sentiment_to_predictions,
                # Claude's analysis text
                'summary': sentiment_result.get('summary', ''),
                'key_events': sentiment_result.get('key_events', []),
                'risks': sentiment_result.get('risks', []),
                'catalysts': sentiment_result.get('catalysts', []),
                'price_impact': sentiment_result.get('price_impact', {}),
                # Recent news headlines
                'recent_news': sentiment_result.get('news_items', [])[:5]
            }
            
            await websocket.send_json({
                'stage': 'backtesting',
                'progress': 92,
                'message': f'📰 Found {sentiment_result.get("news_count", 0)} news items | {sentiment_summary["signal_emoji"]} {sentiment_summary["signal"]}'
            })
            
        except Exception as e:
            print(f"Sentiment analysis error (non-fatal): {e}")
            adjusted_predictions = predictions
            sentiment_summary = {
                'signal': 'NEUTRAL',
                'signal_emoji': '🟡',
                'error': str(e),
                'detected_event_types': [],
            }

        # Apply lightweight, globally-configurable prediction quality tweaks
        live_tweak_config = get_live_tweak_config()
        tuning_meta = {"enabled": bool(getattr(live_tweak_config, "enabled", False))}
        try:
            if adjusted_predictions and tuning_meta["enabled"]:
                adjusted_predictions = apply_prediction_tweaks(adjusted_predictions, live_tweak_config)
                tuning_meta.update(
                    {
                        "neutral_band_pct": float(getattr(live_tweak_config, "neutral_band_pct", 0.0)),
                        "applied_to_days": len(adjusted_predictions),
                    }
                )
                await websocket.send_json({
                    'stage': 'forecasting',
                    'progress': 94,
                    'message': f'🛠️ Applied prediction tuning ({len(adjusted_predictions)} days)'
                })
        except Exception as e:
            print(f"Prediction tuning skipped (non-fatal): {e}")
            tuning_meta = {"enabled": False, "error": str(e)}

        predictions_without_geo = adjusted_predictions
        predictions_with_geo = []
        geo_comparison = {
            "enabled": geo_enabled,
            "applied": False,
            "shock_reason": "No geopolitical shock detected" if geo_enabled else "Geo overlay disabled for this run.",
            "macro_confirmation": {
                "crude": geo_prompt_context.get("crude", {}) if geo_prompt_context else {},
            },
            "stock_health": {},
            "interpretation": {
                "sector_interpretation": "generic",
                "polarity": "neutral" if geo_enabled else "disabled",
                "bullish_for_upstream": False,
                "bearish_for_downstream": False,
                "tailwind_score": 0.0,
                "cashflow_support_score": 0.0,
                "shock_severity_score": 0.0,
                "fuel_hike_magnitude_rs": 0.0,
                "fuel_hike_score": 0.0,
                "circular_debt_relief_score": 0.0,
                "projected_inflation_spike": 0.0,
                "interest_rate_squeeze_score": 0.0,
                "margin_compression_score": 0.0,
                "demand_destruction_score": 0.0,
                "headwind_strength": 0.0,
                "vulnerability_profile": {},
                "evidence_quality": "weak" if geo_enabled else "disabled",
                "reason": (
                    "No sector-specific geopolitical interpretation applied."
                    if geo_enabled
                    else "Geo overlay disabled for this run."
                ),
            },
            "labels": {
                "baseline": "Without Geo Features",
                "geo": "With Geo Features",
            },
        }
        recovery_analysis = {}
        if geo_comparison["enabled"]:
            try:
                from backend.sentiment_math import apply_adjustments_to_predictions
                from backend.geopolitical_features import (
                    get_geopolitical_features_from_news,
                    build_geopolitical_daily_adjustments,
                    build_geo_interpretation,
                    detect_geopolitical_shocks,
                )
                news_items, geo_news_diagnostics = _merge_geo_overlay_news(symbol, sentiment_result)
                geo_features = get_geopolitical_features_from_news(news_items, symbol)

                # Asian market risk-off signal — boost global_risk_off when Asian markets crash
                try:
                    from backend.external_features import fetch_asian_market_realtime
                    asian_status = fetch_asian_market_realtime()
                    asian_signal = asian_status.get("asian_risk_off_signal", 0.0)
                    if asian_signal > 0:
                        # Boost global risk-off score based on Asian market crash severity
                        risk_off_boost = asian_signal * 0.4  # 0.2 for moderate, 0.4 for severe
                        geo_features["geo_global_risk_off"] = min(1.0,
                            geo_features.get("geo_global_risk_off", 0.0) + risk_off_boost)
                        # Sector-specific amplifiers
                        energy_symbols = ['OGDC', 'PPL', 'PSO', 'POL', 'MARI', 'ATRL']
                        banking_symbols = ['HBL', 'MCB', 'UBL', 'BAHL', 'NBP', 'ABL', 'MEBL']
                        if symbol.upper() in energy_symbols:
                            geo_features["geo_energy_supply_risk"] = min(1.0,
                                geo_features.get("geo_energy_supply_risk", 0.0) + asian_signal * 0.3)
                        elif symbol.upper() in banking_symbols:
                            geo_features["geo_global_risk_off"] = min(1.0,
                                geo_features.get("geo_global_risk_off", 0.0) + asian_signal * 0.2)
                        # Add interpretation context
                        severity = asian_status.get("crash_severity", "none")
                        if severity != "none":
                            nk_pct = asian_status.get("nikkei", {}).get("change_pct", 0)
                            ks_pct = asian_status.get("kospi", {}).get("change_pct", 0)
                            news_items.append({
                                "title": f"Asian Market Crash Signal: Nikkei {nk_pct:+.1f}%, KOSPI {ks_pct:+.1f}%",
                                "source": "Asian Markets Monitor",
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "sentiment_score": -0.8 if severity == "severe" else -0.5,
                            })
                except Exception:
                    asian_signal = 0.0

                # Shock detection – emergency multiplier for extreme events
                shock_data = detect_geopolitical_shocks(news_items, symbol, macro_context=geo_prompt_context)

                # A5/A6: compute crude confirmation and stock health for geo gate
                _crude_for_geo = geo_prompt_context.get("crude", {}) if geo_prompt_context else {}
                _stock_health: Optional[dict] = None
                if df is not None and len(df) >= 20 and "Close" in df.columns:
                    _closes = df["Close"].values
                    _cur = float(_closes[-1])
                    _p20 = float(_closes[-20]) if len(_closes) >= 20 else _cur
                    _p60 = float(_closes[-60]) if len(_closes) >= 60 else _cur
                    _stock_health = {
                        "momentum_20d": ((_cur / _p20) - 1.0) * 100.0 if _p20 > 0 else 0.0,
                        "momentum_60d": ((_cur / _p60) - 1.0) * 100.0 if _p60 > 0 else 0.0,
                    }

                geo_interpretation = build_geo_interpretation(
                    news_items=news_items,
                    geo_features=geo_features,
                    shock_data=shock_data,
                    symbol=symbol,
                    crude_data=_crude_for_geo,
                    stock_health=_stock_health,
                    asian_market_data=asian_status if asian_signal > 0 else None,
                )

                geo_adjustment_data = build_geopolitical_daily_adjustments(
                    geo_features,
                    prediction_length=len(predictions_without_geo),
                    symbol=symbol,
                    shock_data=shock_data,
                    interpretation=geo_interpretation,
                    enable_ai_blend=geo_enabled,
                    stock_health=_stock_health,
                    asian_market_data=asian_status if asian_signal > 0 else None,
                )

                current_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
                baseline_with_current = [dict(p, current_price=current_close) for p in predictions_without_geo]
                predictions_with_geo = apply_adjustments_to_predictions(
                    baseline_with_current,
                    geo_adjustment_data.get("adjustments", []),
                )
                # ----- ASIAN MARKET SIGNAL AMPLIFICATION (bidirectional) -----
                # Asian markets open 4-5 hours before PSX and are highly predictive.
                # This is NOT just crash protection — it amplifies in BOTH directions.
                if predictions_with_geo and asian_signal > 0:
                    nk_pct = asian_status.get("nikkei", {}).get("change_pct", 0) or 0
                    ks_pct = asian_status.get("kospi", {}).get("change_pct", 0) or 0
                    asian_avg = (nk_pct + ks_pct) / 2.0
                    severity = asian_status.get("crash_severity", "none")

                    if severity == "severe":
                        # Both Asian markets down 2%+: slash bullish confidence by 50%,
                        # reduce geo adjustment magnitude by 60%
                        for pred in predictions_with_geo:
                            if isinstance(pred, dict):
                                if pred.get("direction", "").lower() in ("up", "bullish"):
                                    if "confidence" in pred:
                                        pred["confidence"] = round(pred["confidence"] * 0.50, 2)
                                if pred.get("sentiment_adjustment_pct", 0) > 0:
                                    pred["sentiment_adjustment_pct"] = round(
                                        pred["sentiment_adjustment_pct"] * 0.40, 4)
                                    if "base_price" in pred and pred["base_price"] > 0:
                                        pred["predicted_price"] = round(
                                            pred["base_price"] * (1 + pred["sentiment_adjustment_pct"] / 100.0), 2)

                    elif severity == "moderate":
                        # One Asian market down 2%+: slash bullish confidence by 30%,
                        # reduce geo adjustment magnitude by 40%
                        for pred in predictions_with_geo:
                            if isinstance(pred, dict):
                                if pred.get("direction", "").lower() in ("up", "bullish"):
                                    if "confidence" in pred:
                                        pred["confidence"] = round(pred["confidence"] * 0.70, 2)
                                if pred.get("sentiment_adjustment_pct", 0) > 0:
                                    pred["sentiment_adjustment_pct"] = round(
                                        pred["sentiment_adjustment_pct"] * 0.60, 4)
                                    if "base_price" in pred and pred["base_price"] > 0:
                                        pred["predicted_price"] = round(
                                            pred["base_price"] * (1 + pred["sentiment_adjustment_pct"] / 100.0), 2)

                    elif severity == "mild":
                        # Asian markets under mild pressure: 15% confidence cut on bullish
                        for pred in predictions_with_geo:
                            if isinstance(pred, dict):
                                if pred.get("direction", "").lower() in ("up", "bullish"):
                                    if "confidence" in pred:
                                        pred["confidence"] = round(pred["confidence"] * 0.85, 2)

                    elif asian_avg > 1.0 and nk_pct > 0.5 and ks_pct > 0.5:
                        # BULLISH AMPLIFICATION: Both Asian markets up >0.5%, avg >1%
                        # Boost bullish confidence by 15%, amplify positive adjustments by 20%
                        for pred in predictions_with_geo:
                            if isinstance(pred, dict):
                                if pred.get("direction", "").lower() in ("up", "bullish"):
                                    if "confidence" in pred:
                                        pred["confidence"] = round(min(0.99, pred["confidence"] * 1.15), 2)
                                if pred.get("sentiment_adjustment_pct", 0) > 0:
                                    pred["sentiment_adjustment_pct"] = round(
                                        pred["sentiment_adjustment_pct"] * 1.20, 4)
                                    if "base_price" in pred and pred["base_price"] > 0:
                                        pred["predicted_price"] = round(
                                            pred["base_price"] * (1 + pred["sentiment_adjustment_pct"] / 100.0), 2)

                    # Recalculate upside_potential after price adjustments
                    for pred in predictions_with_geo:
                        if isinstance(pred, dict) and pred.get("current_price", 0) > 0 and "predicted_price" in pred:
                            pred["upside_potential"] = round(
                                (pred["predicted_price"] / pred["current_price"] - 1) * 100, 2)

                # Build human-readable geo reasoning from LLM + crude data + interpretation
                geo_reasoning_parts = []

                # From LLM trajectory assessment (ticker-specific impact)
                _trajectory_data = (shock_data.get("trajectory") or {}) if isinstance(shock_data.get("trajectory"), dict) else {}
                _llm_data = _trajectory_data.get("llm_assessment") or {}
                if _llm_data.get("ticker_impact_summary"):
                    geo_reasoning_parts.append(_llm_data["ticker_impact_summary"])
                elif _llm_data.get("reasoning"):
                    # Truncate reasoning for display; build impact hint from severity + market_impact
                    _reasoning = _llm_data["reasoning"][:250]
                    _impact_pct = _llm_data.get("market_impact_pct", 0)
                    _severity = _llm_data.get("severity", 0)
                    if _impact_pct and _severity:
                        _impact_dir = "BULLISH" if _impact_pct > 0 else "BEARISH"
                        _reasoning = f"[Severity {_severity}/10, {_impact_dir} {abs(_impact_pct):.1f}%] {_reasoning}"
                    geo_reasoning_parts.append(_reasoning)

                # From crude oil data
                _crude_price = (_crude_for_geo or {}).get("oil_close")
                _crude_change = (_crude_for_geo or {}).get("oil_change_pct")
                if _crude_price and _crude_change:
                    _dir = "up" if _crude_change > 0 else "down"
                    geo_reasoning_parts.append(
                        f"Crude oil at ${_crude_price}/bbl ({_crude_change:+.1f}% {_dir})"
                    )

                # From geo interpretation
                if geo_interpretation.get("reason"):
                    geo_reasoning_parts.append(geo_interpretation["reason"])

                geo_reasoning = " | ".join(geo_reasoning_parts) if geo_reasoning_parts else ""

                geo_comparison.update(
                    {
                        "applied": bool(predictions_with_geo),
                        "geo_features": geo_features,
                        "adjustment_summary": geo_adjustment_data.get("summary", {}),
                        "shock_data": shock_data,
                        "shock_reason": shock_data.get("shock_reason", "No geopolitical shock detected"),
                        "interpretation": geo_interpretation,
                        "overlay_news_diagnostics": geo_news_diagnostics,
                        "asian_market_signal": asian_signal,
                        "macro_confirmation": {
                            "crude": _crude_for_geo,
                        },
                        "stock_health": _stock_health or {},
                        "geo_reasoning": geo_reasoning,
                        "geo_reasoning_parts": geo_reasoning_parts,
                    }
                )

                # Recovery analysis payload (always return a structured status)
                try:
                    from backend.recovery_predictor import get_recovery_analysis
                    recent_high = float(df["Close"].max()) if "Close" in df.columns else current_close
                    recovery_analysis = get_recovery_analysis(
                        symbol=symbol,
                        current_price=current_close,
                        recent_high=recent_high,
                        geo_shock_data=shock_data,
                    )
                    if shock_data.get("shock_detected"):
                        await websocket.send_json({
                            'stage': 'forecasting',
                            'progress': 91,
                            'message': f'🚨 Geopolitical shock detected ({shock_data["max_severity"]:.1f} severity) – generating recovery scenarios'
                        })
                except Exception as rec_err:
                    recovery_analysis = {"enabled": False, "error": str(rec_err)[:200]}
            except Exception as geo_err:
                predictions_with_geo = []
                geo_comparison.update(
                    {
                        "applied": False,
                        "error": str(geo_err)[:200],
                    }
                )

        # 🆕 Generate detailed monthly forecasts with news correlation
        monthly_forecast = []
        forecast_summary = {}
        try:
            from backend.monthly_forecast import generate_monthly_forecast, generate_forecast_summary
            monthly_forecast = generate_monthly_forecast(
                predictions_without_geo,
                sentiment_result,
                df,
                symbol
            )
            forecast_summary = generate_forecast_summary(monthly_forecast)
            
            await websocket.send_json({
                'stage': 'forecasting',
                'progress': 93,
                'message': f'📅 Generated {len(monthly_forecast)} monthly forecasts with {forecast_summary.get("bullish_months", 0)} bullish, {forecast_summary.get("bearish_months", 0)} bearish months'
            })
        except Exception as e:
            print(f"Monthly forecast generation error (non-fatal): {e}")
            monthly_forecast = []
            forecast_summary = {'error': str(e)}
        
        await websocket.send_json({
            'stage': 'finalizing',
            'progress': 95,
            'message': '💰 Calculating final metrics...'
        })
        
        # Calculate backtest metrics from adjusted predictions
        if len(predictions_without_geo) > 0:
            initial_price = df['Close'].iloc[-1]
            final_price = predictions_without_geo[-1]['predicted_price']
            total_return = (final_price - initial_price) / initial_price * 100
        else:
            total_return = 0
        
        # Prepare historical data for charting (last 180 days)
        history_df = df.tail(180)[['Date', 'Close']].copy()
        if not pd.api.types.is_string_dtype(history_df['Date']):
            history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d')
        historical_data = history_df.to_dict('records')

        # Get metrics for response (handle both research and SOTA model formats)
        r2_val = metrics.get('r2', metrics.get('ensemble_accuracy', 0))
        trend_acc = metrics.get('trend_accuracy', metrics.get('ensemble_accuracy', 0))
        mase_val = metrics.get('mase', 0)
        mape_val = metrics.get('mape', 0)

        model_variant = _rcfg.model_variant if _rcfg else os.getenv('MODEL_VARIANT', 'baseline').strip().lower()
        if model_variant not in {'baseline', 'shadow', 'upgraded'}:
            model_variant = 'baseline'

        display_predictions = (
            predictions_with_geo
            if geo_comparison.get("enabled") and geo_comparison.get("applied") and predictions_with_geo
            else predictions_without_geo
        )
        direction_meta = {}
        if display_predictions:
            pivot_idx = 6 if len(display_predictions) >= 7 else len(display_predictions) - 1
            pivot_pred = display_predictions[pivot_idx]
            display_upside_pct = float(pivot_pred.get('upside_potential', 0) or 0)
            display_direction = direction_from_change_pct(
                display_upside_pct,
                neutral_band_pct=float(getattr(live_tweak_config, "neutral_band_pct", 0.0))
            )
            raw_direction = direction_from_change_pct(
                float(pivot_pred.get('upside_potential', 0) or 0),
                neutral_band_pct=float(getattr(live_tweak_config, "neutral_band_pct", 0.0))
            )
            stable_direction = pivot_pred.get('stable_direction', raw_direction)
            logged_direction_source = _rcfg.logged_direction_source if _rcfg else os.getenv('LOGGED_DIRECTION_SOURCE', 'stable').strip().lower()
            if logged_direction_source not in {'stable', 'raw'}:
                logged_direction_source = 'stable'
            logged_direction = stable_direction if logged_direction_source == 'stable' else raw_direction
            stability_note = (
                f"Display direction follows adjusted day-7 upside of {display_upside_pct:+.2f}%, "
                f"while stability state remains {stable_direction} for continuity logging."
                if display_direction != stable_direction
                else ""
            )
            # B1: compute near-term vs day-7 direction and path shape
            near_term_upsides = [
                float(p.get('upside_potential', 0) or 0)
                for p in display_predictions[:min(7, len(display_predictions))]
            ]
            near_term_avg = sum(near_term_upsides) / len(near_term_upsides) if near_term_upsides else 0.0
            _ntband = float(getattr(live_tweak_config, "neutral_band_pct", 0.0))
            near_term_direction = direction_from_change_pct(near_term_avg, neutral_band_pct=_ntband)
            day7_direction = display_direction

            # Path shape: compare near-term trend with later trend
            if len(display_predictions) >= 14:
                later_upsides = [
                    float(p.get('upside_potential', 0) or 0)
                    for p in display_predictions[7:14]
                ]
                later_avg = sum(later_upsides) / len(later_upsides) if later_upsides else 0.0
            else:
                later_avg = near_term_avg

            if near_term_avg < -1.0 and later_avg > 1.0:
                path_shape = "near_term_drop_then_recover"
            elif near_term_avg > 1.0 and later_avg < -1.0:
                path_shape = "near_term_rise_then_decline"
            elif near_term_avg < -1.0 and later_avg < -1.0:
                path_shape = "steady_decline"
            elif near_term_avg > 1.0 and later_avg > 1.0:
                path_shape = "steady_rise"
            else:
                path_shape = "volatile_flat"

            direction_meta = {
                'raw_direction': raw_direction,
                'stable_direction': stable_direction,
                'logged_direction': logged_direction,
                'logged_direction_source': logged_direction_source,
                'display_direction': display_direction,
                'display_upside_pct': round(display_upside_pct, 2),
                'stability_note': stability_note,
                'near_term_direction': near_term_direction,
                'day7_direction': day7_direction,
                'path_shape': path_shape,
            }
        else:
            direction_meta = {
                'raw_direction': 'NEUTRAL',
                'stable_direction': 'NEUTRAL',
                'logged_direction': 'NEUTRAL',
                'logged_direction_source': 'stable',
                'display_direction': 'NEUTRAL',
                'display_upside_pct': 0.0,
                'stability_note': '',
                'near_term_direction': 'NEUTRAL',
                'day7_direction': 'NEUTRAL',
                'path_shape': 'volatile_flat',
            }

        analysis_generated_at = datetime.now()
        analysis_id = job_id
        current_price = float(df['Close'].iloc[-1]) if 'Close' in df.columns else 0.0
        forecast_postmortem = build_forecast_postmortem(
            symbol=symbol,
            current_price=current_price,
            baseline_predictions=predictions_without_geo,
            geo_predictions=(
                predictions_with_geo
                if geo_comparison.get("enabled") and predictions_with_geo
                else (predictions_without_geo if geo_comparison.get("enabled") else [])
            ),
            geo_comparison=geo_comparison,
            prediction_generated_at=analysis_generated_at,
        )

        try:
            from backend.prediction_reasoning import generate_prediction_reasoning
            if USE_RESEARCH_MODEL and research_model is not None:
                reasoning_df = research_model.preprocess(df)
            else:
                from backend.external_features import merge_external_features, is_oil_sector_symbol
                reasoning_df = merge_external_features(
                    df.copy(),
                    symbol=symbol,
                    include_asian_features=geo_enabled,
                    include_oil_features=(geo_enabled or not is_oil_sector_symbol(symbol)),
                )

            reasoning = generate_prediction_reasoning(
                reasoning_df,
                symbol=symbol,
                predicted_upside=direction_meta.get('display_upside_pct', 0.0),
                direction_override=direction_meta.get('display_direction', 'NEUTRAL'),
                apply_stability=False,
                neutral_band_pct=float(getattr(live_tweak_config, "neutral_band_pct", 0.0)),
                horizon_label='Day 7',
            )
        except Exception as e:
            print(f"⚠️ Adjusted reasoning generation failed: {e}")
            reasoning = {'error': str(e)}

        shadow_comparison = {}
        if model_variant == 'shadow':
            # Shadow mode: run upgraded pipeline in parallel, store comparison
            try:
                shadow_comparison = _run_shadow_comparison(
                    symbol,
                    predictions_without_geo,
                    df,
                    sentiment_result,
                    upgraded_predictions=predictions_with_geo if geo_comparison.get("applied") else None,
                    geo_features=geo_comparison.get("geo_features"),
                )
            except Exception as shadow_err:
                shadow_comparison = {
                    'enabled': True,
                    'status': 'error',
                    'error': str(shadow_err)[:200],
                }
        
        await websocket.send_json({
            'stage': 'complete',
            'progress': 100,
            'message': '✅ Research-Backed Analysis Complete with AI Sentiment!',
            'results': {
                'analysis_id': analysis_id,
                'symbol': symbol,
                'model': 'Research Model (SVM + MLP + External Features)' if USE_RESEARCH_MODEL else 'SOTA Ensemble + AI Sentiment',
                'model_variant': model_variant,
                'model_performance': {
                    'r2': float(r2_val),
                    'trend_accuracy': float(trend_acc),
                    'mase': float(mase_val),
                    'mape': float(mape_val)
                },
                'direction_meta': direction_meta,
                'near_term_direction': direction_meta.get('near_term_direction', 'NEUTRAL'),
                'day7_direction': direction_meta.get('day7_direction', 'NEUTRAL'),
                'path_shape': direction_meta.get('path_shape', 'volatile_flat'),
                'forecast_postmortem': forecast_postmortem,
                'sentiment': sentiment_summary,
                'monthly_predictions': predictions_without_geo[:12],  # First 12 months
                'daily_predictions': predictions_without_geo, # Backward-compatible baseline predictions
                'daily_predictions_without_geo': predictions_without_geo,
                'daily_predictions_with_geo': predictions_with_geo,
                'geo_comparison': geo_comparison,
                'recovery_analysis': recovery_analysis,
                'monthly_forecast': monthly_forecast,  # 🆕 Detailed monthly analysis with reasoning
                'forecast_summary': forecast_summary,  # 🆕 Overall forecast summary
                'historical_data': historical_data, # History for charting
                'all_predictions_count': len(predictions_without_geo),
                'backtest': {
                    'total_return': total_return,
                    'prediction_horizon': '24 months to end of 2026'
                },
                'current_price': current_price,
                'data_points': len(df),
                'features_used': len(metrics.get('weights', {})) if USE_RESEARCH_MODEL else 74,
                'external_features_used': USE_RESEARCH_MODEL,
                'prediction_reasoning': reasoning,
                'tuning': tuning_meta,
                'shadow_comparison': shadow_comparison,
            }
        })
        
        # 🆕 Save complete analysis with monthly forecast to JSON
        try:
            complete_analysis_file = Path(__file__).parent.parent / "data" / f"{symbol}_complete_analysis.json"
            import json as json_module
            with open(complete_analysis_file, 'w') as cf:
                json_module.dump({
                    'analysis_id': analysis_id,
                    'symbol': symbol,
                    'generated_at': analysis_generated_at.isoformat(),
                    'model': 'Research Model (SVM + MLP + External Features)' if USE_RESEARCH_MODEL else 'SOTA Ensemble',
                    'model_variant': model_variant,
                    'current_price': current_price,
                    'direction_meta': direction_meta,
                    'near_term_direction': direction_meta.get('near_term_direction', 'NEUTRAL'),
                    'day7_direction': direction_meta.get('day7_direction', 'NEUTRAL'),
                    'path_shape': direction_meta.get('path_shape', 'volatile_flat'),
                    'forecast_postmortem': forecast_postmortem,
                    'sentiment': sentiment_summary,
                    'monthly_forecast': monthly_forecast,  # Detailed monthly analysis
                    'forecast_summary': forecast_summary,  # Overall outlook
                    'prediction_reasoning': reasoning,
                    'daily_predictions': predictions_without_geo,
                    'daily_predictions_without_geo': predictions_without_geo,
                    'daily_predictions_with_geo': predictions_with_geo,
                    'geo_comparison': geo_comparison,
                    'recovery_analysis': recovery_analysis,
                    'daily_predictions_count': len(predictions_without_geo),
                    'tuning': tuning_meta,
                    'shadow_comparison': shadow_comparison,
                }, cf, indent=2)
            print(f"Complete analysis saved to {complete_analysis_file}")
        except Exception as e:
            print(f"WARNING: Failed to save complete analysis: {e}")

        # Log prediction for accuracy tracking
        try:
            from backend.prediction_logger import get_prediction_logger
            logger = get_prediction_logger()
            _log_prediction_variants(
                logger,
                symbol=symbol,
                current_price=current_price,
                baseline_predictions=predictions_without_geo,
                geo_predictions=predictions_with_geo,
                analysis_id=analysis_id,
                prediction_generated_at=analysis_generated_at,
                neutral_band_pct=float(getattr(live_tweak_config, "neutral_band_pct", 0.0)),
                include_geo_variant=bool(geo_comparison.get("enabled")),
            )
            logger.backfill_actuals(symbol=symbol, limit=32)
        except Exception as e:
            print(f"WARNING: Prediction logging skipped: {e}")

        try:
            del progress_data[job_id]
        except KeyError:
            pass
        
    except Exception as e:
        await websocket.send_json({
            'stage': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        })
        import traceback
        traceback.print_exc()
        try:
            del progress_data[job_id]
        except KeyError:
            pass

# This module is imported by main.py, not run standalone
