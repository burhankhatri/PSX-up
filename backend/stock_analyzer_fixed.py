#!/usr/bin/env python3
"""
Backend API for Stock Analysis - Complete & Fixed
Handles data fetching, model training, and progress updates via WebSocket
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import subprocess
import re
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import asyncio
from typing import Union, Optional
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
    )

    comparison: dict = {"enabled": True, "status": "ok"}

    # 1. Compute/reuse geo features
    if geo_features is None:
        news_items = (sentiment_result or {}).get("news_items", [])
        geo_features = get_geopolitical_features_from_news(news_items, symbol)
    comparison["geo_features"] = geo_features or {}

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
                'message': f'⚠️ Data last updated: {last_data_date} ({trading_days_stale} trading day(s) behind). Predictions will start from next trading day after today.'
            })

        # Try to use RESEARCH MODEL (NEW: Based on peer-reviewed PSX studies)
        reasoning = None  # Will be populated if research model is used
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
            research_model = PSXResearchModel(use_wavelet=True, symbol=symbol)
            
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
        
        try:
            from backend.sentiment_analyzer import get_stock_sentiment
            from backend.sentiment_math import get_rigorous_adjustment, apply_adjustments_to_predictions
            
            # Fetch and analyze news with Groq (anti-hallucination prompt)
            sentiment_result = get_stock_sentiment(symbol, use_cache=False)
            
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
                'events_detected': adjustment_data['summary']['events_detected'],
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
            sentiment_summary = {'signal': 'NEUTRAL', 'signal_emoji': '🟡', 'error': str(e)}

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
        geo_enabled = (
            bool(request_geo_toggle)
            if request_geo_toggle is not None
            else (
                bool(_rcfg.enable_geo_features) if _rcfg else (
                    os.getenv("ENABLE_GEO_FEATURES", "false").strip().lower() in {"1", "true", "yes", "on"}
                )
            )
        )
        geo_comparison = {
            "enabled": geo_enabled,
            "applied": False,
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
                    detect_geopolitical_shocks,
                )
                news_items = (sentiment_result or {}).get("news_items", [])
                geo_features = get_geopolitical_features_from_news(news_items, symbol)

                # Shock detection – emergency multiplier for extreme events
                shock_data = detect_geopolitical_shocks(news_items, symbol)

                geo_adjustment_data = build_geopolitical_daily_adjustments(
                    geo_features,
                    prediction_length=len(predictions_without_geo),
                    symbol=symbol,
                    shock_data=shock_data,
                )

                current_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
                baseline_with_current = [dict(p, current_price=current_close) for p in predictions_without_geo]
                predictions_with_geo = apply_adjustments_to_predictions(
                    baseline_with_current,
                    geo_adjustment_data.get("adjustments", []),
                )
                geo_comparison.update(
                    {
                        "applied": bool(predictions_with_geo),
                        "geo_features": geo_features,
                        "adjustment_summary": geo_adjustment_data.get("summary", {}),
                        "shock_data": shock_data,
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

        direction_meta = {}
        if predictions_without_geo:
            # Use day-7 as canonical logged horizon (same as logger)
            pivot_idx = 6 if len(predictions_without_geo) >= 7 else len(predictions_without_geo) - 1
            pivot_pred = predictions_without_geo[pivot_idx]
            raw_direction = direction_from_change_pct(
                float(pivot_pred.get('upside_potential', 0) or 0),
                neutral_band_pct=float(getattr(live_tweak_config, "neutral_band_pct", 0.0))
            )
            stable_direction = pivot_pred.get('stable_direction', raw_direction)
            logged_direction_source = _rcfg.logged_direction_source if _rcfg else os.getenv('LOGGED_DIRECTION_SOURCE', 'stable').strip().lower()
            if logged_direction_source not in {'stable', 'raw'}:
                logged_direction_source = 'stable'
            logged_direction = stable_direction if logged_direction_source == 'stable' else raw_direction
            direction_meta = {
                'raw_direction': raw_direction,
                'stable_direction': stable_direction,
                'logged_direction': logged_direction,
                'logged_direction_source': logged_direction_source,
            }
        else:
            direction_meta = {
                'raw_direction': 'NEUTRAL',
                'stable_direction': 'NEUTRAL',
                'logged_direction': 'NEUTRAL',
                'logged_direction_source': 'stable',
            }

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
                'current_price': float(df['Close'].iloc[-1]),
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
                    'symbol': symbol,
                    'generated_at': datetime.now().isoformat(),
                    'model': 'Research Model (SVM + MLP + External Features)' if USE_RESEARCH_MODEL else 'SOTA Ensemble',
                    'model_variant': model_variant,
                    'current_price': float(df['Close'].iloc[-1]),
                    'direction_meta': direction_meta,
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

            # Log the 7-day prediction for tracking
            if len(predictions_without_geo) >= 7:
                pred_7d = predictions_without_geo[6]  # Day 7 (index 6)
                current_price = float(df['Close'].iloc[-1])

                # Extract Williams signal and sector if available
                williams_signal = pred_7d.get('williams_signal')
                sector = pred_7d.get('sector')

                raw_direction = direction_from_change_pct(
                    float(pred_7d.get('upside_potential', 0) or 0),
                    neutral_band_pct=float(getattr(live_tweak_config, "neutral_band_pct", 0.0))
                )
                stable_direction = pred_7d.get('stable_direction', raw_direction)
                logged_direction_source = os.getenv('LOGGED_DIRECTION_SOURCE', 'stable').strip().lower()
                if logged_direction_source not in {'stable', 'raw'}:
                    logged_direction_source = 'stable'
                logged_direction = stable_direction if logged_direction_source == 'stable' else raw_direction

                logger.log_prediction(
                    symbol=symbol,
                    current_price=current_price,
                    predicted_price=pred_7d['predicted_price'],
                    predicted_direction=logged_direction,
                    confidence=pred_7d.get('confidence', 0.5),
                    horizon_days=7,
                    williams_signal=williams_signal,
                    sector=sector
                )
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
