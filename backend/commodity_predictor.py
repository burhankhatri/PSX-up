#!/usr/bin/env python3
"""
🥈🥇 COMMODITY PREDICTOR - Silver & Gold Price Prediction
Incorporates AI/GPU demand, industrial factors, and macro indicators.

Key Factors for Silver:
- AI/GPU Demand: Semiconductor index, NVDA as proxy
- Solar Panel Demand: TAN ETF
- EV Demand: More silver in EVs than traditional vehicles
- USD Strength: Inverse correlation
- Fed Rates/Inflation: 10Y Treasury, VIX

Key Factors for Gold:
- Safe-haven demand (VIX, geopolitical)
- USD Strength: Strong inverse correlation
- Fed Rates: Higher rates → lower gold
- Inflation expectations

Reference: Silver Institute, WisdomTree (2024-2025)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Data fetching
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Run: pip install yfinance")

# ML imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

COMMODITY_CONFIG = {
    'silver': {
        'ticker': 'SI=F',
        'name': 'Silver',
        'unit': 'oz',
        'color': '#C0C0C0',  # Silver color
        'emoji': '🥈',
        # Silver is heavily influenced by industrial demand
        'industrial_weight': 0.55,
        'safe_haven_weight': 0.25,
        'macro_weight': 0.20,
    },
    'gold': {
        'ticker': 'GC=F',
        'name': 'Gold',
        'unit': 'oz',
        'color': '#FFD700',  # Gold color
        'emoji': '🥇',
        # Gold is more safe-haven focused
        'industrial_weight': 0.15,
        'safe_haven_weight': 0.55,
        'macro_weight': 0.30,
    },
}

# External indicators that affect precious metals
INDUSTRIAL_INDICATORS = {
    'semiconductor': '^SOX',      # Philadelphia Semiconductor Index (AI/GPU demand)
    'nvidia': 'NVDA',             # NVIDIA - GPU demand proxy
    'solar': 'TAN',               # Invesco Solar ETF
    'ev_maker': 'TSLA',           # Tesla - EV demand proxy
}

MACRO_INDICATORS = {
    'usd_index': 'DX-Y.NYB',      # US Dollar Index
    'treasury_10y': '^TNX',       # 10-Year Treasury Yield
    'vix': '^VIX',                # Volatility Index (fear gauge)
    'sp500': '^GSPC',             # S&P 500 (risk sentiment)
}

# For PKR conversion
USD_PKR_TICKER = 'PKR=X'

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "commodity_cache"

# Gold-API.com - Free real-time API (no key required)
GOLD_API_BASE = "https://api.gold-api.com"
GOLD_API_SYMBOLS = {
    'silver': 'XAG',
    'gold': 'XAU'
}


# ============================================================================
# GOLD-API.COM DATA SOURCE (Primary for real-time)
# ============================================================================

def fetch_realtime_from_gold_api(commodity: str) -> Dict:
    """
    Fetch real-time price from gold-api.com (FREE, no key required).
    Returns: {price, name, symbol, updatedAt}
    """
    import subprocess
    import json
    
    symbol = GOLD_API_SYMBOLS.get(commodity.lower())
    if not symbol:
        return {'error': f'Unknown commodity: {commodity}'}
    
    url = f"{GOLD_API_BASE}/price/{symbol}"
    
    try:
        result = subprocess.run(
            ['curl', '-s', url],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'price' in data:
                print(f"  ✅ gold-api.com: {data['name']} = ${data['price']:.2f}")
                return {
                    'price': data['price'],
                    'name': data['name'],
                    'symbol': data['symbol'],
                    'updated_at': data.get('updatedAt'),
                    'source': 'gold-api.com'
                }
            else:
                return {'error': 'No price in response'}
        else:
            return {'error': f'curl failed: {result.stderr}'}
            
    except Exception as e:
        print(f"  ⚠️ gold-api.com error: {e}")
        return {'error': str(e)}


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_commodity_prices(commodity: str, period: str = '2y') -> pd.DataFrame:
    """
    Fetch historical prices for silver or gold.
    
    Args:
        commodity: 'silver' or 'gold'
        period: yfinance period string (1y, 2y, 5y, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not available")
    
    config = COMMODITY_CONFIG.get(commodity.lower())
    if not config:
        raise ValueError(f"Unknown commodity: {commodity}")
    
    ticker = config['ticker']
    
    try:
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Handle multi-level columns (yfinance can return MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Dynamically rename columns based on what yfinance returns
        # Standard columns: Date, Open, High, Low, Close, (Adj Close), Volume
        col_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower or col_lower == 'index':
                col_mapping[col] = 'Date'
            elif 'open' in col_lower:
                col_mapping[col] = 'Open'
            elif 'high' in col_lower:
                col_mapping[col] = 'High'
            elif 'low' in col_lower:
                col_mapping[col] = 'Low'
            elif 'adj' in col_lower and 'close' in col_lower:
                col_mapping[col] = 'Adj Close'
            elif 'close' in col_lower:
                col_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                col_mapping[col] = 'Volume'
        
        data = data.rename(columns=col_mapping)
        
        # Ensure we have required columns
        required = ['Date', 'Open', 'High', 'Low', 'Close']
        for req in required:
            if req not in data.columns:
                raise ValueError(f"Missing required column: {req}")
        
        # Add Volume if missing
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        
        # Add Adj Close if missing
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        
        print(f"✅ Fetched {len(data)} {config['name']} data points")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching {commodity}: {e}")
        raise


def fetch_industrial_indicators(period: str = '2y') -> pd.DataFrame:
    """
    Fetch industrial demand indicators:
    - Semiconductor Index (AI/GPU demand)
    - NVIDIA (GPU proxy)
    - Solar ETF (solar panel demand)
    - Tesla (EV demand proxy)
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    all_data = {}
    
    for name, ticker in INDUSTRIAL_INDICATORS.items():
        try:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty:
                # Handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_data[f'{name}_close'] = data['Close']
                all_data[f'{name}_return'] = data['Close'].pct_change()
                all_data[f'{name}_trend'] = data['Close'] / data['Close'].shift(20) - 1
                print(f"  ✅ Fetched {name} ({ticker})")
        except Exception as e:
            print(f"  ⚠️ Error fetching {name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df.index.name = 'Date'
    return df.reset_index()


def fetch_macro_indicators(period: str = '2y') -> pd.DataFrame:
    """
    Fetch macro indicators:
    - USD Index (inverse correlation with metals)
    - 10Y Treasury Yield (opportunity cost)
    - VIX (fear/greed)
    - S&P 500 (risk sentiment)
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    all_data = {}
    
    for name, ticker in MACRO_INDICATORS.items():
        try:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty:
                # Handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_data[f'{name}_close'] = data['Close']
                all_data[f'{name}_change'] = data['Close'].pct_change()
                print(f"  ✅ Fetched {name} ({ticker})")
        except Exception as e:
            print(f"  ⚠️ Error fetching {name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df.index.name = 'Date'
    return df.reset_index()


def fetch_usd_pkr(period: str = '2y') -> pd.DataFrame:
    """Fetch USD/PKR exchange rate for local price conversion."""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        data = yf.download(USD_PKR_TICKER, period=period, progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df = pd.DataFrame({
                'Date': data.index,
                'usd_pkr': data['Close'].values
            })
            print(f"  ✅ Fetched USD/PKR rate")
            return df
    except Exception as e:
        print(f"  ⚠️ Error fetching USD/PKR: {e}")
    
    return pd.DataFrame()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_commodity_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate commodity-specific technical indicators.
    Adapted from stock indicators but tuned for precious metals.
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # =========================================================================
    # 1. TREND INDICATORS
    # =========================================================================
    
    # Moving Averages (commodities use longer periods)
    for period in [10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = close.rolling(period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # Price vs Moving Averages
    df['price_vs_sma50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-8)
    df['price_vs_sma200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-8)
    
    # Golden/Death Cross signals
    df['sma50_above_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # =========================================================================
    # 2. MOMENTUM INDICATORS
    # =========================================================================
    
    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = (close - close.shift(period)) / (close.shift(period) + 1e-8) * 100
    
    # =========================================================================
    # 3. VOLATILITY INDICATORS
    # =========================================================================
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_percent_b'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma20 + 1e-8)
    
    # Average True Range (ATR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_percent'] = df['atr_14'] / (close + 1e-8) * 100
    
    # Realized Volatility (annualized)
    df['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    
    # =========================================================================
    # 4. COMMODITY-SPECIFIC
    # =========================================================================
    
    # Gold/Silver Ratio (if gold available)
    # Will be calculated during merge
    
    # Seasonal patterns (commodities have strong seasonality)
    if 'Date' in df.columns:
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['is_q4'] = (df['month'] >= 10).astype(int)  # Q4 often bullish (festivals, weddings)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)  # Often slower
    
    # Returns
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))
    
    return df


def merge_external_factors(
    commodity_df: pd.DataFrame,
    industrial_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    usd_pkr_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Merge commodity prices with external factors.
    """
    from backend.external_features import _to_naive_datetime

    df = commodity_df.copy()
    df['Date'] = _to_naive_datetime(df['Date'])

    # Merge industrial indicators
    if not industrial_df.empty:
        industrial_df['Date'] = _to_naive_datetime(industrial_df['Date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            industrial_df.sort_values('Date'),
            on='Date',
            direction='backward'
        )

    # Merge macro indicators
    if not macro_df.empty:
        macro_df['Date'] = _to_naive_datetime(macro_df['Date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            macro_df.sort_values('Date'),
            on='Date',
            direction='backward'
        )

    # Merge USD/PKR
    if usd_pkr_df is not None and not usd_pkr_df.empty:
        usd_pkr_df['Date'] = _to_naive_datetime(usd_pkr_df['Date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            usd_pkr_df.sort_values('Date'),
            on='Date',
            direction='backward'
        )
        # Calculate PKR price
        if 'usd_pkr' in df.columns:
            df['price_pkr'] = df['Close'] * df['usd_pkr']
    
    return df.sort_values('Date').reset_index(drop=True)


# ============================================================================
# PREDICTION MODEL
# ============================================================================

class CommodityPredictor:
    """
    Predicts silver and gold prices using external factors + technicals.
    
    Uses ensemble of:
    - GradientBoostingRegressor (captures non-linear relationships)
    - RandomForestRegressor (robust to noise)
    - Ridge (linear baseline, regularized)
    """
    
    def __init__(self, commodity: str):
        if commodity.lower() not in COMMODITY_CONFIG:
            raise ValueError(f"Unknown commodity: {commodity}")
        
        self.commodity = commodity.lower()
        self.config = COMMODITY_CONFIG[self.commodity]
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_cols = []
        self.is_fitted = False
        self.metrics = {}
        
        # Model weights based on commodity type
        # Silver: More weight on industrial indicators
        # Gold: More weight on macro/safe-haven
        if self.commodity == 'silver':
            self.weights = {'gb': 0.45, 'rf': 0.35, 'ridge': 0.20}
        else:  # gold
            self.weights = {'gb': 0.40, 'rf': 0.40, 'ridge': 0.20}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target."""
        
        # Select feature columns
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                        'Target', 'price_pkr', 'month']
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                continue
            # Check if column has any non-null values
            if df[col].notna().sum() > len(df) * 0.5:
                feature_cols.append(col)
        
        self.feature_cols = feature_cols
        
        # Create target (next day's close)
        df = df.copy()
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN
        df_clean = df.dropna(subset=['Target'] + feature_cols)
        df_clean[feature_cols] = df_clean[feature_cols].ffill().fillna(0)
        
        X = df_clean[feature_cols].values
        y = df_clean['Target'].values
        
        return X, y, feature_cols
    
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """Train the commodity prediction model."""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"{self.config['emoji']} TRAINING {self.config['name'].upper()} PREDICTOR")
            print(f"{'='*60}")
        
        X, y, feature_cols = self.prepare_features(df)
        
        if verbose:
            print(f"📊 Training data: {len(X)} samples, {len(feature_cols)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {name: [] for name in self.weights.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            gb.fit(X_train, y_train)
            cv_scores['gb'].append(r2_score(y_val, gb.predict(X_val)))
            
            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            cv_scores['rf'].append(r2_score(y_val, rf.predict(X_val)))
            
            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            cv_scores['ridge'].append(r2_score(y_val, ridge.predict(X_val)))
        
        # Train final models on all data
        if verbose:
            print("\n🔧 Training final models...")
        
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05, 
            subsample=0.8, random_state=42
        )
        self.models['gb'].fit(X_scaled, y)
        
        self.models['rf'] = RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['rf'].fit(X_scaled, y)
        
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Calculate metrics
        avg_r2 = {name: np.mean(scores) for name, scores in cv_scores.items()}
        ensemble_r2 = sum(avg_r2[n] * self.weights[n] for n in self.weights.keys())
        
        self.metrics = {
            'model_r2': avg_r2,
            'ensemble_r2': ensemble_r2,
            'samples': len(X),
            'features': len(feature_cols)
        }
        
        if verbose:
            print(f"\n📊 Cross-Validation R² Scores:")
            for name, r2 in avg_r2.items():
                print(f"   {name}: {r2:.4f}")
            print(f"   Ensemble: {ensemble_r2:.4f}")
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict(X_scaled)
        
        return predictions
    
    def predict_horizon(self, df: pd.DataFrame, days: int = 180) -> List[Dict]:
        """
        Generate price predictions for specified horizon.
        Uses bounded random walk with mean reversion.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        predictions = []
        base_price = float(df['Close'].iloc[-1])
        base_date = pd.to_datetime(df['Date'].iloc[-1])
        
        # Get model prediction for trend direction
        X = df[self.feature_cols].iloc[-1:].fillna(0).values
        model_pred = self.predict(X)[0]
        trend_return = (model_pred - base_price) / base_price
        trend_return = np.clip(trend_return, -0.15, 0.15)  # Max 15% trend signal
        
        # Historical volatility
        if len(df) > 20:
            hist_vol = float(df['Close'].pct_change().tail(20).std())
        else:
            hist_vol = 0.015  # 1.5% default
        
        daily_vol = max(0.005, min(0.025, hist_vol))
        daily_drift = trend_return / 100 + 0.0001  # Slight positive drift
        
        # AR(1) parameters
        phi = 0.12  # Mean reversion
        current_price = base_price
        prev_return = 0
        
        # Deterministic seed for reproducibility
        rng = np.random.RandomState(int(base_price * 100) % (2**31 - 1))
        
        for day in range(1, days + 1):
            # Confidence decays over time
            confidence = max(0.3, 1.0 - day / 300)
            
            # AR(1) return
            noise = rng.normal(0, daily_vol) * (1 + (1 - confidence) * 0.3)
            daily_return = daily_drift + phi * prev_return + noise
            daily_return = np.clip(daily_return, -0.03, 0.03)  # Max 3% daily
            
            new_price = current_price * (1 + daily_return)
            
            # Bound total return
            total_return = (new_price - base_price) / base_price
            if abs(total_return) > 0.40:  # Max 40% over horizon
                new_price = base_price * (1 + np.sign(total_return) * 0.38)
            
            upside = (new_price - base_price) / base_price * 100
            
            predictions.append({
                'day': day,
                'date': (base_date + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_price': float(round(new_price, 2)),
                'upside_potential': float(round(upside, 2)),
                'confidence': float(round(confidence, 2)),
                'reliability': 'high' if day <= 30 else 'medium' if day <= 90 else 'low'
            })
            
            prev_return = daily_return
            current_price = new_price
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from GradientBoosting model."""
        if 'gb' not in self.models:
            return {}
        
        importances = self.models['gb'].feature_importances_
        return dict(sorted(
            zip(self.feature_cols, importances),
            key=lambda x: x[1],
            reverse=True
        ))


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def analyze_commodity(commodity: str, progress_callback=None) -> Dict:
    """
    Main function to analyze a commodity (silver or gold).
    
    Returns complete analysis with:
    - Current price and historical data
    - External factor analysis
    - Price predictions
    - Key drivers explanation
    """
    commodity = commodity.lower()
    config = COMMODITY_CONFIG.get(commodity)
    
    if not config:
        raise ValueError(f"Unknown commodity: {commodity}. Use 'silver' or 'gold'.")
    
    print(f"\n{'='*60}")
    print(f"{config['emoji']} ANALYZING {config['name'].upper()}")
    print(f"{'='*60}")
    
    # 1. Fetch commodity prices
    print("\n1. Fetching price history...")
    price_df = fetch_commodity_prices(commodity, period='2y')
    
    # 2. Calculate technical indicators
    print("\n2. Calculating technical indicators...")
    price_df = calculate_commodity_indicators(price_df)
    
    # 3. Fetch external factors
    print("\n3. Fetching external factors...")
    print("   Industrial indicators (AI/GPU, Solar, EV)...")
    industrial_df = fetch_industrial_indicators(period='2y')
    
    print("   Macro indicators (USD, Treasury, VIX)...")
    macro_df = fetch_macro_indicators(period='2y')
    
    print("   USD/PKR for local conversion...")
    usd_pkr_df = fetch_usd_pkr(period='2y')
    
    # 4. Merge all data
    print("\n4. Merging external factors...")
    full_df = merge_external_factors(price_df, industrial_df, macro_df, usd_pkr_df)
    print(f"   Final dataset: {len(full_df)} rows x {len(full_df.columns)} features")
    
    # 5. Train model
    print("\n5. Training prediction model...")
    predictor = CommodityPredictor(commodity)
    metrics = predictor.fit(full_df, verbose=True)
    
    # 6. Generate predictions (6 months)
    print("\n6. Generating 6-month predictions...")
    predictions = predictor.predict_horizon(full_df, days=180)
    
    # 7. Get feature importance (key drivers)
    importance = predictor.get_feature_importance()
    top_features = list(importance.items())[:10]
    
    # 8. Build result
    current_price = float(full_df['Close'].iloc[-1])
    current_price_pkr = float(full_df['price_pkr'].iloc[-1]) if 'price_pkr' in full_df.columns else None
    
    final_pred = predictions[-1] if predictions else {}
    
    result = {
        'commodity': commodity,
        'name': config['name'],
        'emoji': config['emoji'],
        'unit': config['unit'],
        'current_price_usd': current_price,
        'current_price_pkr': current_price_pkr,
        'currency': 'USD',
        'predicted_price': final_pred.get('predicted_price', current_price),
        'upside_potential': final_pred.get('upside_potential', 0),
        'prediction_horizon': '6 months',
        'daily_predictions': predictions,
        'historical_data': [
            {'Date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']), 
             'Close': float(row['Close'])}
            for _, row in full_df.tail(180).iterrows()
        ],
        'model_metrics': metrics,
        'key_drivers': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in top_features
        ],
        'factors': build_factors_explanation(full_df, commodity),
        'generated_at': datetime.now().isoformat()
    }
    
    # Save to cache
    save_commodity_analysis(result)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Current: ${current_price:.2f}/{config['unit']}")
    print(f"   Predicted (6mo): ${final_pred.get('predicted_price', 0):.2f}")
    print(f"   Upside: {final_pred.get('upside_potential', 0):+.1f}%")
    
    return result


def build_factors_explanation(df: pd.DataFrame, commodity: str) -> Dict:
    """Build human-readable explanation of what's driving prices."""
    latest = df.iloc[-1]
    factors = {}
    
    # AI/GPU Demand (via semiconductor index)
    if 'semiconductor_trend' in df.columns and pd.notna(latest.get('semiconductor_trend')):
        trend = latest['semiconductor_trend'] * 100
        if trend > 5:
            factors['ai_gpu_demand'] = {
                'status': 'bullish',
                'emoji': '🚀',
                'description': f'Semiconductor index up {trend:.1f}% (AI/GPU demand rising)'
            }
        elif trend < -5:
            factors['ai_gpu_demand'] = {
                'status': 'bearish',
                'emoji': '📉',
                'description': f'Semiconductor index down {abs(trend):.1f}%'
            }
        else:
            factors['ai_gpu_demand'] = {
                'status': 'neutral',
                'emoji': '➡️',
                'description': 'Semiconductor demand stable'
            }
    
    # Solar demand
    if 'solar_trend' in df.columns and pd.notna(latest.get('solar_trend')):
        trend = latest['solar_trend'] * 100
        factors['solar_demand'] = {
            'status': 'bullish' if trend > 3 else 'bearish' if trend < -3 else 'neutral',
            'emoji': '☀️' if trend > 0 else '🌥️',
            'description': f'Solar ETF {trend:+.1f}% (affects silver industrial demand)'
        }
    
    # USD Strength
    if 'usd_index_change' in df.columns and pd.notna(latest.get('usd_index_change')):
        change = latest['usd_index_change'] * 100
        # USD up = metals down (inverse)
        factors['usd_strength'] = {
            'status': 'bearish' if change > 0.5 else 'bullish' if change < -0.5 else 'neutral',
            'emoji': '💵',
            'description': f'USD Index {"strengthening" if change > 0 else "weakening"} ({change:+.2f}%)'
        }
    
    # VIX (Fear gauge)
    if 'vix_close' in df.columns and pd.notna(latest.get('vix_close')):
        vix = latest['vix_close']
        factors['market_fear'] = {
            'status': 'bullish' if vix > 25 else 'neutral' if vix > 15 else 'bearish',
            'emoji': '😰' if vix > 25 else '😐' if vix > 15 else '😎',
            'description': f'VIX at {vix:.1f} ({"high fear - safe haven buying" if vix > 25 else "moderate" if vix > 15 else "low fear"})'
        }
    
    # Treasury Yield
    if 'treasury_10y_close' in df.columns and pd.notna(latest.get('treasury_10y_close')):
        yield_pct = latest['treasury_10y_close']
        factors['interest_rates'] = {
            'status': 'bearish' if yield_pct > 4.5 else 'neutral' if yield_pct > 3.5 else 'bullish',
            'emoji': '🏦',
            'description': f'10Y Treasury at {yield_pct:.2f}% ({"high - headwind for metals" if yield_pct > 4.5 else "moderate" if yield_pct > 3.5 else "low - supportive"})'
        }
    
    # RSI
    if 'rsi_14' in df.columns and pd.notna(latest.get('rsi_14')):
        rsi = latest['rsi_14']
        factors['momentum'] = {
            'status': 'bearish' if rsi > 70 else 'bullish' if rsi < 30 else 'neutral',
            'emoji': '📊',
            'description': f'RSI at {rsi:.0f} ({"overbought - may pull back" if rsi > 70 else "oversold - may bounce" if rsi < 30 else "neutral"})'
        }
    
    return factors


def save_commodity_analysis(result: Dict):
    """Save analysis to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = CACHE_DIR / f"{result['commodity']}_analysis.json"
    
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def load_commodity_analysis(commodity: str) -> Optional[Dict]:
    """Load cached analysis if recent enough."""
    file_path = CACHE_DIR / f"{commodity.lower()}_analysis.json"
    
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if cache is less than 4 hours old
            generated_at = datetime.fromisoformat(data.get('generated_at', '2000-01-01'))
            if datetime.now() - generated_at < timedelta(hours=4):
                return data
        except:
            pass
    
    return None


def get_commodity_quick_data(commodity: str) -> Dict:
    """
    Get quick price data without full analysis.
    Uses gold-api.com (free, no key) as primary source, yfinance as fallback.
    """
    config = COMMODITY_CONFIG.get(commodity.lower())
    if not config:
        return {'error': f'Unknown commodity: {commodity}'}
    
    # Try gold-api.com first (free, no key required)
    api_result = fetch_realtime_from_gold_api(commodity)
    if 'price' in api_result:
        return {
            'commodity': commodity.lower(),
            'name': config['name'],
            'emoji': config['emoji'],
            'current_price': api_result['price'],
            'change_percent': 0,  # gold-api.com doesn't provide change, will update from history if available
            'currency': 'USD',
            'source': 'gold-api.com'
        }
    
    # Fallback to yfinance
    if not YFINANCE_AVAILABLE:
        return {'error': 'No data source available'}
    
    try:
        ticker = yf.Ticker(config['ticker'])
        hist = ticker.history(period='5d')
        
        if hist.empty:
            return {'error': 'No data available'}
        
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
        change = (current - prev) / prev * 100
        
        return {
            'commodity': commodity.lower(),
            'name': config['name'],
            'emoji': config['emoji'],
            'current_price': current,
            'change_percent': change,
            'currency': 'USD',
            'source': 'yfinance'
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🥈🥇 COMMODITY PREDICTOR - TEST")
    print("=" * 70)
    
    # Test silver analysis
    try:
        result = analyze_commodity('silver')
        print(f"\n{'='*50}")
        print("TEST RESULT:")
        print(f"{'='*50}")
        print(f"Commodity: {result['name']}")
        print(f"Current Price: ${result['current_price_usd']:.2f}")
        print(f"Predicted (6mo): ${result['predicted_price']:.2f}")
        print(f"Upside: {result['upside_potential']:+.1f}%")
        print(f"\nKey Drivers:")
        for driver in result['key_drivers'][:5]:
            print(f"  - {driver['feature']}: {driver['importance']:.4f}")
        print(f"\nFactors:")
        for name, factor in result['factors'].items():
            print(f"  {factor['emoji']} {name}: {factor['description']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
