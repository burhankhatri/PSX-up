"""
🔍 PREDICTION REASONING MODULE

Generates human-readable explanations for why predictions are bullish/bearish.
Uses the same features as the model but explains them in plain English.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from backend.post_process._stability import PredictionStabilizer
from backend.post_process._shared import direction_from_change_pct


def generate_prediction_reasoning(
    df: pd.DataFrame,
    symbol: str = None,
    predicted_upside: float = None,
    direction_override: str = None,
    apply_stability: bool = True,
    neutral_band_pct: float = 0.0,
    horizon_label: str = "Day 7",
) -> Dict:
    """
    Analyze current indicators and generate reasons for prediction direction.
    
    Args:
        df: DataFrame with price and indicator data
        symbol: Stock symbol (for sector-specific logic)
        predicted_upside: The actual model's predicted upside % (NEW - use this for direction!)
    
    Returns:
        Dict with bullish_signals, bearish_signals, neutral_signals, and summary
    """
    if len(df) < 20:
        return {'error': 'Not enough data for analysis'}
    
    bullish = []
    bearish = []
    neutral = []
    
    # Get latest values
    latest = df.iloc[-1]
    close = float(latest.get('Close', 0))
    symbol_upper = (symbol or '').upper()
    is_energy_symbol = symbol_upper in {'OGDC', 'PPL', 'POL', 'MARI', 'PSO'}
    energy_shock_regime = float(latest.get('energy_shock_regime', 0) or 0) > 0.5
    
    # =====================================================
    # 0. MODEL PREDICTION (PRIMARY SIGNAL)
    # =====================================================
    # If we have the actual prediction, this is the most important signal
    if predicted_upside is not None:
        if predicted_upside > 15:
            bullish.append({
                'category': 'Model Forecast',
                'signal': f'Strong upside predicted: +{predicted_upside:.1f}%',
                'strength': min(predicted_upside / 30, 2.0)  # Very high weight
            })
        elif predicted_upside > 5:
            bullish.append({
                'category': 'Model Forecast',
                'signal': f'Moderate upside predicted: +{predicted_upside:.1f}%',
                'strength': 1.0
            })
        elif predicted_upside > 0:
            bullish.append({
                'category': 'Model Forecast',
                'signal': f'Slight upside predicted: +{predicted_upside:.1f}%',
                'strength': 0.5
            })
        elif predicted_upside > -5:
            bearish.append({
                'category': 'Model Forecast',
                'signal': f'Slight downside predicted: {predicted_upside:.1f}%',
                'strength': 0.5
            })
        elif predicted_upside > -15:
            bearish.append({
                'category': 'Model Forecast',
                'signal': f'Moderate downside predicted: {predicted_upside:.1f}%',
                'strength': 1.0
            })
        else:
            bearish.append({
                'category': 'Model Forecast',
                'signal': f'Significant decline predicted: {predicted_upside:.1f}%',
                'strength': min(abs(predicted_upside) / 30, 2.0)
            })
    
    # =====================================================
    # 1. PRICE MOMENTUM SIGNALS
    # =====================================================
    
    # Recent price trend (20-day)
    if len(df) > 20:
        returns_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
        if returns_20d > 10:
            bullish.append({
                'category': 'Momentum',
                'signal': f'Strong uptrend: +{returns_20d:.1f}% in 20 days',
                'strength': min(returns_20d / 20, 1.0)
            })
        elif returns_20d > 3:
            bullish.append({
                'category': 'Momentum',
                'signal': f'Positive momentum: +{returns_20d:.1f}% in 20 days',
                'strength': 0.5
            })
        elif returns_20d < -10:
            bearish.append({
                'category': 'Momentum',
                'signal': f'Strong downtrend: {returns_20d:.1f}% in 20 days',
                'strength': min(abs(returns_20d) / 20, 1.0)
            })
        elif returns_20d < -3:
            bearish.append({
                'category': 'Momentum',
                'signal': f'Negative momentum: {returns_20d:.1f}% in 20 days',
                'strength': 0.5
            })
        else:
            neutral.append({
                'category': 'Momentum',
                'signal': f'Sideways movement: {returns_20d:+.1f}% in 20 days'
            })
    
    # =====================================================
    # 2. TECHNICAL INDICATORS
    # =====================================================
    
    # RSI (prioritize TradingView, fallback to local)
    rsi = latest.get('tv_rsi_14', latest.get('RSI_14', latest.get('rsi_14', 50)))
    if pd.notna(rsi):
        if rsi > 70:
            bearish.append({
                'category': 'RSI',
                'signal': f'Overbought (RSI: {rsi:.0f}) - may pullback',
                'strength': (rsi - 70) / 30 * 0.5  # Reduced weight
            })
        elif rsi < 30:
            bullish.append({
                'category': 'RSI',
                'signal': f'Oversold (RSI: {rsi:.0f}) - may bounce',
                'strength': (30 - rsi) / 30 * 0.5  # Reduced weight
            })
        else:
            neutral.append({
                'category': 'RSI',
                'signal': f'Neutral RSI: {rsi:.0f}'
            })
    
    # Williams %R
    williams = latest.get('williams_r', -50)
    if pd.notna(williams):
        if williams > -20:
            bearish.append({
                'category': 'Williams %R',
                'signal': f'Overbought territory ({williams:.0f})',
                'strength': (williams + 20) / 20 * 0.3  # Lower weight
            })
        elif williams < -80:
            bullish.append({
                'category': 'Williams %R',
                'signal': f'Oversold territory ({williams:.0f})',
                'strength': (-80 - williams) / 20 * 0.3
            })
    
    # Price vs EMAs
    ema50 = latest.get('ema_50', close)
    ema100 = latest.get('ema_100', close)
    
    if pd.notna(ema50) and ema50 > 0:
        pct_above_50 = (close / ema50 - 1) * 100
        if pct_above_50 > 5:
            bullish.append({
                'category': 'EMA',
                'signal': f'Trading {pct_above_50:.1f}% above 50-day EMA',
                'strength': min(pct_above_50 / 15, 1.0) * 0.5
            })
        elif pct_above_50 < -5:
            bearish.append({
                'category': 'EMA',
                'signal': f'Trading {abs(pct_above_50):.1f}% below 50-day EMA',
                'strength': min(abs(pct_above_50) / 15, 1.0) * 0.5
            })
    
    # =====================================================
    # 3. EXTERNAL INDICATORS (MACRO)
    # =====================================================
    
    # KSE-100 correlation
    kse_return = latest.get('kse100_return', 0)
    if pd.notna(kse_return) and kse_return != 0:
        if is_energy_symbol and energy_shock_regime and kse_return < -0.01:
            bullish.append({
                'category': 'Regime',
                'signal': 'Energy shock regime: sector may diverge from a falling KSE-100',
                'strength': 0.45
            })
        elif kse_return > 0.01:
            bullish.append({
                'category': 'Market',
                'signal': f'KSE-100 rising (+{kse_return*100:.1f}%)',
                'strength': min(kse_return * 20, 1.0) * 0.5
            })
        elif kse_return < -0.01:
            bearish.append({
                'category': 'Market',
                'signal': f'KSE-100 falling ({kse_return*100:.1f}%)',
                'strength': min(abs(kse_return) * 20, 1.0) * 0.5
            })
    
    # USD/PKR for export companies
    usdpkr_change = latest.get('usdpkr_change', 0)
    if pd.notna(usdpkr_change) and abs(usdpkr_change) > 0.001:
        # PKR weakening is good for exporters, bad for importers
        if usdpkr_change > 0:
            # This is complex - depends on company type
            neutral.append({
                'category': 'Currency',
                'signal': f'PKR weakening ({usdpkr_change*100:.2f}%)'
            })
    
    # Oil price (important for energy stocks like PSO)
    oil_change = latest.get('oil_change', 0)
    if pd.notna(oil_change) and symbol and symbol.upper() in ['PSO', 'PPL', 'OGDC', 'POL']:
        if oil_change > 0.01:
            bullish.append({
                'category': 'Commodities',
                'signal': f'Oil prices rising (+{oil_change*100:.1f}%)',
                'strength': min(oil_change * 10, 1.0) * 0.5
            })
        elif oil_change < -0.01:
            bearish.append({
                'category': 'Commodities',
                'signal': f'Oil prices falling ({oil_change*100:.1f}%)',
                'strength': min(abs(oil_change) * 10, 1.0) * 0.5
            })

    fuel_delta = latest.get('local_fuel_price_delta_rs', 0)
    if pd.notna(fuel_delta) and symbol and symbol_upper in {'OGDC', 'PPL', 'POL', 'MARI', 'PSO'} and abs(float(fuel_delta)) > 0:
        if fuel_delta > 0:
            bullish.append({
                'category': 'Policy',
                'signal': f'Domestic fuel prices hiked by Rs{abs(float(fuel_delta)):.0f}',
                'strength': min(abs(float(fuel_delta)) / 50.0, 1.0) * 0.8
            })
        else:
            bearish.append({
                'category': 'Policy',
                'signal': f'Domestic fuel prices cut by Rs{abs(float(fuel_delta)):.0f}',
                'strength': min(abs(float(fuel_delta)) / 50.0, 1.0) * 0.8
            })

    circular_debt_signal = latest.get('circular_debt_signal', 0)
    if pd.notna(circular_debt_signal) and abs(float(circular_debt_signal)) > 0:
        if circular_debt_signal > 0:
            bullish.append({
                'category': 'Balance Sheet',
                'signal': 'Circular-debt relief/payment signal detected',
                'strength': min(abs(float(circular_debt_signal)), 1.0) * 0.7
            })
        else:
            bearish.append({
                'category': 'Balance Sheet',
                'signal': 'Circular-debt stress signal detected',
                'strength': min(abs(float(circular_debt_signal)), 1.0) * 0.7
            })
    
    # =====================================================
    # 4. VOLATILITY ANALYSIS
    # =====================================================
    
    if len(df) > 20:
        volatility = df['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100
        if volatility > 50:
            neutral.append({
                'category': 'Volatility',
                'signal': f'High volatility ({volatility:.0f}% annualized) - increased risk'
            })
        elif volatility < 20:
            neutral.append({
                'category': 'Volatility',
                'signal': f'Low volatility ({volatility:.0f}%) - stable trading'
            })
    
    # =====================================================
    # 5. VOLUME ANALYSIS
    # =====================================================
    
    if 'Volume' in df.columns and len(df) > 20:
        avg_vol = df['Volume'].tail(20).mean()
        latest_vol = latest.get('Volume', avg_vol)
        if pd.notna(latest_vol) and avg_vol > 0:
            vol_ratio = latest_vol / avg_vol
            if vol_ratio > 2:
                bullish.append({
                    'category': 'Volume',
                    'signal': f'Volume surge ({vol_ratio:.1f}x average) - strong interest',
                    'strength': min((vol_ratio - 1) / 3, 1.0) * 0.3
                })
            elif vol_ratio < 0.5:
                neutral.append({
                    'category': 'Volume',
                    'signal': f'Low volume ({vol_ratio:.1f}x average) - weak conviction'
                })
    
    # =====================================================
    # 6. NEWS EVENTS (Business Recorder + Impact Scoring)
    # =====================================================
    
    if symbol:
        try:
            from backend.brecorder_scraper import get_brecorder_news_for_symbol
        except ImportError:
            try:
                from brecorder_scraper import get_brecorder_news_for_symbol
            except ImportError:
                get_brecorder_news_for_symbol = None
        
        if get_brecorder_news_for_symbol:
            try:
                news_result = get_brecorder_news_for_symbol(symbol)
                impact = news_result.get('impact', {})
                articles = news_result.get('articles', [])[:3]  # Top 3
                
                # Add key events as signals
                for event in impact.get('key_events', [])[:3]:
                    event_type = event.get('type', 'neutral')
                    category = event.get('category', 'news')
                    headline = event.get('headline', '')[:60]
                    duration = event.get('duration_days', 30)
                    
                    signal = {
                        'category': 'News Event',
                        'signal': f'{category.title()}: "{headline}..." (impact ~{duration}d)',
                        'strength': abs(event.get('impact', 0.3))
                    }
                    
                    if event_type == 'positive':
                        bullish.append(signal)
                    elif event_type == 'negative':
                        bearish.append(signal)
                    else:
                        neutral.append(signal)
                
                # Add general news sentiment
                news_bias = impact.get('news_bias', 0)
                if news_bias > 0.2 and not any(s.get('category') == 'News Event' for s in bullish):
                    if articles:
                        bullish.append({
                            'category': 'News Sentiment',
                            'signal': f'Positive news coverage ({len(articles)} recent articles)',
                            'strength': news_bias
                        })
                elif news_bias < -0.2 and not any(s.get('category') == 'News Event' for s in bearish):
                    if articles:
                        bearish.append({
                            'category': 'News Sentiment', 
                            'signal': f'Negative news coverage ({len(articles)} recent articles)',
                            'strength': abs(news_bias)
                        })
            except Exception as e:
                # News fetch failed - continue without news signals
                pass
    
    # =====================================================
    # GENERATE SUMMARY (Model prediction has highest weight)
    # =====================================================
    
    total_bullish = sum(s.get('strength', 0.5) for s in bullish)
    total_bearish = sum(s.get('strength', 0.5) for s in bearish)
    
    # If we have a prediction, use it as primary direction
    if predicted_upside is not None:
        raw_direction = direction_from_change_pct(predicted_upside, neutral_band_pct=neutral_band_pct)
        stability_result = None

        if direction_override:
            direction = direction_override
            smoothed_upside = predicted_upside
        elif apply_stability:
            stabilizer = PredictionStabilizer()
            stability_result = stabilizer.apply_stability(
                symbol or 'UNKNOWN',
                predicted_upside,
                raw_direction
            )
            direction = stability_result['stable_direction']
            smoothed_upside = stability_result['smoothed_prediction']
        else:
            direction = raw_direction
            smoothed_upside = predicted_upside

        if direction == 'BULLISH':
            emoji = '🟢'
            explanation = f"{horizon_label} adjusted forecast points +{smoothed_upside:.1f}% higher. {len(bullish)} supporting signals, {len(bearish)} cautionary signals."
            if stability_result and stability_result['changed_direction']:
                explanation += " ⚠️ Direction changed from previous prediction."
        elif direction == 'BEARISH':
            emoji = '🔴'
            explanation = f"{horizon_label} adjusted forecast points {smoothed_upside:.1f}% lower. {len(bearish)} supporting signals, {len(bullish)} contrary signals."
            if stability_result and stability_result['changed_direction']:
                explanation += " ⚠️ Direction changed from previous prediction."
        else:
            emoji = '🟡'
            explanation = f"{horizon_label} adjusted forecast is {smoothed_upside:+.1f}% with mixed signals: {len(bullish)} bullish, {len(bearish)} bearish."
        
        # Update predicted_upside to smoothed value for consistency
        predicted_upside = smoothed_upside
    else:
        # Fallback to indicator-based direction
        if total_bullish > total_bearish * 1.5:
            direction = 'BULLISH'
            emoji = '🟢'
            explanation = f"Multiple indicators suggest upward momentum. {len(bullish)} bullish signals vs {len(bearish)} bearish."
        elif total_bearish > total_bullish * 1.5:
            direction = 'BEARISH'
            emoji = '🔴'
            explanation = f"Warning signs detected. {len(bearish)} bearish signals vs {len(bullish)} bullish."
        else:
            direction = 'NEUTRAL'
            emoji = '🟡'
            explanation = f"Mixed signals. {len(bullish)} bullish, {len(bearish)} bearish."
    
    # Format top signals for display (prioritize Model Forecast first)
    top_bullish = sorted(bullish, key=lambda x: (x['category'] == 'Model Forecast', x.get('strength', 0)), reverse=True)[:3]
    top_bearish = sorted(bearish, key=lambda x: (x['category'] == 'Model Forecast', x.get('strength', 0)), reverse=True)[:3]
    
    return {
        'direction': direction,
        'emoji': emoji,
        'explanation': explanation,
        'bullish_count': len(bullish),
        'bearish_count': len(bearish),
        'prediction_upside': predicted_upside,
        'bullish_signals': [
            {'category': s['category'], 'signal': s['signal']} 
            for s in top_bullish
        ],
        'bearish_signals': [
            {'category': s['category'], 'signal': s['signal']} 
            for s in top_bearish
        ],
        'neutral_signals': [
            {'category': s['category'], 'signal': s['signal']} 
            for s in neutral[:2]  # Only top 2 neutral
        ]
    }


def format_reasoning_for_display(reasoning: Dict) -> str:
    """Format reasoning as a human-readable string."""
    lines = []
    
    lines.append(f"{reasoning['emoji']} {reasoning['direction']}: {reasoning['explanation']}")
    lines.append("")
    
    if reasoning['bullish_signals']:
        lines.append("✅ Bullish Signals:")
        for s in reasoning['bullish_signals']:
            lines.append(f"   • [{s['category']}] {s['signal']}")
    
    if reasoning['bearish_signals']:
        lines.append("⚠️ Bearish Signals:")
        for s in reasoning['bearish_signals']:
            lines.append(f"   • [{s['category']}] {s['signal']}")
    
    if reasoning['neutral_signals']:
        lines.append("ℹ️ Other Factors:")
        for s in reasoning['neutral_signals']:
            lines.append(f"   • [{s['category']}] {s['signal']}")
    
    return "\n".join(lines)
