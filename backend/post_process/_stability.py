"""
🎯 PREDICTION STABILITY MODULE
Prevents prediction flip-flop using hysteresis and exponential smoothing.

Key Features:
- Hysteresis thresholds to prevent rapid direction changes
- Exponential smoothing with adaptive alpha
- State persistence across predictions
- Per-symbol tracking
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# State file location
STATE_FILE = Path(__file__).parent.parent.parent / "data" / "prediction_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


class PredictionStabilizer:
    """Stabilize predictions using hysteresis and smoothing"""
    
    # Hysteresis thresholds
    BULLISH_THRESHOLD = 7.0   # Require +7% to flip TO BULLISH
    BEARISH_THRESHOLD = -7.0  # Require -7% to flip TO BEARISH
    NEUTRAL_BAND = 5.0        # Stay in previous direction if within ±5%
    
    # Exponential smoothing weights
    ALPHA_DEFAULT = 0.7       # Standard weight for new prediction
    ALPHA_EXTREME = 0.9       # Higher weight for extreme moves (>15%)
    ALPHA_SMALL = 0.5         # Lower weight for small moves (<5%)
    
    def __init__(self):
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load prediction state from file"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load prediction state: {e}")
        
        return {}
    
    def _save_state(self):
        """Save prediction state to file"""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save prediction state: {e}")
    
    def _get_adaptive_alpha(self, raw_prediction: float) -> float:
        """
        Calculate adaptive alpha based on prediction magnitude.
        
        - Extreme moves (>15%): Use alpha=0.9 (trust new data more)
        - Normal moves (5-15%): Use alpha=0.7 (standard)
        - Small moves (<5%): Use alpha=0.5 (trust history more)
        """
        abs_pred = abs(raw_prediction)
        
        if abs_pred > 15.0:
            return self.ALPHA_EXTREME
        elif abs_pred < 5.0:
            return self.ALPHA_SMALL
        else:
            return self.ALPHA_DEFAULT
    
    def _apply_exponential_smoothing(
        self,
        symbol: str,
        raw_prediction: float
    ) -> float:
        """
        Apply exponential smoothing to prediction.
        
        Formula: smoothed = alpha * new + (1 - alpha) * previous
        """
        previous_state = self.state.get(symbol, {})
        previous_smoothed = previous_state.get('smoothed_prediction', raw_prediction)
        
        # Calculate adaptive alpha
        alpha = self._get_adaptive_alpha(raw_prediction)
        
        # Apply smoothing
        smoothed = alpha * raw_prediction + (1 - alpha) * previous_smoothed
        
        return smoothed
    
    def _apply_hysteresis(
        self,
        symbol: str,
        smoothed_prediction: float,
        raw_direction: str
    ) -> str:
        """
        Apply hysteresis to prevent rapid direction changes.
        
        Rules:
        - To flip TO BULLISH: require smoothed_prediction > +7%
        - To flip TO BEARISH: require smoothed_prediction < -7%
        - Stay in previous direction if within ±5% band
        """
        previous_state = self.state.get(symbol, {})
        previous_direction = previous_state.get('stable_direction', 'NEUTRAL')
        
        # Determine new direction with hysteresis
        if smoothed_prediction > self.BULLISH_THRESHOLD:
            new_direction = 'BULLISH'
        elif smoothed_prediction < self.BEARISH_THRESHOLD:
            new_direction = 'BEARISH'
        elif abs(smoothed_prediction) < self.NEUTRAL_BAND:
            # Within neutral band - keep previous direction if it was strong
            if previous_direction in ['BULLISH', 'BEARISH']:
                new_direction = previous_direction
            else:
                new_direction = 'NEUTRAL'
        else:
            # Between neutral band and threshold - use raw direction
            # but prefer previous direction if close
            if previous_direction == 'BULLISH' and smoothed_prediction > 0:
                new_direction = 'BULLISH'
            elif previous_direction == 'BEARISH' and smoothed_prediction < 0:
                new_direction = 'BEARISH'
            else:
                new_direction = raw_direction
        
        return new_direction
    
    def apply_stability(
        self,
        symbol: str,
        raw_prediction: float,
        raw_direction: str
    ) -> Dict:
        """
        Apply full stability logic: smoothing + hysteresis.
        
        Args:
            symbol: Stock symbol (e.g., 'UBL')
            raw_prediction: Raw predicted upside/downside percentage
            raw_direction: Raw direction ('BULLISH', 'BEARISH', 'NEUTRAL')
        
        Returns:
            {
                'smoothed_prediction': float,
                'stable_direction': str,
                'raw_prediction': float,
                'raw_direction': str,
                'alpha_used': float,
                'previous_direction': str,
                'changed_direction': bool
            }
        """
        symbol = symbol.upper()
        
        # Get previous state
        previous_state = self.state.get(symbol, {})
        previous_direction = previous_state.get('stable_direction', 'NEUTRAL')
        
        # Step 1: Apply exponential smoothing
        smoothed_prediction = self._apply_exponential_smoothing(symbol, raw_prediction)
        alpha_used = self._get_adaptive_alpha(raw_prediction)
        
        # Step 2: Apply hysteresis
        stable_direction = self._apply_hysteresis(symbol, smoothed_prediction, raw_direction)
        
        # Check if direction changed
        changed_direction = (stable_direction != previous_direction)
        
        # Update state
        self.state[symbol] = {
            'smoothed_prediction': smoothed_prediction,
            'stable_direction': stable_direction,
            'raw_prediction': raw_prediction,
            'raw_direction': raw_direction,
            'alpha_used': alpha_used,
            'previous_direction': previous_direction,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save state
        self._save_state()
        
        return {
            'smoothed_prediction': round(smoothed_prediction, 2),
            'stable_direction': stable_direction,
            'raw_prediction': round(raw_prediction, 2),
            'raw_direction': raw_direction,
            'alpha_used': alpha_used,
            'previous_direction': previous_direction,
            'changed_direction': changed_direction
        }
    
    def get_state(self, symbol: str) -> Optional[Dict]:
        """Get current state for a symbol"""
        return self.state.get(symbol.upper())
    
    def reset_state(self, symbol: Optional[str] = None):
        """Reset state for a symbol (or all symbols if None)"""
        if symbol:
            self.state.pop(symbol.upper(), None)
        else:
            self.state = {}
        
        self._save_state()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Prediction Stability Module")
    print("=" * 60)
    
    stabilizer = PredictionStabilizer()
    
    # Simulate prediction flip-flop scenario
    symbol = "TEST"
    
    print("\nScenario: Prediction flip-flop test")
    print("-" * 60)
    
    # Initial prediction: BEARISH
    result1 = stabilizer.apply_stability(symbol, -13.8, 'BEARISH')
    print(f"\n1. Raw: {result1['raw_prediction']}% ({result1['raw_direction']})")
    print(f"   Smoothed: {result1['smoothed_prediction']}% ({result1['stable_direction']})")
    print(f"   Direction changed: {result1['changed_direction']}")
    
    # Small positive move (should NOT flip to BULLISH due to hysteresis)
    result2 = stabilizer.apply_stability(symbol, +3.5, 'NEUTRAL')
    print(f"\n2. Raw: {result2['raw_prediction']}% ({result2['raw_direction']})")
    print(f"   Smoothed: {result2['smoothed_prediction']}% ({result2['stable_direction']})")
    print(f"   Direction changed: {result2['changed_direction']}")
    
    # Moderate positive move (should still be cautious)
    result3 = stabilizer.apply_stability(symbol, +6.0, 'NEUTRAL')
    print(f"\n3. Raw: {result3['raw_prediction']}% ({result3['raw_direction']})")
    print(f"   Smoothed: {result3['smoothed_prediction']}% ({result3['stable_direction']})")
    print(f"   Direction changed: {result3['changed_direction']}")
    
    # Strong positive move (NOW should flip to BULLISH)
    result4 = stabilizer.apply_stability(symbol, +10.5, 'BULLISH')
    print(f"\n4. Raw: {result4['raw_prediction']}% ({result4['raw_direction']})")
    print(f"   Smoothed: {result4['smoothed_prediction']}% ({result4['stable_direction']})")
    print(f"   Direction changed: {result4['changed_direction']}")
    
    print("\n" + "=" * 60)
    print("Scenario: Extreme move test")
    print("-" * 60)
    
    # Extreme move (should use higher alpha)
    result5 = stabilizer.apply_stability("EXTREME", +18.5, 'BULLISH')
    print(f"\nRaw: {result5['raw_prediction']}% ({result5['raw_direction']})")
    print(f"Smoothed: {result5['smoothed_prediction']}% ({result5['stable_direction']})")
    print(f"Alpha used: {result5['alpha_used']:.2f} (higher for extreme moves)")
    
    print("\n✅ Done!")
