"""
📊 PREDICTION LOGGER
Tracks predictions vs actual outcomes to measure real-world accuracy.

Features:
- Logs every prediction with timestamp
- Records actual outcomes when available
- Calculates rolling accuracy metrics
- Exports to JSON for analysis
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Log file location
LOG_DIR = Path(__file__).parent.parent / "data" / "prediction_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class PredictionLogger:
    """Track predictions and measure accuracy over time."""

    def __init__(self):
        self.log_file = LOG_DIR / "prediction_log.json"
        self.predictions = self._load_log()

    def _load_log(self) -> List[Dict]:
        """Load existing prediction log."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    raw_entries = json.load(f)
                return [self._normalize_entry(entry) for entry in raw_entries if isinstance(entry, dict)]
            except Exception as e:
                print(f"⚠️ Could not load prediction log: {e}")
        return []

    def _normalize_entry(self, entry: Dict) -> Dict:
        """Backfill additive fields so older logs remain readable."""
        normalized = dict(entry or {})
        symbol = str(normalized.get('symbol', '') or '').upper()
        normalized['symbol'] = symbol

        prediction_date_raw = normalized.get('prediction_date')
        try:
            prediction_dt = datetime.fromisoformat(str(prediction_date_raw))
        except Exception:
            prediction_dt = datetime.now()
        normalized['prediction_date'] = prediction_dt.isoformat()

        horizon_days = int(normalized.get('horizon_days', 0) or 0)
        if horizon_days <= 0:
            target_horizon_raw = str(normalized.get('target_horizon', '') or '').strip().lower()
            if target_horizon_raw.startswith('day_'):
                try:
                    horizon_days = int(target_horizon_raw.split('_', 1)[1])
                except Exception:
                    horizon_days = 0
        normalized['horizon_days'] = horizon_days

        variant = str(normalized.get('variant', 'baseline') or 'baseline').strip().lower()
        if variant not in {'baseline', 'geo'}:
            variant = 'baseline'
        normalized['variant'] = variant

        target_horizon = str(normalized.get('target_horizon', '') or '').strip().lower()
        if not target_horizon:
            target_horizon = f"day_{horizon_days}" if horizon_days > 0 else 'day_unknown'
        normalized['target_horizon'] = target_horizon

        if not normalized.get('id'):
            normalized['id'] = (
                f"{symbol}_{prediction_dt.strftime('%Y%m%d_%H%M%S_%f')}"
                f"_{variant}_{target_horizon}"
            )
        normalized['analysis_id'] = str(normalized.get('analysis_id') or normalized['id'])

        geo_adjustment_pct = normalized.get('geo_adjustment_pct', 0.0)
        try:
            normalized['geo_adjustment_pct'] = round(float(geo_adjustment_pct or 0.0), 4)
        except Exception:
            normalized['geo_adjustment_pct'] = 0.0

        evaluation_date = str(normalized.get('evaluation_date', '') or '').strip()
        if not evaluation_date and horizon_days > 0:
            evaluation_date = (prediction_dt + timedelta(days=horizon_days)).strftime('%Y-%m-%d')
        normalized['evaluation_date'] = evaluation_date

        try:
            normalized['current_price'] = round(float(normalized.get('current_price', 0) or 0), 2)
        except Exception:
            normalized['current_price'] = 0.0
        try:
            normalized['predicted_price'] = round(float(normalized.get('predicted_price', 0) or 0), 2)
        except Exception:
            normalized['predicted_price'] = 0.0
        if normalized.get('predicted_change_pct') is None:
            curr = float(normalized.get('current_price', 0) or 0)
            pred = float(normalized.get('predicted_price', 0) or 0)
            normalized['predicted_change_pct'] = round(((pred - curr) / curr) * 100, 2) if curr > 0 else 0.0

        for key in ('confidence', 'actual_price', 'actual_change_pct', 'error_pct', 'abs_error_pct'):
            value = normalized.get(key)
            if value is None:
                normalized[key] = None
                continue
            try:
                digits = 2 if key in {'confidence', 'actual_price', 'actual_change_pct'} else 4
                normalized[key] = round(float(value), digits)
            except Exception:
                normalized[key] = None

        normalized.setdefault('predicted_direction', 'NEUTRAL')
        normalized.setdefault('williams_signal', None)
        normalized.setdefault('sector', None)
        normalized.setdefault('actual_direction', None)
        normalized.setdefault('direction_correct', None)
        normalized.setdefault('actual_date_used', None)
        normalized.setdefault('evaluated_at', None)
        normalized['evaluated'] = bool(normalized.get('evaluated', False))
        return normalized

    def _save_log(self):
        """Save prediction log to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.predictions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save prediction log: {e}")

    def log_prediction(
        self,
        symbol: str,
        current_price: float,
        predicted_price: float,
        predicted_direction: str,
        confidence: float,
        horizon_days: int = 1,
        williams_signal: Optional[str] = None,
        sector: Optional[str] = None,
        *,
        evaluation_date: Optional[str] = None,
        prediction_date: Optional[datetime] = None,
        analysis_id: Optional[str] = None,
        variant: str = "baseline",
        target_horizon: Optional[str] = None,
        geo_adjustment_pct: float = 0.0,
    ) -> Dict:
        """
        Log a new prediction.

        Args:
            symbol: Stock symbol
            current_price: Current price at prediction time
            predicted_price: Predicted future price
            predicted_direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            confidence: Model confidence (0-1)
            horizon_days: Days until prediction should be evaluated
            williams_signal: Williams %R classifier signal (if available)
            sector: Detected sector (if available)

        Returns:
            The logged prediction entry
        """
        prediction_dt = prediction_date or datetime.now()
        evaluation_date_str = evaluation_date
        if not evaluation_date_str:
            evaluation_date_str = (prediction_dt + timedelta(days=horizon_days)).strftime('%Y-%m-%d')

        variant = str(variant or 'baseline').strip().lower()
        if variant not in {'baseline', 'geo'}:
            variant = 'baseline'

        target_horizon = str(target_horizon or f"day_{horizon_days}").strip().lower()
        entry_id = (
            f"{symbol}_{prediction_dt.strftime('%Y%m%d_%H%M%S_%f')}"
            f"_{variant}_{target_horizon}"
        )

        entry = {
            'id': entry_id,
            'symbol': symbol,
            'prediction_date': prediction_dt.isoformat(),
            'evaluation_date': evaluation_date_str,
            'horizon_days': horizon_days,
            'analysis_id': analysis_id or entry_id,
            'variant': variant,
            'target_horizon': target_horizon,
            'geo_adjustment_pct': round(float(geo_adjustment_pct or 0.0), 4),
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'predicted_change_pct': round(((predicted_price - current_price) / current_price) * 100, 2)
            if current_price else 0.0,
            'predicted_direction': predicted_direction,
            'confidence': round(confidence, 2),
            'williams_signal': williams_signal,
            'sector': sector,
            # To be filled later
            'actual_price': None,
            'actual_change_pct': None,
            'actual_direction': None,
            'direction_correct': None,
            'actual_date_used': None,
            'error_pct': None,
            'abs_error_pct': None,
            'evaluated_at': None,
            'evaluated': False
        }

        self.predictions.append(self._normalize_entry(entry))
        self._save_log()

        print(
            f"📝 Logged prediction: {symbol} {variant}/{target_horizon} "
            f"{predicted_direction} ({confidence:.0%} conf)"
        )
        return self.predictions[-1]

    def _apply_actual_to_entry(
        self,
        pred: Dict,
        actual_price: float,
        actual_date_used: Optional[str] = None,
    ) -> Dict:
        current_price = float(pred.get('current_price', 0) or 0)
        predicted_price = float(pred.get('predicted_price', 0) or 0)
        actual_change = ((actual_price - current_price) / current_price * 100) if current_price > 0 else 0.0
        error_pct = ((predicted_price - actual_price) / actual_price * 100) if actual_price > 0 else None

        pred['actual_price'] = round(actual_price, 2)
        pred['actual_change_pct'] = round(actual_change, 2)
        if abs(actual_change) < 1e-9:
            pred['actual_direction'] = 'NEUTRAL'
        else:
            pred['actual_direction'] = 'BULLISH' if actual_change > 0 else 'BEARISH'

        pred_dir = pred.get('predicted_direction', 'NEUTRAL')
        actual_dir = pred['actual_direction']
        pred['direction_correct'] = (
            (pred_dir == 'BULLISH' and actual_dir == 'BULLISH') or
            (pred_dir == 'BEARISH' and actual_dir == 'BEARISH') or
            (pred_dir == 'NEUTRAL')
        )
        pred['actual_date_used'] = actual_date_used
        pred['error_pct'] = round(error_pct, 4) if error_pct is not None else None
        pred['abs_error_pct'] = round(abs(error_pct), 4) if error_pct is not None else None
        pred['evaluated_at'] = datetime.now().isoformat()
        pred['evaluated'] = True
        return pred

    def update_actual(
        self,
        symbol: str,
        evaluation_date: str,
        actual_price: float,
        *,
        prediction_id: Optional[str] = None,
        actual_date_used: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Update a prediction with actual outcome.

        Args:
            symbol: Stock symbol
            evaluation_date: Date to evaluate (YYYY-MM-DD)
            actual_price: Actual price on evaluation date

        Returns:
            Updated prediction entry or None if not found
        """
        for pred in self.predictions:
            id_match = prediction_id and pred.get('id') == prediction_id
            legacy_match = (
                pred.get('symbol') == symbol and
                pred.get('evaluation_date') == evaluation_date and
                not pred.get('evaluated')
            )
            if not id_match and not legacy_match:
                continue

            self._apply_actual_to_entry(pred, actual_price, actual_date_used=actual_date_used)
            self._save_log()

            status = "✅ CORRECT" if pred['direction_correct'] else "❌ WRONG"
            print(
                f"📊 Evaluated {symbol} {pred.get('variant', 'baseline')}/{pred.get('target_horizon', '')}: "
                f"Predicted {pred.get('predicted_direction')}, Actual {pred.get('actual_direction')} → {status}"
            )

            return pred

        return None

    def backfill_actuals(self, symbol: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Resolve actual closes for overdue predictions using local history or PSX fallback."""
        try:
            from backend.prediction_tuning import _fetch_actual_on_or_after
        except Exception as e:
            print(f"⚠️ Prediction backfill unavailable: {e}")
            return []

        today = datetime.now().strftime('%Y-%m-%d')
        cache = {}
        updated = []

        pending = [
            p for p in self.predictions
            if not p.get('evaluated') and p.get('evaluation_date') and p['evaluation_date'] <= today
            and (symbol is None or p.get('symbol') == symbol)
        ]
        pending.sort(key=lambda p: (p.get('evaluation_date', ''), p.get('prediction_date', '')))

        for pred in pending:
            if limit is not None and len(updated) >= limit:
                break
            try:
                actual_price, actual_date = _fetch_actual_on_or_after(
                    pred['symbol'],
                    pred['evaluation_date'],
                    cache,
                )
            except Exception:
                continue
            if actual_price is None:
                continue

            updated_pred = self.update_actual(
                pred['symbol'],
                pred['evaluation_date'],
                actual_price,
                prediction_id=pred.get('id'),
                actual_date_used=actual_date,
            )
            if updated_pred:
                updated.append(updated_pred)

        return updated

    def get_accuracy_stats(self, symbol: Optional[str] = None, days: int = 30) -> Dict:
        """
        Calculate accuracy statistics.

        Args:
            symbol: Filter by symbol (None = all)
            days: Look back period in days

        Returns:
            Dictionary with accuracy metrics
        """
        cutoff = datetime.now() - timedelta(days=days)

        evaluated = [
            p for p in self.predictions
            if p['evaluated'] and
            datetime.fromisoformat(p['prediction_date']) > cutoff and
            (symbol is None or p['symbol'] == symbol)
        ]

        if not evaluated:
            return {
                'total_predictions': 0,
                'direction_accuracy': None,
                'avg_confidence': None,
                'message': 'No evaluated predictions in period'
            }

        correct = sum(1 for p in evaluated if p['direction_correct'])
        total = len(evaluated)

        # Average error
        errors = [
            abs(p['predicted_change_pct'] - p['actual_change_pct'])
            for p in evaluated
            if p['actual_change_pct'] is not None
        ]

        # Confidence vs accuracy correlation
        high_conf = [p for p in evaluated if p['confidence'] > 0.7]
        high_conf_correct = sum(1 for p in high_conf if p['direction_correct'])

        return {
            'total_predictions': total,
            'direction_accuracy': round(correct / total * 100, 1) if total > 0 else None,
            'avg_confidence': round(np.mean([p['confidence'] for p in evaluated]) * 100, 1),
            'avg_error_pct': round(np.mean(errors), 2) if errors else None,
            'high_confidence_accuracy': round(high_conf_correct / len(high_conf) * 100, 1) if high_conf else None,
            'period_days': days,
            'symbol_filter': symbol
        }

    def get_pending_evaluations(self) -> List[Dict]:
        """Get predictions that need actual price updates."""
        today = datetime.now().strftime('%Y-%m-%d')

        pending = [
            p for p in self.predictions
            if not p['evaluated'] and p['evaluation_date'] <= today
        ]

        return pending

    def get_recent_predictions(self, limit: int = 10, symbol: Optional[str] = None) -> List[Dict]:
        """Get most recent predictions."""
        filtered = [
            p for p in self.predictions
            if symbol is None or p['symbol'] == symbol
        ]

        return sorted(
            filtered,
            key=lambda x: x['prediction_date'],
            reverse=True
        )[:limit]

    def export_to_csv(self, filepath: str = None) -> str:
        """Export prediction log to CSV for analysis."""
        import csv

        if filepath is None:
            filepath = LOG_DIR / f"predictions_export_{datetime.now().strftime('%Y%m%d')}.csv"

        fieldnames = [
            'id', 'analysis_id', 'symbol', 'prediction_date', 'evaluation_date', 'horizon_days',
            'variant', 'target_horizon', 'geo_adjustment_pct',
            'current_price', 'predicted_price', 'predicted_change_pct', 'predicted_direction',
            'confidence', 'williams_signal', 'sector',
            'actual_price', 'actual_change_pct', 'actual_direction', 'direction_correct',
            'actual_date_used', 'error_pct', 'abs_error_pct', 'evaluated_at', 'evaluated'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.predictions)

        print(f"📁 Exported {len(self.predictions)} predictions to {filepath}")
        return str(filepath)


# Singleton instance
_logger = None


def get_prediction_logger() -> PredictionLogger:
    """Get the singleton prediction logger instance."""
    global _logger
    if _logger is None:
        _logger = PredictionLogger()
    return _logger


# ============================================================================
# CLI for checking accuracy
# ============================================================================

if __name__ == "__main__":
    import sys

    logger = get_prediction_logger()

    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        # Show accuracy stats
        symbol = sys.argv[2] if len(sys.argv) > 2 else None
        stats = logger.get_accuracy_stats(symbol=symbol)

        print("\n" + "=" * 60)
        print("📊 PREDICTION ACCURACY STATS")
        print("=" * 60)

        if stats['total_predictions'] == 0:
            print("\n   No evaluated predictions yet.")
        else:
            print(f"\n   Total Predictions: {stats['total_predictions']}")
            print(f"   Direction Accuracy: {stats['direction_accuracy']}%")
            print(f"   Avg Confidence: {stats['avg_confidence']}%")
            print(f"   Avg Error: {stats['avg_error_pct']}%")
            if stats['high_confidence_accuracy']:
                print(f"   High-Conf Accuracy: {stats['high_confidence_accuracy']}%")

    elif len(sys.argv) > 1 and sys.argv[1] == 'pending':
        # Show pending evaluations
        pending = logger.get_pending_evaluations()

        print("\n" + "=" * 60)
        print("⏳ PENDING EVALUATIONS")
        print("=" * 60)

        if not pending:
            print("\n   No pending evaluations.")
        else:
            for p in pending:
                print(f"\n   {p['symbol']}: {p['predicted_direction']} ({p['confidence']:.0%})")
                print(f"      Predicted: {p['predicted_change_pct']:+.1f}%")
                print(f"      Evaluate on: {p['evaluation_date']}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'export':
        # Export to CSV
        logger.export_to_csv()

    else:
        print("\nUsage:")
        print("  python prediction_logger.py stats [symbol]  - Show accuracy stats")
        print("  python prediction_logger.py pending         - Show pending evaluations")
        print("  python prediction_logger.py export          - Export to CSV")
