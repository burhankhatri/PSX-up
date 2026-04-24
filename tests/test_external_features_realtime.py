import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.external_features import fetch_asian_market_realtime


def _market_df(base_price: float) -> pd.DataFrame:
    # Use a fixed start date so list-vs-index lengths always match. Earlier
    # this used pd.bdate_range(end=now, periods=5), but on weekends pandas
    # 3.0 returns only 4 dates for that range — values list of 5 then
    # mismatched the index of 4, raising ValueError in DataFrame.__init__.
    dates = pd.bdate_range(start="2026-01-05", periods=5)
    closes = [base_price - 30.0, base_price - 20.0, base_price - 10.0, base_price, base_price + 15.0]
    opens = [p - 5.0 for p in closes]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [p + 10.0 for p in closes],
            "Low": [p - 10.0 for p in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )


class AsianRealtimeCacheTests(unittest.TestCase):
    def test_invalid_cached_no_data_is_ignored_when_live_fetch_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "asian_markets.json"
            cache_file.write_text(
                json.dumps(
                    {
                        "nikkei": {"current": None, "prev_close": None, "change_pct": None, "open_gap_pct": None, "status": "no_data"},
                        "kospi": {"current": None, "prev_close": None, "change_pct": None, "open_gap_pct": None, "status": "no_data"},
                        "asian_risk_off_signal": 0.0,
                        "crash_warning": False,
                        "crash_severity": "none",
                        "warning_message": "",
                        "checked_at": pd.Timestamp.now().isoformat(),
                    }
                )
            )

            def _download(ticker, period="5d", progress=False):
                if ticker == "^N225":
                    return _market_df(54000.0)
                if ticker == "^KS11":
                    return _market_df(5500.0)
                raise AssertionError(f"unexpected ticker: {ticker}")

            with patch("backend.external_features.ASIAN_CACHE_FILE", cache_file), patch(
                "backend.external_features.yf.download", side_effect=_download
            ):
                result = fetch_asian_market_realtime()

            self.assertIsNotNone(result["nikkei"]["current"])
            self.assertIsNotNone(result["kospi"]["current"])
            self.assertNotEqual(result["nikkei"]["status"], "no_data")
            self.assertNotEqual(result["kospi"]["status"], "no_data")

            cached = json.loads(cache_file.read_text())
            self.assertIsNotNone(cached["nikkei"]["current"])
            self.assertIsNotNone(cached["kospi"]["current"])

    def test_failed_live_fetch_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "asian_markets.json"

            with patch("backend.external_features.ASIAN_CACHE_FILE", cache_file), patch(
                "backend.external_features.yf.download", return_value=pd.DataFrame()
            ):
                result = fetch_asian_market_realtime()

            self.assertEqual(result["nikkei"]["status"], "no_data")
            self.assertEqual(result["kospi"]["status"], "no_data")
            self.assertFalse(cache_file.exists())


if __name__ == "__main__":
    unittest.main()
