from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import pytest

from data.market_data import fetch_market_snapshot


def make_mock_info(overrides=None):
    info = {
        "currentPrice": 150.0,
        "previousClose": 148.0,
        "marketCap": 2_500_000_000_000,
        "volume": 80_000_000,
        "averageVolume10days": 70_000_000,
        "beta": 1.25,
        "trailingPE": 28.5,
        "forwardPE": 25.0,
        "shortPercentOfFloat": 0.007,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "longName": "Apple Inc.",
        "fiftyTwoWeekHigh": 199.0,
        "fiftyTwoWeekLow": 124.0,
    }
    if overrides:
        info.update(overrides)
    return info


def make_mock_history():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    return pd.DataFrame({
        "Open": [149.0] * 252,
        "High": [151.0] * 252,
        "Low": [148.0] * 252,
        "Close": [150.0] * 252,
        "Volume": [80_000_000] * 252,
    }, index=dates)


# ---------------------------------------------------------------------------
# Full happy path
# ---------------------------------------------------------------------------

class TestFetchMarketSnapshot:
    def test_normal_case_all_fields_populated(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info()
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.ticker == "AAPL"
        assert snap.current_price == 150.0
        assert snap.prev_close == 148.0
        assert snap.day_change_pct is not None
        assert snap.market_cap == 2_500_000_000_000
        assert snap.company_name == "Apple Inc."
        assert not snap.ohlcv.empty

    def test_day_change_pct_calculated_correctly(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info({"currentPrice": 150.0, "previousClose": 148.0})
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        expected = (150.0 - 148.0) / 148.0 * 100
        assert abs(snap.day_change_pct - expected) < 1e-6

    def test_current_price_fallback_to_regular_market_price(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info({
            "currentPrice": None,
            "regularMarketPrice": 149.5,
        })
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.current_price == 149.5

    def test_volume_fallback_to_regular_market_volume(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info({"volume": None, "regularMarketVolume": 75_000_000})
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.volume_today == 75_000_000

    def test_avg_volume_fallback(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info({
            "averageVolume10days": None,
            "averageDailyVolume10Day": 68_000_000,
        })
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.avg_volume_10d == 68_000_000

    def test_company_name_fallback_to_short_name(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info({"longName": None, "shortName": "Apple"})
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.company_name == "Apple"

    def test_info_exception_leaves_fields_none(self):
        mock_ticker = MagicMock()
        type(mock_ticker).info = PropertyMock(side_effect=Exception("API error"))
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.current_price is None or snap.current_price == 150.0  # From ohlcv fallback
        assert snap.market_cap is None

    def test_history_exception_leaves_ohlcv_empty(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info()
        mock_ticker.history.side_effect = Exception("Network error")

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.ohlcv.empty

    def test_empty_history_leaves_ohlcv_empty(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.ohlcv.empty

    def test_current_price_filled_from_ohlcv_when_info_fails(self):
        mock_ticker = MagicMock()
        type(mock_ticker).info = PropertyMock(side_effect=Exception("API error"))
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        # current_price should be filled from last OHLCV close
        assert snap.current_price == 150.0

    def test_no_day_change_pct_when_prev_close_zero(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info({"previousClose": 0, "currentPrice": 150.0})
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert snap.day_change_pct is None

    def test_ohlcv_has_correct_columns(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info()
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert list(snap.ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_ohlcv_index_is_datetime(self):
        mock_ticker = MagicMock()
        mock_ticker.info = make_mock_info()
        mock_ticker.history.return_value = make_mock_history()

        with patch("data.market_data.yf.Ticker", return_value=mock_ticker):
            snap = fetch_market_snapshot("AAPL")

        assert pd.api.types.is_datetime64_any_dtype(snap.ohlcv.index)
