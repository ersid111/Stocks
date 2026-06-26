from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from data.options_data import _enrich_chain, fetch_options_chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_chain(n_strikes: int = 3, base_strike: float = 100.0) -> pd.DataFrame:
    strikes = [base_strike - 5, base_strike, base_strike + 5][:n_strikes]
    return pd.DataFrame({
        "strike": strikes,
        "bid": [1.0] * n_strikes,
        "ask": [1.5] * n_strikes,
        "lastPrice": [1.2] * n_strikes,
        "impliedVolatility": [0.25] * n_strikes,
        "openInterest": [100] * n_strikes,
        "volume": [50] * n_strikes,
    })


def make_mock_option_chain(expiry: str, n_strikes: int = 3):
    chain = MagicMock()
    chain.calls = make_raw_chain(n_strikes)
    chain.puts = make_raw_chain(n_strikes)
    return chain


# ---------------------------------------------------------------------------
# _enrich_chain
# ---------------------------------------------------------------------------

class TestEnrichChain:
    def test_dte_non_negative_for_future_expiry(self):
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        df = make_raw_chain()
        enriched = _enrich_chain(df, 100.0, future)
        assert (enriched["dte"] >= 0).all()

    def test_dte_is_zero_for_past_expiry(self):
        past = "2000-01-01"
        df = make_raw_chain()
        enriched = _enrich_chain(df, 100.0, past)
        assert (enriched["dte"] == 0).all()

    def test_mid_is_bid_plus_ask_over_2(self):
        df = make_raw_chain()
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        enriched = _enrich_chain(df, 100.0, future)
        expected_mid = (1.0 + 1.5) / 2
        assert (enriched["mid"] == expected_mid).all()

    def test_mid_falls_back_to_last_price_when_bid_ask_zero(self):
        df = make_raw_chain()
        df["bid"] = 0.0
        df["ask"] = 0.0
        df["lastPrice"] = 1.3
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        enriched = _enrich_chain(df, 100.0, future)
        assert (enriched["mid"] == 1.3).all()

    def test_expiry_column_added(self):
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        df = make_raw_chain()
        enriched = _enrich_chain(df, 100.0, future)
        assert "expiry" in enriched.columns
        assert (enriched["expiry"] == future).all()

    def test_nan_open_interest_filled_with_zero(self):
        df = make_raw_chain()
        df["openInterest"] = float("nan")
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        enriched = _enrich_chain(df, 100.0, future)
        assert (enriched["openInterest"] == 0).all()
        assert enriched["openInterest"].dtype in (int, "int64", "int32")

    def test_nan_volume_filled_with_zero(self):
        df = make_raw_chain()
        df["volume"] = float("nan")
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        enriched = _enrich_chain(df, 100.0, future)
        assert (enriched["volume"] == 0).all()

    def test_missing_columns_filled_with_zeros(self):
        df = pd.DataFrame({"strike": [100.0]})  # Missing all other columns
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        enriched = _enrich_chain(df, 100.0, future)
        for col in ["bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]:
            assert col in enriched.columns

    def test_does_not_mutate_original_df(self):
        df = make_raw_chain()
        original_cols = list(df.columns)
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        _enrich_chain(df, 100.0, future)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# fetch_options_chain
# ---------------------------------------------------------------------------

class TestFetchOptionsChain:
    def test_options_exception_sets_error(self):
        mock_ticker = MagicMock()
        mock_ticker.options = None
        type(mock_ticker).options = property(lambda self: (_ for _ in ()).throw(Exception("No options")))

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0)

        assert chain.error is not None
        assert chain.calls.empty
        assert chain.puts.empty

    def test_empty_expiries_sets_error(self):
        mock_ticker = MagicMock()
        mock_ticker.options = []

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0)

        assert chain.error == "No options data available"

    def test_max_expiries_limits_fetched_expiries(self):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2025-01-17", "2025-02-21", "2025-03-21", "2025-04-17"]
        mock_ticker.option_chain.return_value = make_mock_option_chain("2025-01-17")

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0, max_expiries=2)

        assert mock_ticker.option_chain.call_count == 2
        assert len(chain.expiries) == 2

    def test_single_expiry_failure_skipped_silently(self):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2025-01-17", "2025-02-21"]

        def option_chain_side_effect(expiry):
            if expiry == "2025-01-17":
                raise Exception("Network error")
            return make_mock_option_chain(expiry)

        mock_ticker.option_chain.side_effect = option_chain_side_effect

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0)

        # Second expiry succeeded
        assert not chain.calls.empty

    def test_multiple_expiries_concatenated(self):
        mock_ticker = MagicMock()
        expiries = ["2025-01-17", "2025-02-21"]
        mock_ticker.options = expiries

        def option_chain_side_effect(expiry):
            return make_mock_option_chain(expiry, n_strikes=3)

        mock_ticker.option_chain.side_effect = option_chain_side_effect

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0)

        # 2 expiries × 3 strikes each = 6 rows
        assert len(chain.calls) == 6
        assert len(chain.puts) == 6

    def test_spot_price_stored(self):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2025-01-17"]
        mock_ticker.option_chain.return_value = make_mock_option_chain("2025-01-17")

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 155.0)

        assert chain.spot_price == 155.0

    def test_expiries_stored_on_chain(self):
        mock_ticker = MagicMock()
        expiries = ["2025-01-17", "2025-02-21"]
        mock_ticker.options = expiries
        mock_ticker.option_chain.return_value = make_mock_option_chain("2025-01-17")

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0, max_expiries=2)

        assert chain.expiries == expiries

    def test_empty_calls_from_option_chain_not_appended(self):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2025-01-17"]
        mock_chain = MagicMock()
        mock_chain.calls = pd.DataFrame()  # Empty calls
        mock_chain.puts = make_raw_chain()
        mock_ticker.option_chain.return_value = mock_chain

        with patch("data.options_data.yf.Ticker", return_value=mock_ticker):
            chain = fetch_options_chain("AAPL", 150.0)

        assert chain.calls.empty
        assert not chain.puts.empty
