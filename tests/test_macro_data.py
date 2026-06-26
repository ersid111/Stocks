from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _fetch_fred_series
# ---------------------------------------------------------------------------

class TestFetchFredSeries:
    def test_no_api_key_returns_none(self):
        with patch("data.macro_data.FRED_API_KEY", ""):
            from data.macro_data import _fetch_fred_series
            result = _fetch_fred_series("DFF")
        assert result is None

    def test_http_exception_returns_none(self):
        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", side_effect=Exception("Timeout")):
                from data.macro_data import _fetch_fred_series
                result = _fetch_fred_series("DFF")
        assert result is None

    def test_dot_sentinel_value_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": "."}]}
        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", return_value=mock_resp):
                from data.macro_data import _fetch_fred_series
                result = _fetch_fred_series("DFF")
        assert result is None

    def test_valid_value_returns_float(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": "5.33"}]}
        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", return_value=mock_resp):
                from data.macro_data import _fetch_fred_series
                result = _fetch_fred_series("DFF")
        assert result == 5.33

    def test_empty_observations_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": []}
        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", return_value=mock_resp):
                from data.macro_data import _fetch_fred_series
                result = _fetch_fred_series("DFF")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_macro_context
# ---------------------------------------------------------------------------

class TestFetchMacroContext:
    def test_no_fred_key_sets_error(self):
        with patch("data.macro_data.FRED_API_KEY", ""):
            from data.macro_data import fetch_macro_context
            ctx = fetch_macro_context()
        assert ctx.error is not None
        assert "FRED_API_KEY" in ctx.error
        assert ctx.fed_funds_rate is None

    def test_all_fred_series_fetched(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": "5.0"}]}
        mock_vix_ticker = MagicMock()
        mock_vix_ticker.history.return_value = pd.DataFrame(
            {"Close": [18.5]}, index=pd.date_range("2024-01-01", periods=1)
        )

        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", return_value=mock_resp):
                with patch("data.macro_data.yf.Ticker", return_value=mock_vix_ticker):
                    from data.macro_data import fetch_macro_context
                    ctx = fetch_macro_context()

        assert ctx.fed_funds_rate == 5.0
        assert ctx.inflation_expectations_10y == 5.0
        assert ctx.yield_curve_10y2y == 5.0
        assert ctx.vix == 18.5
        assert ctx.error is None

    def test_vix_exception_leaves_vix_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": "5.0"}]}

        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", return_value=mock_resp):
                with patch("data.macro_data.yf.Ticker", side_effect=Exception("VIX error")):
                    from data.macro_data import fetch_macro_context
                    ctx = fetch_macro_context()

        assert ctx.vix is None
        assert ctx.fed_funds_rate == 5.0  # FRED still worked

    def test_empty_vix_history_leaves_vix_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": "5.0"}]}
        mock_vix_ticker = MagicMock()
        mock_vix_ticker.history.return_value = pd.DataFrame()

        with patch("data.macro_data.FRED_API_KEY", "test_key"):
            with patch("data.macro_data.requests.get", return_value=mock_resp):
                with patch("data.macro_data.yf.Ticker", return_value=mock_vix_ticker):
                    from data.macro_data import fetch_macro_context
                    ctx = fetch_macro_context()

        assert ctx.vix is None


# ---------------------------------------------------------------------------
# fetch_news_sentiment
# ---------------------------------------------------------------------------

class TestFetchNewsSentiment:
    def _make_news(self, titles):
        return [{"title": t} for t in titles]

    def test_yfinance_exception_returns_default(self):
        with patch("data.macro_data.yf.Ticker", side_effect=Exception("API error")):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert ns.sentiment_score == 0.0
        assert ns.summary == "No news data available"

    def test_empty_news_returns_default(self):
        mock_ticker = MagicMock()
        mock_ticker.news = []
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert ns.sentiment_score == 0.0

    def test_positive_headlines_positive_score(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([
            "AAPL stock rise after strong earnings beat",
            "Apple growth record revenue",
            "Analysts upgrade AAPL on bullish outlook",
        ])
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert ns.sentiment_score > 0

    def test_negative_headlines_negative_score(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([
            "AAPL revenue miss disappoints",
            "Apple cut guidance on weak demand",
            "Analysts downgrade AAPL on risk concern",
        ])
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert ns.sentiment_score < 0

    def test_positive_score_summary_contains_positive(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([
            "Apple beat expectations strong growth",
            "AAPL record high bullish trend",
            "Upgrade to buy on outperform gain",
        ] * 5)
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        if ns.sentiment_score > 0.2:
            assert "Positive" in ns.summary

    def test_negative_score_summary_contains_negative(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([
            "Apple miss earnings weak outlook",
            "AAPL fall on downgrade bearish",
            "Risk concern cuts underperform",
        ] * 5)
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        if ns.sentiment_score < -0.2:
            assert "Negative" in ns.summary

    def test_headlines_capped_at_10(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([f"Headline {i}" for i in range(15)])
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert len(ns.headlines) <= 10

    def test_score_bounded_between_minus_one_and_one(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([
            "Apple strong growth beat record gain rise bullish",
        ] * 15)
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert -1.0 <= ns.sentiment_score <= 1.0

    def test_mixed_headlines_neutral_score(self):
        mock_ticker = MagicMock()
        mock_ticker.news = self._make_news([
            "Apple rise in revenue",
            "AAPL miss on guidance",
        ])
        with patch("data.macro_data.yf.Ticker", return_value=mock_ticker):
            from data.macro_data import fetch_news_sentiment
            ns = fetch_news_sentiment("AAPL")
        assert ns.sentiment_score == 0.0  # (1-1)/(1+1) = 0
