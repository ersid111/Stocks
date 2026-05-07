from dataclasses import dataclass, field
from typing import Optional, List, Dict
import requests
from config import FRED_API_KEY, FRED_BASE_URL
import yfinance as yf


@dataclass
class MacroContext:
    fed_funds_rate: Optional[float] = None
    inflation_expectations_10y: Optional[float] = None
    yield_curve_10y2y: Optional[float] = None
    vix: Optional[float] = None
    error: Optional[str] = None


@dataclass
class NewsSentiment:
    headlines: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    summary: str = "No news data available"


def _fetch_fred_series(series_id: str) -> Optional[float]:
    if not FRED_API_KEY:
        return None
    try:
        resp = requests.get(
            FRED_BASE_URL,
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc",
            },
            timeout=10,
        )
        data = resp.json()
        obs = data.get("observations", [])
        if obs:
            val = obs[0].get("value", ".")
            return float(val) if val != "." else None
    except Exception:
        return None


def fetch_macro_context() -> MacroContext:
    ctx = MacroContext()
    if not FRED_API_KEY:
        ctx.error = "FRED_API_KEY not set — macro data unavailable"
        return ctx

    ctx.fed_funds_rate = _fetch_fred_series("DFF")
    ctx.inflation_expectations_10y = _fetch_fred_series("T10YIE")
    ctx.yield_curve_10y2y = _fetch_fred_series("T10Y2Y")

    try:
        vix_data = yf.Ticker("^VIX").history(period="1d")
        if not vix_data.empty:
            ctx.vix = float(vix_data["Close"].iloc[-1])
    except Exception:
        pass

    return ctx


def fetch_news_sentiment(ticker: str) -> NewsSentiment:
    ns = NewsSentiment()
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return ns

        headlines = []
        positive_words = {"up", "rise", "gain", "beat", "strong", "growth", "record", "bullish", "outperform", "upgrade"}
        negative_words = {"down", "fall", "drop", "miss", "weak", "loss", "cut", "bearish", "underperform", "downgrade", "risk", "concern"}

        pos_count = neg_count = 0
        for item in news[:15]:
            title = item.get("title", "")
            if title:
                headlines.append(title)
                title_lower = title.lower()
                if any(w in title_lower for w in positive_words):
                    pos_count += 1
                if any(w in title_lower for w in negative_words):
                    neg_count += 1

        ns.headlines = headlines[:10]
        total = pos_count + neg_count
        if total > 0:
            ns.sentiment_score = (pos_count - neg_count) / total
        else:
            ns.sentiment_score = 0.0

        if ns.sentiment_score > 0.2:
            ns.summary = f"Positive sentiment ({pos_count} positive, {neg_count} negative headlines)"
        elif ns.sentiment_score < -0.2:
            ns.summary = f"Negative sentiment ({pos_count} positive, {neg_count} negative headlines)"
        else:
            ns.summary = f"Mixed/neutral sentiment ({pos_count} positive, {neg_count} negative headlines)"
    except Exception:
        pass

    return ns
