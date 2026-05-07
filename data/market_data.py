from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import yfinance as yf


@dataclass
class MarketSnapshot:
    ticker: str
    current_price: Optional[float] = None
    prev_close: Optional[float] = None
    day_change_pct: Optional[float] = None
    market_cap: Optional[float] = None
    volume_today: Optional[int] = None
    avg_volume_10d: Optional[float] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    short_float: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    company_name: Optional[str] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)


def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    t = yf.Ticker(ticker)
    snap = MarketSnapshot(ticker=ticker)

    try:
        info = t.info
        snap.current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        snap.prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if snap.current_price and snap.prev_close and snap.prev_close != 0:
            snap.day_change_pct = (snap.current_price - snap.prev_close) / snap.prev_close * 100
        snap.market_cap = info.get("marketCap")
        snap.volume_today = info.get("volume") or info.get("regularMarketVolume")
        snap.avg_volume_10d = info.get("averageVolume10days") or info.get("averageDailyVolume10Day")
        snap.beta = info.get("beta")
        snap.pe_ratio = info.get("trailingPE")
        snap.forward_pe = info.get("forwardPE")
        snap.short_float = info.get("shortPercentOfFloat")
        snap.sector = info.get("sector")
        snap.industry = info.get("industry")
        snap.company_name = info.get("longName") or info.get("shortName")
        snap.fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        snap.fifty_two_week_low = info.get("fiftyTwoWeekLow")
    except Exception:
        pass

    try:
        hist = t.history(period="1y")
        if not hist.empty:
            snap.ohlcv = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            snap.ohlcv.index = pd.to_datetime(snap.ohlcv.index)
            if snap.current_price is None and not snap.ohlcv.empty:
                snap.current_price = float(snap.ohlcv["Close"].iloc[-1])
    except Exception:
        pass

    return snap
