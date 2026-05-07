from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import pandas as pd
import yfinance as yf


@dataclass
class OptionsChain:
    ticker: str
    spot_price: float
    expiries: List[str] = field(default_factory=list)
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    puts: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: Optional[str] = None


def _enrich_chain(df: pd.DataFrame, spot: float, expiry_date: str) -> pd.DataFrame:
    df = df.copy()
    expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    dte = max((expiry_dt - datetime.now()).days, 0)
    df["dte"] = dte
    df["expiry"] = expiry_date
    df["mid"] = (df.get("bid", 0) + df.get("ask", 0)) / 2
    df["mid"] = df["mid"].where(df["mid"] > 0, df.get("lastPrice", 0))
    for col in ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
    df["openInterest"] = df["openInterest"].fillna(0).astype(int)
    df["volume"] = df["volume"].fillna(0).astype(int)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(0.0)
    return df


def fetch_options_chain(ticker: str, spot_price: float, max_expiries: int = 6) -> OptionsChain:
    chain = OptionsChain(ticker=ticker, spot_price=spot_price)
    t = yf.Ticker(ticker)

    try:
        all_expiries = t.options
        if not all_expiries:
            chain.error = "No options data available"
            return chain
    except Exception as e:
        chain.error = str(e)
        return chain

    expiries = list(all_expiries[:max_expiries])
    chain.expiries = expiries

    all_calls, all_puts = [], []
    for expiry in expiries:
        try:
            opt = t.option_chain(expiry)
            if not opt.calls.empty:
                all_calls.append(_enrich_chain(opt.calls, spot_price, expiry))
            if not opt.puts.empty:
                all_puts.append(_enrich_chain(opt.puts, spot_price, expiry))
        except Exception:
            continue

    if all_calls:
        chain.calls = pd.concat(all_calls, ignore_index=True)
    if all_puts:
        chain.puts = pd.concat(all_puts, ignore_index=True)

    return chain
