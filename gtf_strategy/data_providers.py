"""
Pluggable OHLCV data providers for the GTF backtester.

The strategy only needs candles. This module hides where they come from behind
one function, so no broker account is required to run a backtest.

Providers
---------
``yfinance``  (default) Yahoo Finance. No account, no API key, no token.
              NSE symbols get a ``.NS`` suffix automatically.
``csv``       Local CSV files. No network at all — use this for broker exports,
              NSE bhavcopy data, or any history you already have.
``fyers``     The Fyers API. Requires an account and an access token.

Every provider returns the same thing: a DataFrame indexed by a timezone-aware
UTC ``Date``, with float ``Open/High/Low/Close/Volume`` columns, sorted ascending
and de-duplicated — so the strategy code never has to care about the source.
"""

import os
import time

import pandas as pd

# --- Interval names used by the strategy, mapped per provider -------------
YF_INTERVALS = {
    "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h", "1d": "1d", "1mo": "1mo",
}

FYERS_INTERVALS = {
    "5m": "5", "15m": "15", "30m": "30", "60m": "60", "1d": "D", "1mo": "M",
}

# How far back Yahoo serves each intraday interval, in calendar days.
# Exceeding these returns an empty frame rather than an error, so we warn first.
YF_LOOKBACK_LIMIT_DAYS = {"5m": 60, "15m": 60, "30m": 60, "60m": 730}

CANONICAL_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

IST = "Asia/Kolkata"


class DataProviderError(Exception):
    """Raised when a provider is misconfigured (bad name, missing client)."""


# =========================================================================
# Normalisation
# =========================================================================


def _normalise(df, source):
    """Coerce any provider's frame into the canonical shape, or return empty."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = df.copy()

    # yfinance can hand back MultiIndex columns ("Close", "RELIANCE.NS")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Match columns case-insensitively; Yahoo's "Adj Close" is dropped
    lookup = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in CANONICAL_COLUMNS if c.lower() not in lookup]
    if missing:
        raise DataProviderError(
            f"{source}: response is missing column(s) {missing}. Got: {list(df.columns)}")
    df = df[[lookup[c.lower()] for c in CANONICAL_COLUMNS]]
    df.columns = CANONICAL_COLUMNS

    # The strategy converts timestamps to IST, so the index must be tz-aware.
    # A naive index from an NSE feed is local exchange time, not UTC.
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST, ambiguous="NaT", nonexistent="NaT")
    df.index = idx.tz_convert("UTC")
    df.index.name = "Date"

    df = df[~df.index.isna()]
    df = df.astype({c: "float64" for c in CANONICAL_COLUMNS}, errors="ignore")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df


def _to_nse_symbol(ticker):
    """RELIANCE -> RELIANCE.NS. Indices (^NSEI) and explicit suffixes pass through."""
    t = ticker.strip().upper()
    if t.startswith("^") or "." in t:
        return t
    return f"{t}.NS"


# =========================================================================
# Providers
# =========================================================================


def _fetch_yfinance(ticker, interval, start_date, end_date, retries=2):
    """Yahoo Finance. No credentials required."""
    try:
        import yfinance as yf
    except ImportError:
        raise DataProviderError(
            "yfinance is not installed. Run: pip install yfinance") from None

    if interval not in YF_INTERVALS:
        print(f"  Interval '{interval}' is not supported by the yfinance provider.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    symbol = _to_nse_symbol(ticker)

    # Yahoo's `end` is exclusive, so push it out a day to include END_DATE itself
    end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    limit = YF_LOOKBACK_LIMIT_DAYS.get(interval)
    if limit is not None:
        age_days = (pd.Timestamp.now(tz=IST).normalize()
                    - pd.to_datetime(start_date).tz_localize(IST)).days
        if age_days > limit:
            print(f"  WARNING: Yahoo serves at most ~{limit} days of {interval} history; "
                  f"{start_date} is {age_days} days back. Expect missing or empty data — "
                  f"use a more recent range, a larger interval, or the 'csv' provider.")

    last_error = None
    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(symbol).history(
                start=start_date, end=end_exclusive,
                interval=YF_INTERVALS[interval],
                auto_adjust=True,     # split/dividend adjusted, so zones stay continuous
                actions=False,
                raise_errors=False,
            )
            if len(df):
                return _normalise(df, "yfinance")
            last_error = "no rows returned"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    print(f"  No data for {symbol} ({interval}) from Yahoo: {last_error}")
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _fetch_csv(ticker, interval, start_date, end_date, csv_dir="csv_data"):
    """Local files: <csv_dir>/<TICKER>_<interval>.csv. No network."""
    path = os.path.join(csv_dir, f"{ticker.strip().upper()}_{interval}.csv")
    if not os.path.exists(path):
        print(f"  CSV not found: {path}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(path)

    date_col = next((c for c in df.columns
                     if str(c).strip().lower() in ("date", "datetime", "timestamp", "time")), None)
    if date_col is None:
        raise DataProviderError(
            f"{path}: no date column found (expected one of Date/Datetime/Timestamp/Time)")

    df = df.set_index(date_col)
    df = _normalise(df, f"csv:{os.path.basename(path)}")

    lo = pd.to_datetime(start_date).tz_localize(IST).tz_convert("UTC")
    hi = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).tz_localize(IST).tz_convert("UTC")
    return df[(df.index >= lo) & (df.index < hi)]


def _fetch_fyers(ticker, interval, start_date, end_date, fyers_model=None):
    """The original Fyers API path. Requires a client and a valid access token."""
    if fyers_model is None:
        raise DataProviderError(
            "The 'fyers' provider needs a logged-in client. Call initialize_fyers() first, "
            "or switch DATA_PROVIDER to 'yfinance'.")

    if interval not in FYERS_INTERVALS:
        print(f"  Interval '{interval}' is not supported by the fyers provider.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    request = {
        "symbol": f"NSE:{ticker.strip().upper()}-EQ",
        "resolution": FYERS_INTERVALS[interval],
        "date_format": "1",
        "range_from": start_date,
        "range_to": end_date,
        "cont_flag": "1",
    }

    try:
        response = fyers_model.history(data=request)
    except Exception as e:
        print(f"  Error fetching Fyers data for {ticker}: {e}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    if response.get("code") != 200 or not response.get("candles"):
        print(f"  Fyers API error for {ticker}: {response.get('message', 'no candles')}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.DataFrame(response["candles"],
                      columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    # Fyers epochs are UTC seconds, so this index is already tz-aware
    df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True)
    return _normalise(df.set_index("Date"), "fyers")


PROVIDERS = {
    "yfinance": _fetch_yfinance,
    "csv": _fetch_csv,
    "fyers": _fetch_fyers,
}


def fetch_ohlcv(ticker, interval, start_date, end_date, provider="yfinance", **kwargs):
    """Fetch candles from the chosen provider in the canonical shape.

    Extra keyword arguments are passed through to the provider:
    ``fyers_model=`` for fyers, ``csv_dir=`` for csv.
    """
    key = str(provider).strip().lower()
    if key not in PROVIDERS:
        raise DataProviderError(
            f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[key](ticker, interval, start_date, end_date, **kwargs)
