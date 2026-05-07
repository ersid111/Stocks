from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class TechnicalIndicators:
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_pct: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    obv: Optional[float] = None
    obv_slope: Optional[float] = None
    adx: Optional[float] = None
    recent_df: pd.DataFrame = None


def compute_technical_indicators(ohlcv: pd.DataFrame) -> TechnicalIndicators:
    ind = TechnicalIndicators()
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 30:
        return ind

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    try:
        import pandas_ta as ta
        df.ta.strategy("All")
    except Exception:
        pass

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    def _get_col(*names):
        for n in names:
            if n in df.columns:
                v = df[n].iloc[-1]
                return None if pd.isna(v) else float(v)
        return None

    # RSI
    ind.rsi = _get_col("RSI_14", "rsi_14")
    if ind.rsi is None:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        v = rsi_series.iloc[-1]
        ind.rsi = None if pd.isna(v) else float(v)

    # MACD
    ind.macd = _get_col("MACD_12_26_9", "macd_12_26_9")
    ind.macd_signal = _get_col("MACDs_12_26_9", "macds_12_26_9")
    ind.macd_histogram = _get_col("MACDh_12_26_9", "macdh_12_26_9")
    if ind.macd is None:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        ind.macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
        ind.macd_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
        if ind.macd and ind.macd_signal:
            ind.macd_histogram = ind.macd - ind.macd_signal

    # EMAs
    ind.ema20 = _get_col("EMA_20", "ema_20")
    ind.ema50 = _get_col("EMA_50", "ema_50")
    ind.ema200 = _get_col("EMA_200", "ema_200")
    if ind.ema20 is None:
        v = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ind.ema20 = float(v) if not pd.isna(v) else None
    if ind.ema50 is None:
        v = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ind.ema50 = float(v) if not pd.isna(v) else None
    if ind.ema200 is None and len(close) >= 200:
        v = close.ewm(span=200, adjust=False).mean().iloc[-1]
        ind.ema200 = float(v) if not pd.isna(v) else None

    # Bollinger Bands
    ind.bb_upper = _get_col("BBU_20_2.0", "bbu_20_2.0")
    ind.bb_middle = _get_col("BBM_20_2.0", "bbm_20_2.0")
    ind.bb_lower = _get_col("BBL_20_2.0", "bbl_20_2.0")
    if ind.bb_upper is None:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        ind.bb_upper = float((sma20 + 2 * std20).iloc[-1])
        ind.bb_middle = float(sma20.iloc[-1])
        ind.bb_lower = float((sma20 - 2 * std20).iloc[-1])
    if ind.bb_upper and ind.bb_lower and ind.bb_upper != ind.bb_lower:
        price = close.iloc[-1]
        ind.bb_pct = (price - ind.bb_lower) / (ind.bb_upper - ind.bb_lower)

    # ATR
    ind.atr = _get_col("ATRr_14", "atr_14")
    if ind.atr is None:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        v = tr.rolling(14).mean().iloc[-1]
        ind.atr = float(v) if not pd.isna(v) else None

    # VWAP (rolling daily approximation over available data)
    ind.vwap = _get_col("VWAP_D", "vwap_d")
    if ind.vwap is None:
        typical = (high + low + close) / 3
        cumvol = volume.cumsum()
        cumtp = (typical * volume).cumsum()
        vwap_series = cumtp / cumvol.replace(0, np.nan)
        v = vwap_series.iloc[-1]
        ind.vwap = float(v) if not pd.isna(v) else None

    # OBV
    ind.obv = _get_col("OBV", "obv")
    if ind.obv is None:
        direction = np.sign(close.diff().fillna(0))
        obv_series = (volume * direction).cumsum()
        ind.obv = float(obv_series.iloc[-1])
        slope_window = obv_series.iloc[-20:]
        if len(slope_window) >= 2:
            x = np.arange(len(slope_window))
            slope = np.polyfit(x, slope_window.values, 1)[0]
            ind.obv_slope = float(slope)
    else:
        obv_col = next((c for c in df.columns if c.lower() in ("obv",)), None)
        if obv_col:
            slope_window = df[obv_col].iloc[-20:]
            if len(slope_window) >= 2:
                x = np.arange(len(slope_window))
                slope = np.polyfit(x, slope_window.values, 1)[0]
                ind.obv_slope = float(slope)

    # ADX
    ind.adx = _get_col("ADX_14", "adx_14")
    if ind.adx is None:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        up_move = high.diff()
        down_move = -low.diff()
        dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        di_plus = 100 * dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
        di_minus = 100 * dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        adx_series = dx.ewm(alpha=1/14, adjust=False).mean()
        v = adx_series.iloc[-1]
        ind.adx = float(v) if not pd.isna(v) else None

    ind.recent_df = df.tail(20)
    return ind
