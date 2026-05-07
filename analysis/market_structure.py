from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from analysis.technical import TechnicalIndicators


@dataclass
class SRLevel:
    price: float
    touches: int
    level_type: str  # "resistance" or "support"


@dataclass
class VolumeProfile:
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    bins: int = 50


@dataclass
class TrendAnalysis:
    primary_trend: str = "unknown"
    secondary_trend: str = "unknown"
    short_term_trend: str = "unknown"
    regime: str = "unknown"
    ema_alignment: str = "unknown"
    rsi_zone: str = "neutral"
    adx_regime: str = "unknown"


@dataclass
class AccumulationSignal:
    signal: str = "neutral"
    obv_slope_positive: bool = False
    up_vol_ratio: Optional[float] = None


def find_support_resistance(ohlcv: pd.DataFrame, window: int = 10, tolerance: float = 0.005) -> List[SRLevel]:
    if ohlcv is None or ohlcv.empty or len(ohlcv) < window * 2 + 1:
        return []

    close = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv["close"]
    high = ohlcv["High"] if "High" in ohlcv.columns else ohlcv["high"]
    low = ohlcv["Low"] if "Low" in ohlcv.columns else ohlcv["low"]

    w = 2 * window + 1
    pivot_highs = high[high == high.rolling(w, center=True).max()].dropna()
    pivot_lows = low[low == low.rolling(w, center=True).min()].dropna()

    def cluster_levels(prices, level_type):
        levels = sorted(prices.tolist())
        clusters = []
        for p in levels:
            placed = False
            for c in clusters:
                if abs(p - c["center"]) / c["center"] <= tolerance:
                    c["count"] += 1
                    c["center"] = (c["center"] * (c["count"] - 1) + p) / c["count"]
                    placed = True
                    break
            if not placed:
                clusters.append({"center": p, "count": 1, "type": level_type})
        return [SRLevel(price=round(c["center"], 2), touches=c["count"], level_type=c["type"])
                for c in clusters if c["count"] >= 2]

    supports = cluster_levels(pivot_lows, "support")
    resistances = cluster_levels(pivot_highs, "resistance")
    all_levels = sorted(supports + resistances, key=lambda x: x.price)
    return all_levels


def classify_trend(ind: TechnicalIndicators, current_price: float) -> TrendAnalysis:
    ta = TrendAnalysis()

    if ind.ema200 and current_price:
        ta.primary_trend = "bullish" if current_price > ind.ema200 else "bearish"
    if ind.ema50 and ind.ema200:
        ta.secondary_trend = "bullish" if ind.ema50 > ind.ema200 else "bearish"
    if ind.ema20 and ind.ema50:
        ta.short_term_trend = "bullish" if ind.ema20 > ind.ema50 else "bearish"

    emas = [v for v in [ind.ema20, ind.ema50, ind.ema200] if v is not None]
    if len(emas) == 3:
        if emas[0] > emas[1] > emas[2]:
            ta.ema_alignment = "fully_bullish"
        elif emas[0] < emas[1] < emas[2]:
            ta.ema_alignment = "fully_bearish"
        else:
            ta.ema_alignment = "mixed"
    elif len(emas) == 2:
        ta.ema_alignment = "bullish" if emas[0] > emas[1] else "bearish"

    if ind.rsi is not None:
        if ind.rsi >= 70:
            ta.rsi_zone = "overbought"
        elif ind.rsi <= 30:
            ta.rsi_zone = "oversold"
        elif ind.rsi >= 55:
            ta.rsi_zone = "bullish"
        elif ind.rsi <= 45:
            ta.rsi_zone = "bearish"
        else:
            ta.rsi_zone = "neutral"

    if ind.adx is not None:
        if ind.adx < 20:
            ta.adx_regime = "ranging"
            ta.regime = "mean_reverting"
        elif ind.adx < 25:
            ta.adx_regime = "emerging_trend"
            ta.regime = "transitioning"
        elif ind.adx < 40:
            ta.adx_regime = "strong_trend"
            ta.regime = "trending"
        else:
            ta.adx_regime = "extreme_trend"
            ta.regime = "trending"

    return ta


def compute_volume_profile(ohlcv: pd.DataFrame, bins: int = 50) -> VolumeProfile:
    vp = VolumeProfile(bins=bins)
    if ohlcv is None or ohlcv.empty:
        return vp

    close = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv["close"]
    volume = ohlcv["Volume"] if "Volume" in ohlcv.columns else ohlcv["volume"]

    price_min, price_max = close.min(), close.max()
    if price_min == price_max:
        return vp

    edges = np.linspace(price_min, price_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_by_bin = np.zeros(bins)

    for price, vol in zip(close, volume):
        idx = min(int((price - price_min) / (price_max - price_min) * bins), bins - 1)
        vol_by_bin[idx] += vol

    vp.poc = float(centers[np.argmax(vol_by_bin)])

    total_vol = vol_by_bin.sum()
    target = total_vol * 0.70
    poc_idx = np.argmax(vol_by_bin)
    lo, hi = poc_idx, poc_idx
    accumulated = vol_by_bin[poc_idx]

    while accumulated < target and (lo > 0 or hi < bins - 1):
        expand_down = (lo > 0) and (hi >= bins - 1 or vol_by_bin[lo - 1] >= vol_by_bin[hi + 1])
        if expand_down:
            lo -= 1
            accumulated += vol_by_bin[lo]
        else:
            hi += 1
            accumulated += vol_by_bin[hi]

    vp.val = float(centers[lo])
    vp.vah = float(centers[hi])
    return vp


def detect_institutional_accumulation(ind: TechnicalIndicators, ohlcv: pd.DataFrame) -> AccumulationSignal:
    sig = AccumulationSignal()
    if ind.obv_slope is not None:
        sig.obv_slope_positive = ind.obv_slope > 0

    if ohlcv is not None and not ohlcv.empty and len(ohlcv) >= 20:
        close = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv["close"]
        volume = ohlcv["Volume"] if "Volume" in ohlcv.columns else ohlcv["volume"]
        recent_close = close.iloc[-20:]
        recent_vol = volume.iloc[-20:]
        up_days = recent_vol[recent_close.diff() > 0].sum()
        down_days = recent_vol[recent_close.diff() < 0].sum()
        total = up_days + down_days
        if total > 0:
            sig.up_vol_ratio = float(up_days / total)

    bullish_points = 0
    bearish_points = 0
    if sig.obv_slope_positive:
        bullish_points += 1
    else:
        bearish_points += 1
    if sig.up_vol_ratio is not None:
        if sig.up_vol_ratio > 0.55:
            bullish_points += 1
        elif sig.up_vol_ratio < 0.45:
            bearish_points += 1

    if bullish_points > bearish_points:
        sig.signal = "accumulation"
    elif bearish_points > bullish_points:
        sig.signal = "distribution"
    else:
        sig.signal = "neutral"

    return sig
