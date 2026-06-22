import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalIndicators
from analysis.market_structure import TrendAnalysis, SRLevel, VolumeProfile
from analysis.options_math import PCMetrics


def make_ohlcv(n: int = 252, seed: int = 42, trending: bool = True) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with no pandas_ta columns."""
    rng = np.random.default_rng(seed)
    if trending:
        prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    else:
        prices = np.full(n, 100.0)
    prices = np.abs(prices) + 1  # Ensure positive

    opens = prices * (1 + rng.uniform(-0.005, 0.005, n))
    highs = prices * (1 + rng.uniform(0.002, 0.015, n))
    lows = prices * (1 - rng.uniform(0.002, 0.015, n))
    closes = prices
    volumes = rng.integers(500_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })
    return df


@pytest.fixture
def ohlcv_252():
    return make_ohlcv(252)


@pytest.fixture
def ohlcv_50():
    return make_ohlcv(50)


@pytest.fixture
def ohlcv_20():
    return make_ohlcv(20)


@pytest.fixture
def sample_calls_df():
    spot = 100.0
    strikes = [90, 95, 100, 105, 110]
    return pd.DataFrame({
        "strike": strikes,
        "openInterest": [200, 300, 500, 400, 250],
        "impliedVolatility": [0.30, 0.25, 0.22, 0.23, 0.26],
        "dte": [30] * 5,
        "expiry": ["2025-01-17"] * 5,
        "bid": [11.0, 6.5, 3.0, 1.2, 0.4],
        "ask": [11.5, 7.0, 3.5, 1.5, 0.6],
        "lastPrice": [11.2, 6.7, 3.2, 1.3, 0.5],
        "volume": [100, 150, 200, 130, 80],
        "mid": [11.25, 6.75, 3.25, 1.35, 0.50],
    })


@pytest.fixture
def sample_puts_df():
    strikes = [90, 95, 100, 105, 110]
    return pd.DataFrame({
        "strike": strikes,
        "openInterest": [250, 350, 600, 350, 200],
        "impliedVolatility": [0.35, 0.28, 0.22, 0.24, 0.27],
        "dte": [30] * 5,
        "expiry": ["2025-01-17"] * 5,
        "bid": [0.3, 0.9, 2.8, 5.8, 10.5],
        "ask": [0.5, 1.1, 3.2, 6.2, 11.0],
        "lastPrice": [0.4, 1.0, 3.0, 6.0, 10.7],
        "volume": [80, 120, 180, 110, 70],
        "mid": [0.40, 1.00, 3.00, 6.00, 10.75],
    })


@pytest.fixture
def sample_technical_indicators():
    ind = TechnicalIndicators()
    ind.rsi = 58.0
    ind.macd = 0.5
    ind.macd_signal = 0.3
    ind.macd_histogram = 0.2
    ind.ema20 = 105.0
    ind.ema50 = 100.0
    ind.ema200 = 95.0
    ind.bb_upper = 112.0
    ind.bb_middle = 103.0
    ind.bb_lower = 94.0
    ind.bb_pct = 0.55
    ind.atr = 2.5
    ind.vwap = 102.0
    ind.obv = 1_500_000.0
    ind.obv_slope = 5000.0
    ind.adx = 28.0
    return ind


@pytest.fixture
def sample_trend_analysis():
    ta = TrendAnalysis()
    ta.primary_trend = "bullish"
    ta.secondary_trend = "bullish"
    ta.short_term_trend = "bullish"
    ta.regime = "trending"
    ta.ema_alignment = "fully_bullish"
    ta.rsi_zone = "bullish"
    ta.adx_regime = "strong_trend"
    return ta


@pytest.fixture
def sample_pc_metrics():
    pc = PCMetrics()
    pc.oi_put_call_ratio = 0.85
    pc.vol_put_call_ratio = 0.90
    pc.dollar_put_call_ratio = 0.80
    pc.total_call_oi = 2000
    pc.total_put_oi = 1700
    return pc


@pytest.fixture
def sample_sr_levels():
    return [
        SRLevel(price=90.0, touches=3, level_type="support"),
        SRLevel(price=95.0, touches=2, level_type="support"),
        SRLevel(price=110.0, touches=2, level_type="resistance"),
        SRLevel(price=115.0, touches=3, level_type="resistance"),
    ]


@pytest.fixture
def sample_volume_profile():
    vp = VolumeProfile()
    vp.poc = 100.0
    vp.vah = 106.0
    vp.val = 94.0
    return vp
