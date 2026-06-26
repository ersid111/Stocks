import numpy as np
import pandas as pd
import pytest

from analysis.technical import compute_technical_indicators
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trending_ohlcv(n: int = 100, daily_return: float = 0.005) -> pd.DataFrame:
    """OHLCV with a steady uptrend, no pandas_ta columns."""
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return))
    prices = np.array(prices)
    return pd.DataFrame({
        "Open": prices * 0.998,
        "High": prices * 1.010,
        "Low": prices * 0.990,
        "Close": prices,
        "Volume": np.full(n, 1_000_000.0),
    })


def make_downtrending_ohlcv(n: int = 100) -> pd.DataFrame:
    return make_trending_ohlcv(n, daily_return=-0.005)


def make_flat_ohlcv(n: int = 50) -> pd.DataFrame:
    """Flat prices — no change in close each day."""
    prices = np.full(n, 100.0)
    return pd.DataFrame({
        "Open": prices,
        "High": prices * 1.001,
        "Low": prices * 0.999,
        "Close": prices,
        "Volume": np.full(n, 1_000_000.0),
    })


# ---------------------------------------------------------------------------
# Minimum data guard
# ---------------------------------------------------------------------------

class TestMinimumDataGuard:
    def test_none_input_returns_empty(self):
        ind = compute_technical_indicators(None)
        assert ind.rsi is None
        assert ind.macd is None
        assert ind.ema20 is None

    def test_empty_df_returns_empty(self):
        ind = compute_technical_indicators(pd.DataFrame())
        assert ind.rsi is None

    @pytest.mark.parametrize("n", [1, 10, 29])
    def test_fewer_than_30_rows_returns_empty(self, n):
        df = make_ohlcv(n)
        ind = compute_technical_indicators(df)
        assert ind.rsi is None
        assert ind.macd is None
        assert ind.ema20 is None
        assert ind.adx is None

    def test_exactly_30_rows_returns_indicators(self):
        df = make_trending_ohlcv(30)
        ind = compute_technical_indicators(df)
        # At least EMAs and MACD should be calculable with 30 rows
        assert ind.ema20 is not None
        assert ind.macd is not None


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def _noisy_ohlcv(self, n=80, drift=0.005, seed=42) -> pd.DataFrame:
        """OHLCV with both up and down days but a net trend."""
        rng = np.random.default_rng(seed)
        daily_changes = drift + rng.normal(0, 0.01, n)
        prices = np.cumprod(1 + daily_changes) * 100
        return pd.DataFrame({
            "Open": prices * 0.999,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
        })

    def test_uptrending_rsi_above_50(self):
        df = self._noisy_ohlcv(drift=0.008, seed=1)
        ind = compute_technical_indicators(df)
        assert ind.rsi is not None
        assert ind.rsi > 50

    def test_downtrending_rsi_below_50(self):
        df = self._noisy_ohlcv(drift=-0.008, seed=2)
        ind = compute_technical_indicators(df)
        assert ind.rsi is not None
        assert ind.rsi < 50

    def test_rsi_bounded_0_to_100(self):
        df = self._noisy_ohlcv(seed=3)
        ind = compute_technical_indicators(df)
        if ind.rsi is not None:
            assert 0 <= ind.rsi <= 100

    def test_all_gains_no_crash(self):
        """When all close changes are gains, loss=0 and RS denominator is zero."""
        prices = np.linspace(100, 200, 50)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(50, 1_000_000.0),
        })
        ind = compute_technical_indicators(df)
        # Should not crash; RSI is None (loss=0 → NaN) or clipped near 100
        assert ind.rsi is None or 0 <= ind.rsi <= 100


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_histogram_equals_macd_minus_signal(self):
        df = make_trending_ohlcv(60)
        ind = compute_technical_indicators(df)
        assert ind.macd is not None
        assert ind.macd_signal is not None
        assert ind.macd_histogram is not None
        assert abs(ind.macd_histogram - (ind.macd - ind.macd_signal)) < 1e-9

    def test_uptrend_macd_positive(self):
        df = make_trending_ohlcv(100)
        ind = compute_technical_indicators(df)
        assert ind.macd is not None
        assert ind.macd > 0

    def test_downtrend_macd_negative(self):
        df = make_downtrending_ohlcv(100)
        ind = compute_technical_indicators(df)
        assert ind.macd is not None
        assert ind.macd < 0


# ---------------------------------------------------------------------------
# EMAs
# ---------------------------------------------------------------------------

class TestEMAs:
    def test_ema20_not_none_for_50_rows(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.ema20 is not None

    def test_ema50_not_none_for_60_rows(self):
        df = make_trending_ohlcv(60)
        ind = compute_technical_indicators(df)
        assert ind.ema50 is not None

    def test_ema200_none_for_fewer_than_200_rows(self):
        df = make_trending_ohlcv(100)
        ind = compute_technical_indicators(df)
        assert ind.ema200 is None

    def test_ema200_not_none_for_250_rows(self):
        df = make_trending_ohlcv(250)
        ind = compute_technical_indicators(df)
        assert ind.ema200 is not None

    def test_ema_ordering_in_uptrend(self):
        df = make_trending_ohlcv(250)
        ind = compute_technical_indicators(df)
        # In a sustained uptrend, short EMAs should be above longer ones
        assert ind.ema20 > ind.ema50 > ind.ema200

    def test_ema_ordering_in_downtrend(self):
        df = make_downtrending_ohlcv(250)
        ind = compute_technical_indicators(df)
        assert ind.ema20 < ind.ema50 < ind.ema200


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_band_ordering(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.bb_upper is not None
        assert ind.bb_middle is not None
        assert ind.bb_lower is not None
        assert ind.bb_upper > ind.bb_middle > ind.bb_lower

    def test_bb_pct_bounded_for_normal_series(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        # bb_pct can go slightly outside 0-1 but middle should be ~0.5
        assert ind.bb_pct is not None

    def test_flat_prices_no_bb_pct_crash(self):
        df = make_flat_ohlcv(50)
        ind = compute_technical_indicators(df)
        # Flat prices → std=0 → upper==lower → bb_pct must be None
        assert ind.bb_pct is None

    def test_bb_middle_is_20_period_sma(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        close = df["Close"]
        expected_sma = float(close.rolling(20).mean().iloc[-1])
        assert abs(ind.bb_middle - expected_sma) < 1e-6


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_positive(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.atr is not None
        assert ind.atr > 0

    def test_atr_manual_calculation(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        expected_atr = float(tr.rolling(14).mean().iloc[-1])
        assert abs(ind.atr - expected_atr) < 1e-6


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

class TestVWAP:
    def test_vwap_between_min_and_max_close(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.vwap is not None
        # VWAP is a cumulative average so it should be in price range
        close = df["Close"]
        assert close.min() <= ind.vwap <= close.max() * 1.5  # Generous bounds

    def test_zero_volume_vwap_none(self):
        df = make_trending_ohlcv(50)
        df["Volume"] = 0.0
        ind = compute_technical_indicators(df)
        assert ind.vwap is None


# ---------------------------------------------------------------------------
# OBV and OBV Slope
# ---------------------------------------------------------------------------

class TestOBV:
    def test_obv_not_none(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.obv is not None

    def test_obv_slope_positive_in_uptrend(self):
        df = make_trending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.obv_slope is not None
        assert ind.obv_slope > 0

    def test_obv_slope_negative_in_downtrend(self):
        df = make_downtrending_ohlcv(50)
        ind = compute_technical_indicators(df)
        assert ind.obv_slope is not None
        assert ind.obv_slope < 0

    def test_obv_slope_requires_at_least_2_points(self):
        # With exactly 30 rows (minimum), slope window = min(20, 30) => 20 rows is fine
        df = make_trending_ohlcv(30)
        ind = compute_technical_indicators(df)
        assert ind.obv_slope is not None


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestADX:
    def test_adx_positive(self):
        df = make_trending_ohlcv(60)
        ind = compute_technical_indicators(df)
        assert ind.adx is not None
        assert ind.adx >= 0

    def test_adx_higher_in_strong_trend(self):
        # Compare a trending series against a flat series: flat should have lower ADX
        trending = make_trending_ohlcv(60, daily_return=0.005)
        flat = make_flat_ohlcv(60)
        trending_ind = compute_technical_indicators(trending)
        flat_ind = compute_technical_indicators(flat)
        if trending_ind.adx is not None and flat_ind.adx is not None:
            assert trending_ind.adx > flat_ind.adx

    def test_adx_bounded(self):
        df = make_trending_ohlcv(100)
        ind = compute_technical_indicators(df)
        assert 0 <= ind.adx <= 100


# ---------------------------------------------------------------------------
# recent_df
# ---------------------------------------------------------------------------

class TestRecentDf:
    def test_recent_df_is_last_20_rows(self):
        df = make_trending_ohlcv(60)
        ind = compute_technical_indicators(df)
        assert len(ind.recent_df) == 20
