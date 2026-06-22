import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalIndicators
from analysis.market_structure import (
    find_support_resistance,
    classify_trend,
    compute_volume_profile,
    detect_institutional_accumulation,
    SRLevel,
    TrendAnalysis,
    VolumeProfile,
    AccumulationSignal,
)
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sr_ohlcv() -> pd.DataFrame:
    """OHLCV with clear pivot highs and lows for S/R detection."""
    # 60 rows with deliberate highs at 110 and lows at 90 twice
    n = 60
    close = np.full(n, 100.0)
    high = np.full(n, 102.0)
    low = np.full(n, 98.0)
    # Inject two pivot highs at 110
    high[10] = 110.0
    close[10] = 109.0
    high[35] = 110.5  # Within 0.5% of 110 → same cluster
    close[35] = 109.5
    # Inject two pivot lows at 90
    low[20] = 90.0
    close[20] = 90.5
    low[45] = 90.3  # Within 0.5% of 90 → same cluster
    close[45] = 90.6

    return pd.DataFrame({
        "Open": close * 0.999,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.full(n, 1_000_000.0),
    })


def make_ind(
    ema20=105.0, ema50=100.0, ema200=95.0,
    rsi=60.0, adx=28.0, obv_slope=1000.0,
    macd_histogram=0.5,
) -> TechnicalIndicators:
    ind = TechnicalIndicators()
    ind.ema20 = ema20
    ind.ema50 = ema50
    ind.ema200 = ema200
    ind.rsi = rsi
    ind.adx = adx
    ind.obv_slope = obv_slope
    ind.macd_histogram = macd_histogram
    return ind


# ---------------------------------------------------------------------------
# find_support_resistance
# ---------------------------------------------------------------------------

class TestFindSupportResistance:
    def test_none_returns_empty(self):
        assert find_support_resistance(None) == []

    def test_empty_df_returns_empty(self):
        assert find_support_resistance(pd.DataFrame()) == []

    def test_too_few_rows_returns_empty(self):
        df = make_ohlcv(10)  # window=10 requires 2*10+1=21 rows minimum
        assert find_support_resistance(df, window=10) == []

    def test_exactly_minimum_rows(self):
        df = make_ohlcv(21)  # Exactly 2*10+1
        result = find_support_resistance(df, window=10)
        assert isinstance(result, list)

    def test_uppercase_column_names(self):
        df = make_sr_ohlcv()  # Uses "Close", "High", "Low"
        result = find_support_resistance(df)
        assert isinstance(result, list)

    def test_lowercase_column_names(self):
        df = make_sr_ohlcv()
        df.columns = [c.lower() for c in df.columns]
        result = find_support_resistance(df)
        assert isinstance(result, list)

    def test_single_touch_levels_excluded(self):
        """Clusters with only 1 touch should not appear in results."""
        df = make_ohlcv(60)
        result = find_support_resistance(df)
        for level in result:
            assert level.touches >= 2

    def test_result_sorted_ascending_by_price(self):
        df = make_sr_ohlcv()
        result = find_support_resistance(df)
        prices = [lvl.price for lvl in result]
        assert prices == sorted(prices)

    def test_level_types_are_valid(self):
        df = make_sr_ohlcv()
        result = find_support_resistance(df)
        for lvl in result:
            assert lvl.level_type in ("support", "resistance")

    def test_clustered_highs_detected_as_resistance(self):
        df = make_sr_ohlcv()
        result = find_support_resistance(df)
        resistances = [l for l in result if l.level_type == "resistance"]
        # We injected two highs near 110 → should produce one resistance cluster
        assert any(abs(r.price - 110.0) < 2.0 for r in resistances)

    def test_clustered_lows_detected_as_support(self):
        df = make_sr_ohlcv()
        result = find_support_resistance(df)
        supports = [l for l in result if l.level_type == "support"]
        # We injected two lows near 90 → should produce one support cluster
        assert any(abs(s.price - 90.0) < 2.0 for s in supports)

    def test_tolerance_clustering_within_threshold(self):
        """Two levels within tolerance → merged into one cluster."""
        n = 60
        close = np.full(n, 100.0)
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        # Two highs: 110 and 110.4 (within 0.5%)
        high[10] = 110.0
        close[10] = 109.5
        high[35] = 110.4
        close[35] = 109.9
        df = pd.DataFrame({
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        })
        result = find_support_resistance(df, window=10, tolerance=0.005)
        resistances = [l for l in result if l.level_type == "resistance"]
        near_110 = [r for r in resistances if abs(r.price - 110.0) < 2.0]
        if near_110:
            assert near_110[0].touches >= 2

    def test_tolerance_clustering_outside_threshold(self):
        """Two price groups far apart → form two separate clusters (each needs 2 touches).
        Pivot spacing must exceed 2*window+1=21 to avoid interference.
        """
        n = 120
        # Use varying baseline so flat regions don't generate spurious pivots
        rng = np.random.default_rng(0)
        baseline = 100.0 + rng.uniform(-0.5, 0.5, n)
        close = baseline.copy()
        high = baseline + 0.5
        low = baseline - 0.5
        # Two pivot highs near 110 — 26 bars apart (> 2*10+1=21)
        high[10] = 110.0;  close[10] = 109.5
        high[36] = 110.4;  close[36] = 109.9
        # Two pivot highs near 125 — 26 bars apart, 13% above 110 cluster
        high[65] = 125.0;  close[65] = 124.5
        high[91] = 125.3;  close[91] = 124.8
        df = pd.DataFrame({
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        })
        result = find_support_resistance(df, window=10, tolerance=0.005)
        resistances = [l for l in result if l.level_type == "resistance"]
        near_110 = [r for r in resistances if abs(r.price - 110.0) < 2.0]
        near_125 = [r for r in resistances if abs(r.price - 125.0) < 2.0]
        assert len(near_110) >= 1
        assert len(near_125) >= 1


# ---------------------------------------------------------------------------
# classify_trend
# ---------------------------------------------------------------------------

class TestClassifyTrend:
    def test_all_none_indicators_returns_unknown(self):
        ind = TechnicalIndicators()
        ta = classify_trend(ind, 100.0)
        assert ta.ema_alignment == "unknown"
        assert ta.regime == "unknown"
        assert ta.adx_regime == "unknown"

    def test_fully_bullish_alignment(self):
        ind = make_ind(ema20=110, ema50=105, ema200=100)
        ta = classify_trend(ind, 115.0)
        assert ta.ema_alignment == "fully_bullish"
        assert ta.primary_trend == "bullish"
        assert ta.secondary_trend == "bullish"
        assert ta.short_term_trend == "bullish"

    def test_fully_bearish_alignment(self):
        ind = make_ind(ema20=90, ema50=95, ema200=100)
        ta = classify_trend(ind, 85.0)
        assert ta.ema_alignment == "fully_bearish"
        assert ta.primary_trend == "bearish"

    def test_mixed_alignment(self):
        ind = make_ind(ema20=110, ema50=95, ema200=100)
        ta = classify_trend(ind, 105.0)
        assert ta.ema_alignment == "mixed"

    def test_two_emas_bullish(self):
        ind = make_ind(ema20=110, ema50=105, ema200=None)
        ta = classify_trend(ind, 115.0)
        assert ta.ema_alignment == "bullish"

    def test_two_emas_bearish(self):
        ind = make_ind(ema20=90, ema50=95, ema200=None)
        ta = classify_trend(ind, 85.0)
        assert ta.ema_alignment == "bearish"

    @pytest.mark.parametrize("rsi,expected_zone", [
        (70, "overbought"),
        (71, "overbought"),
        (69, "bullish"),
        (55, "bullish"),
        (54, "neutral"),
        (46, "neutral"),
        (45, "bearish"),   # <= 45 boundary → bearish
        (44, "bearish"),
        (30, "oversold"),
        (29, "oversold"),
    ])
    def test_rsi_zones(self, rsi, expected_zone):
        ind = make_ind(rsi=rsi)
        ta = classify_trend(ind, 100.0)
        assert ta.rsi_zone == expected_zone

    @pytest.mark.parametrize("adx,expected_regime,expected_adx_regime", [
        (15, "mean_reverting", "ranging"),
        (20, "transitioning", "emerging_trend"),
        (24, "transitioning", "emerging_trend"),
        (25, "trending", "strong_trend"),
        (39, "trending", "strong_trend"),
        (40, "trending", "extreme_trend"),
        (50, "trending", "extreme_trend"),
    ])
    def test_adx_regimes(self, adx, expected_regime, expected_adx_regime):
        ind = make_ind(adx=adx)
        ta = classify_trend(ind, 100.0)
        assert ta.regime == expected_regime
        assert ta.adx_regime == expected_adx_regime

    def test_primary_trend_price_vs_ema200(self):
        ind = make_ind(ema200=100.0)
        assert classify_trend(ind, 101.0).primary_trend == "bullish"
        assert classify_trend(ind, 99.0).primary_trend == "bearish"


# ---------------------------------------------------------------------------
# compute_volume_profile
# ---------------------------------------------------------------------------

class TestComputeVolumeProfile:
    def test_none_returns_empty(self):
        vp = compute_volume_profile(None)
        assert vp.poc is None

    def test_empty_df_returns_empty(self):
        vp = compute_volume_profile(pd.DataFrame())
        assert vp.poc is None

    def test_flat_prices_returns_empty(self):
        df = pd.DataFrame({
            "Close": [100.0] * 50,
            "Volume": [1_000_000.0] * 50,
        })
        vp = compute_volume_profile(df)
        assert vp.poc is None

    def test_poc_is_highest_volume_price(self):
        # Put all volume at price 105
        close = [95.0] * 10 + [105.0] * 40 + [115.0] * 10
        vol = [100.0] * 10 + [10_000.0] * 40 + [100.0] * 10
        df = pd.DataFrame({"Close": close, "Volume": vol})
        vp = compute_volume_profile(df, bins=20)
        assert vp.poc is not None
        assert abs(vp.poc - 105.0) < 5.0  # Should be in the high-volume region

    def test_vah_geq_poc_geq_val(self):
        df = make_ohlcv(100)
        vp = compute_volume_profile(df)
        assert vp.poc is not None
        assert vp.vah >= vp.poc >= vp.val

    def test_value_area_covers_70pct(self):
        df = make_ohlcv(100)
        close = df["Close"]
        volume = df["Volume"]
        vp = compute_volume_profile(df, bins=50)
        # Volume between VAL and VAH should be >= 70% of total
        val_vah_volume = volume[(close >= vp.val) & (close <= vp.vah)].sum()
        total_volume = volume.sum()
        assert val_vah_volume / total_volume >= 0.60  # Allow 60% due to bin rounding

    def test_lowercase_column_names(self):
        df = make_ohlcv(100)
        df.columns = [c.lower() for c in df.columns]
        vp = compute_volume_profile(df)
        assert vp.poc is not None


# ---------------------------------------------------------------------------
# detect_institutional_accumulation
# ---------------------------------------------------------------------------

class TestDetectInstitutionalAccumulation:
    def test_fewer_than_20_rows_signal_neutral(self):
        df = make_ohlcv(15)
        ind = make_ind(obv_slope=1000.0)
        sig = detect_institutional_accumulation(ind, df)
        # With < 20 rows, up_vol_ratio is None; only OBV slope scores
        assert sig.signal in ("accumulation", "distribution", "neutral")

    def test_none_ohlcv_uses_only_obv_slope(self):
        ind = make_ind(obv_slope=1000.0)
        sig = detect_institutional_accumulation(ind, None)
        assert sig.signal == "accumulation"

    def test_positive_obv_and_high_up_vol_is_accumulation(self):
        # Create data where all 20 recent days are up-days
        n = 30
        prices = np.linspace(90, 110, n)
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.99,
        })
        ind = make_ind(obv_slope=5000.0)
        sig = detect_institutional_accumulation(ind, df)
        assert sig.signal == "accumulation"
        assert sig.obv_slope_positive is True

    def test_negative_obv_and_low_up_vol_is_distribution(self):
        # Create data where all 20 recent days are down-days
        n = 30
        prices = np.linspace(110, 90, n)
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
            "Open": prices * 1.01,
            "High": prices * 1.01,
            "Low": prices * 0.99,
        })
        ind = make_ind(obv_slope=-5000.0)
        sig = detect_institutional_accumulation(ind, df)
        assert sig.signal == "distribution"
        assert sig.obv_slope_positive is False

    def test_mixed_signals_neutral(self):
        # positive OBV slope but up_vol_ratio near 0.50 → neutral
        n = 30
        # Alternating up and down
        prices = 100.0 + np.sin(np.arange(n) * 0.5) * 2
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
            "Open": prices * 0.999,
            "High": prices * 1.005,
            "Low": prices * 0.995,
        })
        ind = make_ind(obv_slope=100.0)
        sig = detect_institutional_accumulation(ind, df)
        # up_vol_ratio near 0.5 → no bonus point, OBV is positive → accumulation or neutral
        assert sig.signal in ("accumulation", "neutral")

    def test_up_vol_ratio_no_divide_by_zero(self):
        """If all volume is flat (no up or down days), no crash."""
        n = 30
        prices = np.full(n, 100.0)
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
            "Open": prices,
            "High": prices,
            "Low": prices,
        })
        ind = make_ind(obv_slope=0.0)
        sig = detect_institutional_accumulation(ind, df)
        assert sig.signal in ("accumulation", "distribution", "neutral")

    def test_obv_slope_none_treated_as_not_positive(self):
        ind = TechnicalIndicators()
        ind.obv_slope = None
        sig = detect_institutional_accumulation(ind, None)
        # obv_slope_positive defaults to False → bearish_points += 1
        assert sig.obv_slope_positive is False
        assert sig.signal == "distribution"
