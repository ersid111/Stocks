import math
import numpy as np
import pandas as pd
import pytest

from analysis.technical import TechnicalIndicators
from analysis.market_structure import TrendAnalysis, SRLevel, VolumeProfile
from analysis.options_math import PCMetrics
from analysis.scenarios import _score_direction, build_scenarios


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ind(
    macd_histogram=None,
    adx=None,
    obv_slope=None,
) -> TechnicalIndicators:
    ind = TechnicalIndicators()
    ind.macd_histogram = macd_histogram
    ind.adx = adx
    ind.obv_slope = obv_slope
    return ind


def make_trend(ema_alignment="unknown", rsi_zone="neutral") -> TrendAnalysis:
    ta = TrendAnalysis()
    ta.ema_alignment = ema_alignment
    ta.rsi_zone = rsi_zone
    return ta


def make_pc(oi_put_call_ratio=None) -> PCMetrics:
    pc = PCMetrics()
    pc.oi_put_call_ratio = oi_put_call_ratio
    return pc


def make_ohlcv_for_vol(n=252) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    prices = np.abs(prices) + 1
    return pd.DataFrame({
        "Close": prices,
        "Open": prices * 0.999,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Volume": np.full(n, 1_000_000.0),
    })


# ---------------------------------------------------------------------------
# _score_direction
# ---------------------------------------------------------------------------

class TestScoreDirection:
    def test_all_none_returns_zero(self):
        ind = make_ind()
        trend = make_trend("unknown", "neutral")
        pc = make_pc(None)
        score = _score_direction(ind, trend, pc)
        # weight_total will still be >0 because EMA/RSI/MACD/OBV/PC weights always add
        # but with "unknown" EMA, neutral RSI, None MACD/OBV, None PC → score=0
        assert score == 0.0

    def test_max_bullish_clips_to_one(self):
        ind = make_ind(macd_histogram=1.0, adx=30.0, obv_slope=1000.0)
        trend = make_trend("fully_bullish", "bullish")
        pc = make_pc(0.5)  # < 0.7 → bullish
        score = _score_direction(ind, trend, pc)
        assert score == 1.0

    def test_max_bearish_clips_to_minus_one(self):
        ind = make_ind(macd_histogram=-1.0, adx=30.0, obv_slope=-1000.0)
        trend = make_trend("fully_bearish", "bearish")
        pc = make_pc(1.5)  # > 1.2 → bearish
        score = _score_direction(ind, trend, pc)
        assert score == -1.0

    def test_output_bounded_between_minus_one_and_one(self):
        for ema_align in ["fully_bullish", "fully_bearish", "bullish", "bearish", "mixed", "unknown"]:
            for rsi_zone in ["overbought", "bullish", "neutral", "bearish", "oversold"]:
                ind = make_ind(macd_histogram=0.1, adx=30.0, obv_slope=100.0)
                trend = make_trend(ema_align, rsi_zone)
                pc = make_pc(0.9)
                score = _score_direction(ind, trend, pc)
                assert -1.0 <= score <= 1.0

    def test_adx_multiplier_applied_when_above_25(self):
        ind_low = make_ind(macd_histogram=0.1, adx=20.0, obv_slope=100.0)
        ind_high = make_ind(macd_histogram=0.1, adx=30.0, obv_slope=100.0)
        trend = make_trend("bullish", "bullish")
        pc = make_pc(0.9)
        score_low = _score_direction(ind_low, trend, pc)
        score_high = _score_direction(ind_high, trend, pc)
        # Higher ADX with positive signals → higher absolute score (or same if clipped)
        assert score_high >= score_low

    def test_adx_no_multiplier_at_or_below_25(self):
        ind = make_ind(macd_histogram=0.1, adx=25.0, obv_slope=None)
        trend = make_trend("bullish", "neutral")
        pc = make_pc(None)
        score = _score_direction(ind, trend, pc)
        assert score > 0  # Bullish alignment + positive MACD → positive score

    def test_neutral_pc_ratio_no_score_contribution(self):
        ind = make_ind(macd_histogram=None, adx=None, obv_slope=None)
        trend_with = make_trend("unknown", "neutral")
        # PC ratio 0.9 is between 0.7 and 1.2 → no contribution
        score_with_neutral_pc = _score_direction(ind, trend_with, make_pc(0.9))
        score_with_no_pc = _score_direction(ind, trend_with, make_pc(None))
        assert score_with_neutral_pc == score_with_no_pc

    def test_bearish_pc_ratio_reduces_score(self):
        ind = make_ind(macd_histogram=0.5, adx=20.0, obv_slope=None)
        trend = make_trend("bullish", "bullish")
        score_neutral_pc = _score_direction(ind, trend, make_pc(0.9))
        score_bearish_pc = _score_direction(ind, trend, make_pc(1.5))
        assert score_neutral_pc > score_bearish_pc

    def test_ema_fully_bullish_weight_2(self):
        ind = make_ind()
        trend_fully_bull = make_trend("fully_bullish", "neutral")
        trend_bull = make_trend("bullish", "neutral")
        pc = make_pc(None)
        score_full = _score_direction(ind, trend_fully_bull, pc)
        score_partial = _score_direction(ind, trend_bull, pc)
        assert score_full > score_partial


# ---------------------------------------------------------------------------
# build_scenarios
# ---------------------------------------------------------------------------

class TestBuildScenarios:
    def _base_inputs(self):
        ind = make_ind(macd_histogram=0.2, adx=28.0, obv_slope=500.0)
        trend = make_trend("fully_bullish", "bullish")
        pc = make_pc(0.85)
        sr_levels = [
            SRLevel(price=90.0, touches=3, level_type="support"),
            SRLevel(price=108.0, touches=2, level_type="resistance"),
        ]
        vp = VolumeProfile()
        vp.poc = 100.0
        vp.vah = 106.0
        vp.val = 94.0
        return ind, trend, pc, sr_levels, vp

    def test_probabilities_sum_to_one(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        total = ss.bull.probability + ss.bear.probability + ss.neutral.probability + ss.black_swan.probability
        assert abs(total - 1.0) < 1e-9

    def test_black_swan_probability_always_005(self):
        for direction in [0.0, 0.5, -0.5, 1.0, -1.0]:
            ind, trend, pc, sr, vp = self._base_inputs()
            ind.macd_histogram = direction
            ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
            assert ss.black_swan.probability == 0.05

    def test_bullish_direction_bull_prob_greater_than_bear(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        assert ss.bull.probability > ss.bear.probability

    def test_bearish_direction_bear_prob_greater_than_bull(self):
        ind = make_ind(macd_histogram=-1.0, adx=28.0, obv_slope=-500.0)
        trend = make_trend("fully_bearish", "bearish")
        pc = make_pc(1.5)
        sr = [
            SRLevel(price=90.0, touches=3, level_type="support"),
            SRLevel(price=108.0, touches=2, level_type="resistance"),
        ]
        vp = VolumeProfile()
        vp.poc = 100.0
        vp.vah = 106.0
        vp.val = 94.0
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        assert ss.bear.probability > ss.bull.probability

    def test_neutral_direction_bull_equals_bear(self):
        ind = make_ind()
        trend = make_trend("unknown", "neutral")
        pc = make_pc(None)
        sr = []
        vp = VolumeProfile()
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        assert abs(ss.bull.probability - ss.bear.probability) < 1e-9

    def test_bull_target_uses_nearest_resistance_above_price(self):
        ind, trend, pc, _, vp = self._base_inputs()
        sr = [
            SRLevel(price=108.0, touches=2, level_type="resistance"),
            SRLevel(price=120.0, touches=2, level_type="resistance"),
        ]
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        assert ss.bull.target_price == 108.0

    def test_bear_target_uses_highest_support_below_price(self):
        ind, trend, pc, _, vp = self._base_inputs()
        sr = [
            SRLevel(price=85.0, touches=2, level_type="support"),
            SRLevel(price=92.0, touches=2, level_type="support"),
        ]
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        assert ss.bear.target_price == 92.0

    def test_no_sr_levels_uses_fallback_targets(self):
        ind, trend, pc, _, vp = self._base_inputs()
        ss = build_scenarios(ind, trend, pc, [], vp, 100.0)
        assert abs(ss.bull.target_price - 110.0) < 0.01
        assert abs(ss.bear.target_price - 90.0) < 0.01

    def test_black_swan_target_uses_hist_vol(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ohlcv = make_ohlcv_for_vol(252)
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0, ohlcv)
        assert ss.historical_vol_1y is not None
        assert ss.black_swan.target_price is not None
        # Target should be well below spot (3-sigma down move)
        assert ss.black_swan.target_price < 100.0

    def test_black_swan_fallback_no_ohlcv(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0, None)
        assert ss.black_swan.target_price == 75.0  # 100 * 0.75

    def test_hist_vol_is_annualized(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ohlcv = make_ohlcv_for_vol(252)
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0, ohlcv)
        # Annualized vol for reasonable prices should be 10-200%
        assert 0.01 < ss.historical_vol_1y < 5.0

    def test_scenario_set_has_all_four_scenarios(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        assert ss.bull is not None
        assert ss.bear is not None
        assert ss.neutral is not None
        assert ss.black_swan is not None

    def test_neutral_target_is_midpoint_of_vah_val(self):
        ind, trend, pc, _, _ = self._base_inputs()
        vp = VolumeProfile()
        vp.poc = 100.0
        vp.vah = 110.0
        vp.val = 90.0
        ss = build_scenarios(ind, trend, pc, [], vp, 95.0)
        expected_neutral = (110.0 + 90.0) / 2
        assert abs(ss.neutral.target_price - expected_neutral) < 0.01

    def test_probabilities_rounded_to_3_decimals(self):
        ind, trend, pc, sr, vp = self._base_inputs()
        ss = build_scenarios(ind, trend, pc, sr, vp, 100.0)
        for prob in [ss.bull.probability, ss.bear.probability, ss.neutral.probability]:
            assert round(prob, 3) == prob
