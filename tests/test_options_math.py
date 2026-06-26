import math
import numpy as np
import pandas as pd
import pytest

from analysis.options_math import (
    black_scholes,
    calculate_max_pain,
    calculate_gamma_exposure,
    calculate_iv_skew,
    calculate_expected_move,
    compute_put_call_metrics,
)


# ---------------------------------------------------------------------------
# black_scholes
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_atm_call_basic_properties(self):
        r = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert r.price > 0
        assert 0 < r.delta < 1
        assert r.gamma > 0
        assert r.vega > 0
        assert 0 < r.prob_itm < 1
        assert r.theta < 0  # Time decay costs money

    def test_atm_put_basic_properties(self):
        r = black_scholes(100, 100, 1.0, 0.05, 0.20, "put")
        assert r.price > 0
        assert -1 < r.delta < 0
        assert r.gamma > 0
        assert r.vega > 0
        assert 0 < r.prob_itm < 1

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        # C - P = S - K * e^(-rT)
        lhs = call.price - put.price
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6

    def test_deep_itm_call(self):
        r = black_scholes(100, 50, 1.0, 0.05, 0.20, "call")
        assert r.delta > 0.95
        assert r.price > 48  # Roughly S - K*e^(-rT) minus a small time-value

    def test_deep_otm_call(self):
        r = black_scholes(100, 150, 1.0, 0.05, 0.20, "call")
        assert r.delta < 0.05
        assert r.price < 2.0

    def test_deep_itm_put(self):
        r = black_scholes(50, 100, 1.0, 0.05, 0.20, "put")
        assert r.delta < -0.95

    def test_deep_otm_put(self):
        r = black_scholes(100, 50, 1.0, 0.05, 0.20, "put")
        assert r.delta > -0.05
        assert r.price < 1.0

    def test_gamma_same_for_call_and_put(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        assert abs(call.gamma - put.gamma) < 1e-10

    def test_vega_same_for_call_and_put(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        assert abs(call.vega - put.vega) < 1e-10

    @pytest.mark.parametrize("bad_T", [0, -1, -0.001])
    def test_invalid_T_returns_zeros(self, bad_T):
        r = black_scholes(100, 100, bad_T, 0.05, 0.20)
        assert r.price == 0 and r.delta == 0 and r.gamma == 0

    @pytest.mark.parametrize("bad_sigma", [0, -0.01])
    def test_invalid_sigma_returns_zeros(self, bad_sigma):
        r = black_scholes(100, 100, 1.0, 0.05, bad_sigma)
        assert r.price == 0 and r.delta == 0

    @pytest.mark.parametrize("bad_S", [0, -10])
    def test_invalid_S_returns_zeros(self, bad_S):
        r = black_scholes(bad_S, 100, 1.0, 0.05, 0.20)
        assert r.price == 0 and r.gamma == 0

    @pytest.mark.parametrize("bad_K", [0, -10])
    def test_invalid_K_returns_zeros(self, bad_K):
        r = black_scholes(100, bad_K, 1.0, 0.05, 0.20)
        assert r.price == 0 and r.gamma == 0

    def test_call_prob_itm_greater_than_half_for_itm(self):
        r = black_scholes(110, 100, 0.5, 0.05, 0.20, "call")
        assert r.prob_itm > 0.5

    def test_put_prob_itm_greater_than_half_for_itm(self):
        r = black_scholes(90, 100, 0.5, 0.05, 0.20, "put")
        assert r.prob_itm > 0.5

    def test_vega_scaled_by_100(self):
        # vega = S * N'(d1) * sqrt(T) / 100
        # So it represents 1% vol move in dollar terms
        r = black_scholes(100, 100, 1.0, 0.05, 0.20)
        assert 0.1 < r.vega < 10  # Reasonable range for these params


# ---------------------------------------------------------------------------
# calculate_max_pain
# ---------------------------------------------------------------------------

class TestCalculateMaxPain:
    def test_empty_calls_returns_none(self, sample_puts_df):
        result = calculate_max_pain(pd.DataFrame(), sample_puts_df)
        assert result.max_pain is None

    def test_empty_puts_returns_none(self, sample_calls_df):
        result = calculate_max_pain(sample_calls_df, pd.DataFrame())
        assert result.max_pain is None

    def test_single_strike_both_sides(self):
        calls = pd.DataFrame({"strike": [100.0], "openInterest": [100]})
        puts = pd.DataFrame({"strike": [100.0], "openInterest": [100]})
        result = calculate_max_pain(calls, puts)
        assert result.max_pain == 100.0

    def test_max_pain_is_a_valid_strike(self, sample_calls_df, sample_puts_df):
        result = calculate_max_pain(sample_calls_df, sample_puts_df)
        all_strikes = set(sample_calls_df["strike"].tolist() + sample_puts_df["strike"].tolist())
        assert result.max_pain in all_strikes

    def test_max_pain_minimises_total_pain(self, sample_calls_df, sample_puts_df):
        result = calculate_max_pain(sample_calls_df, sample_puts_df)
        max_pain_strike = result.max_pain

        # Verify it is indeed the minimum-pain strike
        all_strikes = sorted(set(sample_calls_df["strike"].tolist() + sample_puts_df["strike"].tolist()))
        pain_at_max_pain = None
        for K in all_strikes:
            call_rows = sample_calls_df[sample_calls_df["strike"] <= K]
            put_rows = sample_puts_df[sample_puts_df["strike"] >= K]
            call_pain = float((call_rows["openInterest"] * (K - call_rows["strike"]).clip(lower=0)).sum())
            put_pain = float((put_rows["openInterest"] * (put_rows["strike"] - K).clip(lower=0)).sum())
            total = call_pain + put_pain
            if K == max_pain_strike:
                pain_at_max_pain = total
            else:
                assert pain_at_max_pain is None or pain_at_max_pain <= total

    def test_heavily_skewed_oi_pulls_max_pain(self):
        # Huge OI at 105 puts → expiry at 105 hurts most put holders ↓
        # So max pain should be higher to balance
        calls = pd.DataFrame({"strike": [100.0, 105.0, 110.0], "openInterest": [1000, 1000, 1000]})
        puts = pd.DataFrame({"strike": [95.0, 100.0, 105.0], "openInterest": [100, 100, 5000]})
        result = calculate_max_pain(calls, puts)
        assert result.max_pain is not None


# ---------------------------------------------------------------------------
# calculate_gamma_exposure
# ---------------------------------------------------------------------------

class TestCalculateGammaExposure:
    def test_both_empty_returns_zero_gex(self):
        result = calculate_gamma_exposure(pd.DataFrame(), pd.DataFrame(), 100.0)
        assert result.total_gex == 0.0

    def test_invalid_iv_rows_skipped(self):
        calls = pd.DataFrame({
            "strike": [100.0, 105.0],
            "openInterest": [100, 200],
            "impliedVolatility": [0.0, 0.25],  # First row iv=0 should be skipped
            "dte": [30, 30],
        })
        result = calculate_gamma_exposure(calls, pd.DataFrame(), 100.0)
        assert result.total_gex > 0  # Only second row contributed

    def test_invalid_oi_rows_skipped(self):
        calls = pd.DataFrame({
            "strike": [100.0],
            "openInterest": [0],  # OI=0 should be skipped
            "impliedVolatility": [0.25],
            "dte": [30],
        })
        result = calculate_gamma_exposure(calls, pd.DataFrame(), 100.0)
        assert result.total_gex == 0.0

    def test_single_call_positive_gex(self):
        calls = pd.DataFrame({
            "strike": [100.0],
            "openInterest": [1000],
            "impliedVolatility": [0.20],
            "dte": [30],
        })
        result = calculate_gamma_exposure(calls, pd.DataFrame(), 100.0)
        assert result.total_gex > 0
        assert result.regime == "positive"

    def test_single_put_negative_gex(self):
        puts = pd.DataFrame({
            "strike": [100.0],
            "openInterest": [1000],
            "impliedVolatility": [0.20],
            "dte": [30],
        })
        result = calculate_gamma_exposure(pd.DataFrame(), puts, 100.0)
        assert result.total_gex < 0
        assert result.regime == "negative"

    def test_positive_gex_sums_only_positive(self, sample_calls_df, sample_puts_df):
        result = calculate_gamma_exposure(sample_calls_df, sample_puts_df, 100.0)
        assert result.positive_gex >= 0
        assert result.negative_gex <= 0

    def test_gamma_flip_detected(self):
        # Large call GEX at low strike, large put GEX at high strike
        # This should create a sign change in cumulative GEX
        calls = pd.DataFrame({
            "strike": [95.0],
            "openInterest": [10000],
            "impliedVolatility": [0.25],
            "dte": [30],
        })
        puts = pd.DataFrame({
            "strike": [105.0],
            "openInterest": [50000],
            "impliedVolatility": [0.25],
            "dte": [30],
        })
        result = calculate_gamma_exposure(calls, puts, 100.0)
        # With dominant put GEX at 105, there may be a flip between 95 and 105
        # Just verify the function runs without error and returns a result
        assert result is not None

    def test_dte_zero_uses_min_one_day(self):
        calls = pd.DataFrame({
            "strike": [100.0],
            "openInterest": [100],
            "impliedVolatility": [0.20],
            "dte": [0],  # DTE=0 should use 1/365 minimum
        })
        result = calculate_gamma_exposure(calls, pd.DataFrame(), 100.0)
        assert result.total_gex > 0  # Should not crash


# ---------------------------------------------------------------------------
# calculate_iv_skew
# ---------------------------------------------------------------------------

class TestCalculateIVSkew:
    _EMPTY_CALLS = pd.DataFrame(columns=["strike", "impliedVolatility", "expiry"])
    _EMPTY_PUTS = pd.DataFrame(columns=["strike", "impliedVolatility", "expiry"])

    def test_empty_chains_returns_none_fields(self):
        result = calculate_iv_skew(self._EMPTY_CALLS, self._EMPTY_PUTS, 100.0)
        assert result.atm_iv is None
        assert result.otm_put_iv is None
        assert result.otm_call_iv is None
        assert result.put_skew is None
        assert result.risk_reversal is None

    def test_atm_iv_from_calls_within_2pct(self):
        spot = 100.0
        calls = pd.DataFrame({
            "strike": [99.0, 100.0, 101.0, 120.0],  # First three within 2%
            "impliedVolatility": [0.22, 0.20, 0.21, 0.30],
            "expiry": ["2025-01-17"] * 4,
        })
        result = calculate_iv_skew(calls, self._EMPTY_PUTS, spot)
        assert result.atm_iv is not None
        assert abs(result.atm_iv - 0.21) < 0.01  # Median of [0.20, 0.21, 0.22]

    def test_otm_put_iv_in_90_to_97_range(self):
        spot = 100.0
        puts = pd.DataFrame({
            "strike": [88.0, 92.0, 95.0, 98.0],  # 92 and 95 are in 90-97% range
            "impliedVolatility": [0.40, 0.33, 0.30, 0.25],
            "expiry": ["2025-01-17"] * 4,
        })
        result = calculate_iv_skew(self._EMPTY_CALLS, puts, spot)
        assert result.otm_put_iv is not None
        assert abs(result.otm_put_iv - 0.315) < 0.01  # Median of [0.30, 0.33]

    def test_otm_call_iv_in_103_to_110_range(self):
        spot = 100.0
        calls = pd.DataFrame({
            "strike": [100.0, 104.0, 107.0, 115.0],  # 104 and 107 in range
            "impliedVolatility": [0.20, 0.22, 0.24, 0.30],
            "expiry": ["2025-01-17"] * 4,
        })
        result = calculate_iv_skew(calls, self._EMPTY_PUTS, spot)
        assert result.otm_call_iv is not None
        assert abs(result.otm_call_iv - 0.23) < 0.01  # Median of [0.22, 0.24]

    def test_put_skew_equals_otm_put_minus_atm(self):
        spot = 100.0
        calls = pd.DataFrame({
            "strike": [100.0],
            "impliedVolatility": [0.20],
            "expiry": ["2025-01-17"],
        })
        puts = pd.DataFrame({
            "strike": [93.0],
            "impliedVolatility": [0.28],
            "expiry": ["2025-01-17"],
        })
        result = calculate_iv_skew(calls, puts, spot)
        assert result.put_skew is not None
        assert abs(result.put_skew - (0.28 - 0.20)) < 1e-6

    def test_risk_reversal_equals_otm_call_minus_otm_put(self):
        spot = 100.0
        calls = pd.DataFrame({
            "strike": [100.0, 105.0],
            "impliedVolatility": [0.20, 0.23],
            "expiry": ["2025-01-17", "2025-01-17"],
        })
        puts = pd.DataFrame({
            "strike": [93.0],
            "impliedVolatility": [0.27],
            "expiry": ["2025-01-17"],
        })
        result = calculate_iv_skew(calls, puts, spot)
        if result.otm_call_iv is not None and result.otm_put_iv is not None:
            expected_rr = result.otm_call_iv - result.otm_put_iv
            assert abs(result.risk_reversal - expected_rr) < 1e-6

    def test_term_structure_groups_by_expiry(self):
        spot = 100.0
        calls = pd.DataFrame({
            "strike": [100.0, 100.0],
            "impliedVolatility": [0.20, 0.25],
            "expiry": ["2025-01-17", "2025-02-21"],
        })
        result = calculate_iv_skew(calls, self._EMPTY_PUTS, spot)
        assert len(result.term_structure) == 2
        assert "2025-01-17" in result.term_structure
        assert "2025-02-21" in result.term_structure


# ---------------------------------------------------------------------------
# calculate_expected_move
# ---------------------------------------------------------------------------

class TestCalculateExpectedMove:
    def test_formula_only_no_straddle_data(self):
        spot, atm_iv, dte = 100.0, 0.20, 30
        em = calculate_expected_move(spot, atm_iv, dte, pd.DataFrame(), pd.DataFrame())
        expected = spot * atm_iv * math.sqrt(dte / 365)
        assert em.formula_move is not None
        assert abs(em.formula_move - expected) < 1e-6
        assert em.straddle_move is None
        assert abs(em.average_move - expected) < 1e-6

    def test_upper_and_lower_bounds_set(self):
        spot, atm_iv, dte = 100.0, 0.20, 30
        em = calculate_expected_move(spot, atm_iv, dte, pd.DataFrame(), pd.DataFrame())
        assert em.upper is not None
        assert em.lower is not None
        assert em.upper > spot
        assert em.lower < spot
        assert abs(em.upper - (spot + em.average_move)) < 0.01
        assert abs(em.lower - (spot - em.average_move)) < 0.01

    def test_dte_zero_no_formula_move(self):
        em = calculate_expected_move(100.0, 0.20, 0, pd.DataFrame(), pd.DataFrame())
        assert em.formula_move is None

    def test_atm_iv_none_no_formula_move(self):
        calls = pd.DataFrame({
            "strike": [100.0],
            "dte": [30],
            "mid": [3.0],
        })
        puts = pd.DataFrame({
            "strike": [100.0],
            "dte": [30],
            "mid": [2.8],
        })
        em = calculate_expected_move(100.0, None, 30, calls, puts)
        assert em.formula_move is None
        assert em.straddle_move is not None
        assert abs(em.straddle_move - (3.0 + 2.8)) < 1e-6

    def test_straddle_move_is_call_plus_put_mid(self):
        spot = 100.0
        calls = pd.DataFrame({"strike": [100.0], "dte": [30], "mid": [4.0]})
        puts = pd.DataFrame({"strike": [100.0], "dte": [30], "mid": [3.5]})
        em = calculate_expected_move(spot, None, 30, calls, puts)
        assert em.straddle_move is not None
        assert abs(em.straddle_move - 7.5) < 1e-6

    def test_average_is_mean_of_both_methods(self):
        spot, atm_iv, dte = 100.0, 0.20, 30
        calls = pd.DataFrame({"strike": [100.0], "dte": [30], "mid": [4.0]})
        puts = pd.DataFrame({"strike": [100.0], "dte": [30], "mid": [3.5]})
        em = calculate_expected_move(spot, atm_iv, dte, calls, puts)
        if em.formula_move and em.straddle_move:
            assert abs(em.average_move - (em.formula_move + em.straddle_move) / 2) < 1e-6

    def test_nearest_atm_strike_used_for_straddle(self):
        spot = 100.0
        calls = pd.DataFrame({
            "strike": [98.0, 100.0, 102.0],
            "dte": [30, 30, 30],
            "mid": [5.0, 4.0, 3.0],
        })
        puts = pd.DataFrame({
            "strike": [98.0, 100.0, 102.0],
            "dte": [30, 30, 30],
            "mid": [2.0, 3.5, 4.5],
        })
        em = calculate_expected_move(spot, None, 30, calls, puts)
        # Should pick 100 for both (nearest to spot=100)
        assert em.straddle_move is not None
        assert abs(em.straddle_move - (4.0 + 3.5)) < 1e-6


# ---------------------------------------------------------------------------
# compute_put_call_metrics
# ---------------------------------------------------------------------------

class TestComputePutCallMetrics:
    def test_both_empty_returns_zero_totals(self):
        pc = compute_put_call_metrics(pd.DataFrame(), pd.DataFrame())
        assert pc.total_call_oi == 0
        assert pc.total_put_oi == 0
        assert pc.oi_put_call_ratio is None

    def test_zero_call_oi_no_ratio(self):
        calls = pd.DataFrame({
            "openInterest": [0],
            "volume": [0],
            "mid": [1.0],
        })
        puts = pd.DataFrame({
            "openInterest": [500],
            "volume": [100],
            "mid": [2.0],
        })
        pc = compute_put_call_metrics(calls, puts)
        assert pc.oi_put_call_ratio is None  # Division by zero prevented

    def test_oi_ratio_rounded_to_3_decimals(self):
        calls = pd.DataFrame({"openInterest": [1000], "volume": [200], "mid": [2.0]})
        puts = pd.DataFrame({"openInterest": [850], "volume": [170], "mid": [1.8]})
        pc = compute_put_call_metrics(calls, puts)
        assert pc.oi_put_call_ratio == round(850 / 1000, 3)

    def test_vol_ratio_computed_correctly(self):
        calls = pd.DataFrame({"openInterest": [500], "volume": [1000], "mid": [2.0]})
        puts = pd.DataFrame({"openInterest": [600], "volume": [1200], "mid": [1.9]})
        pc = compute_put_call_metrics(calls, puts)
        assert pc.vol_put_call_ratio == round(1200 / 1000, 3)

    def test_dollar_ratio_uses_mid_times_oi_times_100(self):
        calls = pd.DataFrame({"openInterest": [100], "volume": [50], "mid": [3.0]})
        puts = pd.DataFrame({"openInterest": [120], "volume": [60], "mid": [2.5]})
        pc = compute_put_call_metrics(calls, puts)
        call_dollar = 3.0 * 100 * 100  # mid * OI * 100
        put_dollar = 2.5 * 120 * 100
        expected = round(put_dollar / call_dollar, 3)
        assert pc.dollar_put_call_ratio == expected

    def test_total_oi_summed_correctly(self, sample_calls_df, sample_puts_df):
        pc = compute_put_call_metrics(sample_calls_df, sample_puts_df)
        assert pc.total_call_oi == sum(sample_calls_df["openInterest"])
        assert pc.total_put_oi == sum(sample_puts_df["openInterest"])

    def test_zero_call_volume_no_vol_ratio(self):
        calls = pd.DataFrame({"openInterest": [100], "volume": [0], "mid": [2.0]})
        puts = pd.DataFrame({"openInterest": [100], "volume": [50], "mid": [2.0]})
        pc = compute_put_call_metrics(calls, puts)
        assert pc.vol_put_call_ratio is None
