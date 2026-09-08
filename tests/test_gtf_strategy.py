"""Unit tests for the pure helpers of the GTF strategy backtester."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gtf_strategy"))

import gtf_strategy_automation as gtf  # noqa: E402


def candle(o, h, l, c):
    """Build a single feature-annotated candle as a Series."""
    df = gtf.add_candle_features(
        pd.DataFrame([{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000}])
    )
    return df.iloc[0]


# ── is_base / is_exciting ────────────────────────────────────────────────


def test_is_base_true_for_small_body():
    # body 0.5 over range 10 => 0.05 < 0.33
    assert gtf.is_base(candle(100, 105, 95, 100.5))


def test_is_base_false_for_large_body():
    # body 8 over range 10 => 0.8 > 0.33
    assert not gtf.is_base(candle(96, 105, 95, 104))


def test_is_base_false_for_doji_with_zero_body():
    assert not gtf.is_base(candle(100, 105, 95, 100))


def test_is_base_false_for_zero_range():
    assert not gtf.is_base(candle(100, 100, 100, 100))


def test_is_exciting_rally():
    rally, drop = gtf.is_exciting(candle(96, 105, 95, 104))
    assert rally and not drop


def test_is_exciting_drop():
    rally, drop = gtf.is_exciting(candle(104, 105, 95, 96))
    assert drop and not rally


def test_is_exciting_zero_range_is_neither():
    assert gtf.is_exciting(candle(100, 100, 100, 100)) == (False, False)


# ── zone quality ─────────────────────────────────────────────────────────


def test_check_zone_quality_accepts_tight_zone():
    # width 1.0 <= 1.5 * ATR(2.0)
    assert gtf.check_zone_quality(2.0, 101.0, 100.0)


def test_check_zone_quality_rejects_wide_zone():
    # width 10.0 > 1.5 * ATR(2.0)
    assert not gtf.check_zone_quality(2.0, 110.0, 100.0)


def test_check_zone_quality_disabled_by_sentinel(monkeypatch):
    monkeypatch.setattr(gtf, "MAX_ZONE_WIDTH_ATR_MULTIPLE", 999)
    assert gtf.check_zone_quality(0.01, 200.0, 100.0)


# ── wick rejection ───────────────────────────────────────────────────────


def test_wick_rejection_demand_needs_lower_wick():
    # lower wick 5 of range 10 => 0.5 >= 0.4
    assert gtf.check_wick_rejection(candle(105, 108, 100, 107), "DBR")


def test_wick_rejection_demand_fails_without_lower_wick():
    assert not gtf.check_wick_rejection(candle(101, 108, 100, 107), "RBR")


def test_wick_rejection_supply_needs_upper_wick():
    # upper wick 5 of range 10 => 0.5 >= 0.4
    assert gtf.check_wick_rejection(candle(103, 110, 100, 101), "RBD")


def test_wick_rejection_unknown_zone_type_is_false():
    assert not gtf.check_wick_rejection(candle(105, 108, 100, 107), "XYZ")


# ── take profit ──────────────────────────────────────────────────────────


def test_dynamic_tp_long_is_r_multiple_above_entry():
    tp = gtf.get_dynamic_tp(100.0, 98.0, None, 0, "DBR")
    assert tp == pytest.approx(100.0 + 2.0 * gtf.MAX_R_MULTIPLE)


def test_dynamic_tp_short_is_r_multiple_below_entry():
    tp = gtf.get_dynamic_tp(100.0, 102.0, None, 0, "RBD")
    assert tp == pytest.approx(100.0 - 2.0 * gtf.MAX_R_MULTIPLE)


# ── HTF confluence ───────────────────────────────────────────────────────


def make_zone(proximal, distal, is_demand, zone_type="DBR", index=5, base_count=1):
    return gtf.GTFZone(index, zone_type, proximal, distal, is_demand, base_count)


def test_confluence_matches_demand_zone_containing_price():
    zones = [make_zone(100.0, 95.0, True)]
    assert gtf.check_htf_confluence(97.0, zones, True) is zones[0]


def test_confluence_ignores_opposite_direction():
    zones = [make_zone(100.0, 95.0, True)]
    assert gtf.check_htf_confluence(97.0, zones, False) is None


def test_confluence_returns_none_outside_zone():
    zones = [make_zone(100.0, 95.0, True)]
    assert gtf.check_htf_confluence(120.0, zones, True) is None


def test_confluence_matches_supply_zone():
    # For supply, proximal is the lower line and distal the upper one
    zones = [make_zone(100.0, 105.0, False, zone_type="RBD")]
    assert gtf.check_htf_confluence(103.0, zones, False) is zones[0]


# ── location filter ──────────────────────────────────────────────────────


def curve_zones():
    """Fresh demand at 100 and fresh supply at 200 => fifths of 20 points."""
    return [make_zone(100.0, 95.0, True), make_zone(200.0, 205.0, False, zone_type="RBD")]


@pytest.mark.parametrize("price,expected", [
    (195.0, gtf.Location.VERY_HIGH),
    (175.0, gtf.Location.HIGH),
    (150.0, gtf.Location.EQUILIBRIUM),
    (125.0, gtf.Location.LOW),
    (105.0, gtf.Location.VERY_LOW),
])
def test_location_filter_buckets(price, expected):
    assert gtf.check_location_filter(price, curve_zones()) is expected


def test_location_filter_defaults_to_equilibrium_without_both_sides():
    assert gtf.check_location_filter(150.0, [make_zone(100.0, 95.0, True)]) is gtf.Location.EQUILIBRIUM


def test_location_filter_defaults_to_equilibrium_on_inverted_curve():
    zones = [make_zone(200.0, 195.0, True), make_zone(100.0, 105.0, False, zone_type="RBD")]
    assert gtf.check_location_filter(150.0, zones) is gtf.Location.EQUILIBRIUM


# ── base extremes ────────────────────────────────────────────────────────


def test_get_base_extremes_reads_the_base_range_only():
    df = gtf.add_candle_features(pd.DataFrame({
        "Open": [100, 101, 102, 103],
        "High": [110, 104, 106, 108],
        "Low": [90, 99, 100, 101],
        "Close": [101, 102, 103, 104],
        "Volume": [1] * 4,
    }))
    # base = rows 1..2, leg-out at row 3
    (dem_prox, dem_dist), (sup_prox, sup_dist) = gtf.get_base_extremes(df, 3, 2)
    assert dem_prox == 103 and sup_prox == 103   # last base close
    assert dem_dist == 99                        # lowest low of the base
    assert sup_dist == 106                       # highest high of the base


# ── zone detection ───────────────────────────────────────────────────────


def test_find_zones_detects_a_drop_base_rally():
    # 5 filler bars, a drop leg-in, one base, then a rally leg-out
    rows = [{"Open": 100, "High": 101, "Low": 99, "Close": 100.5} for _ in range(5)]
    rows.append({"Open": 100, "High": 100.5, "Low": 90, "Close": 91})    # drop leg-in
    rows.append({"Open": 91, "High": 93, "Low": 90.5, "Close": 91.2})    # base
    rows.append({"Open": 91.2, "High": 101, "Low": 91, "Close": 100})    # rally leg-out
    df = gtf.add_candle_features(pd.DataFrame(rows))
    df["Volume"] = 1

    zones = gtf.find_zones(df)
    demand = [z for z in zones if z.is_demand]
    assert demand, "expected at least one demand zone"
    assert demand[0].type == "DBR"
    assert demand[0].proximal == pytest.approx(91.2)  # last base close
    assert demand[0].distal == pytest.approx(90.5)    # base low


def test_find_zones_returns_nothing_on_flat_data():
    df = gtf.add_candle_features(pd.DataFrame({
        "Open": [100.0] * 20, "High": [100.0] * 20,
        "Low": [100.0] * 20, "Close": [100.0] * 20, "Volume": [1] * 20,
    }))
    assert gtf.find_zones(df) == []


# ── ticker list ──────────────────────────────────────────────────────────


def test_load_tickers_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "stock_list.txt"
    f.write_text("HDFCBANK\n\n# a comment\nRELIANCE\n  SBIN  \n")
    assert gtf.load_tickers(str(f)) == ["HDFCBANK", "RELIANCE", "SBIN"]


def test_load_tickers_missing_file_returns_empty(tmp_path):
    assert gtf.load_tickers(str(tmp_path / "nope.txt")) == []


# ── data fetching ────────────────────────────────────────────────────────


class FakeFyers:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def history(self, data):
        self.last_request = data
        return self.response


def test_get_fyers_data_builds_symbol_and_dataframe():
    fake = FakeFyers({"code": 200, "candles": [[1735689600, 100, 105, 99, 104, 5000]]})
    df = gtf.get_fyers_data(fake, "HDFCBANK", "15m", "2025-09-01", "2025-09-30")

    assert fake.last_request["symbol"] == "NSE:HDFCBANK-EQ"
    assert fake.last_request["resolution"] == "15"
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert len(df) == 1


def test_get_fyers_data_rejects_unsupported_interval():
    fake = FakeFyers({"code": 200, "candles": []})
    assert gtf.get_fyers_data(fake, "HDFCBANK", "3m", "2025-09-01", "2025-09-30").empty
    assert fake.last_request is None


def test_get_fyers_data_handles_api_error():
    fake = FakeFyers({"code": 401, "message": "invalid token"})
    assert gtf.get_fyers_data(fake, "HDFCBANK", "15m", "2025-09-01", "2025-09-30").empty
