"""Unit tests for the pure helpers of the GTF strategy backtester."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gtf_strategy"))

import gtf_strategy_automation as gtf  # noqa: E402
import data_providers as dp  # noqa: E402


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


# ── data providers ───────────────────────────────────────────────────────


class FakeFyers:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def history(self, data):
        self.last_request = data
        return self.response


def test_fyers_provider_builds_symbol_and_dataframe():
    fake = FakeFyers({"code": 200, "candles": [[1735689600, 100, 105, 99, 104, 5000]]})
    df = gtf.get_data("HDFCBANK", "15m", "2025-09-01", "2025-09-30",
                      provider="fyers", fyers_model=fake)

    assert fake.last_request["symbol"] == "NSE:HDFCBANK-EQ"
    assert fake.last_request["resolution"] == "15"
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert str(df.index.tz) == "UTC"
    assert len(df) == 1


def test_fyers_provider_rejects_unsupported_interval():
    fake = FakeFyers({"code": 200, "candles": []})
    df = gtf.get_data("HDFCBANK", "3m", "2025-09-01", "2025-09-30",
                      provider="fyers", fyers_model=fake)
    assert df.empty
    assert fake.last_request is None


def test_fyers_provider_handles_api_error():
    fake = FakeFyers({"code": 401, "message": "invalid token"})
    df = gtf.get_data("HDFCBANK", "15m", "2025-09-01", "2025-09-30",
                      provider="fyers", fyers_model=fake)
    assert df.empty


def test_fyers_provider_without_client_is_an_error():
    with pytest.raises(dp.DataProviderError):
        dp.fetch_ohlcv("HDFCBANK", "15m", "2025-09-01", "2025-09-30", provider="fyers")


def test_unknown_provider_is_an_error():
    with pytest.raises(dp.DataProviderError):
        dp.fetch_ohlcv("HDFCBANK", "15m", "2025-09-01", "2025-09-30", provider="quandl")


@pytest.mark.parametrize("raw,expected", [
    ("RELIANCE", "RELIANCE.NS"),
    ("reliance", "RELIANCE.NS"),
    ("  SBIN  ", "SBIN.NS"),
    ("RELIANCE.NS", "RELIANCE.NS"),   # already suffixed
    ("^NSEI", "^NSEI"),               # index, left alone
    ("AAPL.O", "AAPL.O"),             # explicit non-NSE suffix
])
def test_nse_symbol_mapping(raw, expected):
    assert dp._to_nse_symbol(raw) == expected


# ── normalisation (what every provider is funnelled through) ─────────────


def raw_frame(index, **overrides):
    data = {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100]}
    data.update(overrides)
    return pd.DataFrame(data, index=pd.DatetimeIndex(index, name="Date"))


def test_normalise_localises_a_naive_index_as_exchange_time():
    # A naive 09:15 from an NSE feed is 09:15 IST = 03:45 UTC, not 09:15 UTC
    df = dp._normalise(raw_frame(["2025-09-01 09:15:00"]), "test")
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2025-09-01 03:45:00", tz="UTC")


def test_normalise_converts_an_aware_index_to_utc():
    idx = pd.DatetimeIndex(["2025-09-01 09:15:00"]).tz_localize("Asia/Kolkata")
    df = dp._normalise(raw_frame(idx), "test")
    assert df.index[0] == pd.Timestamp("2025-09-01 03:45:00", tz="UTC")


def test_normalise_flattens_multiindex_columns():
    # yfinance hands back ("Close", "RELIANCE.NS") style columns
    df = raw_frame(["2025-09-01 09:15:00"])
    df.columns = pd.MultiIndex.from_product([df.columns, ["RELIANCE.NS"]])
    out = dp._normalise(df, "test")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalise_drops_adj_close_and_keeps_canonical_order():
    df = raw_frame(["2025-09-01 09:15:00"])
    df["Adj Close"] = 1.45
    out = dp._normalise(df[["Adj Close", "Volume", "Close", "Low", "High", "Open"]], "test")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalise_raises_when_a_price_column_is_missing():
    df = raw_frame(["2025-09-01 09:15:00"]).drop(columns=["Low"])
    with pytest.raises(dp.DataProviderError):
        dp._normalise(df, "test")


def test_normalise_sorts_and_deduplicates():
    idx = ["2025-09-02 09:15:00", "2025-09-01 09:15:00", "2025-09-01 09:15:00"]
    df = pd.DataFrame({"Open": [3.0, 1.0, 9.0], "High": [3.0, 1.0, 9.0],
                       "Low": [3.0, 1.0, 9.0], "Close": [3.0, 1.0, 9.0],
                       "Volume": [1, 2, 3]}, index=pd.DatetimeIndex(idx))
    out = dp._normalise(df, "test")
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert out["Open"].iloc[0] == 1.0   # first of the duplicate pair kept


def test_normalise_empty_frame_returns_canonical_empty():
    out = dp._normalise(pd.DataFrame(), "test")
    assert out.empty
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ── csv provider (works with no network at all) ──────────────────────────


def write_csv(tmp_path, name, rows, date_header="Date"):
    p = tmp_path / name
    lines = [f"{date_header},Open,High,Low,Close,Volume"]
    lines += [f"{d},{o},{h},{l},{c},{v}" for d, o, h, l, c, v in rows]
    p.write_text("\n".join(lines) + "\n")
    return p


def test_csv_provider_reads_and_filters_by_date(tmp_path):
    write_csv(tmp_path, "SBIN_15m.csv", [
        ("2025-08-30 09:15:00", 100, 101, 99, 100.5, 1000),   # before range
        ("2025-09-01 09:15:00", 101, 102, 100, 101.5, 1200),
        ("2025-09-30 15:15:00", 105, 106, 104, 105.5, 1300),
        ("2025-10-02 09:15:00", 110, 111, 109, 110.5, 1400),  # after range
    ])
    df = dp.fetch_ohlcv("SBIN", "15m", "2025-09-01", "2025-09-30",
                        provider="csv", csv_dir=str(tmp_path))
    assert len(df) == 2
    assert str(df.index.tz) == "UTC"
    assert df["Close"].tolist() == [101.5, 105.5]


def test_csv_provider_accepts_alternative_date_headers(tmp_path):
    write_csv(tmp_path, "SBIN_15m.csv",
              [("2025-09-01 09:15:00", 1, 2, 0.5, 1.5, 10)], date_header="timestamp")
    df = dp.fetch_ohlcv("SBIN", "15m", "2025-09-01", "2025-09-30",
                        provider="csv", csv_dir=str(tmp_path))
    assert len(df) == 1


def test_csv_provider_missing_file_returns_empty(tmp_path):
    df = dp.fetch_ohlcv("NOPE", "15m", "2025-09-01", "2025-09-30",
                        provider="csv", csv_dir=str(tmp_path))
    assert df.empty


def test_csv_provider_without_a_date_column_is_an_error(tmp_path):
    (tmp_path / "SBIN_15m.csv").write_text("Open,High,Low,Close,Volume\n1,2,0.5,1.5,10\n")
    with pytest.raises(dp.DataProviderError):
        dp.fetch_ohlcv("SBIN", "15m", "2025-09-01", "2025-09-30",
                       provider="csv", csv_dir=str(tmp_path))


# ── end-to-end through the strategy, no network ──────────────────────────


def test_backtest_runs_end_to_end_on_csv_data(tmp_path, monkeypatch):
    """A full run with the csv provider: no broker, no API, no network."""
    rng = np.random.default_rng(11)

    def series(n, freq, fname):
        start = pd.Timestamp("2025-09-01 09:15:00")
        idx = pd.date_range(start, periods=n, freq=freq)
        price, rows = 1000.0, []
        for ts in idx:
            o = price
            c = max(1.0, o + rng.normal(0, 6))
            h, l = max(o, c) + abs(rng.normal(0, 3)), min(o, c) - abs(rng.normal(0, 3))
            rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), round(o, 2), round(h, 2),
                         round(l, 2), round(c, 2), 10000))
            price = c
        write_csv(tmp_path, fname, rows)

    series(700, "15min", "TESTSTK_15m.csv")
    series(180, "60min", "TESTSTK_60m.csv")

    monkeypatch.setattr(gtf, "DATA_PROVIDER", "csv")
    monkeypatch.setattr(gtf, "CSV_DIR", str(tmp_path))
    monkeypatch.setattr(gtf, "EMA_TP_PERIOD", 50)

    result = gtf.run_backtest("TESTSTK", "2025-09-01", "2025-12-31")

    assert result is not None
    assert result["Status"] == "COMPLETED"
    assert result["Total Trades"] >= 0
    assert result["Final Capital"] == pytest.approx(
        gtf.INITIAL_CAPITAL + result["Total P&L"])


def test_backtest_returns_none_when_the_source_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(gtf, "DATA_PROVIDER", "csv")
    monkeypatch.setattr(gtf, "CSV_DIR", str(tmp_path))
    assert gtf.run_backtest("MISSING", "2025-09-01", "2025-09-30") is None


# ── yfinance provider (mocked — the sandbox cannot reach Yahoo) ───────────


class FakeYFTicker:
    """Stands in for yfinance.Ticker, recording how history() was called."""

    calls = []

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        FakeYFTicker.calls.append({"symbol": self.symbol, **kwargs})
        idx = pd.DatetimeIndex(
            ["2025-09-01 09:15:00", "2025-09-01 09:30:00"]
        ).tz_localize("Asia/Kolkata")
        # Yahoo returns Dividends/Stock Splits alongside the price columns
        return pd.DataFrame({
            "Open": [100.0, 101.0], "High": [102.0, 103.0],
            "Low": [99.0, 100.0], "Close": [101.0, 102.0],
            "Volume": [5000, 6000], "Dividends": [0.0, 0.0], "Stock Splits": [0.0, 0.0],
        }, index=idx)


@pytest.fixture
def fake_yf(monkeypatch):
    FakeYFTicker.calls = []
    module = type(sys)("yfinance")
    module.Ticker = FakeYFTicker
    monkeypatch.setitem(sys.modules, "yfinance", module)
    return FakeYFTicker


def test_yfinance_provider_normalises_the_response(fake_yf):
    df = dp.fetch_ohlcv("RELIANCE", "15m", "2025-09-01", "2025-09-30")

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]  # extras dropped
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2025-09-01 03:45:00", tz="UTC")  # 09:15 IST
    assert df["Close"].tolist() == [101.0, 102.0]


def test_yfinance_provider_appends_the_nse_suffix(fake_yf):
    dp.fetch_ohlcv("RELIANCE", "15m", "2025-09-01", "2025-09-30")
    assert fake_yf.calls[0]["symbol"] == "RELIANCE.NS"


def test_yfinance_provider_maps_60m_to_yahoos_1h(fake_yf):
    dp.fetch_ohlcv("RELIANCE", "60m", "2025-09-01", "2025-09-30")
    assert fake_yf.calls[0]["interval"] == "1h"


def test_yfinance_provider_makes_the_end_date_inclusive(fake_yf):
    # Yahoo's `end` is exclusive, so END_DATE itself would be dropped without this
    dp.fetch_ohlcv("RELIANCE", "15m", "2025-09-01", "2025-09-30")
    assert fake_yf.calls[0]["end"] == "2025-10-01"


def test_yfinance_provider_requests_adjusted_prices(fake_yf):
    # Unadjusted history puts fake gaps at splits, which would forge zones
    dp.fetch_ohlcv("RELIANCE", "1d", "2025-09-01", "2025-09-30")
    assert fake_yf.calls[0]["auto_adjust"] is True


def test_yfinance_provider_rejects_unsupported_interval(fake_yf):
    assert dp.fetch_ohlcv("RELIANCE", "2m", "2025-09-01", "2025-09-30").empty
    assert fake_yf.calls == []


def test_yfinance_provider_retries_then_gives_up_on_empty(monkeypatch):
    attempts = []

    class Empty:
        def __init__(self, symbol):
            pass

        def history(self, **kw):
            attempts.append(1)
            return pd.DataFrame()

    module = type(sys)("yfinance")
    module.Ticker = Empty
    monkeypatch.setitem(sys.modules, "yfinance", module)
    monkeypatch.setattr(dp.time, "sleep", lambda s: None)

    df = dp.fetch_ohlcv("RELIANCE", "15m", "2025-09-01", "2025-09-30")
    assert df.empty
    assert len(attempts) == 3          # initial try plus two retries


def test_yfinance_provider_warns_past_the_intraday_history_limit(fake_yf, capsys):
    # Yahoo serves ~60 days of 15m data; asking for 2015 should say so, not fail silently
    dp.fetch_ohlcv("RELIANCE", "15m", "2015-01-01", "2015-01-31")
    assert "WARNING" in capsys.readouterr().out


def test_yfinance_is_the_default_provider(fake_yf):
    dp.fetch_ohlcv("RELIANCE", "15m", "2025-09-01", "2025-09-30")
    assert len(fake_yf.calls) == 1     # reached yfinance without naming it
