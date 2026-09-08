"""
Ultimate GTF Strategy Backtester
================================

Multi-asset backtester for the GTF (Gap Trading Formula) supply/demand zone
strategy, running on Fyers historical data.

Core idea
---------
1. Build "zones" on a Higher Time Frame (default 60m): a strong leg-in candle
   (rally or drop), followed by 1-3 small-bodied "base" candles, followed by a
   strong leg-out candle. That gives DBR / RBR (demand) and RBD / DBD (supply)
   zones.
2. Repeat the same zone detection on a Lower Time Frame (default 15m) for entry
   signals.
3. A LTF signal is only tradable when it lines up with an HTF zone of the same
   direction (confluence), scores highly enough, sits in an acceptable place on
   the curve, and shows wick rejection at the proximal line.
4. Entries: score >= 7.0 goes in immediately ("set & forget"); lower scores wait
   for a confirmation close beyond the proximal line on a later bar.
5. Stop loss = the zone's distal line. Target = a fixed R multiple of the risk.
   Risk per trade is fixed in rupees, so quantity is derived from the stop
   distance.
6. Intraday mode (any intraday LTF) blocks new entries at/after the EOD cutoff,
   force-closes same-day shorts at the cutoff, and never carries a short
   overnight.

Setup: see README.md in this folder. Run `generate_token.py` first.

Usage:
    python gtf_strategy_automation.py

DISCLAIMER: Educational and research use only. Not investment advice. Past
backtest results do not guarantee future performance.
"""

import math
import os
from datetime import datetime, time
from enum import Enum

import numpy as np
import pandas as pd
import pytz

try:  # the API client is only needed to actually fetch data
    from fyers_apiv3 import fyersModel
except ImportError:  # pragma: no cover - allows importing the strategy helpers
    fyersModel = None

# =========================================================================
# SECTION 1: GLOBAL CONFIGURATION AND STATE
# =========================================================================

# --- CONFIGURATION ---
STOCK_LIST_FILE = "stock_list.txt"
REPORT_FILE = f"Backtest_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
START_DATE = "2025-09-01"
END_DATE = "2025-09-30"

# === FYERS API CREDENTIALS ===
CLIENT_ID = "AAAA1AAAAA-999"  # Use your actual Client ID
TOKEN_FILE = "fyers_access_token.txt"

# === TIMEZONE CONFIGURATION ===
utc_tz = pytz.utc
ist_tz = pytz.timezone("Asia/Kolkata")

# No new entries at/after this time; same-day shorts are squared off here.
EOD_CUTOFF_TIME_IST = time(14, 45)

# === USER CONFIGURABLE TIME FRAME SETTINGS ===
HTF_INTERVAL = "60m"  # Higher time frame: zone/confluence map
LTF_INTERVAL = "15m"  # Lower time frame: entries

# --- STRATEGY TYPE (AUTO-CONFIGURED) ---
# Enables/disables the intraday-only rules based on the LTF chosen above.
IS_INTRADAY_STRATEGY = LTF_INTERVAL in ["5m", "15m", "30m", "60m"]

# --- Strategy Constants (high-probability GTF configuration) ---
INITIAL_CAPITAL = 100000.0
RISK_PER_TRADE = 1000.0  # 1% of initial capital
MAX_BASE_CANDLES = 3
BASE_CANDLE_RATIO = 0.33
EMA_TP_PERIOD = 200

# --- High-Probability Filters ---
MIN_TRADE_SCORE = 4.5
MIN_WICK_TO_RANGE_RATIO = 0.4
MAX_R_MULTIPLE = 3.0

# --- Market Regime Filter Constants ---
ATR_LOOKBACK = 14
MAX_ZONE_WIDTH_ATR_MULTIPLE = 1.5

# --- Global Storage for Results ---
ALL_SUMMARY_RESULTS = []
ALL_COMPLETED_TRADES = []

# Populated by initialize_fyers()
fyers = None


# =========================================================================
# SECTION 2: CUSTOM CLASSES
# =========================================================================


class GTFZone:
    """A supply or demand zone: proximal = entry line, distal = stop line."""

    def __init__(self, index, zone_type, proximal, distal, is_demand, base_count,
                 entry_price=0, sl_price=0):
        self.index = index
        self.type = zone_type
        self.proximal = proximal
        self.distal = distal
        self.is_demand = is_demand
        self.base_count = base_count
        self.hit_count = 0
        self.score = 0.0
        self.consumed = False
        self.entry = entry_price
        self.sl = sl_price


class Trade:
    """A live/closed position."""

    def __init__(self, index, entry_date, trade_type, entry_price, sl_price, tp_price, qty, score):
        self.index = index
        self.entry_date = entry_date
        self.type = trade_type
        self.entry = entry_price
        self.sl = sl_price
        self.tp = tp_price
        self.qty = qty
        self.score = score
        self.status = "OPEN"
        self.pnl = 0.0


class PendingTrade:
    """A signal waiting for a confirmation close beyond the proximal line."""

    def __init__(self, zone, current_date, entry_price, sl_price, tp_price, qty, score):
        self.zone = zone
        self.entry_date = current_date
        self.type = "BUY" if zone.is_demand else "SELL"
        self.entry = entry_price
        self.sl = sl_price
        self.tp = tp_price
        self.qty = qty
        self.score = score


class Location(Enum):
    """Where price sits on the curve between the nearest fresh demand/supply."""

    VERY_HIGH = 1
    HIGH = 2
    EQUILIBRIUM = 3
    LOW = 4
    VERY_LOW = 5


# =========================================================================
# SECTION 3: HELPER FUNCTIONS
# =========================================================================


def get_fyers_data(fyers_model, ticker_symbol, interval, start_date, end_date):
    """Fetch historical candles from Fyers and return them as a DataFrame."""
    # stock_list.txt holds plain NSE symbols ("HDFCBANK"), not "HDFCBANK.NS"
    fyers_symbol = f"NSE:{ticker_symbol}-EQ"

    interval_mapper = {
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "1d": "D",
        "1mo": "M",
    }

    if interval not in interval_mapper:
        print(f"Interval {interval} not supported by this script.")
        return pd.DataFrame()

    fyers_resolution = interval_mapper[interval]

    data = {
        "symbol": fyers_symbol,
        "resolution": fyers_resolution,
        "date_format": "1",  # YYYY-MM-DD
        "range_from": start_date,
        "range_to": end_date,
        "cont_flag": "1",
    }

    try:
        response = fyers_model.history(data=data)
    except Exception as e:
        print(f"Error fetching Fyers data for {ticker_symbol}: {e}")
        return pd.DataFrame()

    if response.get("code") != 200 or not response.get("candles"):
        print(f"Fyers API Error for {ticker_symbol} ({fyers_symbol}): "
              f"{response.get('message', 'No candles data')}")
        return pd.DataFrame()

    df = pd.DataFrame(response["candles"])
    df.rename(columns={0: "Date", 1: "Open", 2: "High", 3: "Low", 4: "Close", 5: "Volume"},
              inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True)
    df.set_index("Date", inplace=True)
    return df


def _scalar(value):
    """Return a plain float whether the input is a numpy scalar or a Python one."""
    try:
        return value.item()
    except AttributeError:
        return value


def add_candle_features(df):
    """Add body/range/wick/ATR columns used everywhere in the strategy."""
    df["Body"] = abs(df["Close"] - df["Open"])
    df["Range"] = df["High"] - df["Low"]
    body_max = np.maximum(df["Open"], df["Close"])
    body_min = np.minimum(df["Open"], df["Close"])
    df["Upper_Wick"] = df["High"] - body_max
    df["Lower_Wick"] = body_min - df["Low"]
    return df


def is_base(candle):
    """A base candle: small body relative to its range."""
    body = _scalar(candle["Body"])
    rng = _scalar(candle["Range"])
    if rng == 0:
        return False
    return (body / rng) < BASE_CANDLE_RATIO and body > 0


def is_exciting(candle):
    """An 'exciting' (leg) candle: body dominates the range. Returns (rally, drop)."""
    body = _scalar(candle["Body"])
    rng = _scalar(candle["Range"])
    if rng == 0:
        return False, False
    strong = (body / rng) > BASE_CANDLE_RATIO
    close_price = _scalar(candle["Close"])
    open_price = _scalar(candle["Open"])
    return (close_price > open_price) and strong, (close_price < open_price) and strong


def check_zone_quality(current_atr, proximal, distal):
    """Reject zones that are too wide for the current volatility regime."""
    if MAX_ZONE_WIDTH_ATR_MULTIPLE == 999:
        return True
    zone_width = abs(proximal - distal)
    max_allowed_width = MAX_ZONE_WIDTH_ATR_MULTIPLE * current_atr
    return zone_width <= max_allowed_width


def get_base_extremes(data, current_index, base_count):
    """Proximal/distal lines for demand and supply readings of the same base."""
    base_start = current_index - base_count
    base_range_data = data.iloc[base_start:current_index]

    dem_prox = _scalar(base_range_data["Close"].iloc[-1])
    dem_dist = _scalar(base_range_data["Low"].min())

    sup_prox = _scalar(base_range_data["Close"].iloc[-1])
    sup_dist = _scalar(base_range_data["High"].max())

    return (dem_prox, dem_dist), (sup_prox, sup_dist)


def find_zones(data, atr_series=None, apply_quality_filter=False):
    """Scan a DataFrame for leg-in / base / leg-out zone patterns."""
    zones = []
    start_bar = MAX_BASE_CANDLES + 2

    for i in range(start_bar, len(data)):
        current_bar = data.iloc[i]
        is_current_rally, is_current_drop = is_exciting(current_bar)

        if not (is_current_rally or is_current_drop):
            continue

        for base_count in range(1, MAX_BASE_CANDLES + 1):
            if i - base_count - 1 < 0:
                continue

            is_valid_base = all(is_base(data.iloc[j]) for j in range(i - base_count, i))
            if not is_valid_base:
                continue

            entry_move = data.iloc[i - base_count - 1]
            is_entry_rally, is_entry_drop = is_exciting(entry_move)
            if not (is_entry_rally or is_entry_drop):
                continue

            (dem_prox, dem_dist), (sup_prox, sup_dist) = get_base_extremes(data, i, base_count)

            bar_atr = 0.0
            if atr_series is not None:
                bar_atr = _scalar(atr_series.iloc[i])
                if pd.isna(bar_atr):
                    continue

            if is_current_rally:
                if apply_quality_filter and not check_zone_quality(bar_atr, dem_prox, dem_dist):
                    continue
                zone_type = "DBR" if is_entry_drop else "RBR"
                zones.append(GTFZone(i, zone_type, dem_prox, dem_dist, True, base_count,
                                     entry_price=dem_prox, sl_price=dem_dist))
            else:
                if apply_quality_filter and not check_zone_quality(bar_atr, sup_prox, sup_dist):
                    continue
                zone_type = "RBD" if is_entry_rally else "DBD"
                zones.append(GTFZone(i, zone_type, sup_prox, sup_dist, False, base_count,
                                     entry_price=sup_prox, sl_price=sup_dist))

    return zones


def get_htf_zones(htf_data):
    """Build the higher-timeframe zone map used for confluence and location."""
    htf_data = add_candle_features(htf_data)
    htf_data["ATR"] = htf_data["Range"].rolling(2).mean()

    htf_zones = find_zones(htf_data, atr_series=None, apply_quality_filter=False)

    # Only zones that have not been over-traded remain usable
    return [z for z in htf_zones if z.hit_count <= 2]


def check_htf_confluence(current_price, fresh_htf_zones, is_demand_signal):
    """Return the HTF zone containing this price in the same direction, if any."""
    for zone in fresh_htf_zones:
        if is_demand_signal != zone.is_demand:
            continue
        if zone.is_demand:
            if zone.distal <= current_price <= zone.proximal:
                return zone
        else:
            if zone.proximal <= current_price <= zone.distal:
                return zone
    return None


def check_trade_score(data, zone, current_index):
    """Score a zone out of 10: freshness, base tightness, leg-out strength, EMA confluence."""
    score = 0.0

    # Freshness
    if zone.hit_count == 0:
        score += 3.0
    elif zone.hit_count == 1:
        score += 1.5

    # Tight base
    if 1 <= zone.base_count <= 3:
        score += 2.0

    # Departure strength
    legout_index = zone.index
    if 0 < legout_index < len(data):
        legout = data.iloc[legout_index]
        prev_close = _scalar(data.iloc[legout_index - 1]["Close"])
        legout_atr = _scalar(legout["ATR"])
        is_gap_open = (not pd.isna(legout_atr)) and \
            abs(_scalar(legout["Open"]) - prev_close) > legout_atr

        is_two_exciting_candles = False
        if legout_index + 1 < len(data):
            legout_body = _scalar(legout["Body"])
            legout_range = _scalar(legout["Range"])
            next_body = _scalar(data.iloc[legout_index + 1]["Body"])
            next_range = _scalar(data.iloc[legout_index + 1]["Range"])
            if legout_range > 0 and next_range > 0:
                if (legout_body / legout_range) > 0.5 and (next_body / next_range) > 0.5:
                    is_two_exciting_candles = True

        if is_gap_open:
            score += 2.0
        elif is_two_exciting_candles:
            score += 2.0
        else:
            score += 1.0

    # Moving-average confluence inside the zone
    try:
        current_ema_20 = _scalar(data["EMA_20"].iloc[current_index])
        current_ema_50 = _scalar(data["EMA_50"].iloc[current_index])
        if (zone.distal < current_ema_20 < zone.proximal) or \
           (zone.distal < current_ema_50 < zone.proximal):
            score += 1.0
    except (ValueError, IndexError, KeyError):
        pass

    return score


def check_wick_rejection(current_bar, zone_type):
    """Require a rejection wick into the zone before entering."""
    rng = _scalar(current_bar["Range"])
    if rng == 0:
        return True

    if zone_type in ["DBR", "RBR"]:
        lower_wick = _scalar(current_bar["Lower_Wick"])
        return (lower_wick / rng) >= MIN_WICK_TO_RANGE_RATIO
    if zone_type in ["RBD", "DBD"]:
        upper_wick = _scalar(current_bar["Upper_Wick"])
        return (upper_wick / rng) >= MIN_WICK_TO_RANGE_RATIO
    return False


def check_location_filter(current_price, fresh_htf_zones):
    """Split the HTF curve into fifths: don't buy at the top, don't sell at the bottom."""
    fresh_demands = [z for z in fresh_htf_zones if z.is_demand and z.hit_count == 0]
    fresh_supplies = [z for z in fresh_htf_zones if not z.is_demand and z.hit_count == 0]

    if not fresh_demands or not fresh_supplies:
        return Location.EQUILIBRIUM

    nearest_sz_prox = max(z.proximal for z in fresh_supplies)
    nearest_dz_prox = min(z.proximal for z in fresh_demands)

    if nearest_sz_prox <= nearest_dz_prox:
        return Location.EQUILIBRIUM

    curve_range = nearest_sz_prox - nearest_dz_prox
    one_fifth = curve_range / 5

    if current_price > (nearest_sz_prox - one_fifth):
        return Location.VERY_HIGH
    if current_price > (nearest_sz_prox - 2 * one_fifth):
        return Location.HIGH
    if current_price > (nearest_dz_prox + 2 * one_fifth):
        return Location.EQUILIBRIUM
    if current_price > (nearest_dz_prox + one_fifth):
        return Location.LOW
    return Location.VERY_LOW


def get_dynamic_tp(entry_price, sl_price, data, current_index, zone_type):
    """Fixed R-multiple target measured from the stop distance."""
    risk_distance = abs(entry_price - sl_price)
    r_multiple = MAX_R_MULTIPLE

    if zone_type in ["DBR", "RBR"]:
        return entry_price + (risk_distance * r_multiple)
    return entry_price - (risk_distance * r_multiple)


# =========================================================================
# SECTION 4: MAIN BACKTEST LOGIC
# =========================================================================


def initialize_fyers():
    """Log in to Fyers with the saved access token. Returns a FyersModel."""
    global fyers

    if fyersModel is None:
        print("--- ERROR: 'fyers-apiv3' is not installed. ---")
        print("Install it with: pip install fyers-apiv3")
        return None

    try:
        with open(TOKEN_FILE, "r") as f:
            access_token = f.read().strip()
    except FileNotFoundError:
        print(f"--- ERROR: '{TOKEN_FILE}' not found. ---")
        print("Please run 'generate_token.py' first to create your access token.")
        return None

    try:
        fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=access_token, log_path=os.getcwd())
        profile = fyers.get_profile()
    except Exception as e:
        print(f"--- Fyers API Initialization Error: {e} ---")
        print(f"This may be due to an expired token. Delete '{TOKEN_FILE}' "
              "and run 'generate_token.py' again.")
        return None

    if profile.get("code") == 200:
        print(f"--- Fyers API Login Successful. Welcome, {profile['data']['name']} ---")
        return fyers

    print(f"--- Fyers API Login Failed: {profile.get('message')} ---")
    print(f"Please delete '{TOKEN_FILE}' and run 'generate_token.py' again.")
    return None


def run_backtest(ticker, start_date, end_date, fyers_model=None):
    """Run the GTF backtest for a single symbol. Returns a summary dict or None."""
    fyers_model = fyers_model or fyers

    current_capital = INITIAL_CAPITAL
    trade_id_counter = 0
    all_zones = []
    open_trades = []
    closed_trades = []
    pending_trades = []

    print(f"\n--- Starting GTF Backtest for {ticker} "
          f"(HTF: {HTF_INTERVAL}, LTF: {LTF_INTERVAL}) ---")

    # --- 1. DATA FETCHING ---
    ltf_data = get_fyers_data(fyers_model, ticker, LTF_INTERVAL, start_date, end_date)
    htf_data = get_fyers_data(fyers_model, ticker, HTF_INTERVAL, start_date, end_date)
    start_dt = pd.to_datetime(start_date).date()

    if ltf_data.empty or htf_data.empty:
        print("Error: Could not fetch enough data for both timeframes.")
        return None

    # --- 2. HTF ZONE PRE-CALCULATION ---
    fresh_htf_zones = get_htf_zones(htf_data)
    print(f"Found {len(fresh_htf_zones)} reusable {HTF_INTERVAL} zones for confluence.")

    # --- 3. LTF FEATURE ENGINEERING ---
    data = add_candle_features(ltf_data.copy())
    data["ATR"] = data["Range"].rolling(ATR_LOOKBACK).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA_200"] = data["Close"].ewm(span=EMA_TP_PERIOD, adjust=False).mean()

    # --- 4. BACKTEST LOOP ---
    start_bar = max(MAX_BASE_CANDLES + 2, EMA_TP_PERIOD + ATR_LOOKBACK + 2)
    if start_bar >= len(data):
        print(f"Error: Not enough LTF data bars available ({len(data)}) to meet the "
              f"indicator warmup period ({start_bar}).")
        return None

    for i in range(start_bar, len(data)):
        current_date = data.index[i]
        current_bar = data.iloc[i]

        if pd.isna(data["ATR"].iloc[i]) or pd.isna(data["EMA_200"].iloc[i]):
            continue

        bar_atr = _scalar(data["ATR"].iloc[i])

        if current_date.date() < start_dt:
            continue

        current_close = _scalar(current_bar["Close"])
        current_time_ist = current_date.astimezone(ist_tz).time()

        # ------------------------------------------------------------------
        # A. ZONE IDENTIFICATION (creation on the LTF)
        # ------------------------------------------------------------------
        is_current_rally, is_current_drop = is_exciting(current_bar)

        if is_current_rally or is_current_drop:
            for base_count in range(1, MAX_BASE_CANDLES + 1):
                if i - base_count - 1 < 0:
                    continue

                is_valid_base = all(is_base(data.iloc[j]) for j in range(i - base_count, i))
                if not is_valid_base:
                    continue

                entry_move = data.iloc[i - base_count - 1]
                is_entry_rally, is_entry_drop = is_exciting(entry_move)
                if not (is_entry_rally or is_entry_drop):
                    continue

                (dem_prox, dem_dist), (sup_prox, sup_dist) = get_base_extremes(data, i, base_count)

                if is_current_rally:
                    if check_zone_quality(bar_atr, dem_prox, dem_dist):
                        zone_type = "DBR" if is_entry_drop else "RBR"
                        all_zones.append(
                            GTFZone(i, zone_type, dem_prox, dem_dist, True, base_count))
                else:
                    if check_zone_quality(bar_atr, sup_prox, sup_dist):
                        zone_type = "RBD" if is_entry_rally else "DBD"
                        all_zones.append(
                            GTFZone(i, zone_type, sup_prox, sup_dist, False, base_count))

        # ------------------------------------------------------------------
        # B1. PENDING TRADES — confirm or invalidate on this bar
        # ------------------------------------------------------------------
        trades_to_activate = []
        pending_to_invalidate = []

        for p_trade in pending_trades:
            confirmed = (
                (p_trade.type == "BUY" and current_close > p_trade.entry)
                or (p_trade.type == "SELL" and current_close < p_trade.entry)
            )

            if confirmed:
                trades_to_activate.append(p_trade)
            elif (p_trade.type == "BUY" and _scalar(current_bar["Low"]) <= p_trade.sl) or \
                 (p_trade.type == "SELL" and _scalar(current_bar["High"]) >= p_trade.sl):
                # Stopped out before it ever confirmed
                pending_to_invalidate.append(p_trade)
                p_trade.zone.consumed = True

        for p_trade in trades_to_activate:
            trade_id_counter += 1
            open_trades.append(Trade(trade_id_counter, p_trade.entry_date, p_trade.type,
                                     p_trade.entry, p_trade.sl, p_trade.tp,
                                     p_trade.qty, p_trade.score))
            pending_trades.remove(p_trade)
            p_trade.zone.consumed = True

        for p_trade in pending_to_invalidate:
            if p_trade in pending_trades:
                pending_trades.remove(p_trade)

        # ------------------------------------------------------------------
        # B2. NEW SIGNALS — proximal touch on a fresh zone
        # ------------------------------------------------------------------
        fresh_zones = [z for z in all_zones if not z.consumed and z.index < i]
        location = check_location_filter(current_close, fresh_htf_zones)

        for zone in fresh_zones:
            entry_price = zone.proximal
            current_low_val = _scalar(current_bar["Low"])
            current_high_val = _scalar(current_bar["High"])

            proximal_hit = (zone.is_demand and current_low_val <= entry_price) or \
                           (not zone.is_demand and current_high_val >= entry_price)
            if not proximal_hit:
                continue

            zone.hit_count += 1

            # --- Filters ---
            htf_confluence_zone = check_htf_confluence(entry_price, fresh_htf_zones, zone.is_demand)
            is_htf_aligned = htf_confluence_zone is not None

            zone.score = check_trade_score(data, zone, i)
            is_tradable_score = zone.score >= MIN_TRADE_SCORE

            is_tradable_location = not (
                (zone.is_demand and location in [Location.VERY_HIGH, Location.HIGH])
                or (not zone.is_demand and location in [Location.VERY_LOW, Location.LOW])
            )

            is_tradable_rejection = check_wick_rejection(current_bar, zone.type)

            if not all([is_htf_aligned, is_tradable_score, is_tradable_location,
                        is_tradable_rejection]):
                zone.consumed = True
                continue

            # --- EOD entry barrier ---
            if IS_INTRADAY_STRATEGY and current_time_ist >= EOD_CUTOFF_TIME_IST:
                zone.consumed = True
                continue

            # --- Position sizing off the stop distance ---
            sl_price = zone.distal
            risk_distance = abs(entry_price - sl_price)
            if risk_distance == 0:
                zone.consumed = True
                continue

            qty = math.floor(RISK_PER_TRADE / risk_distance)
            if qty == 0:
                zone.consumed = True
                continue

            # --- Margin / capital check ---
            trade_value = qty * entry_price
            if IS_INTRADAY_STRATEGY and trade_value > (current_capital * 5):  # 5x MIS leverage
                zone.consumed = True
                continue
            if (not IS_INTRADAY_STRATEGY) and trade_value > current_capital:  # 1x CNC
                zone.consumed = True
                continue

            tp_price = get_dynamic_tp(entry_price, sl_price, data, i, zone.type)

            # --- Entry decision: Type 1 (immediate) vs Type 2/3 (confirmation) ---
            if zone.score >= 7.0:
                trade_id_counter += 1
                open_trades.append(
                    Trade(trade_id_counter, current_date, "BUY" if zone.is_demand else "SELL",
                          entry_price, sl_price, tp_price, qty, zone.score))
                zone.consumed = True
            else:
                pending_trades.append(
                    PendingTrade(zone, current_date, entry_price, sl_price, tp_price,
                                 qty, zone.score))

            if is_htf_aligned and htf_confluence_zone is not None:
                htf_confluence_zone.hit_count += 1

        # ------------------------------------------------------------------
        # C. TRADE MANAGEMENT (closing)
        # ------------------------------------------------------------------
        trades_to_close = []
        high = _scalar(current_bar["High"])
        low = _scalar(current_bar["Low"])

        for trade in open_trades:
            close_status = None
            pnl = 0.0
            exit_price = 0.0

            # 1. Failsafe: an intraday short must never be carried overnight
            if IS_INTRADAY_STRATEGY and trade.type == "SELL" and \
                    trade.entry_date.date() != current_date.date():
                close_status = "OVERNIGHT_SHORT_FAIL"
                exit_price = _scalar(current_bar["Open"])
                pnl = trade.qty * (trade.entry - exit_price)

            # 2. Stop loss
            elif (trade.type == "BUY" and low <= trade.sl) or \
                 (trade.type == "SELL" and high >= trade.sl):
                close_status = "SL"
                pnl = -RISK_PER_TRADE
                exit_price = trade.sl

            # 3. Take profit
            elif (trade.type == "BUY" and high >= trade.tp) or \
                 (trade.type == "SELL" and low <= trade.tp):
                close_status = "TP"
                pnl = trade.qty * abs(trade.tp - trade.entry)
                exit_price = trade.tp

            # 4. EOD force-close for same-day intraday shorts
            elif IS_INTRADAY_STRATEGY and trade.type == "SELL" and \
                    trade.entry_date.date() == current_date.date() and \
                    current_time_ist >= EOD_CUTOFF_TIME_IST:
                close_status = "FORCED_EOD_CLOSE"
                exit_price = current_close
                pnl = trade.qty * (trade.entry - exit_price)

            if not close_status:
                continue

            trade.status = close_status
            trade.pnl = pnl
            current_capital += pnl
            trades_to_close.append(trade)
            closed_trades.append(trade)

            if close_status == "SL":
                r_achieved = -1.0
            else:
                r_achieved = pnl / RISK_PER_TRADE

            entry_time_ist = trade.entry_date.astimezone(ist_tz)
            exit_time_ist = current_date.astimezone(ist_tz)

            ALL_COMPLETED_TRADES.append({
                "Ticker": ticker,
                "ID": trade.index,
                "Zone_Score": trade.score,
                "Entry Date": entry_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "Exit Date": exit_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "Type": trade.type,
                "Entry Price": trade.entry,
                "Exit Price": exit_price,
                "Stop Loss": trade.sl,
                "Take Profit": trade.tp,
                "R_Achieved": r_achieved,
                "Qty": trade.qty,
                "Status": close_status,
                "PnL": pnl,
                "Capital After": current_capital,
            })

        for trade in trades_to_close:
            open_trades.remove(trade)

    # --- 5. RESULTS ---
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades
                      if t.status == "TP" or (t.status == "FORCED_EOD_CLOSE" and t.pnl > 0)]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades else 0
    total_pnl = current_capital - INITIAL_CAPITAL

    print(f"\n--- Backtest Complete for {ticker} ---")
    print(f"Total P&L: Rs {total_pnl:.2f} | Win Rate: {win_rate:.2f}% | "
          f"Total Trades: {total_trades}")

    return {
        "Ticker": ticker,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
        "Total P&L": total_pnl,
        "Final Capital": current_capital,
        "Status": "COMPLETED",
    }


# =========================================================================
# SECTION 5: SCRIPT EXECUTION
# =========================================================================


def load_tickers(stock_list_file=STOCK_LIST_FILE):
    """Read NSE symbols, one per line; '#' starts a comment."""
    try:
        with open(stock_list_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"ERROR: {stock_list_file} not found. Please create the file with stock symbols.")
        return []


def write_report(summary_df, trades_df, report_file=REPORT_FILE):
    """Write the Summary_Report and Detailed_Trades sheets to Excel."""
    try:
        with pd.ExcelWriter(report_file, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary_Report", index=False)
            trades_df.to_excel(writer, sheet_name="Detailed_Trades", index=False)
        print(f"\nDetailed Excel Report Saved: {os.path.abspath(report_file)}")
    except Exception as e:
        print(f"\nFATAL ERROR SAVING EXCEL: {e}")
        print("Please ensure 'openpyxl' is installed: pip install openpyxl")


def execute_multi_backtest():
    tickers = load_tickers()
    if not tickers:
        print(f"ERROR: no symbols to test. Add NSE symbols to {STOCK_LIST_FILE}.")
        return

    if initialize_fyers() is None:
        return

    print(f"\n--- Starting Multi-Asset Backtest on {len(tickers)} Symbols "
          f"(HTF: {HTF_INTERVAL}, LTF: {LTF_INTERVAL}) ---")

    for ticker in tickers:
        result = run_backtest(ticker, START_DATE, END_DATE)
        if result:
            ALL_SUMMARY_RESULTS.append(result)

    if not ALL_SUMMARY_RESULTS:
        print("\nNo completed backtests to report.")
        return

    summary_df = pd.DataFrame(ALL_SUMMARY_RESULTS)
    summary_df = summary_df.sort_values(by="Total P&L", ascending=False).reset_index(drop=True)
    trades_df = pd.DataFrame(ALL_COMPLETED_TRADES)

    total_portfolio_pnl = summary_df["Total P&L"].sum()

    print("\n" + "=" * 80)
    print("CONSOLIDATED MULTI-ASSET BACKTEST REPORT")
    print(f"Total Portfolio P&L (Sum of all stocks): Rs {total_portfolio_pnl:,.2f}")
    print("=" * 80)

    display_df = summary_df.copy()
    display_df["Total P&L"] = display_df["Total P&L"].apply(lambda x: f"Rs {x:,.2f}")
    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x:.2f}%")
    display_df = display_df.drop(columns=["Final Capital"])
    print(display_df.to_markdown(index=True))
    print("=" * 80)

    write_report(display_df, trades_df)


if __name__ == "__main__":
    execute_multi_backtest()
