# Ultimate GTF Strategy Backtester

A multi-asset backtester for the GTF supply/demand zone strategy on NSE equities,
using Fyers historical data.

## Disclaimer (important)

> "Investment in securities market are subject to market risks. Read all the related
> documents carefully before investing."

This software is provided for **educational and research purposes only**. It is not a
recommendation to buy or sell any financial instrument. Backtest results are based on
historical data and do not guarantee future performance.

- **No liability:** the author is not responsible for any financial losses incurred from using this code.
- **API usage:** this script uses the Fyers API. You are responsible for your own API credentials and for complying with Fyers' terms of service.
- **Algorithmic trading:** automated trading carries significant risk. Always test thoroughly in a paper-trading environment first.

## Run it on Google Colab (no local install)

[`GTF_Strategy_Colab.ipynb`](GTF_Strategy_Colab.ipynb) runs the whole backtest in the
browser — dependencies, Fyers login, parameters, results and charts, all as form-driven
cells. Nothing to install locally.

**To open it:**

- In Colab, go to **File → Open notebook → GitHub**, paste this repository's URL, pick the
  branch, and choose `gtf_strategy/GTF_Strategy_Colab.ipynb`; or
- download the `.ipynb` and use **File → Upload notebook**; or
- once this is merged to `main`:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ersid111/Stocks/blob/main/gtf_strategy/GTF_Strategy_Colab.ipynb)

The notebook clones this repo and imports the strategy module from it, so the logic below is
the single source of truth — it is not duplicated inside the notebook.

With the default `yfinance` source there is **nothing to log in to** — open the notebook, pick
your symbols and dates, and run. If the runtime disconnects, just re-run from the top.

## How the strategy works

1. **HTF zone map (default 60m).** A strong leg-in candle, 1-3 small-bodied *base*
   candles, then a strong leg-out candle define a zone:
   - Demand: `DBR` (drop-base-rally), `RBR` (rally-base-rally)
   - Supply: `RBD` (rally-base-drop), `DBD` (drop-base-drop)

   The zone's **proximal** line is the entry side, the **distal** line is the stop side.

2. **LTF signals (default 15m).** The same pattern scan runs on the entry timeframe,
   with a volatility filter that rejects zones wider than `MAX_ZONE_WIDTH_ATR_MULTIPLE`
   times the current ATR.

3. **Filters.** When price touches a fresh LTF zone's proximal line, the signal must pass
   all four checks or the zone is discarded:
   - **HTF confluence** — price must sit inside an HTF zone of the same direction.
   - **Score** — `>= MIN_TRADE_SCORE` (freshness 3.0 / tight base 2.0 / departure
     strength 1.0-2.0 / EMA-in-zone 1.0).
   - **Location** — no buying in the top two fifths of the curve, no selling in the
     bottom two fifths.
   - **Wick rejection** — the touching candle needs a rejection wick of at least
     `MIN_WICK_TO_RANGE_RATIO` of its range.

4. **Entry type.**
   - Score `>= 7.0` → **Type 1**, immediate entry at the proximal line (set & forget).
   - Score below that → **Type 2/3**, pending until a later candle *closes* beyond the
     proximal line. If the stop is hit before confirmation, the pending signal is cancelled.

5. **Risk.** Stop = the distal line, so quantity = `RISK_PER_TRADE / stop distance`.
   Target = `MAX_R_MULTIPLE` times the risk. A margin check caps position value at 5x
   capital intraday (MIS) or 1x for delivery (CNC).

6. **Intraday rules** (auto-enabled for any intraday LTF):
   - No new entries at or after `EOD_CUTOFF_TIME_IST` (14:45 IST).
   - Same-day shorts are force-closed at the cutoff.
   - A short that somehow survives to the next day is closed at that day's open as
     `OVERNIGHT_SHORT_FAIL`.

## Data sources

The backtester only needs candles, so it does not care where they come from. Pick one with
`DATA_PROVIDER` at the top of `gtf_strategy_automation.py`:

| `DATA_PROVIDER` | Account needed | Network | Notes |
| --- | --- | --- | --- |
| `"yfinance"` *(default)* | none | yes | Yahoo Finance. Just works. NSE symbols get `.NS` added for you. |
| `"csv"` | none | **no** | Your own files. Works fully offline. |
| `"fyers"` | Fyers account + token | yes | The original path. See the optional section below. |

Whichever you choose, the strategy receives the same thing: a UTC-indexed OHLCV frame with
prices adjusted for splits, sorted and de-duplicated.

### Know the limits of free Yahoo intraday data

This is the one thing that will bite you. Yahoo caps how far back intraday history goes:

| Interval | History available |
| --- | --- |
| `5m`, `15m`, `30m` | **last ~60 days only** |
| `60m` | last ~730 days |
| `1d` | many years |

The default `15m` entry timeframe therefore **cannot backtest a period more than about two
months old**. Ask for one anyway and you get a printed warning plus an empty result, not a
crash. For older or longer studies, either move to `60m`/`1d`, or use the `csv` provider with
data you supply. Yahoo data is also unofficial and occasionally has bad prints — spot-check
anything surprising before you act on it.

## Installation & setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

That is `pandas`, `numpy`, `yfinance`, `openpyxl`, `pytz` and `tabulate`. No broker SDK.

### 2. Create the stock list

Edit `stock_list.txt` and put one NSE symbol per line — plain symbols, no `.NS` suffix
(the yfinance provider adds it). Lines starting with `#` are ignored.

```
HDFCBANK
RELIANCE
SBIN
```

Indices work too, written the way Yahoo names them (`^NSEI` for Nifty 50).

### 3. Adjust settings

All parameters live in Section 1 of `gtf_strategy_automation.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DATA_PROVIDER` | `"yfinance"` | Where candles come from |
| `START_DATE` / `END_DATE` | `2025-09-01` / `2025-09-30` | Backtest window |
| `HTF_INTERVAL` | `60m` | Higher timeframe for the zone map |
| `LTF_INTERVAL` | `15m` | Entry timeframe |
| `INITIAL_CAPITAL` | `100000.0` | Starting capital per symbol |
| `RISK_PER_TRADE` | `1000.0` | Rupee risk per trade (1% of capital) |
| `MAX_BASE_CANDLES` | `3` | Longest base allowed |
| `BASE_CANDLE_RATIO` | `0.33` | Body/range threshold separating base from leg candles |
| `MIN_TRADE_SCORE` | `4.5` | Minimum zone score to trade |
| `MIN_WICK_TO_RANGE_RATIO` | `0.4` | Rejection wick required at the zone |
| `MAX_R_MULTIPLE` | `3.0` | Reward-to-risk target |
| `ATR_LOOKBACK` | `14` | ATR period |
| `MAX_ZONE_WIDTH_ATR_MULTIPLE` | `1.5` | Widest zone allowed (999 disables the filter) |
| `EOD_CUTOFF_TIME_IST` | `14:45` | Entry barrier and intraday short square-off |

Supported intervals: `5m`, `15m`, `30m`, `60m`, `1d`, `1mo`.

Note the warmup: the loop only starts after `EMA_TP_PERIOD + ATR_LOOKBACK + 2` bars
(216 by default), so a short date range on a slow timeframe will produce no trades.

### 4. Run it

```bash
python gtf_strategy_automation.py
```

## Using your own data (`csv`)

Set `DATA_PROVIDER = "csv"` and drop files into `csv_data/` (or point `CSV_DIR` elsewhere),
named `<TICKER>_<interval>.csv` — so `RELIANCE_15m.csv` and `RELIANCE_60m.csv` for the
default timeframes. Each file needs a date column (`Date`, `Datetime`, `Timestamp` or `Time`)
plus `Open,High,Low,Close,Volume`:

```csv
Date,Open,High,Low,Close,Volume
2025-09-01 09:15:00,1372.5,1378.0,1370.1,1376.4,184320
2025-09-01 09:30:00,1376.4,1381.2,1374.8,1379.9,151204
```

Timestamps without a timezone are read as IST, which is what an NSE export gives you.
This path touches no network at all.

## Optional: using Fyers instead

Only if you want it. Set `DATA_PROVIDER = "fyers"`, then:

```bash
pip install fyers-apiv3
```

- Set `CLIENT_ID` in **both** `generate_token.py` and `gtf_strategy_automation.py`, and
  `SECRET_KEY` in `generate_token.py`.
- Run `python generate_token.py`. It opens the Fyers login page — log in quickly, or it
  expires. You will be redirected to a "page not found" at `http://127.0.0.1`, which is
  normal; copy the full URL from the address bar and paste it back into the terminal. The
  token is saved to `fyers_access_token.txt`.

Tokens last for the trading day. Nothing else in the strategy changes.

## Output

A console summary plus an Excel report named `Backtest_Report_[Date_Time].xlsx` in the
same folder, with two sheets:

- **Summary_Report** — total P&L, win rate and trade count per symbol.
- **Detailed_Trades** — entry/exit time (IST), prices, stop, target, R achieved,
  quantity, status (`TP` / `SL` / `FORCED_EOD_CLOSE` / `OVERNIGHT_SHORT_FAIL`), P&L and
  running capital.

## Troubleshooting

| Message | Cause |
| --- | --- |
| `no LTF data for ... skipping` | Most often the Yahoo intraday limit: `15m` only goes back ~60 days. Use a recent range, a bigger interval, or the `csv` provider. |
| `WARNING: Yahoo serves at most ~60 days...` | Exactly that — your `START_DATE` is too far back for the interval. |
| `Interval ... not supported` | Use `5m`, `15m`, `30m`, `60m`, `1d` or `1mo`. |
| `Not enough LTF data bars available` | The range is shorter than the 216-bar warmup. Widen the dates or lower `EMA_TP_PERIOD`. |
| `yfinance is not installed` | `pip install yfinance`. |
| `CSV not found: ...` | The `csv` provider expects `<CSV_DIR>/<TICKER>_<interval>.csv`, e.g. `csv_data/RELIANCE_15m.csv`. |
| `no date column found` | Your CSV needs a `Date`, `Datetime`, `Timestamp` or `Time` column. |
| `Unknown provider '...'` | `DATA_PROVIDER` must be `yfinance`, `csv` or `fyers`. |
| Backtest completes but zero trades | Normal with strict filters. Try a longer window, a lower `MIN_TRADE_SCORE` or `MIN_WICK_TO_RANGE_RATIO`, or `MAX_ZONE_WIDTH_ATR_MULTIPLE = 999`. |
| `FATAL ERROR SAVING EXCEL` | `pip install openpyxl`. |
| `'fyers_access_token.txt' not found` / `Fyers API Login Failed` | Fyers path only. Re-run `generate_token.py`; tokens last one trading day. |

## Known limitations

Before you trade off these numbers, read the caveats — the backtest does **not** model
transaction costs or slippage, assumes stops fill exactly at the stop price, resets capital
per symbol, and builds its higher-timeframe zone map over the full history up front. See the
project discussion for the full list.
