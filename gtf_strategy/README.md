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

**Two differences from running locally:**

- **The login flow is manual.** Colab has no browser on the VM, so `generate_token.py` cannot
  be used there. The notebook prints a login link instead; you open it, log in, and paste the
  resulting `http://127.0.0.1/...auth_code=...` URL back into the next cell. The
  "site can't be reached" page you land on is expected.
- **The token does not survive a disconnect.** Colab wipes the VM, taking
  `fyers_access_token.txt` with it. After a reconnect, re-run the notebook from the top.

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

## Installation & setup

### 1. Install Python and dependencies

```bash
pip install -r requirements.txt
```

(That installs `pandas`, `numpy`, `fyers-apiv3`, `openpyxl`, `pytz` and `tabulate`.)

### 2. API configuration (crucial step)

You need a valid Fyers API account.

- **Client ID:** set `CLIENT_ID` in **both** `generate_token.py` and
  `gtf_strategy_automation.py` to your actual Fyers Client ID (e.g. `ABCDEFG-101`).
- **Secret key:** set `SECRET_KEY` in `generate_token.py`.
- **Access token:** the backtester *reads* a token, it does not generate one. Run the
  token generator first:

```bash
python generate_token.py
```

This opens the Fyers login page — log in quickly, or the page expires. You will be
redirected to a "page not found" at `http://127.0.0.1`, which is normal; copy the full
URL from the address bar and paste it back into the terminal. The token is saved to
`fyers_access_token.txt` in the same folder.

### 3. Create the stock list

Edit `stock_list.txt` and put one NSE symbol per line — plain symbols, no `.NS` suffix.
Lines starting with `#` are ignored.

```
HDFCBANK
RELIANCE
SBIN
```

### 4. Adjust settings

All parameters live in Section 1 of `gtf_strategy_automation.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
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

### 5. Run it

```bash
python gtf_strategy_automation.py
```

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
| `'fyers_access_token.txt' not found` | You didn't run `generate_token.py`, or the file is named differently / in another folder. |
| `Fyers API Login Failed` | Token expired. Delete `fyers_access_token.txt` and run `generate_token.py` again. |
| `Interval ... not supported` | Use one of `5m`, `15m`, `30m`, `60m`, `1d`, `1mo`. |
| `Not enough LTF data bars available` | The date range is too short for the 216-bar indicator warmup. Widen it or lower `EMA_TP_PERIOD`. |
| Empty report | `stock_list.txt` is empty, or the market was closed across the chosen dates. |
| `FATAL ERROR SAVING EXCEL` | `pip install openpyxl`. |
