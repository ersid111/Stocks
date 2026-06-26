#!/usr/bin/env python3
"""Streamlit web UI for NSE stock analysis powered by Claude AI."""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #4CAF50;
    margin-bottom: 8px;
  }
  .metric-card.red { border-left-color: #f44336; }
  .metric-card.yellow { border-left-color: #FFC107; }
  .metric-card.blue { border-left-color: #2196F3; }
  .scenario-bull {
    background: linear-gradient(135deg, #1b3a1b, #1e2130);
    border: 1px solid #4CAF50;
    border-radius: 10px;
    padding: 20px;
  }
  .scenario-bear {
    background: linear-gradient(135deg, #3a1b1b, #1e2130);
    border: 1px solid #f44336;
    border-radius: 10px;
    padding: 20px;
  }
  .scenario-swan {
    background: linear-gradient(135deg, #1b1b3a, #1e2130);
    border: 1px solid #9c27b0;
    border-radius: 10px;
    padding: 20px;
  }
  .sr-support { color: #4CAF50; font-weight: 600; }
  .sr-resistance { color: #f44336; font-weight: 600; }
  .report-text {
    font-family: 'Inter', sans-serif;
    line-height: 1.7;
    white-space: pre-wrap;
  }
  div[data-testid="stSidebar"] { background: #0e1117; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_inr(value, decimals=2):
    """Format a number as Indian Rupees."""
    if value is None:
        return "N/A"
    return f"₹{value:,.{decimals}f}"


def fmt_pct(value, decimals=1):
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def fmt_num(value, decimals=1):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def color_change(value):
    if value is None:
        return "gray"
    return "green" if value >= 0 else "red"


def nse_ticker(symbol: str) -> str:
    """Append .NS suffix if not already present."""
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        symbol = symbol + ".NS"
    return symbol


# ── Pipeline (runs inside spinner / progress callbacks) ───────────────────────

def run_full_analysis(ticker: str, model: str, status_cb):
    """Run the full analysis pipeline and return all computed objects."""
    results = {}
    ticker = ticker.upper().strip()

    status_cb("Fetching market data...")
    from data.market_data import fetch_market_snapshot
    snap = fetch_market_snapshot(ticker)
    results["snap"] = snap

    if snap.current_price is None:
        raise ValueError(f"Could not fetch price data for **{ticker}**. Check the symbol and try again.")

    status_cb(f"Fetching options chain for {ticker}...")
    from data.options_data import fetch_options_chain
    chain = fetch_options_chain(ticker, snap.current_price)
    results["chain"] = chain

    status_cb("Fetching macro & news data...")
    from data.macro_data import fetch_macro_context, fetch_news_sentiment
    macro = fetch_macro_context()
    news = fetch_news_sentiment(ticker)
    results["macro"] = macro
    results["news"] = news

    status_cb("Computing technical indicators...")
    from analysis.technical import compute_technical_indicators
    ind = compute_technical_indicators(snap.ohlcv)
    results["ind"] = ind

    status_cb("Analysing market structure...")
    from analysis.market_structure import (
        find_support_resistance, classify_trend,
        compute_volume_profile, detect_institutional_accumulation,
    )
    sr_levels = find_support_resistance(snap.ohlcv)
    trend = classify_trend(ind, snap.current_price)
    vp = compute_volume_profile(snap.ohlcv)
    accum = detect_institutional_accumulation(ind, snap.ohlcv)
    results.update(sr_levels=sr_levels, trend=trend, vp=vp, accum=accum)

    status_cb("Computing options analytics...")
    from analysis.options_math import (
        calculate_max_pain, calculate_gamma_exposure,
        calculate_iv_skew, calculate_expected_move, compute_put_call_metrics,
    )
    max_pain = calculate_max_pain(chain.calls, chain.puts)
    gex = calculate_gamma_exposure(chain.calls, chain.puts, snap.current_price)
    iv_skew = calculate_iv_skew(chain.calls, chain.puts, snap.current_price)

    nearest_dte = 30
    if chain.expiries:
        try:
            nearest_dte = max(
                (datetime.strptime(chain.expiries[0], "%Y-%m-%d") - datetime.now()).days, 1
            )
        except Exception:
            pass

    expected_move = calculate_expected_move(
        snap.current_price, iv_skew.atm_iv, nearest_dte, chain.calls, chain.puts
    )
    pc = compute_put_call_metrics(chain.calls, chain.puts)
    results.update(max_pain=max_pain, gex=gex, iv_skew=iv_skew, expected_move=expected_move, pc=pc)

    status_cb("Building scenario model...")
    from analysis.scenarios import build_scenarios
    scenarios = build_scenarios(ind, trend, pc, sr_levels, vp, snap.current_price, snap.ohlcv)
    results["scenarios"] = scenarios

    if os.environ.get("ANTHROPIC_API_KEY"):
        status_cb("Generating AI analysis via Claude...")
        from ai.prompt_builder import build_analysis_payload
        from ai.claude_client import generate_analysis
        payload = build_analysis_payload(
            snap=snap, ind=ind, trend=trend, sr_levels=sr_levels,
            vp=vp, accum=accum, chain=chain, max_pain=max_pain,
            gex=gex, iv_skew=iv_skew, expected_move=expected_move,
            pc=pc, scenarios=scenarios, macro=macro, news=news,
        )
        report_text, _ = generate_analysis(payload, model=model)
        results["report_text"] = report_text
    else:
        status_cb("Skipping AI report (no API key)...")
        results["report_text"] = None

    return results


# ── Chart builders ────────────────────────────────────────────────────────────

def build_candlestick(ohlcv: pd.DataFrame, ticker: str, sr_levels=None, vp=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = ohlcv.tail(120).copy()
    df.index = pd.to_datetime(df.index)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color="#4CAF50",
        decreasing_line_color="#f44336",
    ), row=1, col=1)

    # Volume bars
    colors = ["#4CAF50" if c >= o else "#f44336"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    # S/R levels
    if sr_levels:
        for lvl in sr_levels:
            color = "#4CAF50" if lvl.level_type == "support" else "#f44336"
            fig.add_hline(
                y=lvl.price, line_dash="dot", line_color=color,
                line_width=1, opacity=0.6, row=1, col=1,
                annotation_text=f"{'S' if lvl.level_type == 'support' else 'R'} {lvl.price:.0f}",
                annotation_font_color=color,
                annotation_font_size=10,
            )

    # Volume profile POC
    if vp and vp.poc:
        fig.add_hline(
            y=vp.poc, line_dash="dash", line_color="#FFC107",
            line_width=1.5, opacity=0.8, row=1, col=1,
            annotation_text=f"POC {vp.poc:.0f}",
            annotation_font_color="#FFC107",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def build_technical_chart(ohlcv: pd.DataFrame, ind):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = ohlcv.tail(100).copy()
    df.index = pd.to_datetime(df.index)

    rows = 4
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        subplot_titles=("Price + EMAs + BBands", "RSI", "MACD", "Volume"),
    )

    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#90CAF9", width=1.5)), row=1, col=1)

    # EMAs
    if ind.ema20 is not None:
        ema20_s = df["Close"].ewm(span=20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema20_s, name="EMA20",
                                 line=dict(color="#4CAF50", width=1)), row=1, col=1)
    if ind.ema50 is not None:
        ema50_s = df["Close"].ewm(span=50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema50_s, name="EMA50",
                                 line=dict(color="#FFC107", width=1)), row=1, col=1)
    if ind.ema200 is not None:
        ema200_s = df["Close"].ewm(span=200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema200_s, name="EMA200",
                                 line=dict(color="#f44336", width=1)), row=1, col=1)

    # Bollinger Bands
    if ind.bb_upper is not None:
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper",
                                 line=dict(color="#9E9E9E", width=0.8, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower",
                                 line=dict(color="#9E9E9E", width=0.8, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(158,158,158,0.05)"), row=1, col=1)

    # RSI
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name="RSI",
                             line=dict(color="#CE93D8", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#f44336", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#4CAF50", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", opacity=0.3, row=2, col=1)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                             line=dict(color="#4CAF50", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal",
                             line=dict(color="#f44336", width=1.5)), row=3, col=1)
    bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in histogram]
    fig.add_trace(go.Bar(x=df.index, y=histogram, name="Histogram",
                         marker_color=bar_colors, opacity=0.7), row=3, col=1)

    # Volume
    vol_colors = ["#4CAF50" if c >= o else "#f44336"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol",
                         marker_color=vol_colors, opacity=0.6), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
    )
    fig.update_yaxes(row=2, col=1, range=[0, 100])
    return fig


def build_oi_chart(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float, max_pain_price):
    import plotly.graph_objects as go

    if calls is None or puts is None or calls.empty or puts.empty:
        return None

    calls_g = calls.groupby("strike")["openInterest"].sum().reset_index()
    puts_g = puts.groupby("strike")["openInterest"].sum().reset_index()

    price_range = current_price * 0.3
    calls_g = calls_g[(calls_g["strike"] >= current_price - price_range) &
                       (calls_g["strike"] <= current_price + price_range)]
    puts_g = puts_g[(puts_g["strike"] >= current_price - price_range) &
                    (puts_g["strike"] <= current_price + price_range)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=calls_g["strike"], y=calls_g["openInterest"],
        name="Calls OI", marker_color="#4CAF50", opacity=0.75,
    ))
    fig.add_trace(go.Bar(
        x=-puts_g["openInterest"], y=puts_g["strike"],
        name="Puts OI", marker_color="#f44336", opacity=0.75,
        orientation="h",
    ))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=calls_g["strike"], y=calls_g["openInterest"],
        name="Call OI", marker_color="rgba(76,175,80,0.8)",
    ))
    fig.add_trace(go.Bar(
        x=puts_g["strike"], y=puts_g["openInterest"],
        name="Put OI", marker_color="rgba(244,67,54,0.8)",
    ))

    fig.add_vline(x=current_price, line_dash="solid", line_color="#90CAF9",
                  line_width=2, annotation_text=f"LTP {current_price:.0f}",
                  annotation_font_color="#90CAF9")
    if max_pain_price:
        fig.add_vline(x=max_pain_price, line_dash="dash", line_color="#FFC107",
                      line_width=2, annotation_text=f"MaxPain {max_pain_price:.0f}",
                      annotation_font_color="#FFC107")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        barmode="overlay",
        height=380,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Strike",
        yaxis_title="Open Interest",
        legend=dict(orientation="h"),
    )
    return fig


def build_iv_skew_chart(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float):
    import plotly.graph_objects as go

    if calls is None or puts is None or calls.empty or puts.empty:
        return None

    try:
        calls_filt = calls[calls["impliedVolatility"] > 0].copy()
        puts_filt = puts[puts["impliedVolatility"] > 0].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=calls_filt["strike"], y=calls_filt["impliedVolatility"] * 100,
            mode="markers+lines", name="Call IV",
            marker=dict(color="#4CAF50", size=5),
            line=dict(color="#4CAF50", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=puts_filt["strike"], y=puts_filt["impliedVolatility"] * 100,
            mode="markers+lines", name="Put IV",
            marker=dict(color="#f44336", size=5),
            line=dict(color="#f44336", width=1.5),
        ))
        fig.add_vline(x=current_price, line_dash="solid", line_color="#90CAF9",
                      line_width=2, annotation_text=f"LTP {current_price:.0f}",
                      annotation_font_color="#90CAF9")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=380,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Strike",
            yaxis_title="Implied Volatility (%)",
            legend=dict(orientation="h"),
        )
        return fig
    except Exception:
        return None


def build_scenario_donut(scenarios):
    import plotly.graph_objects as go

    labels, values, colors = [], [], []
    if scenarios.bull:
        labels.append(f"Bullish ({scenarios.bull.probability:.0%})")
        values.append(scenarios.bull.probability)
        colors.append("#4CAF50")
    if scenarios.neutral:
        labels.append(f"Neutral ({scenarios.neutral.probability:.0%})")
        values.append(scenarios.neutral.probability)
        colors.append("#FFC107")
    if scenarios.bear:
        labels.append(f"Bearish ({scenarios.bear.probability:.0%})")
        values.append(scenarios.bear.probability)
        colors.append("#f44336")
    if scenarios.black_swan:
        labels.append(f"Black Swan ({scenarios.black_swan.probability:.0%})")
        values.append(scenarios.black_swan.probability)
        colors.append("#9c27b0")

    if not values:
        return None

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
    )
    return fig


def build_volume_profile_chart(ohlcv: pd.DataFrame, vp, bins: int = 40):
    import plotly.graph_objects as go

    if ohlcv is None or ohlcv.empty:
        return None

    close_col = "Close" if "Close" in ohlcv.columns else "close"
    vol_col = "Volume" if "Volume" in ohlcv.columns else "volume"

    close = ohlcv[close_col].dropna()
    volume = ohlcv[vol_col].reindex(close.index).fillna(0)

    price_min, price_max = close.min(), close.max()
    if price_min >= price_max:
        return None

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_per_bin = np.zeros(bins)
    for i in range(bins):
        mask = (close >= bin_edges[i]) & (close < bin_edges[i + 1])
        vol_per_bin[i] = volume[mask].sum()

    bar_colors = []
    for c in bin_centers:
        if vp and vp.val and vp.vah and vp.val <= c <= vp.vah:
            bar_colors.append("rgba(33,150,243,0.7)")
        else:
            bar_colors.append("rgba(158,158,158,0.4)")

    fig = go.Figure(go.Bar(
        x=vol_per_bin, y=bin_centers,
        orientation="h",
        marker_color=bar_colors,
        name="Volume Profile",
    ))

    if vp and vp.poc:
        fig.add_hline(y=vp.poc, line_dash="dash", line_color="#FFC107",
                      line_width=2, annotation_text=f"POC {vp.poc:.0f}",
                      annotation_font_color="#FFC107")
    if vp and vp.vah:
        fig.add_hline(y=vp.vah, line_dash="dot", line_color="#90CAF9",
                      line_width=1, annotation_text=f"VAH {vp.vah:.0f}",
                      annotation_font_color="#90CAF9")
    if vp and vp.val:
        fig.add_hline(y=vp.val, line_dash="dot", line_color="#90CAF9",
                      line_width=1, annotation_text=f"VAL {vp.val:.0f}",
                      annotation_font_color="#90CAF9")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Volume",
        yaxis_title="Price",
        showlegend=False,
    )
    return fig


# ── Tab renderers ─────────────────────────────────────────────────────────────

def render_overview(r: dict):
    snap = r["snap"]
    ind = r["ind"]
    trend = r["trend"]
    accum = r["accum"]
    expected_move = r["expected_move"]

    # Header row
    price = snap.current_price or 0
    chg = snap.day_change_pct
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("LTP", fmt_inr(price), fmt_pct(chg) if chg is not None else None,
                delta_color="normal" if chg is not None and chg >= 0 else "inverse")
    col2.metric("52W High", fmt_inr(snap.fifty_two_week_high))
    col3.metric("52W Low", fmt_inr(snap.fifty_two_week_low))
    col4.metric("Market Cap", f"₹{snap.market_cap/1e9:.1f}B" if snap.market_cap else "N/A")
    col5.metric("Volume", f"{snap.volume_today:,.0f}" if snap.volume_today else "N/A")

    st.divider()

    # Chart + quick stats
    chart_col, info_col = st.columns([2.5, 1])
    with chart_col:
        if snap.ohlcv is not None and not snap.ohlcv.empty:
            fig = build_candlestick(snap.ohlcv, snap.ticker or "", r["sr_levels"], r["vp"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No OHLCV data available for chart.")

    with info_col:
        st.subheader("Quick Stats")
        stats = {
            "RSI (14)": fmt_num(ind.rsi),
            "ADX": fmt_num(ind.adx),
            "MACD": fmt_num(ind.macd),
            "ATR (14)": fmt_inr(ind.atr),
            "VWAP": fmt_inr(ind.vwap),
            "BB %": fmt_num(ind.bb_pct, 3) if ind.bb_pct else "N/A",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")

        st.divider()
        st.subheader("Trend")
        st.markdown(f"**EMA Alignment:** `{trend.ema_alignment}`")
        st.markdown(f"**Regime:** `{trend.regime}`")
        st.markdown(f"**Primary:** `{trend.primary_trend}`")
        st.markdown(f"**RSI Zone:** `{trend.rsi_zone}`")
        st.markdown(f"**Accumulation:** `{accum.signal}`")

        if expected_move and expected_move.average_move:
            st.divider()
            st.subheader("Expected Move")
            st.markdown(f"**Range:** {fmt_inr(expected_move.lower)} – {fmt_inr(expected_move.upper)}")
            st.markdown(f"**Move ±:** {fmt_inr(expected_move.average_move)}")


def render_technical(r: dict):
    snap = r["snap"]
    ind = r["ind"]

    if snap.ohlcv is not None and not snap.ohlcv.empty:
        fig = build_technical_chart(snap.ohlcv, ind)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No OHLCV data available.")

    # Indicator table
    st.subheader("All Indicators")
    rows = [
        ("RSI (14)", fmt_num(ind.rsi), "0–100; >70 OB, <30 OS"),
        ("MACD", fmt_num(ind.macd), "Momentum oscillator"),
        ("MACD Signal", fmt_num(ind.macd_signal), "9-period EMA of MACD"),
        ("MACD Histogram", fmt_num(ind.macd_histogram), "MACD – Signal"),
        ("EMA 20", fmt_inr(ind.ema20), "Short-term trend"),
        ("EMA 50", fmt_inr(ind.ema50), "Medium-term trend"),
        ("EMA 200", fmt_inr(ind.ema200), "Long-term trend"),
        ("BB Upper", fmt_inr(ind.bb_upper), "Bollinger Band upper"),
        ("BB Middle", fmt_inr(ind.bb_middle), "20-period SMA"),
        ("BB Lower", fmt_inr(ind.bb_lower), "Bollinger Band lower"),
        ("BB %B", fmt_num(ind.bb_pct, 3), "Position within bands"),
        ("ATR (14)", fmt_inr(ind.atr), "Average True Range"),
        ("ADX", fmt_num(ind.adx), "<20 range, >25 trend"),
        ("VWAP", fmt_inr(ind.vwap), "Volume-weighted avg price"),
        ("OBV", f"{ind.obv:,.0f}" if ind.obv else "N/A", "On-Balance Volume"),
        ("OBV Slope", fmt_num(ind.obv_slope), "OBV linear trend"),
    ]
    df_ind = pd.DataFrame(rows, columns=["Indicator", "Value", "Description"])
    st.dataframe(df_ind, use_container_width=True, hide_index=True)


def render_options(r: dict):
    chain = r["chain"]
    max_pain = r["max_pain"]
    gex = r["gex"]
    iv_skew = r["iv_skew"]
    pc = r["pc"]
    snap = r["snap"]

    if chain.error:
        st.warning(f"Options data unavailable: {chain.error}")
        st.info("Options are not available for all NSE symbols (e.g. ETFs, indices may need a different symbol format).")
        return

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max Pain", fmt_inr(max_pain.max_pain))
    c2.metric("GEX Regime", gex.regime or "N/A")
    c3.metric("Gamma Flip", fmt_inr(gex.gamma_flip))
    c4.metric("P/C OI Ratio", fmt_num(pc.oi_put_call_ratio, 3))
    c5.metric("ATM IV", f"{iv_skew.atm_iv*100:.1f}%" if iv_skew.atm_iv else "N/A")

    st.divider()

    oi_col, skew_col = st.columns(2)
    with oi_col:
        st.subheader("Open Interest Distribution")
        fig_oi = build_oi_chart(chain.calls, chain.puts, snap.current_price, max_pain.max_pain)
        if fig_oi:
            st.plotly_chart(fig_oi, use_container_width=True)

    with skew_col:
        st.subheader("IV Skew")
        fig_skew = build_iv_skew_chart(chain.calls, chain.puts, snap.current_price)
        if fig_skew:
            st.plotly_chart(fig_skew, use_container_width=True)

    st.divider()
    st.subheader("IV Skew Details")
    skew_rows = [
        ("ATM IV", f"{iv_skew.atm_iv*100:.2f}%" if iv_skew.atm_iv else "N/A"),
        ("OTM Put IV", f"{iv_skew.otm_put_iv*100:.2f}%" if iv_skew.otm_put_iv else "N/A"),
        ("OTM Call IV", f"{iv_skew.otm_call_iv*100:.2f}%" if iv_skew.otm_call_iv else "N/A"),
        ("Put Skew", f"{iv_skew.put_skew*100:.2f}%" if iv_skew.put_skew else "N/A"),
        ("Risk Reversal", f"{iv_skew.risk_reversal*100:.2f}%" if iv_skew.risk_reversal else "N/A"),
        ("P/C OI Ratio", fmt_num(pc.oi_put_call_ratio, 3)),
        ("P/C Vol Ratio", fmt_num(pc.vol_put_call_ratio, 3)),
        ("P/C $ Ratio", fmt_num(pc.dollar_put_call_ratio, 3)),
        ("Total Call OI", f"{pc.total_call_oi:,.0f}" if pc.total_call_oi else "N/A"),
        ("Total Put OI", f"{pc.total_put_oi:,.0f}" if pc.total_put_oi else "N/A"),
    ]
    df_skew = pd.DataFrame(skew_rows, columns=["Metric", "Value"])
    st.dataframe(df_skew, use_container_width=True, hide_index=True)

    # Options chain tables
    if not chain.calls.empty:
        with st.expander("Calls Chain (nearest expiry)"):
            exp = chain.expiries[0] if chain.expiries else None
            display_cols = ["strike", "openInterest", "impliedVolatility", "bid", "ask", "mid", "dte"]
            calls_exp = chain.calls[chain.calls["expiry"] == exp] if exp else chain.calls
            calls_exp = calls_exp[[c for c in display_cols if c in calls_exp.columns]].head(30)
            st.dataframe(calls_exp, use_container_width=True, hide_index=True)

    if not chain.puts.empty:
        with st.expander("Puts Chain (nearest expiry)"):
            exp = chain.expiries[0] if chain.expiries else None
            display_cols = ["strike", "openInterest", "impliedVolatility", "bid", "ask", "mid", "dte"]
            puts_exp = chain.puts[chain.puts["expiry"] == exp] if exp else chain.puts
            puts_exp = puts_exp[[c for c in display_cols if c in puts_exp.columns]].head(30)
            st.dataframe(puts_exp, use_container_width=True, hide_index=True)


def render_structure(r: dict):
    sr_levels = r["sr_levels"]
    vp = r["vp"]
    accum = r["accum"]
    snap = r["snap"]
    trend = r["trend"]

    st.subheader("Support & Resistance Levels")
    if sr_levels:
        sr_col, vp_col = st.columns([1, 1])
        with sr_col:
            sr_rows = []
            for lvl in reversed(sr_levels):  # highest first
                sr_rows.append({
                    "Price": fmt_inr(lvl.price),
                    "Type": lvl.level_type.title(),
                    "Touches": lvl.touches,
                })
            st.dataframe(pd.DataFrame(sr_rows), use_container_width=True, hide_index=True)

        with vp_col:
            st.subheader("Volume Profile")
            vp_items = [
                ("POC (Point of Control)", fmt_inr(vp.poc)),
                ("VAH (Value Area High)", fmt_inr(vp.vah)),
                ("VAL (Value Area Low)", fmt_inr(vp.val)),
            ]
            for k, v in vp_items:
                st.markdown(f"**{k}:** {v}")

            fig_vp = build_volume_profile_chart(snap.ohlcv, vp)
            if fig_vp:
                st.plotly_chart(fig_vp, use_container_width=True)
    else:
        st.info("No S/R levels detected (insufficient data).")

    st.divider()
    st.subheader("Trend Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**EMA Alignment**")
        st.markdown(f"```{trend.ema_alignment}```")
        st.markdown("**Primary Trend**")
        st.markdown(f"```{trend.primary_trend}```")
    with c2:
        st.markdown("**Regime**")
        st.markdown(f"```{trend.regime}```")
        st.markdown("**ADX Regime**")
        st.markdown(f"```{trend.adx_regime}```")
    with c3:
        st.markdown("**RSI Zone**")
        st.markdown(f"```{trend.rsi_zone}```")
        st.markdown("**Secondary Trend**")
        st.markdown(f"```{trend.secondary_trend}```")

    st.divider()
    st.subheader("Institutional Activity")
    acc_color = {"accumulation": "green", "distribution": "red"}.get(accum.signal, "gray")
    st.markdown(f"**Signal:** :{acc_color}[{accum.signal.upper()}]")
    st.markdown(f"**OBV Slope Positive:** {accum.obv_slope_positive}")
    if hasattr(accum, "up_vol_ratio") and accum.up_vol_ratio is not None:
        st.markdown(f"**Up Volume Ratio:** {accum.up_vol_ratio:.3f}")


def render_scenarios(r: dict):
    scenarios = r["scenarios"]
    snap = r["snap"]

    donut_col, detail_col = st.columns([1, 2])
    with donut_col:
        st.subheader("Probability Distribution")
        fig_d = build_scenario_donut(scenarios)
        if fig_d:
            st.plotly_chart(fig_d, use_container_width=True)

    with detail_col:
        if scenarios.bull:
            st.markdown(f"""
<div class="scenario-bull">
<h4>Bullish Scenario — {scenarios.bull.probability:.0%}</h4>
<p><b>Target:</b> {fmt_inr(scenarios.bull.target_price)}</p>
<p><b>Timeframe:</b> {scenarios.bull.timeframe}</p>
<p><b>Catalyst:</b> {scenarios.bull.catalyst}</p>
</div>
""", unsafe_allow_html=True)

        if scenarios.bear:
            st.markdown(f"""
<div class="scenario-bear" style="margin-top:12px">
<h4>Bearish Scenario — {scenarios.bear.probability:.0%}</h4>
<p><b>Target:</b> {fmt_inr(scenarios.bear.target_price)}</p>
<p><b>Timeframe:</b> {scenarios.bear.timeframe}</p>
<p><b>Catalyst:</b> {scenarios.bear.catalyst}</p>
</div>
""", unsafe_allow_html=True)

        if scenarios.neutral:
            st.markdown(f"""
<div class="metric-card yellow" style="margin-top:12px">
<h4>Neutral / Consolidation — {scenarios.neutral.probability:.0%}</h4>
<p><b>Target:</b> {fmt_inr(scenarios.neutral.target_price)}</p>
<p><b>Timeframe:</b> {scenarios.neutral.timeframe}</p>
<p><b>Catalyst:</b> {scenarios.neutral.catalyst}</p>
</div>
""", unsafe_allow_html=True)

        if scenarios.black_swan:
            st.markdown(f"""
<div class="scenario-swan" style="margin-top:12px">
<h4>Black Swan / Tail Risk — {scenarios.black_swan.probability:.0%}</h4>
<p><b>Target:</b> {fmt_inr(scenarios.black_swan.target_price)}</p>
<p><b>Timeframe:</b> {scenarios.black_swan.timeframe}</p>
<p><b>Catalyst:</b> {scenarios.black_swan.catalyst}</p>
</div>
""", unsafe_allow_html=True)


def render_macro_news(r: dict):
    macro = r["macro"]
    news = r["news"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Macro Environment")
        if macro.error:
            st.warning(f"Macro data unavailable: {macro.error}")
        macro_items = [
            ("Fed Funds Rate", f"{macro.fed_funds_rate:.2f}%" if macro.fed_funds_rate else "N/A"),
            ("10Y Treasury Yield", f"{macro.treasury_10y:.2f}%" if macro.treasury_10y else "N/A"),
            ("CPI (YoY)", f"{macro.cpi_yoy:.2f}%" if macro.cpi_yoy else "N/A"),
            ("VIX", fmt_num(macro.vix)),
        ]
        for k, v in macro_items:
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.subheader("News Sentiment")
        score = news.score if news else 0
        if score > 0.2:
            sentiment_label = ":green[Positive]"
        elif score < -0.2:
            sentiment_label = ":red[Negative]"
        else:
            sentiment_label = ":gray[Neutral]"

        st.markdown(f"**Sentiment:** {sentiment_label}")
        st.markdown(f"**Score:** {score:.3f}" if news else "N/A")
        if news and news.summary:
            st.markdown(f"**Summary:** {news.summary}")
        if news and news.headline_count:
            st.markdown(f"**Headlines Analysed:** {news.headline_count}")


def render_ai_report(r: dict):
    report_text = r.get("report_text")
    snap = r["snap"]

    st.subheader(f"AI Analysis — {snap.company_name or snap.ticker}")

    if not report_text:
        st.info(
            "**AI report not generated** — `ANTHROPIC_API_KEY` is not set.\n\n"
            "To enable this tab:\n"
            "1. Get a free API key at https://console.anthropic.com\n"
            "2. Add it to your `.env` file as `ANTHROPIC_API_KEY=sk-ant-...`\n"
            "3. On Streamlit Cloud: go to **App settings → Secrets** and add the key there.\n\n"
            "All other tabs (Overview, Technical, Options, Structure, Scenarios, Macro) "
            "work fully without an API key."
        )
        return

    st.caption(f"Generated {datetime.now().strftime('%d %b %Y %H:%M IST')} | Powered by Claude")
    st.markdown(f'<div class="report-text">{report_text}</div>', unsafe_allow_html=True)

    st.download_button(
        label="Download Report (Markdown)",
        data=report_text,
        file_name=f"{snap.ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}_analysis.md",
        mime="text/markdown",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("NSE Stock Analyser")
        st.caption("Institutional-grade analysis powered by Claude AI")
        st.divider()

        symbol_input = st.text_input(
            "NSE Symbol",
            placeholder="e.g. RELIANCE, INFY, NIFTY50",
            help="Enter the NSE ticker symbol. .NS suffix is added automatically.",
        ).strip().upper()

        model = st.selectbox(
            "Claude Model",
            options=[
                "claude-opus-4-8",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
            index=0,
            help="Choose the Claude model for AI analysis. Opus gives best quality.",
        )

        api_key_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if not api_key_ok:
            st.caption("No ANTHROPIC_API_KEY — AI Report tab will be skipped. All other tabs work.")

        st.divider()
        analyse_btn = st.button("Analyse", type="primary", use_container_width=True,
                                disabled=not symbol_input)

        if symbol_input:
            full_ticker = nse_ticker(symbol_input)
            st.caption(f"Will fetch: `{full_ticker}`")

        st.divider()
        st.markdown("""
**How to use**
1. Enter an NSE symbol (e.g. `RELIANCE`, `TCS`, `HDFCBANK`)
2. Choose Claude model
3. Click **Analyse** and wait ~30–60s
4. Explore each tab for full analysis

**Indices**
- NIFTY 50: `^NSEI`
- BANK NIFTY: `^NSEBANK`
        """)

    return symbol_input, model, analyse_btn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    symbol_input, model, analyse_btn = sidebar()

    if "results" not in st.session_state:
        st.session_state.results = None
    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = None
    if "error" not in st.session_state:
        st.session_state.error = None

    if analyse_btn and symbol_input:
        ticker = nse_ticker(symbol_input)
        st.session_state.error = None
        st.session_state.results = None

        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        steps = [
            "Fetching market data...",
            "Fetching options chain...",
            "Fetching macro & news data...",
            "Computing technical indicators...",
            "Analysing market structure...",
            "Computing options analytics...",
            "Building scenario model...",
            "Generating AI analysis via Claude...",
        ]
        total = len(steps)
        step_counter = [0]

        def status_cb(msg: str):
            step_counter[0] += 1
            pct = min(step_counter[0] / total, 1.0)
            status_placeholder.info(f"**{msg}**")
            progress_placeholder.progress(pct)

        try:
            results = run_full_analysis(ticker, model, status_cb)
            st.session_state.results = results
            st.session_state.last_ticker = ticker
        except Exception as e:
            st.session_state.error = str(e)
        finally:
            status_placeholder.empty()
            progress_placeholder.empty()

    # Show results or landing
    if st.session_state.error:
        st.error(f"Analysis failed: {st.session_state.error}")

    if st.session_state.results:
        r = st.session_state.results
        snap = r["snap"]

        ticker_display = snap.ticker or st.session_state.last_ticker or ""
        company_display = snap.company_name or ""
        price_display = snap.current_price or 0
        chg = snap.day_change_pct

        chg_str = f" ({fmt_pct(chg)})" if chg is not None else ""
        st.title(f"{company_display}  —  {fmt_inr(price_display)}{chg_str}")
        st.caption(f"`{ticker_display}` · NSE · Analysis generated {datetime.now().strftime('%d %b %Y %H:%M')}")

        tab_overview, tab_tech, tab_opts, tab_struct, tab_scen, tab_macro, tab_ai = st.tabs([
            "Overview",
            "Technical",
            "Options",
            "Market Structure",
            "Scenarios",
            "Macro & News",
            "AI Report",
        ])

        with tab_overview:
            render_overview(r)
        with tab_tech:
            render_technical(r)
        with tab_opts:
            render_options(r)
        with tab_struct:
            render_structure(r)
        with tab_scen:
            render_scenarios(r)
        with tab_macro:
            render_macro_news(r)
        with tab_ai:
            render_ai_report(r)

    else:
        # Landing page
        st.title("NSE Stock Analyser")
        st.markdown("""
### Institutional-Grade Analysis for NSE Stocks

Enter an NSE symbol in the sidebar and click **Analyse** to get:

| Tab | What you get |
|-----|-------------|
| **Overview** | Price chart with S/R overlays, key metrics |
| **Technical** | RSI, MACD, EMAs, Bollinger Bands, ATR, OBV, ADX |
| **Options** | OI distribution, IV skew, Max Pain, GEX, P/C ratios |
| **Market Structure** | S/R levels, volume profile, trend classification, institutional activity |
| **Scenarios** | Bull / Bear / Black Swan probabilities with targets |
| **Macro & News** | Fed rates, VIX, CPI, news sentiment |
| **AI Report** | Claude-generated institutional analysis report |

---
**Popular symbols:** `RELIANCE` · `TCS` · `INFY` · `HDFCBANK` · `ICICIBANK` · `BAJFINANCE` · `WIPRO` · `HINDUNILVR`

> Options data is not available for all NSE symbols. The rest of the analysis will work regardless.
        """)


if __name__ == "__main__":
    main()
