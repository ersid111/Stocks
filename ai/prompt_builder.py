import json
from typing import Any, Optional
from data.market_data import MarketSnapshot
from data.options_data import OptionsChain
from data.macro_data import MacroContext, NewsSentiment
from analysis.technical import TechnicalIndicators
from analysis.market_structure import TrendAnalysis, SRLevel, VolumeProfile, AccumulationSignal
from analysis.options_math import MaxPainResult, GEXResult, IVSkew, ExpectedMove, PCMetrics
from analysis.scenarios import ScenarioSet


def _fmt(value: Any, precision: int = 2) -> Any:
    if value is None:
        return "data unavailable"
    if isinstance(value, float):
        return round(value, precision)
    return value


def build_analysis_payload(
    snap: MarketSnapshot,
    ind: TechnicalIndicators,
    trend: TrendAnalysis,
    sr_levels: list,
    vp: VolumeProfile,
    accum: AccumulationSignal,
    chain: OptionsChain,
    max_pain: MaxPainResult,
    gex: GEXResult,
    iv_skew: IVSkew,
    expected_move: ExpectedMove,
    pc: PCMetrics,
    scenarios: ScenarioSet,
    macro: MacroContext,
    news: NewsSentiment,
) -> str:
    nearest_dte = None
    if chain.expiries:
        from datetime import datetime
        try:
            nearest_dte = (datetime.strptime(chain.expiries[0], "%Y-%m-%d") - datetime.now()).days
        except Exception:
            nearest_dte = None

    payload = {
        "ticker": snap.ticker,
        "company": _fmt(snap.company_name),
        "sector": _fmt(snap.sector),
        "industry": _fmt(snap.industry),
        "market_cap_usd": _fmt(snap.market_cap),
        "price": {
            "current": _fmt(snap.current_price),
            "prev_close": _fmt(snap.prev_close),
            "day_change_pct": _fmt(snap.day_change_pct),
            "52w_high": _fmt(snap.fifty_two_week_high),
            "52w_low": _fmt(snap.fifty_two_week_low),
        },
        "fundamentals": {
            "pe_ratio": _fmt(snap.pe_ratio),
            "forward_pe": _fmt(snap.forward_pe),
            "beta": _fmt(snap.beta),
            "short_float_pct": _fmt(snap.short_float, 4) if snap.short_float else "data unavailable",
        },
        "volume": {
            "today": _fmt(snap.volume_today),
            "avg_10d": _fmt(snap.avg_volume_10d),
            "ratio_vs_avg": _fmt(
                round(snap.volume_today / snap.avg_volume_10d, 2)
                if snap.volume_today and snap.avg_volume_10d and snap.avg_volume_10d > 0
                else None
            ),
        },
        "technical_indicators": {
            "rsi_14": _fmt(ind.rsi),
            "macd": _fmt(ind.macd, 4),
            "macd_signal": _fmt(ind.macd_signal, 4),
            "macd_histogram": _fmt(ind.macd_histogram, 4),
            "ema_20": _fmt(ind.ema20),
            "ema_50": _fmt(ind.ema50),
            "ema_200": _fmt(ind.ema200),
            "bb_upper": _fmt(ind.bb_upper),
            "bb_middle": _fmt(ind.bb_middle),
            "bb_lower": _fmt(ind.bb_lower),
            "bb_pct_b": _fmt(ind.bb_pct, 3),
            "atr_14": _fmt(ind.atr),
            "vwap": _fmt(ind.vwap),
            "obv": _fmt(ind.obv, 0),
            "obv_slope_20d": _fmt(ind.obv_slope, 2),
            "adx_14": _fmt(ind.adx),
        },
        "market_structure": {
            "primary_trend": trend.primary_trend,
            "secondary_trend": trend.secondary_trend,
            "short_term_trend": trend.short_term_trend,
            "ema_alignment": trend.ema_alignment,
            "regime": trend.regime,
            "rsi_zone": trend.rsi_zone,
            "adx_regime": trend.adx_regime,
            "support_levels": [
                {"price": l.price, "touches": l.touches}
                for l in sr_levels if l.level_type == "support"
            ][:5],
            "resistance_levels": [
                {"price": l.price, "touches": l.touches}
                for l in sr_levels if l.level_type == "resistance"
            ][:5],
            "volume_profile": {
                "poc": _fmt(vp.poc),
                "vah": _fmt(vp.vah),
                "val": _fmt(vp.val),
            },
            "accumulation_signal": accum.signal,
            "up_volume_ratio": _fmt(accum.up_vol_ratio, 3),
        },
        "options_analytics": {
            "expiries_analyzed": chain.expiries,
            "nearest_dte": _fmt(nearest_dte),
            "max_pain": {
                "level": _fmt(max_pain.max_pain),
                "distance_from_spot_pct": _fmt(
                    round((max_pain.max_pain - snap.current_price) / snap.current_price * 100, 2)
                    if max_pain.max_pain and snap.current_price else None
                ),
            },
            "gamma_exposure": {
                "total_gex": _fmt(gex.total_gex, 0),
                "positive_gex": _fmt(gex.positive_gex, 0),
                "negative_gex": _fmt(gex.negative_gex, 0),
                "gamma_flip_level": _fmt(gex.gamma_flip),
                "regime": gex.regime,
            },
            "iv_skew": {
                "atm_iv_pct": _fmt(round(iv_skew.atm_iv * 100, 1) if iv_skew.atm_iv else None),
                "otm_put_iv_pct": _fmt(round(iv_skew.otm_put_iv * 100, 1) if iv_skew.otm_put_iv else None),
                "otm_call_iv_pct": _fmt(round(iv_skew.otm_call_iv * 100, 1) if iv_skew.otm_call_iv else None),
                "put_skew": _fmt(round(iv_skew.put_skew * 100, 1) if iv_skew.put_skew else None),
                "risk_reversal": _fmt(round(iv_skew.risk_reversal * 100, 1) if iv_skew.risk_reversal else None),
                "term_structure": {k: round(v * 100, 1) for k, v in iv_skew.term_structure.items()},
            },
            "expected_move": {
                "straddle_derived": _fmt(expected_move.straddle_move),
                "formula_derived": _fmt(expected_move.formula_move),
                "average": _fmt(expected_move.average_move),
                "upper_target": _fmt(expected_move.upper),
                "lower_target": _fmt(expected_move.lower),
            },
            "put_call_ratios": {
                "oi_ratio": _fmt(pc.oi_put_call_ratio, 3),
                "volume_ratio": _fmt(pc.vol_put_call_ratio, 3),
                "dollar_ratio": _fmt(pc.dollar_put_call_ratio, 3),
                "total_call_oi": pc.total_call_oi,
                "total_put_oi": pc.total_put_oi,
            },
        },
        "scenarios": {
            "bull": {
                "probability": _fmt(scenarios.bull.probability, 3) if scenarios.bull else "data unavailable",
                "target_price": _fmt(scenarios.bull.target_price) if scenarios.bull else "data unavailable",
                "timeframe": scenarios.bull.timeframe if scenarios.bull else "data unavailable",
                "catalyst": scenarios.bull.catalyst if scenarios.bull else "data unavailable",
            },
            "bear": {
                "probability": _fmt(scenarios.bear.probability, 3) if scenarios.bear else "data unavailable",
                "target_price": _fmt(scenarios.bear.target_price) if scenarios.bear else "data unavailable",
                "timeframe": scenarios.bear.timeframe if scenarios.bear else "data unavailable",
                "catalyst": scenarios.bear.catalyst if scenarios.bear else "data unavailable",
            },
            "neutral": {
                "probability": _fmt(scenarios.neutral.probability, 3) if scenarios.neutral else "data unavailable",
                "target_price": _fmt(scenarios.neutral.target_price) if scenarios.neutral else "data unavailable",
                "timeframe": scenarios.neutral.timeframe if scenarios.neutral else "data unavailable",
            },
            "black_swan": {
                "probability": 0.05,
                "target_price": _fmt(scenarios.black_swan.target_price) if scenarios.black_swan else "data unavailable",
                "timeframe": scenarios.black_swan.timeframe if scenarios.black_swan else "data unavailable",
                "catalyst": scenarios.black_swan.catalyst if scenarios.black_swan else "data unavailable",
            },
            "historical_vol_1y_pct": _fmt(
                round(scenarios.historical_vol_1y * 100, 1) if scenarios.historical_vol_1y else None
            ),
        },
        "macro": {
            "fed_funds_rate_pct": _fmt(macro.fed_funds_rate),
            "inflation_expectations_10y_pct": _fmt(macro.inflation_expectations_10y),
            "yield_curve_10y2y_spread": _fmt(macro.yield_curve_10y2y),
            "vix": _fmt(macro.vix),
            "macro_data_status": macro.error if macro.error else "available",
        },
        "news_sentiment": {
            "summary": news.summary,
            "sentiment_score": _fmt(news.sentiment_score, 3),
            "recent_headlines": news.headlines[:5],
        },
    }

    return json.dumps(payload, indent=2, default=str)
