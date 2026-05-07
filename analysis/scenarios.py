from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from analysis.technical import TechnicalIndicators
from analysis.market_structure import TrendAnalysis, SRLevel, VolumeProfile
from analysis.options_math import PCMetrics


@dataclass
class Scenario:
    name: str
    probability: float
    target_price: Optional[float]
    timeframe: str
    catalyst: str


@dataclass
class ScenarioSet:
    bull: Scenario = None
    bear: Scenario = None
    neutral: Scenario = None
    black_swan: Scenario = None
    historical_vol_1y: Optional[float] = None


def _score_direction(ind: TechnicalIndicators, trend: TrendAnalysis, pc: PCMetrics) -> float:
    """Returns a score from -1 (max bearish) to +1 (max bullish)."""
    score = 0.0
    weight_total = 0.0

    # EMA alignment (weight 2)
    w = 2.0
    if trend.ema_alignment == "fully_bullish":
        score += w
    elif trend.ema_alignment == "fully_bearish":
        score -= w
    elif trend.ema_alignment == "bullish":
        score += w * 0.5
    elif trend.ema_alignment == "bearish":
        score -= w * 0.5
    weight_total += w

    # RSI zone (weight 1.5)
    w = 1.5
    rsi_scores = {"overbought": 0.5, "bullish": 1.0, "neutral": 0.0, "bearish": -1.0, "oversold": -0.5}
    score += w * rsi_scores.get(trend.rsi_zone, 0.0)
    weight_total += w

    # MACD histogram direction (weight 1.5)
    w = 1.5
    if ind.macd_histogram is not None:
        score += w * (1.0 if ind.macd_histogram > 0 else -1.0)
    weight_total += w

    # ADX / trend regime: only scale magnitude, not direction
    # (high ADX amplifies the existing directional signals)
    adx_multiplier = 1.0
    if ind.adx and ind.adx > 25:
        adx_multiplier = 1.2

    # OBV slope (weight 1)
    w = 1.0
    if ind.obv_slope is not None:
        score += w * (1.0 if ind.obv_slope > 0 else -1.0)
    weight_total += w

    # Put/call ratio (weight 1) — high P/C = bearish sentiment
    w = 1.0
    if pc.oi_put_call_ratio is not None:
        if pc.oi_put_call_ratio > 1.2:
            score -= w
        elif pc.oi_put_call_ratio < 0.7:
            score += w
    weight_total += w

    if weight_total == 0:
        return 0.0
    return float(np.clip(score * adx_multiplier / weight_total, -1.0, 1.0))


def build_scenarios(
    ind: TechnicalIndicators,
    trend: TrendAnalysis,
    pc: PCMetrics,
    sr_levels: List[SRLevel],
    vp: VolumeProfile,
    current_price: float,
    ohlcv=None,
) -> ScenarioSet:

    ss = ScenarioSet()

    # Historical volatility from daily returns (1 year)
    hist_vol = None
    if ohlcv is not None and not ohlcv.empty:
        close = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv["close"]
        returns = close.pct_change().dropna()
        if len(returns) > 20:
            hist_vol = float(returns.std() * np.sqrt(252))
            ss.historical_vol_1y = hist_vol

    direction_score = _score_direction(ind, trend, pc)

    # Probability model: score maps to bull/bear allocation
    # Bull base: 50% + 40% * direction_score
    bull_raw = 0.50 + 0.40 * direction_score
    bear_raw = 0.50 - 0.40 * direction_score
    neutral_raw = 0.25
    black_swan_raw = 0.05

    # Normalize to sum to 1 after reserving black swan
    remaining = 1.0 - black_swan_raw
    total_raw = bull_raw + bear_raw + neutral_raw
    bull_prob = (bull_raw / total_raw) * remaining
    bear_prob = (bear_raw / total_raw) * remaining
    neutral_prob = (neutral_raw / total_raw) * remaining

    # Targets
    resistances = sorted([l.price for l in sr_levels if l.level_type == "resistance" and l.price > current_price])
    supports = sorted([l.price for l in sr_levels if l.level_type == "support" and l.price < current_price], reverse=True)

    bull_target = resistances[0] if resistances else round(current_price * 1.10, 2)
    bear_target = supports[0] if supports else round(current_price * 0.90, 2)
    neutral_high = vp.vah if vp.vah and vp.vah > current_price else round(current_price * 1.03, 2)
    neutral_low = vp.val if vp.val and vp.val < current_price else round(current_price * 0.97, 2)

    bs_target = None
    if hist_vol:
        monthly_vol = hist_vol / np.sqrt(12)
        bs_target = round(current_price * (1 - 3 * monthly_vol), 2)
    else:
        bs_target = round(current_price * 0.75, 2)

    ss.bull = Scenario(
        name="Bull Case",
        probability=round(bull_prob, 3),
        target_price=bull_target,
        timeframe="4-8 weeks",
        catalyst="Momentum continuation, breakout above resistance, positive earnings/macro catalysts",
    )
    ss.bear = Scenario(
        name="Bear Case",
        probability=round(bear_prob, 3),
        target_price=bear_target,
        timeframe="4-8 weeks",
        catalyst="Breakdown below support, macro deterioration, negative sector rotation",
    )
    ss.neutral = Scenario(
        name="Neutral / Consolidation",
        probability=round(neutral_prob, 3),
        target_price=(neutral_high + neutral_low) / 2,
        timeframe="2-4 weeks",
        catalyst="Sideways chop within value area, low-conviction market, awaiting catalyst",
    )
    ss.black_swan = Scenario(
        name="Black Swan / Tail Risk",
        probability=0.05,
        target_price=bs_target,
        timeframe="1-4 weeks",
        catalyst="Macro shock, credit event, geopolitical escalation, liquidity crisis",
    )

    return ss
