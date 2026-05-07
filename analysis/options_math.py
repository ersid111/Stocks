from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BSResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    prob_itm: float


@dataclass
class MaxPainResult:
    max_pain: Optional[float] = None
    current_spot: Optional[float] = None
    distance_pct: Optional[float] = None


@dataclass
class GEXResult:
    total_gex: float = 0.0
    gamma_flip: Optional[float] = None
    positive_gex: float = 0.0
    negative_gex: float = 0.0
    regime: str = "unknown"


@dataclass
class IVSkew:
    atm_iv: Optional[float] = None
    otm_put_iv: Optional[float] = None
    otm_call_iv: Optional[float] = None
    put_skew: Optional[float] = None
    risk_reversal: Optional[float] = None
    term_structure: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExpectedMove:
    straddle_move: Optional[float] = None
    formula_move: Optional[float] = None
    average_move: Optional[float] = None
    upper: Optional[float] = None
    lower: Optional[float] = None


@dataclass
class PCMetrics:
    oi_put_call_ratio: Optional[float] = None
    vol_put_call_ratio: Optional[float] = None
    dollar_put_call_ratio: Optional[float] = None
    total_call_oi: int = 0
    total_put_oi: int = 0


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> BSResult:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return BSResult(0, 0, 0, 0, 0, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        prob_itm = norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        prob_itm = norm.cdf(-d2)

    return BSResult(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, prob_itm=prob_itm)


def calculate_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> MaxPainResult:
    result = MaxPainResult()
    if calls.empty or puts.empty:
        return result

    try:
        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        if not all_strikes:
            return result

        pain = {}
        for K in all_strikes:
            call_rows = calls[calls["strike"] <= K]
            put_rows = puts[puts["strike"] >= K]
            call_pain = float((call_rows["openInterest"] * (K - call_rows["strike"]).clip(lower=0)).sum())
            put_pain = float((put_rows["openInterest"] * (put_rows["strike"] - K).clip(lower=0)).sum())
            pain[K] = call_pain + put_pain

        result.max_pain = min(pain, key=pain.get)
    except Exception:
        pass

    return result


def calculate_gamma_exposure(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, r: float = 0.05) -> GEXResult:
    result = GEXResult()
    if calls.empty and puts.empty:
        return result

    strike_gex: Dict[float, float] = {}

    def process_chain(df: pd.DataFrame, sign: float):
        for _, row in df.iterrows():
            try:
                K = float(row["strike"])
                oi = int(row["openInterest"])
                iv = float(row["impliedVolatility"])
                dte = int(row["dte"])
                T = max(dte / 365, 1 / 365)
                if iv <= 0 or oi <= 0:
                    continue
                bs = black_scholes(spot, K, T, r, iv)
                gex = bs.gamma * oi * 100 * (spot ** 2) * 0.01 * sign
                strike_gex[K] = strike_gex.get(K, 0.0) + gex
            except Exception:
                continue

    process_chain(calls, 1.0)
    process_chain(puts, -1.0)

    if not strike_gex:
        return result

    result.total_gex = sum(strike_gex.values())
    result.positive_gex = sum(v for v in strike_gex.values() if v > 0)
    result.negative_gex = sum(v for v in strike_gex.values() if v < 0)
    result.regime = "positive" if result.total_gex >= 0 else "negative"

    strikes_sorted = sorted(strike_gex.keys())
    cumulative = 0.0
    for k in strikes_sorted:
        prev = cumulative
        cumulative += strike_gex[k]
        if prev < 0 < cumulative or prev > 0 > cumulative:
            result.gamma_flip = k
            break

    return result


def calculate_iv_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> IVSkew:
    skew = IVSkew()
    if calls.empty and puts.empty:
        return skew

    # ATM IV from calls near spot
    atm_window = spot * 0.02
    atm_calls = calls[(calls["strike"] - spot).abs() <= atm_window].sort_values("impliedVolatility")
    if not atm_calls.empty:
        iv = atm_calls["impliedVolatility"].median()
        skew.atm_iv = float(iv) if not pd.isna(iv) else None

    # OTM put IV (~5-10% below spot)
    otm_put_low = spot * 0.90
    otm_put_high = spot * 0.97
    otm_puts = puts[(puts["strike"] >= otm_put_low) & (puts["strike"] <= otm_put_high)]
    if not otm_puts.empty:
        iv = otm_puts["impliedVolatility"].median()
        skew.otm_put_iv = float(iv) if not pd.isna(iv) else None

    # OTM call IV (~3-10% above spot)
    otm_call_low = spot * 1.03
    otm_call_high = spot * 1.10
    otm_calls = calls[(calls["strike"] >= otm_call_low) & (calls["strike"] <= otm_call_high)]
    if not otm_calls.empty:
        iv = otm_calls["impliedVolatility"].median()
        skew.otm_call_iv = float(iv) if not pd.isna(iv) else None

    if skew.atm_iv and skew.otm_put_iv:
        skew.put_skew = skew.otm_put_iv - skew.atm_iv
    if skew.otm_call_iv and skew.otm_put_iv:
        skew.risk_reversal = skew.otm_call_iv - skew.otm_put_iv

    # Term structure: ATM IV per expiry
    for expiry, grp in calls.groupby("expiry"):
        atm_grp = grp[(grp["strike"] - spot).abs() <= atm_window]
        if not atm_grp.empty:
            iv = atm_grp["impliedVolatility"].median()
            if not pd.isna(iv):
                skew.term_structure[str(expiry)] = round(float(iv), 4)

    return skew


def calculate_expected_move(spot: float, atm_iv: Optional[float], dte: int,
                             calls: pd.DataFrame, puts: pd.DataFrame) -> ExpectedMove:
    em = ExpectedMove()
    if atm_iv and atm_iv > 0 and dte > 0:
        T = dte / 365
        em.formula_move = spot * atm_iv * np.sqrt(T)

    # Straddle price from nearest ATM options
    if not calls.empty and not puts.empty and dte > 0:
        try:
            atm_calls = calls[calls["dte"] == dte].copy() if not calls.empty else calls
            atm_puts = puts[puts["dte"] == dte].copy() if not puts.empty else puts
            if atm_calls.empty:
                atm_calls = calls
            if atm_puts.empty:
                atm_puts = puts

            best_call = atm_calls.iloc[(atm_calls["strike"] - spot).abs().argsort().iloc[0:1]]
            best_put = atm_puts.iloc[(atm_puts["strike"] - spot).abs().argsort().iloc[0:1]]
            if not best_call.empty and not best_put.empty:
                call_mid = float(best_call["mid"].iloc[0])
                put_mid = float(best_put["mid"].iloc[0])
                straddle = call_mid + put_mid
                em.straddle_move = straddle
        except Exception:
            pass

    if em.formula_move and em.straddle_move:
        em.average_move = (em.formula_move + em.straddle_move) / 2
    elif em.formula_move:
        em.average_move = em.formula_move
    elif em.straddle_move:
        em.average_move = em.straddle_move

    if em.average_move:
        em.upper = round(spot + em.average_move, 2)
        em.lower = round(spot - em.average_move, 2)

    return em


def compute_put_call_metrics(calls: pd.DataFrame, puts: pd.DataFrame) -> PCMetrics:
    pc = PCMetrics()
    if calls.empty and puts.empty:
        return pc

    pc.total_call_oi = int(calls["openInterest"].sum()) if not calls.empty else 0
    pc.total_put_oi = int(puts["openInterest"].sum()) if not puts.empty else 0

    if pc.total_call_oi > 0:
        pc.oi_put_call_ratio = round(pc.total_put_oi / pc.total_call_oi, 3)

    call_vol = int(calls["volume"].sum()) if not calls.empty else 0
    put_vol = int(puts["volume"].sum()) if not puts.empty else 0
    if call_vol > 0:
        pc.vol_put_call_ratio = round(put_vol / call_vol, 3)

    try:
        call_dollar = float((calls["mid"] * calls["openInterest"] * 100).sum()) if not calls.empty else 0
        put_dollar = float((puts["mid"] * puts["openInterest"] * 100).sum()) if not puts.empty else 0
        if call_dollar > 0:
            pc.dollar_put_call_ratio = round(put_dollar / call_dollar, 3)
    except Exception:
        pass

    return pc
