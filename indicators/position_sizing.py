"""
Position Sizing Indicators
==========================
Formulas from core reference:

  f* = (p · b - q) / b               Kelly Criterion
  f  = α · f*,  α ∈ (0, 1]           Fractional Kelly
  VaR = μ - 1.645 · σ                Value at Risk 95%
  MDD = (Peak - Trough) / Peak        Max Drawdown
"""

import numpy as np
import pandas as pd
from typing import Optional


def kelly_criterion(p: float, b: float) -> float:
    """
    f* = (p · b - q) / b    where q = 1 - p
    Returns optimal fraction of bankroll to wager.
    Clamped to [0, 1]; negative → no bet.
    """
    q = 1 - p
    if b <= 0:
        return 0.0
    f_star = (p * b - q) / b
    return max(0.0, min(1.0, f_star))


def fractional_kelly(p: float, b: float, alpha: float = 0.25) -> float:
    """
    f = α · f*
    alpha : Kelly fraction, default 0.25 (conservative quarter-Kelly)
    """
    return alpha * kelly_criterion(p, b)


def position_size(p: float, b: float, bankroll: float, alpha: float = 0.25) -> float:
    """
    Dollar / yen amount to allocate based on fractional Kelly.
    """
    f = fractional_kelly(p, b, alpha)
    return f * bankroll


def value_at_risk_95(returns: pd.Series) -> float:
    """
    VaR = μ - 1.645 · σ   (parametric, 95% confidence, 1-day)
    Returns the VaR as a negative return (expected maximum daily loss).
    """
    mu    = float(returns.mean())
    sigma = float(returns.std())
    return mu - 1.645 * sigma


def max_drawdown(close: pd.Series) -> float:
    """
    MDD = (Peak - Trough) / Peak
    Returns the maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
    """
    if close.empty:
        return 0.0
    peak = close.cummax()
    drawdown = (close - peak) / peak
    return float(abs(drawdown.min()))


def current_drawdown(close: pd.Series) -> float:
    """Current drawdown from all-time high in the series."""
    if close.empty:
        return 0.0
    peak = float(close.max())
    current = float(close.iloc[-1])
    if peak == 0:
        return 0.0
    return (peak - current) / peak


def sizing_metrics(close: pd.Series, bankroll: float = 1_000_000,
                   alpha: float = 0.25) -> dict:
    """
    Compute all position-sizing metrics from a price series.
    """
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return {}

    # Derive p and b from historical returns
    win_mask   = returns > 0
    p          = float(win_mask.mean())
    gains      = returns[win_mask]
    losses     = returns[~win_mask]
    avg_gain   = float(gains.mean())   if len(gains)  > 0 else 0.0
    avg_loss   = float(abs(losses.mean())) if len(losses) > 0 else 1e-9
    b          = avg_gain / avg_loss

    f_star     = kelly_criterion(p, b)
    f_frac     = fractional_kelly(p, b, alpha)
    size       = f_frac * bankroll
    var_95     = value_at_risk_95(returns)
    mdd        = max_drawdown(close)
    curr_dd    = current_drawdown(close)

    return {
        "kelly_full":      round(f_star,  4),
        "kelly_frac":      round(f_frac,  4),
        "position_size":   round(size,    0),
        "var_95_daily":    round(var_95,  6),
        "max_drawdown":    round(mdd,     4),
        "current_drawdown": round(curr_dd, 4),
        "mdd_breach":      mdd > 0.08,   # True → block new trades
    }
