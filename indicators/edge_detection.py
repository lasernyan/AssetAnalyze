"""
Edge Detection Indicators
=========================
Formulas from core reference:

  EV   = p · b - (1 - p)            Expected Value
  edge = p_model - p_mkt             Market Edge
  P(H|E) = P(E|H)·P(H) / P(E)       Bayes Update
  BS   = (1/n) · Σ(p_i - o_i)²      Brier Score
  δ    = (p_model - p_mkt) / σ       Mispricing Score (Z-score)
"""

import numpy as np
import pandas as pd
from typing import Optional


def expected_value(p: float, b: float) -> float:
    """
    EV = p · b - (1 - p)
    p : model probability of win
    b : decimal odds - 1  (net profit per unit staked)
    """
    return p * b - (1 - p)


def market_edge(p_model: float, p_mkt: float) -> float:
    """
    edge = p_model - p_mkt
    Positive → market underestimates the move; consider LONG.
    Negative → market overestimates; consider SHORT.
    """
    return p_model - p_mkt


def bayes_update(prior: float, likelihood: float, evidence: float) -> float:
    """
    P(H|E) = P(E|H) · P(H) / P(E)
    prior      : P(H)   – prior probability of hypothesis
    likelihood : P(E|H) – probability of evidence given hypothesis
    evidence   : P(E)   – total probability of evidence
    """
    if evidence == 0:
        return prior
    return (likelihood * prior) / evidence


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """
    BS = (1/n) · Σ(p_i - o_i)²
    Lower is better; perfect calibration → 0.
    probabilities : array of predicted probabilities
    outcomes      : array of binary outcomes (0 or 1)
    """
    n = len(probabilities)
    if n == 0:
        return np.nan
    return float(np.mean((probabilities - outcomes) ** 2))


def mispricing_score(p_model: float, p_mkt: float, sigma: float) -> float:
    """
    δ = (p_model - p_mkt) / σ
    Z-score of model vs market divergence.
    |δ| > 1.5 → significant mispricing signal.
    """
    if sigma == 0:
        return 0.0
    return (p_model - p_mkt) / sigma


# ── Price-based helpers ────────────────────────────────────────────────────────

def price_edge_from_series(close: pd.Series, lookback: int = 252) -> dict:
    """
    Derive edge metrics directly from a price series.

    p_model: probability that next return > 0, estimated from historical win rate
    p_mkt:   implied probability from recent momentum (last-N-day positive fraction)
    b:       average gain/loss ratio
    """
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return {}

    hist = returns.iloc[-lookback:]
    recent = returns.iloc[-20:]  # last ~1 month

    win_mask = hist > 0
    p_model = float(win_mask.mean())
    p_mkt   = float((recent > 0).mean())

    gains  = hist[win_mask]
    losses = hist[~win_mask]
    avg_gain = gains.mean() if len(gains) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9

    b = avg_gain / avg_loss  # net profit ratio (used as decimal odds - 1)

    ev    = expected_value(p_model, b)
    edge  = market_edge(p_model, p_mkt)
    sigma = float(hist.std())
    delta = mispricing_score(p_model, p_mkt, sigma) if sigma > 0 else 0.0

    return {
        "p_model": round(p_model, 4),
        "p_mkt":   round(p_mkt,   4),
        "b":       round(b,        4),
        "ev":      round(ev,       4),
        "edge":    round(edge,     4),
        "sigma":   round(sigma,    6),
        "mispricing_score": round(delta, 4),
    }
