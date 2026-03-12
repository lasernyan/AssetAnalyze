"""
Performance & Technical Indicators
===================================
Formulas from core reference:

  SR  = (E[R] - Rf) / σ(R)           Sharpe Ratio
  PF  = gross_profit / gross_loss     Profit Factor

Plus standard technical indicators:
  - SMA / EMA (Golden Cross / Death Cross)
  - RSI (Relative Strength Index)
  - ATR (Average True Range)
  - Bollinger Bands
  - MACD
"""

import numpy as np
import pandas as pd
from config import (
    RISK_FREE_RATE, SHORT_MA, LONG_MA,
    RSI_PERIOD, ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    RSI_OVERBOUGHT, RSI_OVERSOLD,
)


# ── Core performance metrics ───────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    SR = (E[R] - Rf) / σ(R)
    Annualised. Target SR > 2.0 for a healthy strategy.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / 252  # daily risk-free
    return float(np.sqrt(252) * excess.mean() / excess.std())


def profit_factor(returns: pd.Series) -> float:
    """
    PF = gross_profit / gross_loss
    Healthy bot maintains PF > 1.5.
    """
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


# ── Technical indicators ───────────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI via Wilder's smoothing."""
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def bollinger_bands(close: pd.Series,
                    period: int = BOLLINGER_PERIOD,
                    num_std: float = BOLLINGER_STD) -> pd.DataFrame:
    """Returns DataFrame with columns: middle, upper, lower, %B."""
    mid   = sma(close, period)
    std   = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"middle": mid, "upper": upper,
                          "lower": lower, "pct_b": pct_b})


def macd(close: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line   = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line,
                          "signal": signal_line,
                          "histogram": histogram})


# ── Composite signal generator ────────────────────────────────────────────────

def technical_metrics(df: pd.DataFrame) -> dict:
    """
    Given an OHLCV DataFrame, compute all technical & performance metrics
    and return the latest values plus a composite signal.

    Returns dict with keys:
      price, change_1d, change_5d, change_20d,
      ma50, ma200, golden_cross,
      rsi, atr,
      bb_upper, bb_lower, bb_pct_b,
      macd_val, macd_hist,
      sharpe, profit_factor,
      signal (BUY / SELL / HOLD), signal_score (-100 .. +100)
    """
    if df is None or len(df) < LONG_MA:
        return {}

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    ret   = close.pct_change().dropna()

    # ── Price & returns ──────────────────────────────────────────────────────
    price     = float(close.iloc[-1])
    ch_1d     = float(ret.iloc[-1])    if len(ret) >= 1  else 0.0
    ch_5d     = float((close.iloc[-1] / close.iloc[-6]  - 1)) if len(close) >= 6  else 0.0
    ch_20d    = float((close.iloc[-1] / close.iloc[-21] - 1)) if len(close) >= 21 else 0.0

    # ── Moving averages ──────────────────────────────────────────────────────
    ma50_s  = sma(close, SHORT_MA)
    ma200_s = sma(close, LONG_MA)
    ma50    = float(ma50_s.iloc[-1])  if not ma50_s.isna().iloc[-1]  else None
    ma200   = float(ma200_s.iloc[-1]) if not ma200_s.isna().iloc[-1] else None

    # Golden cross: MA50 recently crossed above MA200
    golden_cross = None
    if ma50 is not None and ma200 is not None:
        prev50  = float(ma50_s.iloc[-2])  if len(ma50_s) >= 2  else ma50
        prev200 = float(ma200_s.iloc[-2]) if len(ma200_s) >= 2 else ma200
        if prev50 < prev200 and ma50 >= ma200:
            golden_cross = "GOLDEN"
        elif prev50 > prev200 and ma50 <= ma200:
            golden_cross = "DEATH"
        else:
            golden_cross = "NONE"

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_s   = rsi(close)
    rsi_val = float(rsi_s.iloc[-1]) if not rsi_s.isna().iloc[-1] else None

    # ── ATR ──────────────────────────────────────────────────────────────────
    atr_s   = atr(high, low, close)
    atr_val = float(atr_s.iloc[-1]) if not atr_s.isna().iloc[-1] else None
    atr_pct = (atr_val / price * 100) if (atr_val and price) else None

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bb      = bollinger_bands(close)
    bb_up   = float(bb["upper"].iloc[-1])
    bb_lo   = float(bb["lower"].iloc[-1])
    bb_pb   = float(bb["pct_b"].iloc[-1]) if not bb["pct_b"].isna().iloc[-1] else None

    # ── MACD ─────────────────────────────────────────────────────────────────
    mc      = macd(close)
    macd_v  = float(mc["macd"].iloc[-1])     if not mc["macd"].isna().iloc[-1]      else None
    macd_h  = float(mc["histogram"].iloc[-1]) if not mc["histogram"].isna().iloc[-1] else None

    # ── Performance metrics ──────────────────────────────────────────────────
    sr  = sharpe_ratio(ret)
    pf  = profit_factor(ret)

    # ── Composite signal score (+100 = strong BUY, -100 = strong SELL) ──────
    score = 0

    # Trend (MA) signals
    if ma50 and ma200:
        if ma50 > ma200:          score += 20   # uptrend
        else:                     score -= 20   # downtrend
    if golden_cross == "GOLDEN":  score += 15
    elif golden_cross == "DEATH": score -= 15

    # Momentum (RSI)
    if rsi_val is not None:
        if rsi_val < RSI_OVERSOLD:   score += 25   # oversold → bullish
        elif rsi_val > RSI_OVERBOUGHT: score -= 25  # overbought → bearish
        elif rsi_val < 50:           score += 5
        else:                        score -= 5

    # MACD
    if macd_h is not None:
        if macd_h > 0:   score += 15
        else:            score -= 15

    # Bollinger %B
    if bb_pb is not None:
        if bb_pb < 0.0:    score += 15   # below lower band → oversold
        elif bb_pb > 1.0:  score -= 15   # above upper band → overbought

    # Recent momentum
    if ch_5d > 0.02:   score += 10
    elif ch_5d < -0.02: score -= 10

    score = max(-100, min(100, score))

    if score >= 30:
        signal = "BUY"
    elif score <= -30:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "price":       round(price,   4),
        "change_1d":   round(ch_1d,   4),
        "change_5d":   round(ch_5d,   4),
        "change_20d":  round(ch_20d,  4),
        "ma50":        round(ma50,    4) if ma50 is not None else None,
        "ma200":       round(ma200,   4) if ma200 is not None else None,
        "golden_cross": golden_cross,
        "rsi":         round(rsi_val, 2) if rsi_val is not None else None,
        "atr_pct":     round(atr_pct, 4) if atr_pct is not None else None,
        "bb_upper":    round(bb_up,   4),
        "bb_lower":    round(bb_lo,   4),
        "bb_pct_b":    round(bb_pb,   4) if bb_pb is not None else None,
        "macd":        round(macd_v,  6) if macd_v is not None else None,
        "macd_hist":   round(macd_h,  6) if macd_h is not None else None,
        "sharpe":      round(sr,      3),
        "profit_factor": round(pf,    3),
        "signal":      signal,
        "signal_score": score,
    }
