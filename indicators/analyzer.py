"""
Asset Analyzer
==============
Orchestrates data fetching + indicator computation for a single symbol.
Returns a unified result dict that the report layer can render.
"""

from data.fetcher import fetch
from indicators.edge_detection import price_edge_from_series
from indicators.position_sizing import sizing_metrics
from indicators.performance import technical_metrics
from config import DEFAULT_BANKROLL, KELLY_FRACTION, LOOKBACK_DAYS


def analyze_symbol(symbol: str, name: str,
                   bankroll: float = DEFAULT_BANKROLL,
                   alpha: float = KELLY_FRACTION) -> dict:
    """
    Full analysis for one symbol.

    Returns:
    {
      symbol, name,
      # from technical_metrics
      price, change_1d, change_5d, change_20d,
      ma50, ma200, golden_cross,
      rsi, atr_pct,
      bb_upper, bb_lower, bb_pct_b,
      macd, macd_hist,
      sharpe, profit_factor,
      signal, signal_score,
      # from edge detection
      p_model, p_mkt, b, ev, edge, sigma, mispricing_score,
      # from position sizing
      kelly_full, kelly_frac, position_size,
      var_95_daily, max_drawdown, current_drawdown, mdd_breach,
      # error flag
      error (str or None)
    }
    """
    result: dict = {"symbol": symbol, "name": name, "error": None}

    df = fetch(symbol, period="2y", interval="1d")
    if df is None or df.empty or len(df) < 60:
        result["error"] = "データ取得失敗"
        return result

    close = df["Close"].dropna()

    # ── Technical + performance ──────────────────────────────────────────────
    tech = technical_metrics(df)
    result.update(tech)

    # ── Edge detection ───────────────────────────────────────────────────────
    edge = price_edge_from_series(close, lookback=min(LOOKBACK_DAYS, len(close)))
    result.update(edge)

    # ── Position sizing ──────────────────────────────────────────────────────
    sizing = sizing_metrics(close, bankroll=bankroll, alpha=alpha)
    result.update(sizing)

    return result


def analyze_asset_class(symbols: dict[str, str],
                         bankroll: float = DEFAULT_BANKROLL,
                         alpha: float = KELLY_FRACTION) -> list[dict]:
    """Analyze all symbols in an asset class dict {symbol: name}."""
    results = []
    for symbol, name in symbols.items():
        results.append(analyze_symbol(symbol, name, bankroll, alpha))
    return results
