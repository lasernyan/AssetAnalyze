"""
Data fetcher using yfinance with mock fallback.
Downloads OHLCV data and caches it for the session.
Falls back to synthetic GBM data when the network is unavailable.
"""

import yfinance as yf
import pandas as pd
from typing import Optional

_cache: dict[str, pd.DataFrame] = {}

# Set to True to always use mock data (e.g. for offline demos)
FORCE_MOCK: bool = False


def fetch(symbol: str, period: str = "1y", interval: str = "1d",
          mock_fallback: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a symbol.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Falls back to mock data if the network request fails.
    """
    cache_key = f"{symbol}:{period}:{interval}"
    if cache_key in _cache:
        return _cache[cache_key]

    if not FORCE_MOCK:
        import os, io, contextlib
        try:
            ticker = yf.Ticker(symbol)
            # Suppress yfinance's own stderr messages
            with contextlib.redirect_stderr(io.StringIO()):
                df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                _cache[cache_key] = df
                return df
        except Exception:
            pass

    if mock_fallback or FORCE_MOCK:
        from data.mock import generate_ohlcv
        # Map period string to approximate number of trading days
        period_days = {"1y": 252, "2y": 504, "5y": 1260, "6mo": 126, "3mo": 63}
        n_days = period_days.get(period, 252)
        df = generate_ohlcv(symbol, n_days=n_days)
        df.index = pd.to_datetime(df.index)
        _cache[cache_key] = df
        return df

    return None


def fetch_current_price(symbol: str) -> Optional[float]:
    """Return the latest closing price for a symbol."""
    df = fetch(symbol)
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def clear_cache() -> None:
    """Clear the in-memory cache (useful for long-running sessions)."""
    _cache.clear()


def is_mock(symbol: str) -> bool:
    """Return True if the cached data for this symbol is synthetic."""
    cache_key = f"{symbol}:2y:1d"
    if FORCE_MOCK:
        return True
    if cache_key not in _cache:
        return False
    # Heuristic: if yfinance returned data, timezone info will be present
    idx = _cache[cache_key].index
    return idx.tzinfo is None and not hasattr(idx, "tz") or idx.tz is None
