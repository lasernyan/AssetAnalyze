"""
Mock data generator for offline / demo mode.
Generates realistic OHLCV price series using geometric Brownian motion.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Representative starting prices and drift/vol parameters per symbol
_PARAMS: dict[str, tuple[float, float, float]] = {
    # symbol: (start_price, annual_drift, annual_vol)
    "^N225":   (38_500,   0.08, 0.18),
    "^TOPX":   (2_700,    0.07, 0.16),
    "7203.T":  (3_200,    0.10, 0.22),
    "6758.T":  (13_000,   0.12, 0.25),
    "9984.T":  (9_500,    0.15, 0.35),
    "^GSPC":   (5_200,    0.10, 0.15),
    "^IXIC":   (16_400,   0.12, 0.18),
    "^DJI":    (38_800,   0.08, 0.14),
    "AAPL":    (185,      0.09, 0.22),
    "NVDA":    (780,      0.30, 0.50),
    "^TNX":    (4.3,      -0.05, 0.20),
    "^FVX":    (4.1,      -0.04, 0.18),
    "^IRX":    (5.2,       0.01, 0.08),
    "GC=F":    (2_350,    0.08, 0.12),
    "GLD":     (219,      0.08, 0.12),
    "CL=F":    (78,       0.05, 0.30),
    "NG=F":    (2.0,      -0.10, 0.50),
    "HG=F":    (4.5,      0.06, 0.25),
    "PDBC":    (14,       0.04, 0.20),
    "^VIX":    (18,       -0.02, 0.60),
    "^VXN":    (20,       -0.02, 0.65),
    "JPY=X":   (150,      0.03, 0.08),
    "EURJPY=X": (162,     0.02, 0.08),
    "EURUSD=X": (1.08,   -0.01, 0.06),
    "GBPJPY=X": (190,     0.03, 0.09),
}

_DEFAULT = (100.0, 0.05, 0.20)


def generate_ohlcv(symbol: str, n_days: int = 500, seed: int | None = None) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a symbol using GBM.
    n_days: number of trading days to generate
    """
    start_price, mu, sigma = _PARAMS.get(symbol, _DEFAULT)
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))

    dt     = 1 / 252
    daily_mu    = (mu - 0.5 * sigma**2) * dt
    daily_sigma = sigma * np.sqrt(dt)

    log_returns = rng.normal(daily_mu, daily_sigma, n_days)
    prices      = start_price * np.exp(np.cumsum(log_returns))

    # Add some intraday noise for H/L
    high_noise = rng.uniform(0.001, 0.015, n_days)
    low_noise  = rng.uniform(0.001, 0.015, n_days)
    vol_base   = int(start_price * 1_000_000)

    closes = prices
    opens  = np.roll(closes, 1)
    opens[0] = start_price

    highs  = np.maximum(opens, closes) * (1 + high_noise)
    lows   = np.minimum(opens, closes) * (1 - low_noise)
    vols   = rng.integers(int(vol_base * 0.5), int(vol_base * 1.5), n_days)

    end_date = datetime(2025, 3, 1)
    dates = [end_date - timedelta(days=n_days - 1 - i) for i in range(n_days)]

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": vols,
    }, index=pd.DatetimeIndex(dates))

    return df
