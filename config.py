"""
Multi-asset monitoring configuration.
Defines symbols for each asset class and global parameters.
"""

# ── Asset universe ─────────────────────────────────────────────────────────────

ASSETS = {
    "日本株": {
        "^N225":  "日経225",
        "^TOPX":  "TOPIX",
        "7203.T": "トヨタ",
        "6758.T": "ソニー",
        "9984.T": "ソフトバンクG",
    },
    "米国株": {
        "^GSPC":  "S&P500",
        "^IXIC":  "NASDAQ",
        "^DJI":   "ダウ",
        "AAPL":   "Apple",
        "NVDA":   "NVIDIA",
    },
    "金利": {
        "^TNX":   "米10年債利回り",
        "^FVX":   "米5年債利回り",
        "^IRX":   "米3ヶ月TB",
    },
    "金": {
        "GC=F":   "金先物",
        "GLD":    "金ETF(GLD)",
    },
    "資源": {
        "CL=F":   "WTI原油",
        "NG=F":   "天然ガス",
        "HG=F":   "銅先物",
        "PDBC":   "コモディティETF",
    },
    "VIX": {
        "^VIX":   "VIX(恐怖指数)",
        "^VXN":   "VXN(NASDAQ VIX)",
    },
    "為替": {
        "JPY=X":  "USD/JPY",
        "EURJPY=X": "EUR/JPY",
        "EURUSD=X": "EUR/USD",
        "GBPJPY=X": "GBP/JPY",
    },
}

# ── Model parameters ───────────────────────────────────────────────────────────

# Edge detection
EDGE_THRESHOLD = 0.04          # Trade only when |edge| > 4%
MISPRICING_THRESHOLD = 1.5     # Z-score threshold for mispricing signal

# Position sizing
KELLY_FRACTION = 0.25          # Fractional Kelly α (25% = conservative)
DEFAULT_BANKROLL = 1_000_000   # ¥1M default bankroll for sizing display
RISK_FREE_RATE = 0.04          # Annual risk-free rate (US 3M TB ~ 4%)

# Risk limits
MAX_DRAWDOWN_LIMIT = 0.08      # Block new trades if MDD > 8%
VAR_CONFIDENCE = 0.95          # VaR at 95% confidence (1.645σ)
TARGET_SHARPE = 2.0            # Target Sharpe Ratio

# Lookback windows
LOOKBACK_DAYS = 252            # 1 trading year for stats
SHORT_MA = 50                  # Short moving average
LONG_MA = 200                  # Long moving average (Golden/Death cross)
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
