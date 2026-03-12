"""
FastAPI Web Dashboard
=====================
Serves the multi-asset trading indicator web UI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, (np.bool_,)):     return bool(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        return super().default(obj)


def _jsonify(data) -> str:
    return json.dumps(data, cls=_NumpyEncoder, ensure_ascii=False)

from config import ASSETS
from data.fetcher import fetch
from indicators.analyzer import analyze_symbol
from indicators.parallel_world import full_parallel_world_analysis

# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="マルチアセット売買指標ダッシュボード")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ── Market overview (フラクタル解析ヘッダー用) ────────────────────────────────

OVERVIEW_SYMBOLS = {
    "株価":   ("^GSPC",  "米国株"),
    "金利":   ("^TNX",   "金利"),
    "金":     ("GC=F",   "金"),
    "日経":   ("^N225",  "日本株"),
    "資源":   ("CL=F",   "資源"),
    "心理":   ("^VIX",   "VIX"),
}

OVERVIEW_FLAGS = {
    "株価":  "🇺🇸",
    "金利":  "🇺🇸",
    "金":    "🥇",
    "日経":  "🇯🇵",
    "資源":  "🔷",
    "心理":  "😱",
}


def _overview_sentiment(symbol: str, name: str) -> dict:
    """Return sentiment for one overview symbol."""
    df = fetch(symbol, period="2y")
    if df is None or df.empty:
        return {"name": name, "sentiment": "不明", "icon": "❓", "bullish": None}

    close   = df["Close"].dropna()
    ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1)) if len(close) >= 21 else 0
    ret_5d  = float((close.iloc[-1] / close.iloc[-6]  - 1)) if len(close) >= 6  else 0

    # VIX: high = bearish market psychology
    if symbol == "^VIX":
        val = float(close.iloc[-1])
        if val > 30:
            return {"name": name, "sentiment": "警戒", "icon": "🚨", "bullish": False}
        elif val > 20:
            return {"name": name, "sentiment": "注意", "icon": "⚠️",  "bullish": None}
        else:
            return {"name": name, "sentiment": "安心", "icon": "✅", "bullish": True}

    combined = ret_20d * 0.6 + ret_5d * 0.4
    if combined > 0.02:
        return {"name": name, "sentiment": "上昇", "icon": "📈", "bullish": True}
    elif combined < -0.02:
        return {"name": name, "sentiment": "弱気", "icon": "🐻", "bullish": False}
    else:
        return {"name": name, "sentiment": "横ばい", "icon": "➡️", "bullish": None}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Build list of (symbol, display_name) for selector
    symbol_list = []
    for cls, syms in ASSETS.items():
        for sym, name in syms.items():
            symbol_list.append({"symbol": sym, "name": f"{name} ({cls})", "class": cls})
    return templates.TemplateResponse("index.html", {
        "request":     request,
        "symbol_list": symbol_list,
    })


@app.get("/api/overview")
async def api_overview():
    """フラクタル解析ヘッダー用: 6アセットの市場概況"""
    overview = []
    for key, (symbol, name) in OVERVIEW_SYMBOLS.items():
        data = _overview_sentiment(symbol, name)
        data["key"]  = key
        data["flag"] = OVERVIEW_FLAGS.get(key, "")
        overview.append(data)
    return Response(_jsonify({"overview": overview}), media_type="application/json")


@app.get("/api/analysis/{symbol:path}")
async def api_analysis(symbol: str, demo: bool = False):
    """
    指定シンボルのフル分析。
    Returns: technical indicators + parallel world win rates + action + trajectory
    """
    import data.fetcher as _fetcher
    if demo:
        _fetcher.FORCE_MOCK = True

    # Find display name
    name = symbol
    for cls, syms in ASSETS.items():
        if symbol in syms:
            name = syms[symbol]
            break

    df = fetch(symbol, period="5y")
    if df is None or df.empty:
        return JSONResponse({"error": "データ取得失敗"}, status_code=404)

    close = df["Close"].dropna()

    # Technical analysis
    tech = analyze_symbol(symbol, name)

    # Parallel world
    pw = full_parallel_world_analysis(close)

    # Total trading days in history
    total_days = int(len(close))

    payload = {
        "symbol":      symbol,
        "name":        name,
        "total_days":  total_days,
        "technical":   tech,
        "win_rates":   pw["win_rates"],
        "action":      pw["action"],
        "trajectory":  pw["trajectory"],
        "matched":     pw["matched_count"],
    }
    return Response(_jsonify(payload), media_type="application/json")


@app.get("/api/symbols")
async def api_symbols():
    """全シンボル一覧"""
    result = []
    for cls, syms in ASSETS.items():
        for sym, name in syms.items():
            result.append({"symbol": sym, "name": name, "class": cls})
    return Response(_jsonify({"symbols": result}), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True,
                app_dir=BASE_DIR)
