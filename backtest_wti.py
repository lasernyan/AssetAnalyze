"""
WTI原油 (CL=F) バックテストスクリプト
========================================
Goldバックテスト手法をベースに、WTI固有の特性を加味した戦略を検証。
目標: プロフィットファクター >= 1.5 (実用圏)

WTI固有パラメータ (GBMモック):
  開始価格 $78 / 年間ドリフト μ=5% / 年間ボラ σ=30%
  ※ Goldより低ドリフト・高ボラのため、トレンドフォローが難しい

戦略:
  S1: MACDゼロクロス + MAフィルター (Goldベースライン移植)
  S2: MACDゼロクロス + 週足EMAフィルター [改善3]
  S3: MACDゼロクロス + スウィングSL [改善4]
  S4: MACDゼロクロス + 週足EMA + スウィングSL [改善3+4]
  S5: RSI逆張り + ATRボラフィルター (WTI特化)
  S6: ドンチャン + ATR急騰フィルター (WTI特化ブレイクアウト)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

try:
    import yfinance as yf
    _YFINANCE_OK = True
except Exception:
    _YFINANCE_OK = False

# ── パラメータ ────────────────────────────────────────────────────────────────
SYMBOL       = "CL=F"
PERIOD       = "5y"
ATR_TP_MULT  = 3.0
ATR_SL_MULT  = 1.5
SWING_TP_RR  = 2.0
SWING_N      = 10
WARMUP_BARS  = 220

# ── テクニカル指標 ─────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"]  - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d  = close.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    ag = g.ewm(com=period - 1, min_periods=period).mean()
    al = l.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ml = _ema(close, fast) - _ema(close, slow)
    return ml - _ema(ml, signal)


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"]        = calc_atr(df)
    df["atr_pct"]    = df["atr"] / df["Close"] * 100      # ATR%（ボラ正規化）
    df["atr_avg20"]  = df["atr_pct"].rolling(20).mean()   # ATR%の20日平均
    df["rsi"]        = calc_rsi(df["Close"])
    df["macd_hist"]  = calc_macd_hist(df["Close"])
    df["ema50"]      = _ema(df["Close"], 50)
    df["ema200"]     = _ema(df["Close"], 200)
    df["high_20"]    = df["Close"].shift(1).rolling(20).max()
    df["low_20"]     = df["Close"].shift(1).rolling(20).min()
    df["swing_low"]  = df["Low"].rolling(SWING_N).min()
    df["swing_high"] = df["High"].rolling(SWING_N).max()

    # 週足EMA20（改善3: 上位足トレンドフィルター）
    df_tz = df.copy()
    if df_tz.index.tz is None:
        df_tz.index = df_tz.index.tz_localize("UTC")
    weekly_close   = df_tz["Close"].resample("W").last()
    weekly_ema20   = _ema(weekly_close, 20)
    wema_daily     = weekly_ema20.reindex(df_tz.index, method="ffill")
    wema_daily.index = df.index
    df["weekly_ema20"] = wema_daily

    return df.dropna(subset=["ema200", "atr", "rsi", "weekly_ema20", "swing_low"])


# ── トレード記録 ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  str
    exit_date:   str
    direction:   str
    entry_price: float
    exit_price:  float
    exit_reason: str
    pnl_pct:     float


# ── ATRベースバックテストエンジン ─────────────────────────────────────────────

def backtest(df: pd.DataFrame, signal_func) -> List[Trade]:
    trades: List[Trade] = []
    in_trade = False
    direction = entry_price = tp = sl = entry_date = None

    for i in range(WARMUP_BARS, len(df)):
        row   = df.iloc[i]
        date  = df.index[i]
        close = float(row["Close"])
        high  = float(row["High"])
        low   = float(row["Low"])
        atr_v = float(row["atr"])

        if in_trade:
            ep = exit_price = exit_reason = None
            if direction == "LONG":
                if high >= tp:   ep, exit_reason = tp,    "TP"
                elif low <= sl:  ep, exit_reason = sl,    "SL"
                else:
                    s = signal_func(df, i)
                    if s == "SHORT": ep, exit_reason = close, "SIGNAL"
            else:
                if low <= tp:    ep, exit_reason = tp,    "TP"
                elif high >= sl: ep, exit_reason = sl,    "SL"
                else:
                    s = signal_func(df, i)
                    if s == "LONG":  ep, exit_reason = close, "SIGNAL"

            if ep is not None:
                pnl = ((ep - entry_price) / entry_price * 100
                       if direction == "LONG"
                       else (entry_price - ep) / entry_price * 100)
                trades.append(Trade(str(entry_date.date()), str(date.date()),
                                    direction, round(entry_price, 2), round(ep, 2),
                                    exit_reason, round(pnl, 4)))
                in_trade = False
        else:
            s = signal_func(df, i)
            if s in ("LONG", "SHORT"):
                in_trade    = True
                direction   = s
                entry_price = close
                entry_date  = date
                if s == "LONG":
                    tp = entry_price + ATR_TP_MULT * atr_v
                    sl = entry_price - ATR_SL_MULT * atr_v
                else:
                    tp = entry_price - ATR_TP_MULT * atr_v
                    sl = entry_price + ATR_SL_MULT * atr_v

    return trades


# ── スウィングSLバックテストエンジン（改善4）─────────────────────────────────

def backtest_swing(df: pd.DataFrame, signal_func) -> List[Trade]:
    trades: List[Trade] = []
    in_trade = False
    direction = entry_price = tp = sl = entry_date = None

    for i in range(WARMUP_BARS, len(df)):
        row   = df.iloc[i]
        date  = df.index[i]
        close = float(row["Close"])
        high  = float(row["High"])
        low   = float(row["Low"])
        atr_v = float(row["atr"])

        if in_trade:
            ep = exit_reason = None
            if direction == "LONG":
                if high >= tp:   ep, exit_reason = tp,    "TP"
                elif low <= sl:  ep, exit_reason = sl,    "SL"
                else:
                    s = signal_func(df, i)
                    if s == "SHORT": ep, exit_reason = close, "SIGNAL"
            else:
                if low <= tp:    ep, exit_reason = tp,    "TP"
                elif high >= sl: ep, exit_reason = sl,    "SL"
                else:
                    s = signal_func(df, i)
                    if s == "LONG":  ep, exit_reason = close, "SIGNAL"

            if ep is not None:
                pnl = ((ep - entry_price) / entry_price * 100
                       if direction == "LONG"
                       else (entry_price - ep) / entry_price * 100)
                trades.append(Trade(str(entry_date.date()), str(date.date()),
                                    direction, round(entry_price, 2), round(ep, 2),
                                    exit_reason, round(pnl, 4)))
                in_trade = False
        else:
            s = signal_func(df, i)
            if s in ("LONG", "SHORT"):
                sl_val = (float(row["swing_low"])  if s == "LONG"
                          else float(row["swing_high"]))
                if s == "LONG"  and sl_val >= close: continue
                if s == "SHORT" and sl_val <= close: continue
                sl_dist = abs(close - sl_val)
                min_sl  = atr_v * 0.5
                sl_dist = max(sl_dist, min_sl)
                in_trade    = True
                direction   = s
                entry_price = close
                entry_date  = date
                if s == "LONG":
                    sl = close - sl_dist
                    tp = close + sl_dist * SWING_TP_RR
                else:
                    sl = close + sl_dist
                    tp = close - sl_dist * SWING_TP_RR

    return trades


# ── シグナル関数群 ────────────────────────────────────────────────────────────

def signal_macd_ma(df: pd.DataFrame, i: int) -> Optional[str]:
    """S1: MACDゼロクロス + EMA50/200フィルター + RSI (Goldベースライン移植)"""
    mh_c = df["macd_hist"].iloc[i]
    mh_p = df["macd_hist"].iloc[i - 1]
    e50  = df["ema50"].iloc[i]
    e200 = df["ema200"].iloc[i]
    rsi  = df["rsi"].iloc[i]
    if any(pd.isna(x) for x in [mh_c, mh_p, e50, e200, rsi]): return None
    if mh_p < 0 and mh_c >= 0 and e50 > e200 and 40 <= rsi <= 70: return "LONG"
    if mh_p > 0 and mh_c <= 0 and e50 < e200 and 30 <= rsi <= 60: return "SHORT"
    return None


def signal_macd_weekly(df: pd.DataFrame, i: int) -> Optional[str]:
    """S2: MACDゼロクロス + 週足EMA20フィルター [改善3]"""
    mh_c  = df["macd_hist"].iloc[i]
    mh_p  = df["macd_hist"].iloc[i - 1]
    e50   = df["ema50"].iloc[i]
    e200  = df["ema200"].iloc[i]
    rsi   = df["rsi"].iloc[i]
    close = df["Close"].iloc[i]
    wema  = df["weekly_ema20"].iloc[i]
    if any(pd.isna(x) for x in [mh_c, mh_p, e50, e200, rsi, wema]): return None
    if mh_p < 0 and mh_c >= 0 and e50 > e200 and 40 <= rsi <= 70 and close > wema: return "LONG"
    if mh_p > 0 and mh_c <= 0 and e50 < e200 and 30 <= rsi <= 60 and close < wema: return "SHORT"
    return None


def signal_rsi_reversion(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S5: RSI逆張り + ATRボラフィルター (WTI特化)
    WTIは高ボラ(σ=30%)で急反発/急落しやすい。
    LONG : RSIが35を下向きクロス後、35を上抜け回復 かつ ATR%が平均以下(低ボラ期)
    SHORT: RSIが65を上向きクロス後、65を下抜け かつ ATR%が平均以下
    ATRが平均以上の日はスキップ（過度な混乱期のダマシを排除）
    """
    rsi_c    = df["rsi"].iloc[i]
    rsi_p    = df["rsi"].iloc[i - 1]
    e50      = df["ema50"].iloc[i]
    e200     = df["ema200"].iloc[i]
    atr_pct  = df["atr_pct"].iloc[i]
    atr_avg  = df["atr_avg20"].iloc[i]
    if any(pd.isna(x) for x in [rsi_c, rsi_p, e50, e200, atr_pct, atr_avg]): return None

    # ATRが平均の1.5倍超ならスキップ（EIA発表後や急変動期）
    if atr_pct > atr_avg * 1.5: return None

    # RSI回復 + 順トレンド方向のみ
    if rsi_p < 35 and rsi_c >= 35 and e50 > e200: return "LONG"
    if rsi_p > 65 and rsi_c <= 65 and e50 < e200: return "SHORT"
    return None


def signal_donchian_atr(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S6: ドンチャン20日ブレイクアウト + ATR急騰フィルター (WTI特化)
    WTIは需給ショック時に強いブレイクアウトが起きる。
    ATRが平均より大きい(=ボラ拡大)タイミングのブレイクのみトレード。
    LONG : 終値が直近20日高値を上抜け かつ ATR%が平均以上
    SHORT: 終値が直近20日安値を下抜け かつ ATR%が平均以上
    """
    close    = df["Close"].iloc[i]
    prev_c   = df["Close"].iloc[i - 1]
    high_20  = df["high_20"].iloc[i]
    low_20   = df["low_20"].iloc[i]
    atr_pct  = df["atr_pct"].iloc[i]
    atr_avg  = df["atr_avg20"].iloc[i]
    rsi      = df["rsi"].iloc[i]
    if any(pd.isna(x) for x in [high_20, low_20, atr_pct, atr_avg, rsi]): return None

    # ATRが平均以上 = ボラ拡大期のブレイク（本物のブレイクアウト）
    if atr_pct < atr_avg: return None

    if prev_c <= high_20 and close > high_20 and rsi < 75: return "LONG"
    if prev_c >= low_20  and close < low_20  and rsi > 25: return "SHORT"
    return None


# ── パフォーマンス集計 ────────────────────────────────────────────────────────

def _pf_label(pf: float) -> str:
    if pf >= 2.0: return "★ 優秀"
    if pf >= 1.5: return "○ 実用圏"
    if pf >= 1.2: return "△ 要改善"
    return "✗ 実用不可"


def calc_metrics(trades: List[Trade], name: str) -> dict:
    if not trades:
        return {"strategy": name, "trades": 0, "profit_factor": 0.0,
                "win_rate": 0.0, "sharpe": 0.0, "max_dd": 0.0, "total_return": 0.0}

    pnl    = [t.pnl_pct for t in trades]
    wins   = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]

    win_rate     = len(wins) / len(pnl) * 100
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = pd.Series([100.0])
    for p in pnl:
        equity = pd.concat([equity, pd.Series([equity.iloc[-1] * (1 + p / 100)])],
                           ignore_index=True)
    rets   = equity.pct_change().dropna()
    sharpe = (float(np.sqrt(252) * rets.mean() / rets.std())
              if rets.std() > 0 else 0.0)
    max_dd = float(((equity - equity.cummax()) / equity.cummax() * 100).min())

    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    tp_h = sum(1 for t in trades if t.exit_reason == "TP")
    sl_h = sum(1 for t in trades if t.exit_reason == "SL")
    sg_h = sum(1 for t in trades if t.exit_reason == "SIGNAL")

    return {
        "strategy":      name,
        "trades":        len(trades),
        "win_rate":      round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "sharpe":        round(sharpe, 3),
        "max_dd":        round(max_dd, 2),
        "total_return":  round(sum(pnl), 2),
        "avg_win_pct":   round(avg_win, 3),
        "avg_loss_pct":  round(avg_loss, 3),
        "rr_ratio":      round(avg_win / abs(avg_loss), 2) if avg_loss != 0 else 0,
        "tp_hits": tp_h, "sl_hits": sl_h, "signal_exits": sg_h,
        "long_trades":  sum(1 for t in trades if t.direction == "LONG"),
        "short_trades": sum(1 for t in trades if t.direction == "SHORT"),
    }


# ── 表示 ──────────────────────────────────────────────────────────────────────

def print_header(label: str, tp=None, sl=None):
    sep2 = "═" * 72
    rr   = (tp / sl) if tp and sl else SWING_TP_RR
    tp_s = f"TP=ATR×{tp}  SL=ATR×{sl}  (R:R={rr:.1f})" if tp else f"スウィングSL  R:R={rr:.1f}"
    print(f"\n{sep2}")
    print(f"  WTI ({SYMBOL}) {label}  {tp_s}")
    print(f"{sep2}")


def print_metrics(m: dict):
    sep = "─" * 72
    pf  = m["profit_factor"]
    n   = m["trades"]
    print(f"\n  戦略: {m['strategy']}")
    print(f"  {sep}")
    print(f"  {'トレード数':<20}: {n:>5}  {'(統計不十分)' if n < 20 else ''}")
    print(f"  {'勝率':<20}: {m['win_rate']:>5.1f} %")
    print(f"  {'プロフィットファクター':<18}: {pf:>6.3f}  {_pf_label(pf)}")
    print(f"  {'シャープレシオ':<20}: {m['sharpe']:>6.3f}")
    print(f"  {'最大ドローダウン':<19}: {m['max_dd']:>6.2f} %")
    print(f"  {'累積リターン':<20}: {m['total_return']:>+7.2f} %")
    print(f"  {'平均利益/損失':<19}: {m['avg_win_pct']:>+7.3f}% / {m['avg_loss_pct']:>+7.3f}%")
    print(f"  {'実R:R':<20}: {m['rr_ratio']:>6.2f}")
    print(f"  {'TP/SL/シグナル終了':<17}: {m['tp_hits']} / {m['sl_hits']} / {m['signal_exits']}")
    print(f"  {'Long / Short':<20}: {m['long_trades']} / {m['short_trades']}")


def print_summary_table(results: list, title: str):
    sep2 = "═" * 72
    sep  = "─" * 72
    print(f"\n{sep2}")
    print(f"  {title}")
    print(f"{sep2}")
    for m in results:
        print_metrics(m)
    print(f"\n{sep2}")
    print("  【判定】  PF≥2.0: 優秀  PF≥1.5: 実用圏  PF≥1.2: 要改善  PF<1.2: 実用不可")
    print(f"{sep2}")
    valid = [m for m in results if m["trades"] >= 10]
    if valid:
        best  = max(valid, key=lambda x: x["profit_factor"])
        pf    = best["profit_factor"]
        print(f"\n  ▶ セクションベスト: 【{best['strategy']}】")
        print(f"    PF={pf:.3f}  {_pf_label(pf)}  勝率={best['win_rate']}%  MDD={best['max_dd']}%")
        reach = [m for m in valid if m["profit_factor"] >= 1.5]
        if reach:
            print(f"  ✅ {len(reach)}戦略が実用圏 (PF≥1.5) 到達！")
        else:
            print(f"  ❌ 実用圏未達 (不足={1.5 - pf:.3f})")


def print_trades(trades: List[Trade], name: str, n: int = 15):
    print(f"\n  ── 【{name}】 直近{n}トレード ──")
    print(f"  {'Entry':>12}  {'Exit':>12}  {'Dir':>5}  {'Entry$':>7}  {'Exit$':>7}  {'P/L%':>8}  {'理由'}")
    print(f"  {'─' * 68}")
    for t in trades[-n:]:
        sign = "+" if t.pnl_pct >= 0 else ""
        flag = "✓" if t.pnl_pct >= 0 else "✗"
        print(f"  {t.entry_date:>12}  {t.exit_date:>12}  {t.direction:>5}  "
              f"{t.entry_price:>7.2f}  {t.exit_price:>7.2f}  "
              f"{sign}{t.pnl_pct:>7.3f}%  {t.exit_reason} {flag}")


def param_sweep(df: pd.DataFrame, signal_func, name: str):
    """TP/SLパラメータ最適化スイープ"""
    global ATR_TP_MULT, ATR_SL_MULT
    tp_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    sl_range = [1.0, 1.2, 1.5, 2.0]

    print(f"\n  ── パラメータスイープ: 【{name}】 ──")
    print(f"  {'TP':>5}  {'SL':>5}  {'R:R':>5}  {'Trades':>7}  {'WR%':>6}  {'PF':>7}  {'MDD%':>7}  {'判定'}")
    print(f"  {'─' * 66}")

    best_pf = 0.0
    best_p  = None
    for tp in tp_range:
        for sl in sl_range:
            ATR_TP_MULT = tp
            ATR_SL_MULT = sl
            tr = backtest(df, signal_func)
            m  = calc_metrics(tr, name)
            pf = m["profit_factor"]
            if m["trades"] >= 10:
                hl = "  ◀ 実用圏!" if pf >= 1.5 else ""
                print(f"  {tp:>5.1f}  {sl:>5.1f}  {tp/sl:>5.2f}  {m['trades']:>7}  "
                      f"{m['win_rate']:>6.1f}  {pf:>7.3f}  {m['max_dd']:>7.2f}  {_pf_label(pf)}{hl}")
                if pf > best_pf:
                    best_pf = pf
                    best_p  = (tp, sl, m)

    ATR_TP_MULT = 3.0
    ATR_SL_MULT = 1.5
    print()
    if best_p:
        tp_b, sl_b, m_b = best_p
        print(f"  最適: TP={tp_b}×ATR  SL={sl_b}×ATR  (R:R={tp_b/sl_b:.2f})")
        print(f"  最高PF={m_b['profit_factor']:.3f}  勝率={m_b['win_rate']}%  MDD={m_b['max_dd']}%")
        if best_pf >= 1.5:
            print(f"  ✅ 実用圏 (PF≥1.5) 到達！")
        else:
            print(f"  ❌ 最大PF={best_pf:.3f}  実用圏まで {1.5 - best_pf:.3f} 不足")
    return best_p


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n  ━━━ WTI原油 Backtest ({SYMBOL}) ━━━")
    print(f"  データ取得中 ({PERIOD}) ...")

    df_raw = None
    if _YFINANCE_OK:
        try:
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()):
                df_raw = yf.Ticker(SYMBOL).history(period=PERIOD, interval="1d",
                                                   auto_adjust=True)
        except Exception as e:
            print(f"  ※ yfinanceエラー: {e}")

    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
        print("  ※ GBMモックデータを使用 (CL=F: 開始$78, μ=5%/年, σ=30%/年)")
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "mock_module", pathlib.Path(__file__).parent / "data" / "mock.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df_raw = mod.generate_ohlcv(SYMBOL, n_days=1260, seed=7)

    print(f"  データ期間: {df_raw.index[0].date()} ～ {df_raw.index[-1].date()}"
          f"  ({len(df_raw)}本)")

    df = prepare_indicators(df_raw)
    print(f"  指標計算後: {len(df)}本  (有効バー: {len(df) - WARMUP_BARS}本)")

    # WTI のボラ確認
    atr_pct_mean = df["atr_pct"].iloc[WARMUP_BARS:].mean()
    print(f"  平均ATR%: {atr_pct_mean:.2f}%  "
          f"(Gold≈0.9%と比較して{'高' if atr_pct_mean > 1.5 else '普通'}ボラ)\n")

    # ── フェーズ1: Goldベースライン移植 + 改善3+4 ──────────────────────────
    phase1 = [
        ("S1: MACDゼロクロス+MAフィルター",         signal_macd_ma,     "ATR"),
        ("S2: MACDゼロクロス+週足EMA [改善3]",       signal_macd_weekly, "ATR"),
        ("S3: MACDゼロクロス+スウィングSL [改善4]",  signal_macd_ma,     "SWING"),
        ("S4: MACDゼロクロス+週足+スウィング [改3+4]", signal_macd_weekly, "SWING"),
    ]

    results_p1 = []
    for name, func, sl_type in phase1:
        trades = backtest_swing(df, func) if sl_type == "SWING" else backtest(df, func)
        results_p1.append(calc_metrics(trades, name))

    print_summary_table(results_p1,
        f"【フェーズ1】 Goldベースライン戦略をWTIに移植  "
        f"(TP=ATR×{ATR_TP_MULT} / SL=ATR×{ATR_SL_MULT} / SwingR:R={SWING_TP_RR})")

    # ── フェーズ2: WTI特化戦略 ───────────────────────────────────────────────
    phase2_defs = [
        ("S5: RSI逆張り+ATRボラフィルター",          signal_rsi_reversion, "ATR"),
        ("S5s: RSI逆張り+スウィングSL",              signal_rsi_reversion, "SWING"),
        ("S6: ドンチャン+ATR急騰フィルター",          signal_donchian_atr,  "ATR"),
        ("S6s: ドンチャン+ATR急騰+スウィングSL",      signal_donchian_atr,  "SWING"),
    ]

    results_p2 = []
    all_trades_p2 = []
    for name, func, sl_type in phase2_defs:
        trades = backtest_swing(df, func) if sl_type == "SWING" else backtest(df, func)
        results_p2.append(calc_metrics(trades, name))
        all_trades_p2.append((calc_metrics(trades, name), trades))

    print_summary_table(results_p2, "【フェーズ2】 WTI特化戦略")

    # ── ベスト戦略のトレード一覧 ──────────────────────────────────────────────
    all_results = results_p1 + results_p2
    all_valid   = [(m, t) for m, (_, t) in
                   zip(results_p1, [(n, backtest_swing(df, f) if s == "SWING" else backtest(df, f))
                                    for n, f, s in phase1])
                   if m["trades"] >= 10]
    all_valid  += [(m, t) for m, t in all_trades_p2 if m["trades"] >= 10]

    if all_valid:
        best_m, best_t = max(all_valid, key=lambda x: x[0]["profit_factor"])
        print_trades(best_t, best_m["strategy"])

    # ── ベスト戦略のパラメータスイープ ───────────────────────────────────────
    best_atr_m = max([m for m in all_results if m["trades"] >= 10 and "スウィング" not in m["strategy"]],
                     key=lambda x: x["profit_factor"], default=None)
    if best_atr_m:
        # ATRベースのベスト戦略を特定してスイープ
        for name, func, sl_type in (phase1 + [(d[0], d[1], d[2]) for d in phase2_defs]):
            if name == best_atr_m["strategy"] and sl_type == "ATR":
                param_sweep(df, func, name)
                break

    # ── 最終サマリー ──────────────────────────────────────────────────────────
    sep2 = "═" * 72
    print(f"\n{sep2}")
    print("  最終サマリー（全戦略）")
    print(f"{sep2}")
    print(f"  {'戦略':<42}  {'Trades':>7}  {'勝率':>6}  {'PF':>7}  {'MDD':>7}")
    print(f"  {'─' * 70}")

    valid_all = [m for m in all_results if m["trades"] >= 10]
    best_overall = max(valid_all, key=lambda x: x["profit_factor"]) if valid_all else None
    for m in all_results:
        flag  = " ◀ BEST" if best_overall and m["strategy"] == best_overall["strategy"] else ""
        reach = " 🎯実用圏!" if m["profit_factor"] >= 1.5 else ""
        print(f"  {m['strategy']:<42}  {m['trades']:>7}  "
              f"{m['win_rate']:>5.1f}%  {m['profit_factor']:>7.3f}  {m['max_dd']:>6.2f}%"
              f"{flag}{reach}")

    print(f"  {'─' * 70}")
    if best_overall:
        pf = best_overall["profit_factor"]
        print(f"\n  ▶ 全戦略ベスト: 【{best_overall['strategy']}】")
        print(f"    PF={pf:.3f}  {_pf_label(pf)}  勝率={best_overall['win_rate']}%  "
              f"MDD={best_overall['max_dd']}%")
        if pf >= 1.5:
            print(f"\n  ✅ 実用圏 (PF≥1.5) 到達！")
        else:
            print(f"\n  ❌ 実用圏未達 (不足={1.5 - pf:.3f})")

    # 必要勝率の計算
    rr = ATR_TP_MULT / ATR_SL_MULT
    target_wr = 100 * 1.5 / (1.5 + rr)
    print(f"\n  R:R={rr:.1f} → PF1.5達成に必要な勝率: {target_wr:.1f}%")
    if best_overall:
        print(f"  現状ベスト勝率: {best_overall['win_rate']}% "
              f"(不足: {max(0, target_wr - best_overall['win_rate']):.1f}pp)")

    print(f"\n  【WTI特有の課題と改善案】")
    print(f"  ・σ=30%（Gold 12%の2.5倍）→ GBMモックではノイズが支配的")
    print(f"  ・リアルデータには EIA在庫発表・OPECショックの方向性あり")
    print(f"  ・EIAフィルター（毎週水曜 14:30 ET 前後除外）を追加すると改善見込み")
    print(f"  ・週足/月足の需給サイクルに合わせた上位足フィルター強化が有効")
    print()


if __name__ == "__main__":
    main()
