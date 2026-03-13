"""
Gold (GC=F) Backtesting Script
================================
5つの戦略でゴールド先物データをバックテスト。
目標: プロフィットファクター >= 1.5 (実用圏)

データ:
  実データ: yfinance (GC=F, 5年日足)
  フォールバック: GBMモックデータ (開始$2350, μ=8%/年, σ=12%/年)

戦略:
  S1: EMA20/50クロス + MACD確認
  S2: RSI50クロス + EMAトレンドフィルター
  S3: MACDゼロクロス + MAフィルター
  S4: ロング限定・押し目買い (RSI<45 + 上昇トレンド)
  S5: ダブルMA(50/200)クロス + MACD確認

TP = ATR × 3.0  /  SL = ATR × 1.5  (R:R = 2.0)
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
SYMBOL       = "GC=F"
PERIOD       = "5y"
ATR_TP_MULT  = 3.0
ATR_SL_MULT  = 1.5
WARMUP_BARS  = 210   # MA200 + バッファ

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
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(com=period - 1, min_periods=period).mean()
    al = l.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml - sl


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"]       = calc_atr(df)
    df["rsi"]       = calc_rsi(df["Close"])
    df["macd_hist"] = calc_macd_hist(df["Close"])
    df["ema20"]     = _ema(df["Close"], 20)
    df["ema50"]     = _ema(df["Close"], 50)
    df["ema200"]    = _ema(df["Close"], 200)
    df["sma50"]     = df["Close"].rolling(50).mean()
    df["sma200"]    = df["Close"].rolling(200).mean()
    return df.dropna(subset=["ema200", "atr", "rsi"])


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


# ── バックテストエンジン ───────────────────────────────────────────────────────

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
            exit_price = exit_reason = None
            if direction == "LONG":
                if high >= tp:
                    exit_price, exit_reason = tp, "TP"
                elif low <= sl:
                    exit_price, exit_reason = sl, "SL"
                else:
                    s = signal_func(df, i)
                    if s == "SHORT":
                        exit_price, exit_reason = close, "SIGNAL"
            else:
                if low <= tp:
                    exit_price, exit_reason = tp, "TP"
                elif high >= sl:
                    exit_price, exit_reason = sl, "SL"
                else:
                    s = signal_func(df, i)
                    if s == "LONG":
                        exit_price, exit_reason = close, "SIGNAL"

            if exit_price is not None:
                pnl = ((exit_price - entry_price) / entry_price * 100
                       if direction == "LONG"
                       else (entry_price - exit_price) / entry_price * 100)
                trades.append(Trade(
                    str(entry_date.date()), str(date.date()),
                    direction,
                    round(entry_price, 2), round(exit_price, 2),
                    exit_reason, round(pnl, 4)
                ))
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


# ── 5つの戦略シグナル ─────────────────────────────────────────────────────────

def signal_ema_cross_macd(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S1: EMA20/50クロス + MACD確認
    LONG : EMA20がEMA50を上抜け かつ MACDhist>0
    SHORT: EMA20がEMA50を下抜け かつ MACDhist<0
    """
    e20_c = df["ema20"].iloc[i]
    e20_p = df["ema20"].iloc[i - 1]
    e50_c = df["ema50"].iloc[i]
    e50_p = df["ema50"].iloc[i - 1]
    mh    = df["macd_hist"].iloc[i]

    if pd.isna(e20_c) or pd.isna(e50_c) or pd.isna(mh):
        return None

    if e20_p < e50_p and e20_c >= e50_c and mh > 0:
        return "LONG"
    if e20_p > e50_p and e20_c <= e50_c and mh < 0:
        return "SHORT"
    return None


def signal_rsi50_cross_ema(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S2: RSI50クロス + EMAトレンドフィルター
    LONG : RSIが50を上向きクロス かつ EMA50>EMA200
    SHORT: RSIが50を下向きクロス かつ EMA50<EMA200
    """
    rsi_c = df["rsi"].iloc[i]
    rsi_p = df["rsi"].iloc[i - 1]
    e50   = df["ema50"].iloc[i]
    e200  = df["ema200"].iloc[i]

    if pd.isna(rsi_c) or pd.isna(e50) or pd.isna(e200):
        return None

    if rsi_p < 50 and rsi_c >= 50 and e50 > e200:
        return "LONG"
    if rsi_p > 50 and rsi_c <= 50 and e50 < e200:
        return "SHORT"
    return None


def signal_macd_zerocross(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S3: MACDゼロクロス + MAフィルター
    LONG : MACDhistが負→正 かつ EMA50>EMA200
    SHORT: MACDhistが正→負 かつ EMA50<EMA200
    """
    mh_c = df["macd_hist"].iloc[i]
    mh_p = df["macd_hist"].iloc[i - 1]
    e50  = df["ema50"].iloc[i]
    e200 = df["ema200"].iloc[i]
    rsi  = df["rsi"].iloc[i]

    if any(pd.isna(x) for x in [mh_c, mh_p, e50, e200, rsi]):
        return None

    if mh_p < 0 and mh_c >= 0 and e50 > e200 and 40 <= rsi <= 70:
        return "LONG"
    if mh_p > 0 and mh_c <= 0 and e50 < e200 and 30 <= rsi <= 60:
        return "SHORT"
    return None


def signal_dip_buy_long_only(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S4: ロング限定・押し目買い
    LONG : EMA50>EMA200 (上昇トレンド) かつ RSI<45 (押し目)
           かつ 終値がEMA50より上 (完全崩壊していない)
    SHORT: なし (ロング限定)
    ※ ゴールドの正のドリフト(μ=8%)を活かす
    """
    e50  = df["ema50"].iloc[i]
    e200 = df["ema200"].iloc[i]
    rsi  = df["rsi"].iloc[i]
    rsi_p = df["rsi"].iloc[i - 1]
    close = df["Close"].iloc[i]
    mh   = df["macd_hist"].iloc[i]

    if any(pd.isna(x) for x in [e50, e200, rsi, mh]):
        return None

    # RSIが45を下から上にクロス → 押し目からの回復
    if (e50 > e200                         # 上昇トレンド確認
            and rsi_p < 45 and rsi >= 45   # RSI回復クロス
            and close > e50 * 0.97         # EMA50から離れすぎていない
            and mh > 0):                   # MACDポジティブ
        return "LONG"

    return None


def signal_sma_cross_confirmed(df: pd.DataFrame, i: int) -> Optional[str]:
    """
    S5: SMA50/200クロス + RSI + MACD確認
    LONG : SMA50がSMA200を上抜け かつ RSI>50 かつ MACDhist>0
    SHORT: SMA50がSMA200を下抜け かつ RSI<50 かつ MACDhist<0
    ※ トレンド転換の初動を捉える
    """
    s50_c = df["sma50"].iloc[i]
    s50_p = df["sma50"].iloc[i - 1]
    s200_c = df["sma200"].iloc[i]
    s200_p = df["sma200"].iloc[i - 1]
    rsi   = df["rsi"].iloc[i]
    mh    = df["macd_hist"].iloc[i]

    if any(pd.isna(x) for x in [s50_c, s50_p, s200_c, s200_p, rsi, mh]):
        return None

    if s50_p < s200_p and s50_c >= s200_c and rsi > 50 and mh > 0:
        return "LONG"
    if s50_p > s200_p and s50_c <= s200_c and rsi < 50 and mh < 0:
        return "SHORT"
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
                "win_rate": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "total_return": 0.0}

    pnl    = [t.pnl_pct for t in trades]
    wins   = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]

    win_rate     = len(wins) / len(pnl) * 100
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = pd.Series([100.0])
    for p in pnl:
        equity = pd.concat([equity,
                            pd.Series([equity.iloc[-1] * (1 + p / 100)])],
                           ignore_index=True)

    rets   = equity.pct_change().dropna()
    sharpe = (float(np.sqrt(252) * rets.mean() / rets.std())
              if rets.std() > 0 else 0.0)

    rolling_max = equity.cummax()
    max_dd      = float(((equity - rolling_max) / rolling_max * 100).min())

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
        "tp_hits":       tp_h,
        "sl_hits":       sl_h,
        "signal_exits":  sg_h,
        "long_trades":   sum(1 for t in trades if t.direction == "LONG"),
        "short_trades":  sum(1 for t in trades if t.direction == "SHORT"),
    }


# ── 表示 ──────────────────────────────────────────────────────────────────────

def print_summary(results: List[dict]):
    sep  = "─" * 72
    sep2 = "═" * 72

    print(f"\n{sep2}")
    print(f"  Gold ({SYMBOL}) バックテスト  "
          f"TP=ATR×{ATR_TP_MULT}  SL=ATR×{ATR_SL_MULT}  (R:R={ATR_TP_MULT/ATR_SL_MULT:.1f})")
    print(f"{sep2}")

    for m in results:
        pf    = m["profit_factor"]
        label = _pf_label(pf)
        n     = m["trades"]
        stat  = "(統計不十分)" if n < 20 else ""
        print(f"\n  戦略: {m['strategy']}")
        print(f"  {sep}")
        print(f"  {'トレード数':<20}: {n:>5}  {stat}")
        print(f"  {'勝率':<20}: {m['win_rate']:>5.1f} %")
        print(f"  {'プロフィットファクター':<18}: {pf:>6.3f}  {label}")
        print(f"  {'シャープレシオ':<20}: {m['sharpe']:>6.3f}")
        print(f"  {'最大ドローダウン':<19}: {m['max_dd']:>6.2f} %")
        print(f"  {'累積リターン':<20}: {m['total_return']:>+7.2f} %")
        print(f"  {'平均利益/損失':<19}: {m['avg_win_pct']:>+7.3f}% / {m['avg_loss_pct']:>+7.3f}%")
        print(f"  {'実R:R':<20}: {m['rr_ratio']:>6.2f}")
        print(f"  {'TP/SL/シグナル終了':<17}: {m['tp_hits']} / {m['sl_hits']} / {m['signal_exits']}")
        print(f"  {'Long / Short':<20}: {m['long_trades']} / {m['short_trades']}")

    print(f"\n{sep2}")
    print("  【判定】  PF≥2.0: 優秀  PF≥1.5: 実用圏  PF≥1.2: 要改善  PF<1.2: 実用不可")
    print(f"{sep2}")

    valid = [m for m in results if m["trades"] >= 10]
    if valid:
        best  = max(valid, key=lambda x: x["profit_factor"])
        print(f"\n  ▶ ベスト戦略: 【{best['strategy']}】")
        print(f"    PF = {best['profit_factor']:.3f}  {_pf_label(best['profit_factor'])}")
        print(f"    勝率 {best['win_rate']}%  Sharpe {best['sharpe']}  MDD {best['max_dd']}%\n")

        # 実用圏到達の有無
        reach = [m for m in valid if m["profit_factor"] >= 1.5]
        if reach:
            print(f"  ✅ {len(reach)}戦略が実用圏 (PF≥1.5) に到達しました。")
        else:
            best_close = max(valid, key=lambda x: x["profit_factor"])
            gap = 1.5 - best_close["profit_factor"]
            print(f"  ❌ 実用圏未達。ベストPF={best_close['profit_factor']:.3f} (不足={gap:.3f})")
            if gap < 0.3:
                print("     → パラメータ調整・フィルター追加で到達可能な水準です。")
            else:
                print("     → データ期間延長・戦略見直しを推奨します。")
    print()


def print_trades(trades: List[Trade], name: str, n: int = 15):
    print(f"  ── 【{name}】 直近{n}トレード ──")
    print(f"  {'Entry':>12}  {'Exit':>12}  {'Dir':>5}  {'Entry$':>8}  {'Exit$':>8}  {'P/L%':>8}  {'理由'}")
    print(f"  {'─' * 72}")
    for t in trades[-n:]:
        sign = "+" if t.pnl_pct >= 0 else ""
        flag = "✓" if t.pnl_pct >= 0 else "✗"
        print(f"  {t.entry_date:>12}  {t.exit_date:>12}  {t.direction:>5}  "
              f"{t.entry_price:>8.2f}  {t.exit_price:>8.2f}  "
              f"{sign}{t.pnl_pct:>7.3f}%  {t.exit_reason} {flag}")
    print()


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n  ━━━ Gold Backtest ({SYMBOL}) ━━━")
    print(f"  データ取得中 ({PERIOD}) ...")

    df_raw = None
    if _YFINANCE_OK:
        try:
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()):
                df_raw = yf.Ticker(SYMBOL).history(
                    period=PERIOD, interval="1d", auto_adjust=True)
        except Exception as e:
            print(f"  ※ yfinanceエラー: {e}")

    if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
        print("  ※ GBMモックデータを使用 (GC=F: 開始$2350, μ=8%/年, σ=12%/年)")
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "mock_module",
            pathlib.Path(__file__).parent / "data" / "mock.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df_raw = mod.generate_ohlcv(SYMBOL, n_days=1260, seed=42)

    print(f"  データ期間: {df_raw.index[0].date()} ～ {df_raw.index[-1].date()}"
          f"  ({len(df_raw)}本)")

    df = prepare_indicators(df_raw)
    print(f"  指標計算後: {len(df)}本  (有効バー: {len(df) - WARMUP_BARS}本)\n")

    strategies = [
        ("S1: EMA20/50クロス+MACD",        signal_ema_cross_macd),
        ("S2: RSI50クロス+EMAトレンド",      signal_rsi50_cross_ema),
        ("S3: MACDゼロクロス+MAフィルター",  signal_macd_zerocross),
        ("S4: ロング専用・押し目買い",        signal_dip_buy_long_only),
        ("S5: SMA50/200クロス+MACD確認",    signal_sma_cross_confirmed),
    ]

    results = []
    for name, func in strategies:
        trades = backtest(df, func)
        m      = calc_metrics(trades, name)
        results.append((name, func, trades, m))

    print_summary([r[3] for r in results])

    # ベスト戦略のトレード一覧 + パラメータスイープ
    valid = [(n, f, t, m) for n, f, t, m in results if m["trades"] >= 10]
    if valid:
        best = max(valid, key=lambda x: x[3]["profit_factor"])
        print_trades(best[2], best[0])
        param_sweep(df, best[1], best[0])

    # ── 診断レポート ──────────────────────────────────────────────────────────
    print("  ── 診断レポート ──")
    best_all = max(results, key=lambda x: x[3]["profit_factor"])
    bm = best_all[3]
    rr = ATR_TP_MULT / ATR_SL_MULT
    min_wr = 100 / (rr + 1)
    target_wr = 100 * 1.5 / (1.5 + rr)

    print(f"  R:R = {rr:.1f} → PF1.5達成に必要な勝率: {target_wr:.1f}%")
    print(f"  現状ベスト勝率: {bm['win_rate']}%  (不足: {max(0, target_wr - bm['win_rate']):.1f}pp)")
    print()
    print("  【実用圏到達のヒント】")
    if bm["profit_factor"] < 1.2:
        print("  ・ショートトレードの除外 (ゴールドは長期的に上昇バイアス)")
        print("  ・TP/SL比率を TP×4/SL×1.5 に広げる")
        print("  ・週足トレンドフィルターを追加する")
    elif bm["profit_factor"] < 1.5:
        print("  ・現在の戦略はほぼ実用圏。追加フィルターで改善可能")
        print("  ・ATR倍率の微調整 (TP×3.5/SL×1.5 など)")
        print("  ・出来高フィルター or VIX相関フィルターの追加")
    else:
        print("  ・実用圏到達済み。リアルデータでの追加検証を推奨")
    print()


def param_sweep(df: pd.DataFrame, signal_func, name: str):
    """TP/SLパラメータ最適化スイープ (S3戦略に適用)"""
    global ATR_TP_MULT, ATR_SL_MULT

    tp_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    sl_range = [1.0, 1.2, 1.5, 2.0]

    print(f"  ── パラメータスイープ: 【{name}】 ──")
    print(f"  {'TP':>5}  {'SL':>5}  {'R:R':>5}  {'Trades':>7}  "
          f"{'WR%':>6}  {'PF':>7}  {'MDD%':>7}  {'判定'}")
    print(f"  {'─' * 66}")

    best_pf    = 0.0
    best_param = None

    for tp in tp_range:
        for sl in sl_range:
            ATR_TP_MULT = tp
            ATR_SL_MULT = sl
            trades = backtest(df, signal_func)
            m      = calc_metrics(trades, name)
            pf     = m["profit_factor"]
            rr     = tp / sl

            if m["trades"] >= 10:
                label = _pf_label(pf)
                hl = "  ◀ 実用圏!" if pf >= 1.5 else ""
                print(f"  {tp:>5.1f}  {sl:>5.1f}  {rr:>5.2f}  {m['trades']:>7}  "
                      f"{m['win_rate']:>6.1f}  {pf:>7.3f}  {m['max_dd']:>7.2f}  {label}{hl}")

                if pf > best_pf:
                    best_pf    = pf
                    best_param = (tp, sl, m)

    # 元に戻す
    ATR_TP_MULT = 3.0
    ATR_SL_MULT = 1.5

    print()
    if best_param:
        tp_b, sl_b, m_b = best_param
        print(f"  最適パラメータ: TP={tp_b}×ATR  SL={sl_b}×ATR  "
              f"(R:R={tp_b/sl_b:.2f})")
        print(f"  最高PF = {m_b['profit_factor']:.3f}  勝率={m_b['win_rate']}%  "
              f"MDD={m_b['max_dd']}%")
        if best_pf >= 1.5:
            print(f"  ✅ 実用圏 (PF≥1.5) 到達！最適パラメータで運用可能です。")
        else:
            print(f"  ❌ 最大PF={best_pf:.3f}。実用圏まで {1.5 - best_pf:.3f} 不足。")
    print()


    print("  【実用圏到達の改善案】")
    print("  1. ショートトレードの除外 → ゴールドの上昇バイアスを活かす")
    print("  2. TP×4/SL×1.5 (R:R=2.67) → 必要勝率が 36%に下がる")
    print("  3. 週足EMAを上位フィルターに追加 → 偽シグナルを削減")
    print("  4. ATRの代わりにスウィング高値/安値ベースのSL → 構造的な損切り")
    print("  5. リアルデータ (yfinance) での検証 → GBMより強いトレンドを確認")
    print()


if __name__ == "__main__":
    main()
