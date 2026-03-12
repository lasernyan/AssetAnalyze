"""
パラレルワールド (Parallel World Pattern Matching)
===================================================
過去の価格系列から「現在に最も似た局面」を抽出し、
その後の値動き統計から各時間軸の勝率・目標・ストップを算出する。

アルゴリズム:
  1. 直近 N 日間の価格変動率パターンを正規化
  2. 全履歴を同じウィンドウサイズでスキャンし DTW 距離を計算
  3. 距離が小さい上位 K 件を「類似パターン (パラレルワールド)」として選定
  4. 各時間軸 (1h/4h/1d/1w/1m/1y) で勝率・期待リターンを集計
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── パラメータ ─────────────────────────────────────────────────────────────────
PATTERN_WINDOW    = 20    # 類似判定に使う直近日数
TOP_K             = 100   # 抽出する類似パターン件数
MIN_HISTORY_DAYS  = 300   # 最低必要履歴日数

# 時間軸定義: (ラベル, 日数換算, 説明)
TIME_HORIZONS = [
    ("1時間後",   0.042, "直近2ヶ月・テクニカル類似"),
    ("4時間後",   0.167, "直近2ヶ月・テクニカル類似"),
    ("1日後",     1,     "究極マクロ・パラレルワールド抽出"),
    ("1週間後",   5,     "究極マクロ・パラレルワールド抽出"),
    ("1ヶ月後",   21,    "究極マクロ・パラレルワールド抽出"),
    ("1年後",     252,   "究極マクロ・パラレルワールド抽出"),
]


def _normalize(arr: np.ndarray) -> np.ndarray:
    """ゼロ平均・単位分散に正規化 (分散=0の場合はゼロ配列を返す)"""
    std = arr.std()
    if std < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def find_similar_patterns(
    close: pd.Series,
    pattern_window: int = PATTERN_WINDOW,
    top_k: int = TOP_K,
) -> tuple[list[int], np.ndarray]:
    """
    直近 pattern_window 日間のパターンに最も近い過去の開始インデックスを返す。
    Returns (indices, distances)
    """
    returns = close.pct_change().fillna(0).values  # shape: (N,)
    n = len(returns)

    current_pattern = _normalize(returns[-pattern_window:])

    distances = []
    # スキャン範囲: 0 .. n-pattern_window-252 (未来が十分にある箇所のみ)
    max_start = n - pattern_window - 252
    if max_start < pattern_window:
        max_start = n - pattern_window - 5  # 最低5日は未来が必要
    if max_start <= 0:
        return [], np.array([])

    for i in range(pattern_window, max_start):
        window = _normalize(returns[i - pattern_window: i])
        dist   = _euclidean_distance(current_pattern, window)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    top = distances[:top_k]
    indices   = [x[0] for x in top]
    dists_arr = np.array([x[1] for x in top])
    return indices, dists_arr


def compute_win_rates(
    close: pd.Series,
    indices: list[int],
    time_horizons: list[tuple] = TIME_HORIZONS,
) -> list[dict]:
    """
    各時間軸について類似パターン後の勝率・期待リターンを算出。
    """
    prices = close.values
    n      = len(prices)
    results = []

    for label, days, method in time_horizons:
        horizon_int = max(1, round(days))
        future_rets = []

        for idx in indices:
            end_idx = idx + horizon_int
            if end_idx >= n:
                continue
            ret = (prices[end_idx] - prices[idx]) / prices[idx]
            future_rets.append(ret)

        if len(future_rets) < 5:
            results.append({
                "label":      label,
                "method":     method,
                "days":       days,
                "buy_pct":    50.0,
                "sell_pct":   50.0,
                "avg_ret":    0.0,
                "win_count":  0,
                "total":      0,
            })
            continue

        arr      = np.array(future_rets)
        win_cnt  = int((arr > 0).sum())
        total    = len(arr)
        buy_pct  = round(win_cnt / total * 100, 1)
        sell_pct = round(100 - buy_pct, 1)
        avg_ret  = float(arr.mean())

        results.append({
            "label":     label,
            "method":    method,
            "days":      days,
            "buy_pct":   buy_pct,
            "sell_pct":  sell_pct,
            "avg_ret":   round(avg_ret * 100, 3),
            "win_count": win_cnt,
            "total":     total,
        })

    return results


def compute_action(
    close: pd.Series,
    win_rates: list[dict],
    indices: list[int],
) -> dict:
    """
    最適な推奨アクション (時間軸・方向・ターゲット・ストップ) を算出。
    最も勝率が高い (50%から最も離れた) 時間軸を推奨する。
    """
    current_price = float(close.iloc[-1])
    prices        = close.values

    # 最大エッジの時間軸を選定
    best = max(win_rates, key=lambda x: abs(x["buy_pct"] - 50))
    is_long   = best["buy_pct"] >= 50
    edge      = abs(best["buy_pct"] - 50)
    win_pct   = best["buy_pct"] if is_long else best["sell_pct"]

    # 類似パターン後の値動き統計でターゲット・ストップを計算
    horizon_int = max(1, round(best["days"]))
    gains, losses = [], []

    for idx in indices:
        end_idx = idx + horizon_int
        if end_idx >= len(prices):
            continue
        # 区間内の最大値・最小値
        segment = prices[idx: end_idx + 1]
        max_move = (segment.max() - prices[idx]) / prices[idx]
        min_move = (segment.min() - prices[idx]) / prices[idx]
        gains.append(max_move)
        losses.append(min_move)

    avg_gain  = float(np.mean(gains))  if gains  else 0.003
    avg_loss  = float(np.mean(losses)) if losses else -0.003
    max_gain  = float(np.percentile(gains,  25)) if gains  else avg_gain
    max_loss  = float(np.percentile(losses, 75)) if losses else avg_loss

    if is_long:
        target_price = current_price * (1 + abs(avg_gain))
        stop_price   = current_price * (1 + avg_loss)      # avg_loss は負
        hist_avg     = avg_gain * current_price
        max_reached  = max_gain * current_price
        avg_adverse  = abs(avg_loss) * current_price
        max_adverse  = abs(max_loss) * current_price
    else:
        target_price = current_price * (1 - abs(avg_gain))
        stop_price   = current_price * (1 + abs(avg_loss))
        hist_avg     = -avg_gain * current_price
        max_reached  = -max_gain * current_price
        avg_adverse  = abs(avg_loss) * current_price
        max_adverse  = abs(max_loss) * current_price

    direction = "ロング(買い)" if is_long else "ショート(売り)"

    return {
        "label":        best["label"],
        "direction":    direction,
        "is_long":      is_long,
        "win_pct":      round(win_pct, 1),
        "edge":         round(edge, 1),
        "current_price": round(current_price, 4),
        "target_price": round(target_price, 4),
        "stop_price":   round(stop_price, 4),
        "hist_avg_move": round(hist_avg, 4),
        "max_reached":  round(max_reached, 4),
        "avg_adverse":  round(avg_adverse, 4),
        "max_adverse":  round(max_adverse, 4),
    }


def compute_trajectory(
    close: pd.Series,
    indices: list[int],
    horizon_days: int = 5,
) -> list[dict]:
    """
    類似パターン後の中央値軌道を計算 (未来軌道チャート用)。
    Returns list of {day, price, above_current} for day 0..horizon_days
    """
    current_price = float(close.iloc[-1])
    prices        = close.values
    n             = len(prices)

    trajectories = []
    for idx in indices:
        traj = []
        for d in range(horizon_days + 1):
            end = idx + d
            if end >= n:
                break
            traj.append((prices[end] - prices[idx]) / prices[idx])
        if len(traj) == horizon_days + 1:
            trajectories.append(traj)

    if not trajectories:
        return [{"day": d, "price": current_price, "above_current": False}
                for d in range(horizon_days + 1)]

    arr    = np.array(trajectories)
    median = np.median(arr, axis=0)

    result = []
    for d, ret in enumerate(median):
        price = current_price * (1 + ret)
        result.append({
            "day":           d,
            "price":         round(price, 4),
            "above_current": price >= current_price,
        })
    return result


def full_parallel_world_analysis(
    close: pd.Series,
    pattern_window: int = PATTERN_WINDOW,
    top_k: int          = TOP_K,
    trajectory_days: int = 5,
) -> dict:
    """
    パラレルワールド分析の全結果を返す。
    """
    if len(close) < MIN_HISTORY_DAYS:
        # 履歴不足: ダミーを返す
        buy_pct = 50.0
        return {
            "matched_count":  0,
            "win_rates":      [{
                "label": lbl, "method": meth, "days": d,
                "buy_pct": 50.0, "sell_pct": 50.0,
                "avg_ret": 0.0, "win_count": 0, "total": 0,
            } for lbl, d, meth in TIME_HORIZONS],
            "action":     {},
            "trajectory": [],
        }

    indices, distances = find_similar_patterns(close, pattern_window, top_k)
    win_rates  = compute_win_rates(close, indices)
    action     = compute_action(close, win_rates, indices)
    trajectory = compute_trajectory(close, indices, trajectory_days)

    return {
        "matched_count": len(indices),
        "win_rates":     win_rates,
        "action":        action,
        "trajectory":    trajectory,
    }
