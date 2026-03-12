#!/usr/bin/env python3
"""
マルチアセット売買指標ツール
============================
日米株 / 金利 / 金 / 資源 / VIX / 為替 の各アセットを一括監視し、
以下のシグナルを表示します (自動売買機能なし):

  - BUY / SELL / HOLD シグナル & スコア
  - Edge (p_model − p_mkt)
  - Mispricing Score δ
  - Kelly ポジションサイズ
  - VaR 95% / 最大ドローダウン
  - Sharpe Ratio / Profit Factor
  - RSI / MA cross / MACD / Bollinger %B

Usage:
  python main.py                  # 全アセット (実データ / 取得失敗時はモック)
  python main.py --demo           # モックデータで強制デモ実行
  python main.py --class 日本株   # 特定クラスのみ
  python main.py --bankroll 5000000  # バンクロール変更 (円)
  python main.py --alpha 0.5      # Kelly 分数変更 (0〜1)
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import data.fetcher as _fetcher
from config import ASSETS, DEFAULT_BANKROLL, KELLY_FRACTION
from indicators.analyzer import analyze_asset_class
from report.renderer import render_report, console


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="マルチアセット売買指標ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--class", dest="asset_class", default=None,
        choices=list(ASSETS.keys()),
        help="特定のアセットクラスのみ分析 (デフォルト: 全クラス)",
    )
    p.add_argument(
        "--bankroll", type=float, default=DEFAULT_BANKROLL,
        help=f"バンクロール (円) [デフォルト: {DEFAULT_BANKROLL:,.0f}]",
    )
    p.add_argument(
        "--alpha", type=float, default=KELLY_FRACTION,
        help=f"Fractional Kelly α [デフォルト: {KELLY_FRACTION}]",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="モックデータで強制デモ実行 (ネット不要)",
    )
    p.add_argument(
        "--no-parallel", action="store_true",
        help="並列ダウンロードを無効化",
    )
    return p.parse_args()


def fetch_all_parallel(target: dict[str, dict[str, str]],
                        bankroll: float, alpha: float) -> dict[str, list[dict]]:
    """
    アセットクラスごとにスレッドプールで並列ダウンロード + 分析。
    """
    all_results: dict[str, list[dict]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        total = sum(len(v) for v in target.values())
        task  = progress.add_task("データ取得中...", total=total)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for cls, symbols in target.items():
                for symbol, name in symbols.items():
                    from indicators.analyzer import analyze_symbol
                    fut = pool.submit(analyze_symbol, symbol, name, bankroll, alpha)
                    futures[fut] = cls

            per_class: dict[str, list[dict]] = {cls: [] for cls in target}
            for fut in as_completed(futures):
                cls = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    sym = "?"
                    result = {"symbol": sym, "name": sym, "error": str(e)}
                per_class[cls].append(result)
                progress.advance(task)

    # Preserve original symbol order
    for cls, symbols in target.items():
        ordered = []
        sym_to_result = {r["symbol"]: r for r in per_class[cls]}
        for sym in symbols:
            if sym in sym_to_result:
                ordered.append(sym_to_result[sym])
        all_results[cls] = ordered

    return all_results


def fetch_all_sequential(target: dict[str, dict[str, str]],
                           bankroll: float, alpha: float) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for cls, symbols in target.items():
            task = progress.add_task(f"{cls} 取得中...", total=len(symbols))
            results = []
            for symbol, name in symbols.items():
                from indicators.analyzer import analyze_symbol
                results.append(analyze_symbol(symbol, name, bankroll, alpha))
                progress.advance(task)
            all_results[cls] = results
    return all_results


def main() -> None:
    args = parse_args()

    # Demo / mock mode
    if args.demo:
        _fetcher.FORCE_MOCK = True
        console.print("[bold yellow]★ デモモード: モックデータを使用[/bold yellow]")

    # Select target asset classes
    if args.asset_class:
        target = {args.asset_class: ASSETS[args.asset_class]}
    else:
        target = ASSETS

    console.print(f"\n[bold white]バンクロール:[/bold white] ¥{args.bankroll:,.0f}  "
                  f"[bold white]Kelly α:[/bold white] {args.alpha}  "
                  f"[bold white]対象:[/bold white] {', '.join(target.keys())}\n")

    if args.no_parallel:
        all_results = fetch_all_sequential(target, args.bankroll, args.alpha)
    else:
        all_results = fetch_all_parallel(target, args.bankroll, args.alpha)

    render_report(all_results)


if __name__ == "__main__":
    main()
