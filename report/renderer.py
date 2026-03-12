"""
Rich terminal renderer for multi-asset indicator report.
"""

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text
from rich.rule import Rule

console = Console()

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _signal_style(signal: str) -> str:
    return {"BUY": "bold green", "SELL": "bold red", "HOLD": "bold yellow"}.get(signal, "white")


def _pct(v, digits: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.{digits}f}%"


def _float(v, digits: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _price(v) -> str:
    if v is None:
        return "N/A"
    if v >= 1000:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def _rsi_style(rsi) -> str:
    if rsi is None:
        return "white"
    if rsi < 30:
        return "bold green"
    if rsi > 70:
        return "bold red"
    return "white"


def _cross_style(cross: str) -> str:
    return {"GOLDEN": "bold green", "DEATH": "bold red", "NONE": "dim white"}.get(cross or "", "dim")


def _mdd_style(mdd_breach: bool) -> str:
    return "bold red" if mdd_breach else "green"


def _score_bar(score: int, width: int = 20) -> str:
    """Simple ASCII progress bar for signal score."""
    filled = int(abs(score) / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    side   = "+" if score >= 0 else "-"
    return f"[{side}{abs(score):3d}] {bar}"


# ── Per-asset-class table ──────────────────────────────────────────────────────

def build_class_table(class_name: str, results: list[dict]) -> Table:
    t = Table(
        title=f"[bold cyan]{class_name}[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="cyan",
        expand=True,
    )

    t.add_column("銘柄",          style="bold white", no_wrap=True, min_width=14)
    t.add_column("価格",          justify="right",    no_wrap=True, min_width=10)
    t.add_column("1日",           justify="right",    no_wrap=True, min_width=7)
    t.add_column("5日",           justify="right",    no_wrap=True, min_width=7)
    t.add_column("20日",          justify="right",    no_wrap=True, min_width=7)
    t.add_column("RSI",           justify="right",    no_wrap=True, min_width=6)
    t.add_column("MA trend",      justify="center",   no_wrap=True, min_width=8)
    t.add_column("MACD",          justify="center",   no_wrap=True, min_width=7)
    t.add_column("BB %B",         justify="right",    no_wrap=True, min_width=7)
    t.add_column("Edge",          justify="right",    no_wrap=True, min_width=7)
    t.add_column("Mispricing δ",  justify="right",    no_wrap=True, min_width=10)
    t.add_column("Kelly f",       justify="right",    no_wrap=True, min_width=7)
    t.add_column("VaR 95%",       justify="right",    no_wrap=True, min_width=8)
    t.add_column("MDD",           justify="right",    no_wrap=True, min_width=7)
    t.add_column("Sharpe",        justify="right",    no_wrap=True, min_width=7)
    t.add_column("PF",            justify="right",    no_wrap=True, min_width=5)
    t.add_column("シグナル",      justify="center",   no_wrap=True, min_width=7)
    t.add_column("スコア",        justify="left",     no_wrap=True, min_width=26)

    for r in results:
        if r.get("error"):
            t.add_row(
                r["name"], "[red]" + r["error"] + "[/red]",
                *["—"] * 16,
            )
            continue

        signal = r.get("signal", "HOLD")
        score  = r.get("signal_score", 0)
        rsi_v  = r.get("rsi")
        cross  = r.get("golden_cross", "NONE")
        mdd_b  = r.get("mdd_breach", False)

        ch1d_style  = "green" if (r.get("change_1d") or 0) >= 0 else "red"
        ch5d_style  = "green" if (r.get("change_5d") or 0) >= 0 else "red"
        ch20d_style = "green" if (r.get("change_20d") or 0) >= 0 else "red"
        macd_style  = "green" if (r.get("macd_hist") or 0) >= 0 else "red"
        edge_style  = "green" if (r.get("edge") or 0) >= 0 else "red"

        t.add_row(
            r["name"],
            _price(r.get("price")),
            f"[{ch1d_style}]{_pct(r.get('change_1d'))}[/{ch1d_style}]",
            f"[{ch5d_style}]{_pct(r.get('change_5d'))}[/{ch5d_style}]",
            f"[{ch20d_style}]{_pct(r.get('change_20d'))}[/{ch20d_style}]",
            f"[{_rsi_style(rsi_v)}]{_float(rsi_v)}[/{_rsi_style(rsi_v)}]",
            f"[{_cross_style(cross)}]{cross or '—'}[/{_cross_style(cross)}]",
            f"[{macd_style}]{'▲' if (r.get('macd_hist') or 0) >= 0 else '▼'}[/{macd_style}]",
            _float(r.get("bb_pct_b")),
            f"[{edge_style}]{_float(r.get('edge'), 4)}[/{edge_style}]",
            _float(r.get("mispricing_score"), 3),
            _pct(r.get("kelly_frac")),
            _pct(r.get("var_95_daily")),
            f"[{_mdd_style(mdd_b)}]{_pct(r.get('max_drawdown'))}[/{_mdd_style(mdd_b)}]",
            _float(r.get("sharpe"), 2),
            _float(r.get("profit_factor"), 2),
            f"[{_signal_style(signal)}]{signal}[/{_signal_style(signal)}]",
            _score_bar(score),
        )

    return t


# ── Summary panel ──────────────────────────────────────────────────────────────

def build_summary_panel(all_results: dict[str, list[dict]]) -> Panel:
    lines = []
    for cls, results in all_results.items():
        buys  = sum(1 for r in results if r.get("signal") == "BUY")
        sells = sum(1 for r in results if r.get("signal") == "SELL")
        holds = sum(1 for r in results if r.get("signal") == "HOLD")
        errs  = sum(1 for r in results if r.get("error"))
        total = len(results)
        lines.append(
            f"[bold cyan]{cls:8s}[/bold cyan] "
            f"[green]BUY {buys:2d}[/green] "
            f"[red]SELL {sells:2d}[/red] "
            f"[yellow]HOLD {holds:2d}[/yellow] "
            f"[dim](エラー {errs}/{total})[/dim]"
        )

    content = "\n".join(lines)
    return Panel(content, title="[bold]サマリー[/bold]", border_style="white", expand=False)


# ── Legend panel ───────────────────────────────────────────────────────────────

LEGEND = """[bold]指標の説明[/bold]
[cyan]Edge[/cyan]       = p_model − p_mkt  (>0.04 で売買機会)
[cyan]Mispricing δ[/cyan] = (p_model − p_mkt) / σ  (|δ|>1.5 で顕著な乖離)
[cyan]Kelly f[/cyan]    = 1/4 Kelly (推奨ポジションサイズ / bankroll)
[cyan]VaR 95%[/cyan]    = μ − 1.645σ  (日次最大損失の目安)
[cyan]MDD[/cyan]        = 最大ドローダウン (>8% で[red]警告[/red])
[cyan]Sharpe[/cyan]     = (E[R]−Rf) / σ  (目標 > 2.0)
[cyan]PF[/cyan]         = 総利益 / 総損失  (目標 > 1.5)
[cyan]RSI[/cyan]        [green]<30 売られ過ぎ[/green] / [red]>70 買われ過ぎ[/red]
[cyan]MA trend[/cyan]   [green]GOLDEN[/green]=上昇クロス / [red]DEATH[/red]=下落クロス
[cyan]スコア[/cyan]     −100(強SELL) ～ +100(強BUY) , ±30以上でシグナル点灯"""


# ── Main render ────────────────────────────────────────────────────────────────

def render_report(all_results: dict[str, list[dict]]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    console.print()
    console.print(Rule(f"[bold white] マルチアセット売買指標レポート  {now} [/bold white]",
                       style="bright_blue"))
    console.print()

    for cls, results in all_results.items():
        table = build_class_table(cls, results)
        console.print(table)
        console.print()

    console.print(build_summary_panel(all_results))
    console.print()
    console.print(Panel(LEGEND, border_style="dim", expand=False))
    console.print()
