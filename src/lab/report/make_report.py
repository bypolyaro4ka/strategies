"""Markdown-отчёт с графиком equity (01_PROTOCOL.md, раздел 11)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без экрана - сохраняем сразу в файл, не пытаемся открыть окно
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.metrics import performance as perf
from lab.metrics.dsr import deflated_sharpe_ratio

STAT_CAVEAT = (
    "**Статистическая оговорка.** Окно ~9 месяцев короткое: стандартная ошибка годового "
    "Шарпа ≈ √((1 + SR²/2) / T), при T ≈ 0.73 это около ±1.2. Шарп 1 на таком окне почти "
    "неотличим от нуля (01_PROTOCOL.md, раздел 8)."
)


def _plot_equity(equity: pd.Series, benchmarks: dict[str, pd.Series], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Стратегия", linewidth=1.5)
    for name, bench_equity in benchmarks.items():
        ax.plot(bench_equity.index, bench_equity.values, label=name, linewidth=1, alpha=0.7)
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _fmt(v: float, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def make_report(
    *,
    title: str,
    strategy_result,
    protocol_cfg,
    out_dir: Path,
    benchmark_results: dict[str, object] | None = None,
    random_sharpes: np.ndarray | None = None,
    n_trials: int | None = None,
    pnl_by_symbol: dict[str, float] | None = None,
    verdict_note: str = "",
) -> Path:
    """strategy_result / значения benchmark_results — BacktestResult (equity + trades).
    Пишет report.md и equity.png в out_dir, возвращает путь к .md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_results = benchmark_results or {}
    ann_days = protocol_cfg.metrics.annualization_days

    metrics = perf.compute_all(
        strategy_result.equity, strategy_result.trades, ann_days,
        protocol_cfg.metrics.risk_free, pnl_by_symbol,
    )

    daily = perf.daily_returns(strategy_result.equity)
    if n_trials:
        metrics["dsr"] = deflated_sharpe_ratio(daily, n_trials)

    if random_sharpes is not None and len(random_sharpes) > 0:
        metrics["random_pct"] = perf.percentile_rank(metrics["sharpe"], random_sharpes)

    _plot_equity(
        strategy_result.equity, {name: r.equity for name, r in benchmark_results.items()},
        out_dir / "equity.png",
    )

    lines = [f"# {title}\n"]
    lines.append(f"Прогонов учтено (для DSR): {n_trials if n_trials else '—'}\n")
    lines.append("![equity](equity.png)\n")

    lines.append("## Метрики\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    rows = [
        ("Total return", _fmt(metrics["total_return"], pct=True)),
        ("CAGR", _fmt(metrics["cagr"], pct=True)),
        ("Vol", _fmt(metrics["vol"], pct=True)),
        ("Sharpe", _fmt(metrics["sharpe"])),
        ("Sortino", _fmt(metrics["sortino"])),
        ("MaxDD", _fmt(metrics["maxdd"], pct=True)),
        ("Calmar", _fmt(metrics["calmar"])),
        ("Trades", str(metrics["trades"])),
        ("Win rate", _fmt(metrics["win_rate"], pct=True)),
        ("Profit factor", _fmt(metrics["profit_factor"])),
        ("Avg trade gross, б.п.", _fmt(metrics["avg_trade_gross_bp"], digits=1)),
        ("Avg trade net, б.п.", _fmt(metrics["avg_trade_net_bp"], digits=1)),
        ("Cost share", _fmt(metrics["cost_share"], pct=True)),
        ("Exposure", _fmt(metrics["exposure"], pct=True)),
        ("Funding paid", _fmt(metrics["funding_paid"], digits=2)),
        ("DSR", _fmt(metrics.get("dsr", float("nan")))),
        ("Random percentile", _fmt(metrics.get("random_pct", float("nan")), digits=1)),
    ]
    if "breadth" in metrics:
        rows.append(("Breadth", _fmt(metrics["breadth"], pct=True)))
    for name, value in rows:
        lines.append(f"| {name} | {value} |")

    if benchmark_results:
        lines.append("\n## Бенчмарки\n")
        lines.append("| Бенчмарк | Total return | Sharpe | MaxDD |")
        lines.append("|---|---|---|---|")
        for name, r in benchmark_results.items():
            b_daily = perf.daily_returns(r.equity)
            lines.append(
                f"| {name} | {_fmt(perf.total_return(r.equity), pct=True)} | "
                f"{_fmt(perf.sharpe(b_daily, ann_days))} | {_fmt(perf.max_drawdown(r.equity), pct=True)} |"
            )

    monthly = perf.monthly_returns(strategy_result.equity)
    if len(monthly) > 0:
        lines.append("\n## Помесячная доходность\n")
        lines.append("| Месяц | Доходность |")
        lines.append("|---|---|")
        for idx, v in monthly.items():
            lines.append(f"| {idx:%Y-%m} | {_fmt(v, pct=True)} |")

    lines.append(f"\n{STAT_CAVEAT}\n")

    if verdict_note:
        lines.append(f"\n## Вывод\n\n{verdict_note}\n")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
