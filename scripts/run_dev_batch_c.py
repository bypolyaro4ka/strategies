"""Этап 6: прогон V0 (дефолты) партии C (S10, S11) на dev 2023-2025.

Та же схема, что и run_dev_batch_a.py (см. его докстринг) - копия структуры, не общий
код, чтобы каждую партию можно было прогнать и почитать независимо.

Запуск: python scripts/run_dev_batch_c.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from lab.benchmarks.buy_hold import BuyHoldStrategy, run_buy_hold_portfolio
from lab.benchmarks.flat import FlatStrategy
from lab.benchmarks.random_entry import random_benchmark_sharpes, trade_profile
from lab.config import load_protocol
from lab.data.loader import load_bars, load_funding
from lab.engine.backtest import run_backtest
from lab.metrics import performance as perf
from lab.metrics.dsr import deflated_sharpe_ratio
from lab.registry import log_run
from lab.report.make_report import make_report
from lab.strategies.base import Context
from lab.strategies.s10_rsi2 import S10
from lab.strategies.s11_bb_adx import S11

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_START, DEV_END = "2023-01-01", "2025-12-31"
RUN_RANDOM_BENCHMARK = False  # см. решение в run_dev_batch_a.py / JOURNAL - не нужно для
# санити-чека партии, отдельный прогон позже.
N_RANDOM_SEEDS = 30

STRATEGIES = [S10(), S11()]


def load_universe():
    with (PROJECT_ROOT / "config" / "universe.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [c["binance_symbol"] for c in cfg["pool"]]


def load_symbol_data(symbol: str, tf: str, warmup_bars: int):
    bars_1h = load_bars(symbol, "1h", "2022-01-01", DEV_END)
    bars_tf = load_bars(symbol, tf, "2022-01-01", DEV_END)
    funding = load_funding(symbol, "2022-01-01", DEV_END)
    return bars_1h, bars_tf, funding


def run_one(strategy, symbols, cfg):
    tf = strategy.timeframes[0]
    warmup_days = strategy.required_history(tf)
    print(f"\n=== {strategy.id} ({strategy.name}), tf={tf}, warmup={warmup_days} ===")

    bars_1h_by_symbol, bars_tf_by_symbol, funding_by_symbol, prepared_by_symbol = {}, {}, {}, {}
    for s in symbols:
        try:
            b1h, btf, fnd = load_symbol_data(s, tf, warmup_days)
        except FileNotFoundError:
            print(f"  {s}: нет данных, пропуск")
            continue
        if len(btf) < warmup_days + 5:
            print(f"  {s}: истории мало ({len(btf)} баров), пропуск")
            continue
        bars_1h_by_symbol[s] = b1h
        bars_tf_by_symbol[s] = btf
        funding_by_symbol[s] = fnd
        ctx = Context(params=strategy.params, protocol=cfg)
        prepared_by_symbol[s] = strategy.prepare(btf, ctx)

    dev_start_ts = __import__("pandas").Timestamp(DEV_START, tz="UTC")

    # --- портфель (основной режим) ---
    max_slots = cfg.account.max_slots_portfolio
    portfolio_result = run_backtest(
        strategy, bars_1h_by_symbol, prepared_by_symbol, funding_by_symbol, tf, cfg, 10_000, max_slots,
    )
    portfolio_result.equity = portfolio_result.equity.loc[dev_start_ts:]

    flat_result = run_backtest(
        FlatStrategy(), bars_1h_by_symbol, {s: b for s, b in bars_tf_by_symbol.items()},
        funding_by_symbol, tf, cfg, 10_000, max_slots,
    )
    flat_result.equity = flat_result.equity.loc[dev_start_ts:]
    bh_result = run_buy_hold_portfolio(bars_1h_by_symbol, cfg, 10_000)
    bh_result.equity = bh_result.equity.loc[dev_start_ts:]

    profile = trade_profile(portfolio_result.trades)
    random_sharpes = None
    if RUN_RANDOM_BENCHMARK and profile["n_trades"] >= 3:
        tf_hours = {"1d": 24, "12h": 12, "4h": 4, "1h": 1}[tf]
        avg_holding_bars = profile["avg_holding_hours"] / tf_hours
        random_sharpes = random_benchmark_sharpes(
            bars_1h_by_symbol, bars_tf_by_symbol, funding_by_symbol, tf, cfg, 10_000, max_slots,
            n_trades=profile["n_trades"], avg_holding_bars=avg_holding_bars,
            long_fraction=profile["long_fraction"], n_seeds=N_RANDOM_SEEDS,
        )

    n_trials = 1  # только V0 в этом прогоне; полный счёт испытаний - после Этапа 7 (walk-forward)
    out_dir = PROJECT_ROOT / "reports" / "dev" / strategy.id
    report_path = make_report(
        title=f"{strategy.id} — {strategy.name} (V0, портфель, dev {DEV_START}..{DEV_END})",
        strategy_result=portfolio_result,
        protocol_cfg=cfg,
        out_dir=out_dir,
        benchmark_results={"Flat": flat_result, "Buy & Hold": bh_result},
        random_sharpes=random_sharpes,
        n_trials=n_trials,
    )

    metrics = perf.compute_all(portfolio_result.equity, portfolio_result.trades, cfg.metrics.annualization_days)
    log_run(
        {
            "run_id": str(uuid.uuid4())[:8], "timestamp_utc": __import__("pandas").Timestamp.now('UTC').isoformat(),
            "strategy_id": strategy.id, "strategy_version": strategy.version, "variant": "V0",
            "mode": "portfolio", "tf": tf, "symbols": ";".join(bars_1h_by_symbol.keys()),
            "period_start": DEV_START, "period_end": DEV_END, "stage": "dev", "debug": "False",
            "artifacts_path": str(out_dir.relative_to(PROJECT_ROOT)),
        },
        metrics,
    )
    print(f"  портфель: Sharpe={metrics['sharpe']:.2f} return={metrics['total_return']*100:.1f}% "
          f"trades={metrics['trades']} -> {report_path}")

    # --- пара (диагностика, компактно - без отдельных report.md на каждую монету) ---
    pair_rows = []
    for s in bars_1h_by_symbol:
        strat_pair = type(strategy)(params=strategy.params)
        ctx = Context(params=strat_pair.params, protocol=cfg)
        prepared = strat_pair.prepare(bars_tf_by_symbol[s], ctx)
        r = run_backtest(strat_pair, {s: bars_1h_by_symbol[s]}, {s: prepared}, {s: funding_by_symbol[s]}, tf, cfg, 10_000, 1)
        r.equity = r.equity.loc[dev_start_ts:]
        m = perf.compute_all(r.equity, r.trades, cfg.metrics.annualization_days)
        pair_rows.append((s, m))
        log_run(
            {
                "run_id": str(uuid.uuid4())[:8], "timestamp_utc": __import__("pandas").Timestamp.now('UTC').isoformat(),
                "strategy_id": strategy.id, "strategy_version": strategy.version, "variant": "V0",
                "mode": "pair", "tf": tf, "symbols": s, "period_start": DEV_START, "period_end": DEV_END,
                "stage": "dev", "debug": "False",
            },
            m,
        )
    print(f"  пара: {len(pair_rows)} монет прогнано")
    return metrics, pair_rows, report_path


def main():
    cfg = load_protocol()
    symbols = load_universe()
    print(f"Монеты: {symbols}")

    summary = []
    t0 = time.time()
    for strategy in STRATEGIES:
        portfolio_metrics, pair_rows, report_path = run_one(strategy, symbols, cfg)
        summary.append((strategy, portfolio_metrics, pair_rows, report_path))

    lines = ["# Партия C (возврат к среднему) — V0 на dev 2023-01-01..2025-12-31\n",
             "Режим «портфель» (основной для рейтинга) + «пара» по каждой монете (диагностика).\n"]
    lines.append("## Портфель (V0, дефолты)\n")
    lines.append("| ID | Sharpe | Return | MaxDD | Trades | Cost share | Random pct | Отчёт |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for strategy, m, _, report_path in summary:
        rel = report_path.relative_to(PROJECT_ROOT / "reports" / "dev")
        rp = m.get("random_pct", float("nan"))
        lines.append(
            f"| {strategy.id} | {m['sharpe']:.2f} | {m['total_return']*100:.1f}% | {m['maxdd']*100:.1f}% | "
            f"{m['trades']} | {m['cost_share']*100:.1f}% | "
            f"{'—' if rp != rp else f'{rp:.0f}'} | [{rel}]({rel}) |"
        )

    lines.append("\n## Пара — Sharpe по монетам\n")
    lines.append("| ID | " + " | ".join(symbols) + " |")
    lines.append("|---|" + "---|" * len(symbols))
    for strategy, _, pair_rows, _ in summary:
        by_symbol = dict(pair_rows)
        cells = [f"{by_symbol[s]['sharpe']:.2f}" if s in by_symbol and by_symbol[s]['sharpe'] == by_symbol[s]['sharpe'] else "—" for s in symbols]
        lines.append(f"| {strategy.id} | " + " | ".join(cells) + " |")

    out_path = PROJECT_ROOT / "reports" / "dev" / "batch_C.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nГотово за {time.time()-t0:.0f}с. Сводка: {out_path}")


if __name__ == "__main__":
    main()
