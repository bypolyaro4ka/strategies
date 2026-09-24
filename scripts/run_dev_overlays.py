"""Этап 6, надстройки: O1 (фильтр режима BTC) на S01b/S02/S06, O2 (таргетирование
волатильности) на S02/S03 - список пар фиксирован в 03_STRATEGIES.md, дефолтные
параметры (O1: sma_days=50, O2: sigma_star=0.5, W=90). Только портфельный режим (та же
логика, что и у обычных партий, но здесь варианты уже протестированных стратегий, не
новые правила - отдельный прогон "пара" не добавляет новой информации).

ID варианта = "S02+BtcRegimeFilter" и т.п. (WithOverlays сам их так называет) -
соответствует формату `S02+O1` из 03_STRATEGIES.md по смыслу, не по буквальному тексту.

Запуск: python scripts/run_dev_overlays.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from lab.benchmarks.buy_hold import run_buy_hold_portfolio
from lab.benchmarks.flat import FlatStrategy
from lab.config import load_protocol
from lab.data.loader import load_bars, load_funding
from lab.engine.backtest import run_backtest
from lab.metrics import performance as perf
from lab.registry import log_run
from lab.report.make_report import make_report
from lab.strategies.base import Context, WithOverlays
from lab.strategies.overlays import BtcRegimeFilter, VolTarget
from lab.strategies.s01_donchian import S01b
from lab.strategies.s02_tsmom import S02
from lab.strategies.s03_sma import S03
from lab.strategies.s06_supertrend import S06

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_START, DEV_END = "2023-01-01", "2025-12-31"

COMBOS = [
    WithOverlays(S01b(), [BtcRegimeFilter(sma_days=50)]),
    WithOverlays(S02(), [BtcRegimeFilter(sma_days=50)]),
    WithOverlays(S06(), [BtcRegimeFilter(sma_days=50)]),
    WithOverlays(S02(), [VolTarget(sigma_star=0.5, W=90)]),
    WithOverlays(S03(), [VolTarget(sigma_star=0.5, W=90)]),
]


def load_universe():
    with (PROJECT_ROOT / "config" / "universe.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [c["binance_symbol"] for c in cfg["pool"]]


def run_one(strategy, symbols, cfg, btc_bars):
    tf = strategy.timeframes[0]
    warmup = strategy.required_history(tf)
    print(f"\n=== {strategy.id}, tf={tf}, warmup={warmup} ===")

    bars_1h_by_symbol, prepared_by_symbol, funding_by_symbol = {}, {}, {}
    for s in symbols:
        try:
            b1h = load_bars(s, "1h", "2022-01-01", DEV_END)
            btf = load_bars(s, tf, "2022-01-01", DEV_END)
            fnd = load_funding(s, "2022-01-01", DEV_END)
        except FileNotFoundError:
            print(f"  {s}: нет данных, пропуск")
            continue
        if len(btf) < warmup + 5:
            print(f"  {s}: истории мало ({len(btf)} баров), пропуск")
            continue
        bars_1h_by_symbol[s] = b1h
        funding_by_symbol[s] = fnd
        ctx = Context(params=strategy.strategy.params, protocol=cfg, btc_bars=btc_bars)
        prepared_by_symbol[s] = strategy.prepare(btf, ctx)

    dev_start_ts = pd.Timestamp(DEV_START, tz="UTC")
    max_slots = cfg.account.max_slots_portfolio
    result = run_backtest(
        strategy, bars_1h_by_symbol, prepared_by_symbol, funding_by_symbol, tf, cfg, 10_000, max_slots,
    )
    result.equity = result.equity.loc[dev_start_ts:]

    flat_result = run_backtest(
        FlatStrategy(), bars_1h_by_symbol, prepared_by_symbol, funding_by_symbol, tf, cfg, 10_000, max_slots,
    )
    flat_result.equity = flat_result.equity.loc[dev_start_ts:]
    bh_result = run_buy_hold_portfolio(bars_1h_by_symbol, cfg, 10_000)
    bh_result.equity = bh_result.equity.loc[dev_start_ts:]

    out_dir = PROJECT_ROOT / "reports" / "dev" / strategy.id.replace("+", "_")
    report_path = make_report(
        title=f"{strategy.id} (V0, портфель, dev {DEV_START}..{DEV_END})",
        strategy_result=result,
        protocol_cfg=cfg,
        out_dir=out_dir,
        benchmark_results={"Flat": flat_result, "Buy & Hold": bh_result},
        random_sharpes=None,
        n_trials=1,
    )

    metrics = perf.compute_all(result.equity, result.trades, cfg.metrics.annualization_days)
    log_run(
        {
            "run_id": str(uuid.uuid4())[:8], "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
            "strategy_id": strategy.id, "strategy_version": strategy.version, "variant": "V0+overlay",
            "mode": "portfolio", "tf": tf, "symbols": ";".join(bars_1h_by_symbol.keys()),
            "period_start": DEV_START, "period_end": DEV_END, "stage": "dev", "debug": "False",
            "artifacts_path": str(out_dir.relative_to(PROJECT_ROOT)),
        },
        metrics,
    )
    print(f"  портфель: Sharpe={metrics['sharpe']:.2f} return={metrics['total_return']*100:.1f}% "
          f"trades={metrics['trades']} -> {report_path}")
    return metrics, report_path


def main():
    cfg = load_protocol()
    symbols = load_universe()
    print(f"Монеты: {symbols}")
    btc_bars = load_bars("BTCUSDT", "1d", "2022-01-01", DEV_END)

    summary = []
    t0 = time.time()
    for strategy in COMBOS:
        metrics, report_path = run_one(strategy, symbols, cfg, btc_bars)
        summary.append((strategy, metrics, report_path))

    lines = ["# Надстройки O1/O2 — V0 на dev 2023-01-01..2025-12-31\n",
             "Портфельный режим. Дефолты надстроек: O1 sma_days=50, O2 sigma_star=0.5, W=90.\n"]
    lines.append("| Вариант | Sharpe | Return | MaxDD | Trades | Cost share | Отчёт |")
    lines.append("|---|---|---|---|---|---|---|")
    for strategy, m, report_path in summary:
        rel = report_path.relative_to(PROJECT_ROOT / "reports" / "dev")
        lines.append(
            f"| {strategy.id} | {m['sharpe']:.2f} | {m['total_return']*100:.1f}% | {m['maxdd']*100:.1f}% | "
            f"{m['trades']} | {m['cost_share']*100:.1f}% | [{rel}]({rel}) |"
        )
    out_path = PROJECT_ROOT / "reports" / "dev" / "overlays.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nГотово за {time.time()-t0:.0f}с. Сводка: {out_path}")


if __name__ == "__main__":
    main()
