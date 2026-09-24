"""Этап 7: walk-forward для S02 (tsmom) — первый прогон, обкатывает
optimize/walk_forward.py целиком (train-сетка, плато, отбор монет, test "с нуля",
склейка WF-OOS, выбор основного варианта, отчёт, prereg).

УПРОЩЕНИЕ первого прохода (см. JOURNAL): сетка НЕ ищет по `tf` (у S02 в 03_STRATEGIES.md
tf тоже часть сетки: {1d, 12h}) - тестируется только дефолтный tf=1d. Разные tf требуют
разных bars_tf и по сути отдельного под-прогона walk-forward на каждый tf - решили
сначала проверить сам движок на параметрах, ТФ-поиск - отдельным заходом, если
понадобится, а не блокировать весь Этап 7 на этом сразу для всех 13 стратегий.
lookback_days трактуется как порядковый параметр в том порядке, в котором перечислен
в 03_STRATEGIES.md (соседи - по позиции в списке, не по числовому смыслу кортежа).

Запуск: python scripts/run_wf_s02.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from lab.config import load_protocol
from lab.data.loader import load_bars, load_funding
from lab.optimize.walk_forward import run_walk_forward
from lab.registry import log_run
from lab.strategies.base import Context
from lab.strategies.s02_tsmom import S02

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TF = "1d"  # фиксирован в этом прогоне - см. докстринг

LOOKBACK_GRID = [(3, 7, 14), (7, 14, 28), (14, 28, 56), (7,), (14,), (28,)]
REBALANCE_GRID = ["every_bar", "weekly"]


def load_universe():
    with (PROJECT_ROOT / "config" / "universe.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [c["binance_symbol"] for c in cfg["pool"]]


def make_strategy(params: dict) -> S02:
    return S02(params={**params, "tf": TF})


def prepare_fn(strategy, bars_by_symbol, ctx):
    return {s: strategy.prepare(b, ctx) for s, b in bars_by_symbol.items()}


def main():
    cfg = load_protocol()
    symbols = load_universe()
    print(f"Монеты: {symbols}")

    strategy = S02()
    warmup = strategy.required_history(TF)
    bars_1h, bars_tf, funding = {}, {}, {}
    for s in symbols:
        try:
            b1h = load_bars(s, "1h", "2022-01-01", cfg.periods.dev_end)
            btf = load_bars(s, TF, "2022-01-01", cfg.periods.dev_end)
            fnd = load_funding(s, "2022-01-01", cfg.periods.dev_end)
        except FileNotFoundError:
            print(f"  {s}: нет данных, пропуск")
            continue
        if len(btf) < warmup + 5:
            print(f"  {s}: истории мало ({len(btf)} баров), пропуск")
            continue
        bars_1h[s], bars_tf[s], funding[s] = b1h, btf, fnd

    param_grid_points = [
        {"lookback_days": lb, "rebalance": rb}
        for lb, rb in product(LOOKBACK_GRID, REBALANCE_GRID)
    ]
    print(f"Точек сетки: {len(param_grid_points)}, монет: {len(bars_1h)}, фолдов: {len(cfg.folds)}")

    def on_trial(run_meta, metrics):
        log_run({**run_meta, "strategy_version": strategy.version}, metrics)

    wf = run_walk_forward(
        strategy_id="S02",
        make_strategy=make_strategy,
        prepare_fn=prepare_fn,
        default_params={"lookback_days": (7, 14, 28), "rebalance": "every_bar"},
        param_grid_points=param_grid_points,
        ordinal_grids={"lookback_days": LOOKBACK_GRID},
        categorical_params=("rebalance",),
        bars_1h_by_symbol=bars_1h, bars_tf_by_symbol=bars_tf, funding_by_symbol=funding,
        tf=TF, cfg=cfg, symbols_pool=list(bars_1h.keys()),
        on_trial=on_trial,
    )

    print("\n=== Итог по фолдам ===")
    for f in wf.folds:
        print(f"  {f.fold_id}: V1 params={f.v1_plateau.params} (score={f.v1_plateau.score:.3f}) "
              f"| test Sharpe V0={f.v0_test['sharpe']:.2f} V1={f.v1_test['sharpe']:.2f} V2={f.v2_test['sharpe']:.2f} "
              f"| монет V2={'все' if not f.coin_selection or not f.coin_selection.applied else len(f.coin_selection.symbols)}")

    print("\n=== WF-OOS (склеенные test-периоды F1-F4) ===")
    for variant in ("V0", "V1", "V2"):
        m = wf.oos_metrics[variant]
        print(f"  {variant}: Sharpe={m['sharpe']:.3f} return={m['total_return']*100:.1f}% "
              f"trades={m['trades']} folds_better_than_v0={wf.folds_better_than_v0.get(variant, '-')}")
    print(f"\nОсновной вариант (04_OPTIMIZATION.md §6): {wf.primary_variant}")

    out_dir = PROJECT_ROOT / "reports" / "wf"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# S02 — walk-forward (Этап 7)\n", f"ТФ зафиксирован на {TF} в этом прогоне (см. докстринг скрипта).\n"]
    lines.append("## По фолдам\n")
    lines.append("| Фолд | V1 params | plateau score | V0 test Sharpe | V1 test Sharpe | V2 test Sharpe | Монет V2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for f in wf.folds:
        coins = "все" if not f.coin_selection or not f.coin_selection.applied else str(len(f.coin_selection.symbols))
        lines.append(f"| {f.fold_id} | {f.v1_plateau.params} | {f.v1_plateau.score:.3f} | "
                      f"{f.v0_test['sharpe']:.2f} | {f.v1_test['sharpe']:.2f} | {f.v2_test['sharpe']:.2f} | {coins} |")
    lines.append("\n## WF-OOS (склеенные test-периоды)\n")
    lines.append("| Вариант | Sharpe | Return | Trades | Фолдов лучше V0 |")
    lines.append("|---|---|---|---|---|")
    for variant in ("V0", "V1", "V2"):
        m = wf.oos_metrics[variant]
        lines.append(f"| {variant} | {m['sharpe']:.3f} | {m['total_return']*100:.1f}% | {m['trades']} | "
                      f"{wf.folds_better_than_v0.get(variant, '-')} |")
    lines.append(f"\n**Основной вариант: {wf.primary_variant}**\n")
    (out_dir / "S02.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт: {out_dir / 'S02.md'}")


if __name__ == "__main__":
    main()
