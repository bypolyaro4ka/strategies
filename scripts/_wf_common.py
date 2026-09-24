"""Общий код для run_wf_<ID>.py - загрузка пула, запуск walk-forward, печать, отчёт.
Не тестируется отдельно юнит-тестами (это скрипт-обвязка, не библиотечный код) -
логика, которую стоит проверять, живёт в lab/optimize/walk_forward.py и уже покрыта.

Во всех драйверах ТФ сетки зафиксирован на дефолтном ТФ стратегии (см. JOURNAL,
запись про run_wf_s02.py) - поиск по ТФ не входит в первый проход Этапа 7."""

from __future__ import annotations

from pathlib import Path

import yaml

from lab.config import load_protocol
from lab.data.loader import load_bars, load_funding
from lab.optimize.walk_forward import run_walk_forward
from lab.registry import log_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_universe() -> list[str]:
    with (PROJECT_ROOT / "config" / "universe.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [c["binance_symbol"] for c in cfg["pool"]]


def load_pool(symbols, tf, warmup, dev_end):
    bars_1h, bars_tf, funding = {}, {}, {}
    for s in symbols:
        try:
            b1h = load_bars(s, "1h", "2022-01-01", dev_end)
            btf = load_bars(s, tf, "2022-01-01", dev_end)
            fnd = load_funding(s, "2022-01-01", dev_end)
        except FileNotFoundError:
            print(f"  {s}: нет данных, пропуск")
            continue
        if len(btf) < warmup + 5:
            print(f"  {s}: истории мало ({len(btf)} баров), пропуск")
            continue
        bars_1h[s], bars_tf[s], funding[s] = b1h, btf, fnd
    return bars_1h, bars_tf, funding


def prepare_fn_base(strategy, bars_by_symbol, ctx):
    return {s: strategy.prepare(b, ctx) for s, b in bars_by_symbol.items()}


def prepare_fn_portfolio(strategy, bars_by_symbol, ctx):
    """Для PortfolioStrategy (S13) - один вызов prepare() со всем словарём сразу,
    не цикл по монетам, как у BaseStrategy."""
    return strategy.prepare(bars_by_symbol, ctx)


def make_prepare_fn_with_funding(funding_by_symbol):
    """Для стратегий вроде S12, которым нужен ctx.funding - в отличие от btc_bars,
    фандинг свой у каждой монеты, поэтому единого ctx на весь пул (как для btc_bars)
    недостаточно: строим свой Context на каждый символ внутри prepare_fn."""
    from lab.strategies.base import Context

    def prepare_fn(strategy, bars_by_symbol, ctx):
        out = {}
        for s, b in bars_by_symbol.items():
            per_symbol_ctx = Context(params=ctx.params, protocol=ctx.protocol, btc_bars=ctx.btc_bars,
                                      funding=funding_by_symbol.get(s))
            out[s] = strategy.prepare(b, per_symbol_ctx)
        return out

    return prepare_fn


def run_and_report(
    strategy_id, strategy_cls, tf, default_params, grid_points, ordinal_grids,
    categorical_params=(), apply_coin_selection=True, use_btc_filter=False,
    fixed_extra_params=None, prepare_fn_factory=None, prepare_fn=None,
):
    """fixed_extra_params - параметры, зафиксированные у каждой точки сетки, но не
    входящие в сам перебор (обычно tf). prepare_fn_factory(funding_by_symbol) -> prepare_fn
    - для стратегий, которым нужен фандинг (см. make_prepare_fn_with_funding).
    prepare_fn - готовая функция напрямую (напр. prepare_fn_portfolio для S13/
    PortfolioStrategy), приоритет выше factory. По умолчанию - prepare_fn_base."""
    cfg = load_protocol()
    symbols = load_universe()
    print(f"Монеты: {symbols}")

    fixed_extra_params = fixed_extra_params or {}

    def make_strategy(params):
        return strategy_cls(params={**params, **fixed_extra_params})

    probe = make_strategy(default_params)
    warmup = probe.required_history(tf)
    bars_1h, bars_tf, funding = load_pool(symbols, tf, warmup, cfg.periods.dev_end)
    print(f"Точек сетки: {len(grid_points)}, монет: {len(bars_1h)}, фолдов: {len(cfg.folds)}")

    resolved_prepare_fn = prepare_fn or (prepare_fn_factory(funding) if prepare_fn_factory else prepare_fn_base)

    btc_bars = None
    if use_btc_filter:
        btc_bars = load_bars(cfg.btc_filter.symbol, "1d", "2022-01-01", cfg.periods.dev_end)

    def on_trial(run_meta, metrics):
        log_run({**run_meta, "strategy_version": probe.version}, metrics)

    wf = run_walk_forward(
        strategy_id=strategy_id, make_strategy=make_strategy, prepare_fn=resolved_prepare_fn,
        default_params=default_params, param_grid_points=grid_points,
        ordinal_grids=ordinal_grids, categorical_params=categorical_params,
        bars_1h_by_symbol=bars_1h, bars_tf_by_symbol=bars_tf, funding_by_symbol=funding,
        tf=tf, cfg=cfg, symbols_pool=list(bars_1h.keys()), btc_bars=btc_bars,
        apply_coin_selection=apply_coin_selection, on_trial=on_trial,
    )

    print("\n=== Итог по фолдам ===")
    for f in wf.folds:
        coins = "все" if not f.coin_selection or not f.coin_selection.applied else len(f.coin_selection.symbols)
        print(f"  {f.fold_id}: V1 params={f.v1_plateau.params} (score={f.v1_plateau.score:.3f}) "
              f"| test Sharpe V0={f.v0_test['sharpe']:.2f} V1={f.v1_test['sharpe']:.2f} V2={f.v2_test['sharpe']:.2f} "
              f"| монет V2={coins}")

    print("\n=== WF-OOS (склеенные test-периоды F1-F4) ===")
    for variant in ("V0", "V1", "V2"):
        m = wf.oos_metrics[variant]
        print(f"  {variant}: Sharpe={m['sharpe']:.3f} return={m['total_return']*100:.1f}% "
              f"trades={m['trades']} folds_better_than_v0={wf.folds_better_than_v0.get(variant, '-')}")
    print(f"\nОсновной вариант (04_OPTIMIZATION.md §6): {wf.primary_variant}")

    out_dir = PROJECT_ROOT / "reports" / "wf"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {strategy_id} — walk-forward (Этап 7)\n", f"ТФ зафиксирован на {tf} в этом прогоне.\n"]
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
    (out_dir / f"{strategy_id}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт: {out_dir / f'{strategy_id}.md'}")
    return wf
