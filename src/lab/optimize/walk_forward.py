"""Walk-forward оптимизация по 04_OPTIMIZATION.md §4-§8. Один и тот же код гоняет любую
стратегию (BaseStrategy или PortfolioStrategy) — специфика инкапсулирована в `prepare_fn`,
которую передаёт вызывающий скрипт (`scripts/run_wf_<ID>.py`)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from lab.engine.backtest import run_backtest
from lab.metrics import performance as perf
from lab.optimize.selection import CoinSelectionResult, PlateauResult, plateau_select, select_coins
from lab.strategies.base import Context


def _params_key(params: dict) -> tuple:
    return tuple(sorted(params.items(), key=lambda kv: kv[0]))


def evaluate_window(
    strategy, bars_1h: dict, signal_bars: dict, funding: dict, tf: str, cfg,
    window_start, window_end, max_slots: int, btc_bars=None, cold_start: bool = False,
):
    """Прогон на произвольном окне [window_start, window_end]. Бары обрезаются по
    window_end (стратегия не может видеть данные за пределами окна), но НЕ по
    window_start снизу — прогрев индикаторов берётся из предшествующих данных (прошлое,
    не подглядывание), только у самого события ещё не могло быть сигнала в этот момент.

    cold_start=True — test-окно walk-forward (04_OPTIMIZATION.md §4, шаг 4): позиция
    должна быть гарантированно плоской на window_start, даже если бы стратегия
    "теоретически" открыла позицию раньше на тех же данных — движок это обеспечивает
    через `signals_start` (см. engine/backtest.py)."""
    ws = pd.Timestamp(window_start, tz="UTC") if pd.Timestamp(window_start).tzinfo is None else pd.Timestamp(window_start)
    we = pd.Timestamp(window_end, tz="UTC") if pd.Timestamp(window_end).tzinfo is None else pd.Timestamp(window_end)

    bars_1h_w = {s: b.loc[:we] for s, b in bars_1h.items()}
    signal_bars_w = {s: b.loc[:we] for s, b in signal_bars.items()}
    funding_w = {s: f.loc[:we] for s, f in funding.items()}

    result = run_backtest(
        strategy, bars_1h_w, signal_bars_w, funding_w, tf, cfg, cfg.account.initial_equity,
        max_slots, btc_bars=btc_bars, signals_start=ws if cold_start else None,
    )
    result.equity = result.equity.loc[ws:we]
    return result


def _metrics(result, cfg) -> dict:
    return perf.compute_all(result.equity, result.trades, cfg.metrics.annualization_days, cfg.metrics.risk_free)


@dataclass
class FoldResult:
    fold_id: str
    train_grid: dict[tuple, dict]  # params_key -> train metrics, для тепловых карт в отчёте
    v1_plateau: PlateauResult
    coin_selection: CoinSelectionResult | None
    v0_test: dict
    v1_test: dict
    v2_test: dict


@dataclass
class WalkForwardResult:
    strategy_id: str
    folds: list[FoldResult] = field(default_factory=list)
    oos_equity: dict[str, pd.Series] = field(default_factory=dict)  # "V0"/"V1"/"V2" -> склеенная кривая
    oos_metrics: dict[str, dict] = field(default_factory=dict)  # метрики по стыкованной кривой + сделки всех фолдов
    folds_better_than_v0: dict[str, int] = field(default_factory=dict)
    primary_variant: str = "V0"


def run_walk_forward(
    strategy_id: str,
    make_strategy,  # Callable[[dict], strategy]
    prepare_fn,  # Callable[[strategy, dict[symbol, DataFrame], Context], dict[symbol, DataFrame]]
    default_params: dict,
    param_grid_points: list[dict],
    ordinal_grids: dict[str, list],
    categorical_params: tuple[str, ...],
    bars_1h_by_symbol: dict[str, pd.DataFrame],
    bars_tf_by_symbol: dict[str, pd.DataFrame],
    funding_by_symbol: dict[str, pd.DataFrame],
    tf: str,
    cfg,
    symbols_pool: list[str],
    btc_bars=None,
    apply_coin_selection: bool = True,
    on_trial=None,  # Callable[[dict_run_meta, dict_metrics], None] - для реестра (stage=wf)
) -> WalkForwardResult:
    max_slots = cfg.account.max_slots_portfolio
    wf = WalkForwardResult(strategy_id=strategy_id)
    test_equities = {"V0": [], "V1": [], "V2": []}
    test_trades = {"V0": [], "V1": [], "V2": []}

    def log(variant, mode, fold_id, window, symbols, params, metrics):
        if on_trial is None:
            return
        on_trial(
            {
                "strategy_id": strategy_id, "variant": variant, "mode": mode, "tf": tf,
                "symbols": ";".join(symbols), "period_start": str(window[0]), "period_end": str(window[1]),
                "fold": fold_id, "stage": "wf", "debug": "False", "params_json": str(params),
            },
            metrics,
        )

    # prepare() для каждой точки сетки не зависит от фолда (только от параметров) -
    # считаем один раз на весь walk-forward, не по 4 раза на каждый фолд. evaluate_window
    # обрезает результат по train_end/test_end каждый раз заново, не мутируя кэш.
    grid_prepared_cache: dict[tuple, dict] = {}
    for params in param_grid_points:
        strat = make_strategy(params)
        ctx = Context(params=params, protocol=cfg, btc_bars=btc_bars)
        grid_prepared_cache[_params_key(params)] = prepare_fn(strat, bars_tf_by_symbol, ctx)

    for fold in cfg.folds:
        train_start, train_end = fold.train
        test_start, test_end = fold.test

        # --- 1. train: сетка на всём пуле, портфельный режим ---
        train_grid: dict[tuple, dict] = {}
        for params in param_grid_points:
            key = _params_key(params)
            strat = make_strategy(params)
            result = evaluate_window(
                strat, bars_1h_by_symbol, grid_prepared_cache[key], funding_by_symbol, tf, cfg,
                train_start, train_end, max_slots, btc_bars, cold_start=False,
            )
            m = _metrics(result, cfg)
            train_grid[key] = m
            log("V1_grid", "portfolio", fold.id, (train_start, train_end), symbols_pool, params, m)

        # --- 2. выбор V1 по правилу плато, только среди точек с trades >= порога ---
        eligible = [p for p in param_grid_points if train_grid[_params_key(p)]["trades"] >= cfg.selection.min_trades_train_portfolio]
        if not eligible:
            v1_params = dict(default_params)
            plateau = PlateauResult(params=v1_params, score=float("nan"), raw_sharpe=float("nan"))
        else:
            eligible_sharpes = [train_grid[_params_key(p)]["sharpe"] for p in eligible]
            plateau = plateau_select(eligible, eligible_sharpes, ordinal_grids, categorical_params)
            v1_params = plateau.params

        # --- 3. отбор монет (V2), режим "пара" на train с параметрами V1 ---
        coin_sel = None
        v2_symbols = list(symbols_pool)
        if apply_coin_selection:
            pair_metrics = {}
            for s in symbols_pool:
                strat_pair = make_strategy(v1_params)
                ctx = Context(params=v1_params, protocol=cfg, btc_bars=btc_bars)
                prepared_pair = prepare_fn(strat_pair, {s: bars_tf_by_symbol[s]}, ctx)
                r = evaluate_window(
                    strat_pair, {s: bars_1h_by_symbol[s]}, prepared_pair, {s: funding_by_symbol[s]},
                    tf, cfg, train_start, train_end, 1, btc_bars, cold_start=False,
                )
                m = _metrics(r, cfg)
                pair_metrics[s] = {"sharpe": m["sharpe"], "trades": m["trades"]}
                log("V2_pair", "pair", fold.id, (train_start, train_end), [s], v1_params, m)
            coin_sel = select_coins(
                pair_metrics, cfg.selection.coin_min_sharpe_train, cfg.selection.coin_min_trades_train,
                cfg.selection.min_coins_after_selection, symbols_pool,
            )
            v2_symbols = coin_sel.symbols

        # --- 4. test: V0/V1/V2 на test-окне, каждый "с нуля" (cold_start=True) ---
        def test_variant(params, symbols):
            strat = make_strategy(params)
            ctx = Context(params=params, protocol=cfg, btc_bars=btc_bars)
            key = _params_key(params)
            if key in grid_prepared_cache:
                # V0 (дефолты) и V1 (из сетки) почти всегда уже посчитаны в кэше -
                # просто берём нужные символы, не пересчитываем prepare() заново
                prepared = {s: grid_prepared_cache[key][s] for s in symbols}
            else:
                prepared = prepare_fn(strat, {s: bars_tf_by_symbol[s] for s in symbols}, ctx)
            b1h = {s: bars_1h_by_symbol[s] for s in symbols}
            fnd = {s: funding_by_symbol[s] for s in symbols}
            return evaluate_window(strat, b1h, prepared, fnd, tf, cfg, test_start, test_end,
                                    max_slots, btc_bars, cold_start=True)

        v0_result = test_variant(default_params, symbols_pool)
        v1_result = test_variant(v1_params, symbols_pool)
        v2_result = test_variant(v1_params, v2_symbols)

        v0_m, v1_m, v2_m = _metrics(v0_result, cfg), _metrics(v1_result, cfg), _metrics(v2_result, cfg)
        log("V0", "portfolio", fold.id, (test_start, test_end), symbols_pool, default_params, v0_m)
        log("V1", "portfolio", fold.id, (test_start, test_end), symbols_pool, v1_params, v1_m)
        log("V2", "portfolio", fold.id, (test_start, test_end), v2_symbols, v1_params, v2_m)

        for variant, result in (("V0", v0_result), ("V1", v1_result), ("V2", v2_result)):
            test_equities[variant].append(result.equity)
            test_trades[variant].extend(result.trades)

        wf.folds.append(FoldResult(
            fold_id=fold.id, train_grid=train_grid, v1_plateau=plateau, coin_selection=coin_sel,
            v0_test=v0_m, v1_test=v1_m, v2_test=v2_m,
        ))

    # --- 5. склейка test-доходностей F1-F4 в WF-OOS кривую на вариант ---
    for variant in ("V0", "V1", "V2"):
        stitched = _stitch_equity(test_equities[variant], cfg.account.initial_equity)
        wf.oos_equity[variant] = stitched
        wf.oos_metrics[variant] = perf.compute_all(
            stitched, test_trades[variant], cfg.metrics.annualization_days, cfg.metrics.risk_free,
        )

    # --- 6. основной вариант (04_OPTIMIZATION.md §6) ---
    v0_sharpe = wf.oos_metrics["V0"]["sharpe"]
    for variant in ("V1", "V2"):
        better_folds = sum(
            1 for f in wf.folds
            if (f.v1_test if variant == "V1" else f.v2_test)["sharpe"] > f.v0_test["sharpe"]
        )
        wf.folds_better_than_v0[variant] = better_folds

    candidates = []
    for variant in ("V1", "V2"):
        gain = wf.oos_metrics[variant]["sharpe"] - v0_sharpe
        if gain >= cfg.selection.min_wf_sharpe_gain_vs_v0 and wf.folds_better_than_v0[variant] >= cfg.selection.min_folds_better:
            candidates.append(variant)
    wf.primary_variant = max(candidates, key=lambda v: wf.oos_metrics[v]["sharpe"]) if candidates else "V0"

    return wf


def _stitch_equity(pieces: list[pd.Series], equity0: float) -> pd.Series:
    """Склеивает test-доходности F1-F4 в одну кривую WF-OOS: каждый фолд начинается со
    своих 10000 (evaluate_window(cold_start=True) гарантирует это), поэтому склейка -
    это перемножение относительных приростов каждого фолда, а не конкатенация уровней
    equity "как есть" (иначе разрыв между концом одного фолда и стартом следующего внёс
    бы фиктивный скачок)."""
    out = []
    level = equity0
    for piece in pieces:
        if piece.empty:
            continue
        relative = piece / piece.iloc[0]
        scaled = relative * level
        out.append(scaled)
        level = scaled.iloc[-1]
    if not out:
        return pd.Series(dtype=float)
    return pd.concat(out)
