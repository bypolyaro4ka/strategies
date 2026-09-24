"""Тесты для optimize/walk_forward.py — Этап 7 (движок walk-forward).

Стратегия-заглушка и все прогоны здесь на ТФ=1h, чтобы bars_1h (для движка) и
signal_bars (для сигналов) были буквально одним и тем же DataFrame — не нужно
резать реальный дневной ресемплинг, чтобы проверить сам walk-forward движок."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.optimize.walk_forward import run_walk_forward
from lab.strategies.base import BaseStrategy, Decision


class _ThresholdStub(BaseStrategy):
    """Простая параметризуемая стратегия для теста движка walk-forward: лонг, если
    close выше threshold, иначе плоско. Один числовой параметр - удобно для плато."""
    id = "STUB_THRESHOLD"
    name = "threshold"
    version = "1.0.0"
    timeframes = ["1h"]
    direction = "long_only"
    default_params = {"threshold": 100}

    def required_history(self, tf):
        return 0

    def prepare(self, bars, ctx):
        return bars

    def on_bar(self, t, row, pos, ctx):
        if row["close"] > self.params["threshold"]:
            return Decision(target=1.0, tag="above")
        return Decision(target=0.0, tag="below")


def _cfg(folds, min_trades=0, min_coins=1):
    return SimpleNamespace(
        folds=folds,
        account=SimpleNamespace(initial_equity=10_000, slot_fraction=0.10, max_slots_portfolio=3),
        exchange=SimpleNamespace(fee_taker=0.0005, slippage=0.0002),
        selection=SimpleNamespace(
            min_trades_train_portfolio=min_trades, coin_min_trades_train=1,
            coin_min_sharpe_train=-10.0, min_coins_after_selection=min_coins,
            min_wf_sharpe_gain_vs_v0=0.3, min_folds_better=2,
        ),
        metrics=SimpleNamespace(annualization_days=365, risk_free=0.0),
    )


def _fold(fold_id, train, test):
    return SimpleNamespace(id=fold_id, train=train, test=test)


def _rising_bars(n_hours, price0=90.0, step=0.02, start="2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    close = [price0 + i * step for i in range(n_hours)]
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True}, index=idx)


def _empty_funding():
    return pd.DataFrame({"funding_rate": []}, index=pd.DatetimeIndex([], tz="UTC"))


def _make_pool(n_hours=60 * 24, seed=0):
    rng = np.random.default_rng(seed)
    bars = {}
    for i, sym in enumerate(["AAA", "BBB"]):
        bars[sym] = _rising_bars(n_hours, price0=90.0 + i * 5, step=0.02 + rng.uniform(-0.004, 0.004))
    funding = {s: _empty_funding() for s in bars}
    return bars, bars, funding  # bars_1h, bars_tf (одно и то же на ТФ=1h), funding


def _prepare_fn(strat, bars_by_symbol, ctx):
    return {s: strat.prepare(b, ctx) for s, b in bars_by_symbol.items()}


def test_walk_forward_runs_all_folds_and_returns_stitched_curves():
    bars_1h, bars_tf, funding = _make_pool(60 * 24)
    folds = [
        _fold("F1", ("2023-01-01", "2023-01-20"), ("2023-01-21", "2023-01-31")),
        _fold("F2", ("2023-01-01", "2023-01-31"), ("2023-02-01", "2023-02-15")),
    ]
    cfg = _cfg(folds)
    wf = run_walk_forward(
        strategy_id="STUB_THRESHOLD",
        make_strategy=lambda params: _ThresholdStub(params=params),
        prepare_fn=_prepare_fn,
        default_params={"threshold": 100},
        param_grid_points=[{"threshold": t} for t in (90, 95, 100, 105, 110)],
        ordinal_grids={"threshold": [90, 95, 100, 105, 110]},
        categorical_params=(),
        bars_1h_by_symbol=bars_1h, bars_tf_by_symbol=bars_tf, funding_by_symbol=funding,
        tf="1h", cfg=cfg, symbols_pool=["AAA", "BBB"],
    )
    assert len(wf.folds) == 2
    assert set(wf.oos_equity.keys()) == {"V0", "V1", "V2"}
    assert wf.primary_variant in ("V0", "V1", "V2")
    for variant in ("V0", "V1", "V2"):
        assert not wf.oos_equity[variant].empty


def test_cold_start_no_leaked_position_at_test_window_start():
    """Ключевое свойство test-окна (04_OPTIMIZATION.md §4, шаг 4): позиция должна быть
    плоской РОВНО на начало test-окна, даже если стратегия непрерывно в рынке весь train."""
    bars_1h, bars_tf, funding = _make_pool(60 * 24)
    folds = [_fold("F1", ("2023-01-01", "2023-01-20"), ("2023-01-21", "2023-01-31"))]
    cfg = _cfg(folds)
    wf = run_walk_forward(
        strategy_id="STUB_THRESHOLD",
        make_strategy=lambda params: _ThresholdStub(params=params),
        prepare_fn=_prepare_fn,
        default_params={"threshold": 89},  # почти всегда "above" - была бы в рынке весь train
        param_grid_points=[{"threshold": 89}],
        ordinal_grids={"threshold": [89]},
        categorical_params=(),
        bars_1h_by_symbol=bars_1h, bars_tf_by_symbol=bars_tf, funding_by_symbol=funding,
        tf="1h", cfg=cfg, symbols_pool=["AAA", "BBB"],
    )
    v0_equity = wf.oos_equity["V0"]
    # первая точка test-окна должна равняться стартовому депозиту (10000) - если бы
    # позиция "унаследовалась" из train, equity в первой точке отличалась бы от 10000
    assert v0_equity.iloc[0] == 10_000


def test_primary_variant_stays_v0_when_grid_does_not_help():
    """Все точки сетки дают одинаковые (нулевые) результаты - оптимизация не должна
    "заслужить" замену V0 (04_OPTIMIZATION.md §6)."""
    idx = pd.date_range("2023-01-01", periods=60 * 24, freq="1h", tz="UTC")
    flat_low = pd.Series([50.0] * len(idx), index=idx)
    bars = {"AAA": pd.DataFrame({"open": flat_low, "high": flat_low, "low": flat_low, "close": flat_low,
                                  "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)}
    funding = {"AAA": _empty_funding()}
    folds = [_fold("F1", ("2023-01-01", "2023-01-20"), ("2023-01-21", "2023-01-31"))]
    cfg = _cfg(folds)
    wf = run_walk_forward(
        strategy_id="STUB_THRESHOLD",
        make_strategy=lambda params: _ThresholdStub(params=params),
        prepare_fn=_prepare_fn,
        default_params={"threshold": 100},
        param_grid_points=[{"threshold": t} for t in (90, 100, 110)],  # ничто не пробивает close=50
        ordinal_grids={"threshold": [90, 100, 110]},
        categorical_params=(),
        bars_1h_by_symbol=bars, bars_tf_by_symbol=bars, funding_by_symbol=funding,
        tf="1h", cfg=cfg, symbols_pool=["AAA"],
    )
    assert wf.primary_variant == "V0"
