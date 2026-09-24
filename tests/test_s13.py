"""Тесты для S13 (03_STRATEGIES.md) — Этап 6, партия D часть 2 (PortfolioStrategy)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe_portfolio, assert_truncation_safe_portfolio
from lab.strategies.base import Context, PositionState
from lab.strategies.s13_cross_sectional_momentum import S13


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _daily_bars(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})


def _six_symbol_pool(n_days=60, seed=0) -> dict[str, pd.DataFrame]:
    """6 монет с разной динамикой - однозначный ранкинг: WINNER растёт быстрее всех,
    LOSER падает быстрее всех, остальные почти без движения (шум)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="1D", tz="UTC")  # начинается с воскресенья
    pool = {}
    pool["WINNER"] = pd.Series(100 * np.cumprod(1 + np.full(n_days, 0.02)), index=idx)
    pool["LOSER"] = pd.Series(100 * np.cumprod(1 + np.full(n_days, -0.02)), index=idx)
    for name in ["MID1", "MID2", "MID3", "MID4"]:
        noise = rng.normal(0.0, 0.003, n_days)
        pool[name] = pd.Series(100 * np.cumprod(1 + noise), index=idx)
    return {s: _daily_bars(c) for s, c in pool.items()}


def test_hold_on_non_sunday():
    bars = _six_symbol_pool(40, seed=1)
    strat = S13(params={"L": 21, "k": 2, "skip": 0})
    prepared = strat.prepare(bars, _ctx())
    monday = [t for t in next(iter(prepared.values())).index if t.dayofweek == 0][5]
    rows = {s: df.loc[monday] for s, df in prepared.items()}
    positions = {s: PositionState(symbol=s, target=0.5) for s in bars}  # искусственное "уже держим"
    decisions = strat.on_bar(monday, rows, positions, _ctx())
    for s in bars:
        assert decisions[s].target == 0.5  # понедельник - держим то, что было
        assert decisions[s].tag == "hold"


def test_sunday_ranks_winner_long_loser_short():
    bars = _six_symbol_pool(60, seed=2)
    strat = S13(params={"L": 21, "k": 2, "skip": 0})
    prepared = strat.prepare(bars, _ctx())
    sundays = [t for t in next(iter(prepared.values())).index if t.dayofweek == 6]
    t = sundays[-1]  # последнее воскресенье - точно после прогрева L=21
    rows = {s: df.loc[t] for s, df in prepared.items()}
    positions = {s: PositionState(symbol=s) for s in bars}
    decisions = strat.on_bar(t, rows, positions, _ctx())
    assert decisions["WINNER"].target == 1.0
    assert decisions["LOSER"].target == -1.0
    # k=2, значит один из MID* тоже должен войти в топ/боттом - не проверяем, какой именно
    nonzero = [s for s, d in decisions.items() if d.target != 0]
    assert len(nonzero) == 4  # 2*k


def test_not_enough_history_gives_flat():
    idx = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")  # мало истории для L=21
    bars = {"A": _daily_bars(pd.Series([100.0] * 10, index=idx)),
            "B": _daily_bars(pd.Series([100.0] * 10, index=idx)),
            "C": _daily_bars(pd.Series([100.0] * 10, index=idx))}
    strat = S13(params={"L": 21, "k": 2, "skip": 0})
    prepared = strat.prepare(bars, _ctx())
    sunday = next(t for t in idx if t.dayofweek == 6)
    rows = {s: df.loc[sunday] for s, df in prepared.items()}
    positions = {s: PositionState(symbol=s) for s in bars}
    decisions = strat.on_bar(sunday, rows, positions, _ctx())
    assert all(d.target == 0.0 for d in decisions.values())
    assert all(d.tag == "not_enough_history" for d in decisions.values())


def test_s13_truncation_and_future_poison_safe():
    bars = _six_symbol_pool(80, seed=3)
    strat = S13(params={"L": 21, "k": 2, "skip": 0})
    ctx = _ctx()
    assert_truncation_safe_portfolio(strat, bars, ctx, n_checks=15)
    assert_future_poison_safe_portfolio(strat, bars, ctx, n_checks=15)
