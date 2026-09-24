"""Тесты для S05 (03_STRATEGIES.md) — Этап 5."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s05_bollinger import S05


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.02, n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_breakout_above_upper_band_enters_long():
    idx = pd.date_range("2023-01-01", periods=21, freq="1D", tz="UTC")
    close = pd.Series([100.0] * 20 + [130.0], index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S05(params={"n": 20, "k": 2})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == 1.0


def test_exit_rule_mid_closes_before_opposite_band():
    idx = pd.date_range("2023-01-01", periods=21, freq="1D", tz="UTC")
    close = list(np.linspace(100, 130, 20)) + [None]  # заполним последний ниже
    bars_close = pd.Series(np.linspace(100, 130, 21), index=idx)
    bars = pd.DataFrame({"open": bars_close, "high": bars_close, "low": bars_close, "close": bars_close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat_mid = S05(params={"n": 20, "k": 1.0, "exit_rule": "mid"})
    out = strat_mid.prepare(bars, _ctx())
    row = out.iloc[-1].copy()
    row["close"] = row["mid"] - 0.01  # чуть ниже средней, но выше нижней полосы
    pos = PositionState(symbol="X", target=1.0)
    d_mid = strat_mid.on_bar(idx[-1], row, pos, _ctx())
    assert d_mid.target == 0.0  # exit_rule=mid уже закрыл

    strat_opp = S05(params={"n": 20, "k": 1.0, "exit_rule": "opposite"})
    out2 = strat_opp.prepare(bars, _ctx())
    row2 = out2.iloc[-1].copy()
    row2["close"] = row2["mid"] - 0.01
    d_opp = strat_opp.on_bar(idx[-1], row2, pos, _ctx())
    assert d_opp.target == 1.0  # exit_rule=opposite - ещё держит, полоса не пробита


def test_s05_truncation_and_future_poison_safe():
    bars = _synthetic_bars(150, seed=3)
    strat = S05(params={"n": 20, "k": 2, "exit_rule": "mid"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
