"""Тесты для S04 (03_STRATEGIES.md) — Этап 5."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s04_turtle import S04


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"open": np.roll(close, 1), "high": high, "low": low, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_manual_breakout_sets_entry_and_stop():
    idx = pd.date_range("2023-01-01", periods=25, freq="4h", tz="UTC")
    close = pd.Series([100.0] * 24 + [130.0], index=idx)
    high = close.copy()
    low = close - 1
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S04(params={"n_in": 20, "n_out": 10, "k": 2, "atr_n": 20})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == 1.0
    assert d.stop_price < close.iloc[-1]  # стоп ниже цены входа для лонга
    assert d.stop_price == pytest.approx(close.iloc[-1] - 2 * out["atr"].iloc[-1])


def test_manual_breakdown_sets_short_and_stop_above():
    idx = pd.date_range("2023-01-01", periods=25, freq="4h", tz="UTC")
    close = pd.Series([100.0] * 24 + [70.0], index=idx)
    high = close + 1
    low = close.copy()
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S04(params={"n_in": 20, "n_out": 10, "k": 2, "atr_n": 20})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == -1.0
    assert d.stop_price > close.iloc[-1]


def test_hold_keeps_target_and_stop_when_no_exit_signal():
    bars = _synthetic_bars(60, seed=1)
    strat = S04(params={"n_in": 5, "n_out": 3, "k": 2, "atr_n": 5})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X", target=1.0, stop_price=1.0)  # искусственно "в позиции", стоп не заденет
    row = out.iloc[-1]
    d = strat.on_bar(out.index[-1], row, pos, _ctx())
    if row["close"] >= row["exit_low"]:
        assert d.target == 1.0
        assert d.stop_price == 1.0


def test_s04_truncation_and_future_poison_safe():
    bars = _synthetic_bars(300, seed=2)
    strat = S04(params={"n_in": 20, "n_out": 10, "k": 2, "atr_n": 20})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
