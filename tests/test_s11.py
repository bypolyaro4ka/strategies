"""Тесты для S11 (03_STRATEGIES.md) — Этап 6, партия C."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s11_bb_adx import S11


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _ranging_bars(n=100, seed=0) -> pd.DataFrame:
    """Боковик - невысокая амплитуда без тренда, чтобы ADX оставался низким."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 + rng.normal(0, 1.0, n).cumsum() * 0.05  # почти без дрейфа
    close = np.clip(close, 90, 110)
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def _trending_bars(n=100, seed=0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 + np.arange(n) * 1.5  # устойчивый монотонный тренд -> высокий ADX
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_strong_trend_blocks_entry_even_if_band_touched():
    bars = _trending_bars(80, seed=1)
    strat = S11(params={"adx_thr": 20, "k": 2, "s": 2, "n_bb": 20, "n_adx": 14})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    row = out.iloc[-1]
    assert row["adx"] >= 20  # монотонный тренд - ADX должен быть высоким
    d = strat.on_bar(out.index[-1], row, pos, _ctx())
    assert d.target == 0.0


def test_exit_at_mid_band_closes_position():
    bars = _ranging_bars(100, seed=2)
    strat = S11(params={"adx_thr": 90, "k": 2, "s": 2, "n_bb": 20, "n_adx": 14, "time_stop": 20})
    out = strat.prepare(bars, _ctx())
    row = out.iloc[-1].copy()
    row["close"] = row["mid"] + 0.01  # чуть выше средней
    pos = PositionState(symbol="X", target=1.0, bars_in_trade=1, stop_price=row["close"] - 10)
    d = strat.on_bar(out.index[-1], row, pos, _ctx())
    assert d.target == 0.0
    assert d.tag == "exit_long"


def test_time_stop_forces_exit():
    bars = _ranging_bars(100, seed=3)
    strat = S11(params={"adx_thr": 90, "k": 2, "s": 2, "n_bb": 20, "n_adx": 14, "time_stop": 5})
    out = strat.prepare(bars, _ctx())
    row = out.iloc[-1]
    pos = PositionState(symbol="X", target=1.0, bars_in_trade=5, stop_price=row["close"] - 10)
    d = strat.on_bar(out.index[-1], row, pos, _ctx())
    assert d.target == 0.0
    assert d.tag == "exit_long"


def test_s11_truncation_and_future_poison_safe():
    bars = _ranging_bars(200, seed=4)
    strat = S11(params={"adx_thr": 20, "k": 2, "s": 2, "n_bb": 20, "n_adx": 14})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
