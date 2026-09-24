"""Тесты для S10 (03_STRATEGIES.md) — Этап 6, партия C."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s10_rsi2 import S10


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_deep_pullback_in_uptrend_enters_long_no_filter():
    # устойчивый рост (SMA200 фильтр отключён, чтобы не тащить 200+ баров), затем резкий
    # однодневный откат - RSI(2) должен уйти ниже 10
    idx = pd.date_range("2023-01-01", periods=21, freq="4h", tz="UTC")
    close = pd.Series(list(np.linspace(100, 150, 20)) + [140.0], index=idx)  # откат в конце
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S10(params={"lo": 10, "hi": 90, "filter_sma200": False, "T": 10})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    if out["rsi2"].iloc[-1] < 10:
        assert d.target == 1.0


def test_exit_on_time_stop_even_without_sma5_cross():
    bars = _synthetic_bars(60, seed=1)
    strat = S10(params={"lo": 10, "hi": 90, "filter_sma200": False, "T": 3})
    out = strat.prepare(bars, _ctx())
    row = out.iloc[-1]
    pos = PositionState(symbol="X", target=1.0, bars_in_trade=5)  # уже дольше тайм-стопа
    d = strat.on_bar(out.index[-1], row, pos, _ctx())
    assert d.target == 0.0
    assert d.tag == "exit_long"


def test_s10_truncation_and_future_poison_safe():
    bars = _synthetic_bars(300, seed=2)
    strat = S10(params={"lo": 10, "hi": 90, "filter_sma200": False, "T": 10})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)


def test_s10_with_sma200_filter_truncation_and_future_poison_safe():
    bars = _synthetic_bars(400, seed=3)
    strat = S10(params={"lo": 10, "hi": 90, "filter_sma200": True, "T": 10})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=15)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=15)
