"""Тесты для S03 (03_STRATEGIES.md) — Этап 5."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s03_sma import S03


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.02, n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_above_sma_gives_long():
    idx = pd.date_range("2023-01-01", periods=25, freq="1D", tz="UTC")
    close = pd.Series([100.0] * 24 + [200.0], index=idx)  # резкий скачок выше SMA
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S03(params={"n_days": 20, "band": 0.0, "mode": "LS"})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == 1.0


def test_below_sma_ls_mode_gives_short():
    idx = pd.date_range("2023-01-01", periods=25, freq="1D", tz="UTC")
    close = pd.Series([100.0] * 24 + [50.0], index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S03(params={"n_days": 20, "mode": "LS"})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == -1.0


def test_below_sma_lf_mode_gives_flat():
    idx = pd.date_range("2023-01-01", periods=25, freq="1D", tz="UTC")
    close = pd.Series([100.0] * 24 + [50.0], index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S03(params={"n_days": 20, "mode": "LF"})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == 0.0


def test_inside_band_holds_previous_target():
    idx = pd.date_range("2023-01-01", periods=25, freq="1D", tz="UTC")
    close = pd.Series([100.0] * 25, index=idx)  # ровно на SMA - внутри любой полосы
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S03(params={"n_days": 20, "band": 0.05, "mode": "LS"})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X", target=0.7)
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == 0.7


def test_s03_truncation_and_future_poison_safe():
    bars = _synthetic_bars(150)
    strat = S03(params={"n_days": 20, "band": 0.01, "mode": "LS"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
