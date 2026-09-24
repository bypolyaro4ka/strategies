"""Тесты для S07 (03_STRATEGIES.md) — Этап 6, партия B."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s07_ema_cross import S07


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.01, n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_golden_cross_gives_long():
    idx = pd.date_range("2023-01-01", periods=15, freq="4h", tz="UTC")
    close = pd.Series(list(np.linspace(100, 100, 10)) + list(np.linspace(101, 150, 5)), index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S07(params={"fast": 3, "slow": 8, "mode": "LS"})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    d = strat.on_bar(idx[-1], out.iloc[-1], pos, _ctx())
    assert d.target == 1.0


def test_death_cross_ls_gives_short_lf_gives_flat():
    idx = pd.date_range("2023-01-01", periods=15, freq="4h", tz="UTC")
    close = pd.Series(list(np.linspace(100, 100, 10)) + list(np.linspace(99, 50, 5)), index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    pos = PositionState(symbol="X")

    strat_ls = S07(params={"fast": 3, "slow": 8, "mode": "LS"})
    out_ls = strat_ls.prepare(bars, _ctx())
    d_ls = strat_ls.on_bar(idx[-1], out_ls.iloc[-1], pos, _ctx())
    assert d_ls.target == -1.0

    strat_lf = S07(params={"fast": 3, "slow": 8, "mode": "LF"})
    out_lf = strat_lf.prepare(bars, _ctx())
    d_lf = strat_lf.on_bar(idx[-1], out_lf.iloc[-1], pos, _ctx())
    assert d_lf.target == 0.0


def test_s07_truncation_and_future_poison_safe():
    bars = _synthetic_bars(400, seed=1)
    strat = S07(params={"fast": 20, "slow": 50, "mode": "LS"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
