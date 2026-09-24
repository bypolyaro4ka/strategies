"""Тесты для S06 (03_STRATEGIES.md) — Этап 6, партия B."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s06_supertrend import S06


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


def test_trend_never_zero_outside_warmup():
    bars = _synthetic_bars(200, seed=1)
    strat = S06(params={"n": 10, "m": 3})
    out = strat.prepare(bars, _ctx())
    after_warmup = out["trend"].iloc[strat.required_history("4h"):]
    assert set(after_warmup.unique()) <= {-1, 1}


def test_flat_then_strong_jump_flips_trend_up():
    idx = pd.date_range("2023-01-01", periods=40, freq="4h", tz="UTC")
    close = pd.Series([100.0] * 30 + list(np.linspace(101, 200, 10)), index=idx)
    bars = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S06(params={"n": 10, "m": 3})
    out = strat.prepare(bars, _ctx())
    assert out["trend"].iloc[-1] == 1


def test_on_bar_reads_trend_column_directly():
    bars = _synthetic_bars(60, seed=2)
    strat = S06(params={"n": 10, "m": 3})
    out = strat.prepare(bars, _ctx())
    pos = PositionState(symbol="X")
    row = out.iloc[-1]
    d = strat.on_bar(out.index[-1], row, pos, _ctx())
    assert d.target == float(row["trend"]) or (row["trend"] == 0 and d.target == 0.0)


def test_s06_truncation_and_future_poison_safe():
    bars = _synthetic_bars(200, seed=3)
    strat = S06(params={"n": 10, "m": 3})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=15)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=15)
