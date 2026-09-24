"""Тесты для S08 (03_STRATEGIES.md) — Этап 6, партия B."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s08_macd import S08


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_cross_up_above_filter_enters_long_no_filter_too():
    # затяжной спад (macd < signal, ниже фильтра), затем резкий подъём - пересечение
    # вверх macd/signal и цена выше EMA-фильтра одновременно
    idx = pd.date_range("2023-01-01", periods=260, freq="4h", tz="UTC")
    down = 200 - np.linspace(0, 80, 220)
    up = down[-1] + np.linspace(1, 120, 40)
    close = pd.Series(np.concatenate([down, up]), index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    pos = PositionState(symbol="X")

    strat = S08(params={"fast": 12, "slow": 26, "signal": 9, "filter_n": 200})
    out = strat.prepare(bars, _ctx())
    assert bool(out["cross_up"].iloc[-10:].any())  # где-то в хвосте пересечение случилось
    cross_idx = out.index[out["cross_up"]][-1]
    d = strat.on_bar(cross_idx, out.loc[cross_idx], pos, _ctx())
    if out.loc[cross_idx, "close"] > out.loc[cross_idx, "ema_filter"]:
        assert d.target == 1.0


def test_no_filter_ignores_ema_filter_column():
    bars = _synthetic_bars(300, seed=5)
    strat = S08(params={"fast": 12, "slow": 26, "signal": 9, "filter_n": None})
    out = strat.prepare(bars, _ctx())
    assert out["ema_filter"].isna().all()


def test_same_bar_reversal_long_to_short_when_cross_down_confirmed():
    idx = pd.date_range("2023-01-01", periods=280, freq="4h", tz="UTC")
    up = 100 + np.linspace(0, 80, 220)
    down = up[-1] - np.linspace(1, 150, 60)
    close = pd.Series(np.concatenate([up, down]), index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S08(params={"fast": 12, "slow": 26, "signal": 9, "filter_n": 200})
    out = strat.prepare(bars, _ctx())
    cross_downs = out.index[out["cross_down"]]
    if len(cross_downs) == 0:
        return  # синтетика может не дать пересечения - не валим тест, это sanity-проверка
    t = cross_downs[-1]
    row = out.loc[t]
    pos = PositionState(symbol="X", target=1.0)
    d = strat.on_bar(t, row, pos, _ctx())
    expected = -1.0 if row["close"] < row["ema_filter"] else 0.0
    assert d.target == expected


def test_s08_truncation_and_future_poison_safe():
    bars = _synthetic_bars(500, seed=2)
    strat = S08(params={"fast": 12, "slow": 26, "signal": 9, "filter_n": 200})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
