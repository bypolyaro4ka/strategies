"""Тесты для S09 (03_STRATEGIES.md) — Этап 6, партия B."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s09_volatility_breakout import S09


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _week_with_breakout_on_day6(breakout_hour=5) -> pd.DataFrame:
    """6 плоских дней (для прогрева R и SMA5d), седьмой день - пробой вверх на
    breakout_hour, держим до конца дня."""
    idx = pd.date_range("2023-01-01", periods=7 * 24, freq="1h", tz="UTC")
    close, high, low, open_ = [], [], [], []
    for i in range(7 * 24):
        day, hour = divmod(i, 24)
        if day < 6 or hour < breakout_hour:
            c, h, l, o = 100.0, 101.0, 99.0, 100.0
        elif hour == breakout_hour:
            c, h, l, o = 105.0, 105.0, 100.0, 100.0  # пробой: close > up (100 + 0.5*2 = 101)
        else:
            c, h, l, o = 105.0, 105.5, 104.5, 105.0
        close.append(c); high.append(h); low.append(l); open_.append(o)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def _synthetic_bars(days=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.006, n))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"open": np.roll(close, 1), "high": high, "low": low, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_no_entry_before_breakout_then_entry_then_hold_then_day_end_exit():
    bars = _week_with_breakout_on_day6(breakout_hour=5)
    strat = S09(params={"k": 0.5, "mode": "LS", "trend_filter": "none"})
    out = strat.prepare(bars, _ctx())
    day6 = out.iloc[6 * 24:7 * 24]
    pos = SimpleNamespace(target=0.0)
    decisions = []
    for t, row in day6.iterrows():
        d = strat.on_bar(t, row, pos, _ctx())
        decisions.append((t.hour, d.target))
        pos = SimpleNamespace(target=d.target)
    hours, targets = zip(*decisions)
    assert all(targets[h] == 0.0 for h in range(5))  # до пробоя - не входили
    assert targets[5] == 1.0  # пробой - вход
    assert all(targets[h] == 1.0 for h in range(6, 23))  # держим весь день
    assert targets[23] == 0.0  # принудительный выход на последнем часовом баре дня


def test_not_more_than_one_entry_per_day():
    bars = _week_with_breakout_on_day6(breakout_hour=3)
    strat = S09(params={"k": 0.5, "mode": "LS", "trend_filter": "none"})
    out = strat.prepare(bars, _ctx())
    day6 = out.iloc[6 * 24:7 * 24]
    pos = SimpleNamespace(target=0.0)
    entries = 0
    prev_target = 0.0
    for t, row in day6.iterrows():
        d = strat.on_bar(t, row, pos, _ctx())
        if prev_target == 0.0 and d.target != 0.0:
            entries += 1
        prev_target = d.target
        pos = SimpleNamespace(target=d.target)
    assert entries == 1


def test_long_only_mode_ignores_downside_breakout():
    idx = pd.date_range("2023-01-01", periods=7 * 24, freq="1h", tz="UTC")
    close, high, low, open_ = [], [], [], []
    for i in range(7 * 24):
        day, hour = divmod(i, 24)
        if day < 6 or hour < 5:
            c, h, l, o = 100.0, 101.0, 99.0, 100.0
        elif hour == 5:
            c, h, l, o = 95.0, 100.0, 95.0, 100.0  # пробой ВНИЗ: close < dn (100 - 1 = 99)
        else:
            c, h, l, o = 95.0, 95.5, 94.5, 95.0
        close.append(c); high.append(h); low.append(l); open_.append(o)
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)
    strat = S09(params={"k": 0.5, "mode": "long_only", "trend_filter": "none"})
    out = strat.prepare(bars, _ctx())
    day6 = out.iloc[6 * 24:7 * 24]
    pos = SimpleNamespace(target=0.0)
    for t, row in day6.iterrows():
        d = strat.on_bar(t, row, pos, _ctx())
        pos = SimpleNamespace(target=d.target)
    assert pos.target == 0.0  # long_only - пробой вниз игнорируется, входа не было


def test_s09_truncation_and_future_poison_safe():
    bars = _synthetic_bars(days=20, seed=1)
    strat = S09(params={"k": 0.5, "mode": "LS", "trend_filter": "none"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
