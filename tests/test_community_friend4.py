"""Тесты для портированных стратегий 4-го друга — Этап 9 (стратегии друзей)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.community.friend4.strategies import (
    Friend4BbRegimeReversion,
    Friend4BbSqueezeBreakout,
    Friend4EmaAdxTrend,
    Friend4IchimokuRsi,
    Friend4RsiPullback,
    Friend4Supertrend,
)

STRATEGIES = [
    Friend4Supertrend(), Friend4EmaAdxTrend(), Friend4BbSqueezeBreakout(),
    Friend4RsiPullback(), Friend4BbRegimeReversion(), Friend4IchimokuRsi(),
]


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=800, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.normal(1000, 300, n))
    return pd.DataFrame({"open": np.roll(close, 1), "high": high, "low": low, "close": close,
                          "volume": volume, "quote_volume": volume * close, "trades": 1,
                          "complete": True}, index=idx)


def test_all_have_author_friend4():
    for s in STRATEGIES:
        assert s.author == "Друг4"


def test_truncation_and_future_poison_safe():
    # 1200 часовых баров = 50 дней - хватает и для дневных (1d) семейств с их прогревом
    bars = _synthetic_bars(1600)
    ctx = _ctx()
    for strat in STRATEGIES:
        assert_truncation_safe(strat, bars, ctx, n_checks=12)
        assert_future_poison_safe(strat, bars, ctx, n_checks=12)


def test_prepare_runs_without_error_on_short_history():
    bars = _synthetic_bars(50)
    for strat in STRATEGIES:
        out = strat.prepare(bars, _ctx())
        assert len(out) == len(bars)


def test_stateful_families_never_reenter_before_exit_signal():
    """На синтетике без экстремальных условий stateful-семейства не должны стрелять
    target отличным от {-1,0,1} - sanity на то, что _stateful_step() возвращает
    дискретные значения, не что-то промежуточное."""
    bars = _synthetic_bars(1600, seed=2)
    stateful = [Friend4BbSqueezeBreakout(), Friend4RsiPullback(), Friend4BbRegimeReversion(), Friend4IchimokuRsi()]
    for strat in stateful:
        out = strat.prepare(bars, _ctx())
        pos = SimpleNamespace(target=0.0)
        for t, row in out.iterrows():
            d = strat.on_bar(t, row, pos, _ctx())
            assert d.target in (-1.0, 0.0, 1.0)
            pos = SimpleNamespace(target=d.target)


def test_memoryless_families_only_emit_discrete_targets():
    bars = _synthetic_bars(1600, seed=3)
    memoryless = [Friend4Supertrend(), Friend4EmaAdxTrend()]
    for strat in memoryless:
        out = strat.prepare(bars, _ctx())
        pos = SimpleNamespace(target=0.0)
        for t, row in out.iterrows():
            d = strat.on_bar(t, row, pos, _ctx())
            assert d.target in (-1.0, 0.0, 1.0)
