"""Тесты для портированных стратегий Славы — Этап 9 (стратегии друзей)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.community.slava.c2_local_compression import SlavaC2LocalCompression
from lab.strategies.community.slava.eth_strong_breakout import SlavaEthStrongBreakout
from lab.strategies.community.slava.sol_frozen_v2 import SlavaSolFrozenV2

STRATEGIES = [SlavaSolFrozenV2(), SlavaEthStrongBreakout(), SlavaC2LocalCompression()]


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


def test_all_have_author_slava():
    for s in STRATEGIES:
        assert s.author == "Слава"


def test_truncation_and_future_poison_safe():
    bars = _synthetic_bars(800)
    ctx = _ctx()
    for strat in STRATEGIES:
        assert_truncation_safe(strat, bars, ctx, n_checks=15)
        assert_future_poison_safe(strat, bars, ctx, n_checks=15)


def test_prepare_runs_without_error_on_short_history():
    bars = _synthetic_bars(50)
    for strat in STRATEGIES:
        out = strat.prepare(bars, _ctx())
        assert len(out) == len(bars)
