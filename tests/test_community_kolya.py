"""Тесты для портированных стратегий Коли — 9 семейств (⚠ видели holdout, см. DECLARATION.md)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.community.kolya.strategies import ALL_STRATEGIES, KolyaFundingContrarian


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n=700, seed=0, with_funding=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.normal(1000, 300, n))
    df = pd.DataFrame({"open": np.roll(close, 1), "high": high, "low": low, "close": close,
                        "volume": volume, "quote_volume": volume * close, "trades": 1,
                        "complete": True}, index=idx)
    if with_funding:
        df["funding_rate"] = rng.normal(0.0001, 0.0003, n)
    return df


@pytest.mark.parametrize("cls", ALL_STRATEGIES)
def test_author_is_kolya(cls):
    assert cls().author == "Коля"


@pytest.mark.parametrize("cls", ALL_STRATEGIES)
def test_truncation_and_future_poison_safe(cls):
    strat = cls()
    with_funding = isinstance(strat, KolyaFundingContrarian)
    bars = _synthetic_bars(700, with_funding=with_funding)
    ctx = _ctx()
    assert_truncation_safe(strat, bars, ctx, n_checks=12)
    assert_future_poison_safe(strat, bars, ctx, n_checks=12)


@pytest.mark.parametrize("cls", ALL_STRATEGIES)
def test_prepare_produces_signal_column(cls):
    bars = _synthetic_bars(300)
    out = cls().prepare(bars, _ctx())
    assert "signal" in out.columns
    assert "atr" in out.columns


def test_funding_contrarian_without_funding_column_stays_flat():
    strat = KolyaFundingContrarian()
    bars = _synthetic_bars(300, with_funding=False)
    out = strat.prepare(bars, _ctx())
    assert (out["signal"] == 0).all()
