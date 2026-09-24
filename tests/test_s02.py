"""Тесты для S02 (03_STRATEGIES.md) — Этап 5."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.s02_tsmom import S02


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


def _synthetic_bars(n_days=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="1D", tz="UTC")  # с понедельника
    ret = rng.normal(0.0003, 0.02, n_days)
    close = 100 * np.cumprod(1 + ret)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)


def test_manual_sign_mean():
    idx = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 120.0], index=idx)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True})
    strat = S02(params={"lookback_days": (2, 5), "tf": "1d"})
    out = strat.prepare(bars, _ctx())
    # на последнем баре: r_2 = 120/107-1>0, r_5=120/104-1>0 -> mean(sign)=1.0
    assert out["target"].iloc[-1] == pytest.approx(1.0)


def test_target_range_matches_lookback_count():
    bars = _synthetic_bars(150)
    strat = S02(params={"lookback_days": (7, 14, 28), "tf": "1d"})
    out = strat.prepare(bars, _ctx())
    allowed = {-1.0, -1 / 3, 1 / 3, 1.0, 0.0}
    vals = set(round(v, 6) for v in out["target"].dropna().unique())
    assert vals <= {round(v, 6) for v in allowed}


def test_weekly_rebalance_holds_between_sundays():
    bars = _synthetic_bars(60)
    strat = S02(params={"lookback_days": (7,), "rebalance": "weekly", "tf": "1d"})
    out = strat.prepare(bars, _ctx())
    # понедельник-суббота: target должен совпадать с предыдущим воскресеньем (ffill)
    is_sunday = out.index.dayofweek == 6
    non_sunday_targets = out.loc[~is_sunday, "target"].dropna()
    sunday_targets = out.loc[is_sunday, "target"]
    # каждое значение между воскресеньями должно встречаться и среди значений по воскресеньям
    assert set(non_sunday_targets.round(6)) <= set(sunday_targets.dropna().round(6)) | {0.0}


def test_every_bar_vs_weekly_differ():
    bars = _synthetic_bars(60, seed=5)
    every_bar = S02(params={"lookback_days": (7,), "rebalance": "every_bar"}).prepare(bars, _ctx())["target"]
    weekly = S02(params={"lookback_days": (7,), "rebalance": "weekly"}).prepare(bars, _ctx())["target"]
    assert not every_bar.equals(weekly)


def test_s02_truncation_and_future_poison_safe():
    bars = _synthetic_bars(150, seed=2)
    strat = S02(params={"lookback_days": (7, 14, 28), "tf": "1d"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)


def test_s02_weekly_truncation_safe():
    bars = _synthetic_bars(150, seed=3)
    strat = S02(params={"lookback_days": (7,), "rebalance": "weekly"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)
