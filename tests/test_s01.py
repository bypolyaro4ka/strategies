"""Тесты для S01/S01b (03_STRATEGIES.md) — Этап 5."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.data.resample import resample_ohlcv
from lab.engine.backtest import run_backtest
from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.s01_donchian import S01, S01b, _track_long, _track_short


def _cfg():
    return SimpleNamespace(
        account=SimpleNamespace(slot_fraction=0.10, min_target_change=0.05, max_slots_portfolio=5),
        exchange=SimpleNamespace(fee_taker=0.0005, slippage=0.0002),
        metrics=SimpleNamespace(annualization_days=365, risk_free=0.0),
    )


def _ctx():
    return Context(params={}, protocol=_cfg())


# --- ручной сценарий: ряд, который пробивает максимум на известном баре ---

def test_track_long_manual_breakout_scenario():
    # close[0..4] = 100 (окно "истории"), затем пробой на баре 5, затем откат ниже mid -> выход
    close = np.array([100, 100, 100, 100, 100, 110, 108, 90], dtype=float)
    upper = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 100, 100, 100])
    lower = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 100, 100, 100])
    on = _track_long(close, upper, lower)
    assert list(on) == [False, False, False, False, False, True, True, False]


def test_track_short_manual_breakdown_scenario():
    close = np.array([100, 100, 100, 100, 100, 90, 92, 111], dtype=float)
    upper = np.array([np.nan] * 5 + [100, 100, 100])
    lower = np.array([np.nan] * 5 + [100, 100, 100])
    on = _track_short(close, upper, lower)
    assert list(on) == [False, False, False, False, False, True, True, False]


def test_track_long_trailing_stop_ratchets_up():
    # вход на баре 5 (stop=100), mid поднимается до 110 на баре 6 - стоп подтягивается.
    # На баре 7 цена=105: выше первоначального стопа (100), но ниже подтянутого (110) -
    # с трейлингом позиция закрывается, без него - осталась бы открытой.
    close = np.array([100, 100, 100, 100, 100, 110, 130, 105], dtype=float)
    upper = np.array([np.nan] * 5 + [100, 115, 115])
    lower = np.array([np.nan] * 5 + [100, 105, 105])
    on = _track_long(close, upper, lower)
    assert list(on) == [False, False, False, False, False, True, True, False]


# --- синтетические дневные бары для truncation/future-poison/интеграции ---

def _synthetic_bars(n_days=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="1D", tz="UTC")
    ret = rng.normal(0.0003, 0.02, n_days)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    open_ = np.roll(close, 1)
    open_[0] = 100
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)


def test_s01_truncation_and_future_poison_safe():
    bars = _synthetic_bars(300)
    strat = S01(params={"lookback_set": "short", "vol_target": None, "tf": "1d"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)


def test_s01b_truncation_and_future_poison_safe():
    bars = _synthetic_bars(300, seed=1)
    strat = S01b(params={"lookback_set": "short", "vol_target": None, "tf": "1d"})
    assert_truncation_safe(strat, bars, _ctx(), n_checks=20)
    assert_future_poison_safe(strat, bars, _ctx(), n_checks=20)


def test_s01_integration_runs_through_backtest():
    bars_1d = _synthetic_bars(400)
    idx_1h = pd.date_range(bars_1d.index[0], periods=len(bars_1d) * 24, freq="1h", tz="UTC")
    bars_1h = bars_1d.reindex(bars_1d.index.repeat(24))
    bars_1h.index = idx_1h  # грубая, но достаточная для теста сборка 1h из дневных close
    cfg = _cfg()
    strat = S01(params={"lookback_set": "short", "vol_target": 0.25, "tf": "1d"})
    prepared = strat.prepare(bars_1d, _ctx())
    result = run_backtest(strat, {"X": bars_1h}, {"X": prepared}, {}, "1d", cfg, 10_000, 1)
    assert len(result.equity) > 0
    assert not result.equity.isna().any()


def test_s01_target_bounded_0_1_and_s01b_bounded_minus1_1():
    bars = _synthetic_bars(400, seed=2)
    s01 = S01(params={"vol_target": None}).prepare(bars, _ctx())
    s01b = S01b(params={"vol_target": None}).prepare(bars, _ctx())
    assert s01["target"].dropna().between(0, 1).all()
    assert s01b["target"].dropna().between(-1, 1).all()


def test_s01_vol_target_scales_down_exposure():
    bars = _synthetic_bars(400, seed=3)
    no_scale = S01(params={"vol_target": None}).prepare(bars, _ctx())["target"]
    scaled = S01(params={"vol_target": 0.1}).prepare(bars, _ctx())["target"]  # низкая цель - должно урезать
    assert scaled.fillna(0).abs().sum() <= no_scale.fillna(0).abs().sum() + 1e-9
