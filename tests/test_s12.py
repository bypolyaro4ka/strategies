"""Тесты для S12 (03_STRATEGIES.md) — Этап 6, партия D."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context, PositionState
from lab.strategies.s12_funding_contrarian import S12


def _ctx(funding=None):
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg, funding=funding)


def _synthetic_bars(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.01, n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def _synthetic_funding(bars_idx, seed=0, spike_at=None, spike_rate=0.0008) -> pd.DataFrame:
    """Отметки фандинга каждые 8h, обычная ставка ~0.0001 с небольшим шумом, опционально
    один экстремальный всплеск (для проверки входа)."""
    rng = np.random.default_rng(seed)
    f_idx = pd.date_range(bars_idx[0], bars_idx[-1] + pd.Timedelta(hours=8), freq="8h", tz="UTC")
    rate = rng.normal(0.0001, 0.00002, len(f_idx))
    if spike_at is not None:
        rate[spike_at] = spike_rate
    return pd.DataFrame({"funding_rate": rate}, index=f_idx).rename_axis("funding_time")


def test_no_funding_stays_flat():
    bars = _synthetic_bars(200, seed=1)
    strat = S12()
    out = strat.prepare(bars, _ctx(funding=None))
    assert out["z"].isna().all()
    pos = PositionState(symbol="X")
    d = strat.on_bar(out.index[-1], out.iloc[-1], pos, _ctx(funding=None))
    assert d.target == 0.0
    assert d.tag == "warmup"


def test_extreme_positive_funding_spike_triggers_short():
    bars = _synthetic_bars(300, seed=2)
    funding = _synthetic_funding(bars.index, seed=2, spike_at=len(bars) // 16, spike_rate=0.002)
    strat = S12(params={"z_in": 2.0, "hold": 18, "confirm": "none"})
    ctx = _ctx(funding=funding)
    out = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    # где-то после всплеска z должен превысить порог хотя бы один раз
    triggered = False
    for t, row in out.iterrows():
        d = strat.on_bar(t, row, pos, ctx)
        if d.target == -1.0:
            triggered = True
            break
    assert triggered


def test_exit_when_z_reverts_or_time_stop():
    bars = _synthetic_bars(100, seed=3)
    strat = S12(params={"z_in": 2.0, "hold": 5, "confirm": "none"})
    out = strat.prepare(bars, _ctx(funding=None))
    row = out.iloc[-1].copy()
    row["z"], row["f8"] = 0.1, 0.0001  # z вернулся к норме
    pos = PositionState(symbol="X", target=-1.0, bars_in_trade=1)
    d = strat.on_bar(out.index[-1], row, pos, _ctx(funding=None))
    assert d.target == 0.0
    assert d.tag == "exit"

    row2 = row.copy()
    row2["z"] = 3.0  # всё ещё экстремально, но пересидели hold
    pos2 = PositionState(symbol="X", target=-1.0, bars_in_trade=5)
    d2 = strat.on_bar(out.index[-1], row2, pos2, _ctx(funding=None))
    assert d2.target == 0.0
    assert d2.tag == "exit"


def test_s12_truncation_and_future_poison_safe():
    bars = _synthetic_bars(400, seed=4)
    funding = _synthetic_funding(bars.index, seed=4, spike_at=150, spike_rate=0.0015)
    strat = S12(params={"z_in": 2.0, "hold": 18, "confirm": "none"})
    ctx = _ctx(funding=funding)
    assert_truncation_safe(strat, bars, ctx, n_checks=20)
    assert_future_poison_safe(strat, bars, ctx, n_checks=20)
