"""Тесты для O1/O2 (03_STRATEGIES.md) — Этап 6, надстройки."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.strategies.base import BaseStrategy, Context, Decision, PositionState, WithOverlays
from lab.strategies.overlays import BtcRegimeFilter, VolTarget


def _ctx(btc_bars=None):
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg, btc_bars=btc_bars)


class _AlwaysLongStub(BaseStrategy):
    id, name, version = "STUB", "always_long", "1.0.0"
    timeframes = ["4h"]
    direction = "long_only"

    def prepare(self, bars, ctx):
        return bars

    def on_bar(self, t, row, pos, ctx):
        return Decision(target=1.0, tag="entry")


def _bars_4h(n=60, price0=100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    price = [price0] * n
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def _btc_daily(n=20, trend="up") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    if trend == "up":
        close = [100.0 + i * 5 for i in range(n)]  # уверенно выше своей SMA
    else:
        close = [100.0 - i * 5 for i in range(n)]  # уверенно ниже своей SMA
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)


def test_o1_blocks_long_when_btc_below_sma():
    bars = _bars_4h(60)
    btc = _btc_daily(20, trend="down")
    overlay = BtcRegimeFilter(sma_days=5)
    strat = WithOverlays(_AlwaysLongStub(), [overlay])
    ctx = _ctx(btc_bars=btc)
    prepared = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    row = prepared.iloc[-1]
    assert row["btc_regime"] == -1.0  # BTC в нисходящем режиме
    d = strat.on_bar(prepared.index[-1], row, pos, ctx)
    assert d.target == 0.0
    assert "O1_blocked_bear" in d.tag


def test_o1_allows_long_when_btc_above_sma():
    bars = _bars_4h(60)
    btc = _btc_daily(20, trend="up")
    overlay = BtcRegimeFilter(sma_days=5)
    strat = WithOverlays(_AlwaysLongStub(), [overlay])
    ctx = _ctx(btc_bars=btc)
    prepared = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    row = prepared.iloc[-1]
    assert row["btc_regime"] == 1.0
    d = strat.on_bar(prepared.index[-1], row, pos, ctx)
    assert d.target == 1.0  # не заблокировано


def test_o1_no_btc_bars_is_noop():
    bars = _bars_4h(60)
    overlay = BtcRegimeFilter(sma_days=5)
    strat = WithOverlays(_AlwaysLongStub(), [overlay])
    ctx = _ctx(btc_bars=None)
    prepared = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    d = strat.on_bar(prepared.index[-1], prepared.iloc[-1], pos, ctx)
    assert d.target == 1.0  # без BTC-данных надстройка ничего не блокирует


def test_o2_scales_down_high_volatility_never_up():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=150, freq="1D", tz="UTC")
    # высокая волатильность - большие случайные скачки
    close = 100 * np.cumprod(1 + rng.normal(0, 0.08, 150))
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)
    overlay = VolTarget(sigma_star=0.5, W=90)
    strat = WithOverlays(_AlwaysLongStub(), [overlay])
    ctx = _ctx()
    prepared = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    row = prepared.iloc[-1]
    assert pd.notna(row["vol_scale"])
    assert row["vol_scale"] <= 1.0  # никогда не увеличивает экспозицию
    d = strat.on_bar(prepared.index[-1], row, pos, ctx)
    assert 0.0 <= d.target <= 1.0
    if row["vol_scale"] < 1.0:
        assert d.target < 1.0  # реально урезало таргет


def test_o2_low_volatility_keeps_full_target():
    idx = pd.date_range("2023-01-01", periods=150, freq="1D", tz="UTC")
    close = [100.0 + i * 0.001 for i in range(150)]  # почти неподвижная цена - низкая vol
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True}, index=idx)
    overlay = VolTarget(sigma_star=0.5, W=90)
    strat = WithOverlays(_AlwaysLongStub(), [overlay])
    ctx = _ctx()
    prepared = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    row = prepared.iloc[-1]
    d = strat.on_bar(prepared.index[-1], row, pos, ctx)
    assert d.target == 1.0  # вола ниже целевой - масштаб 1.0, targeting не режет


def test_combined_overlays_apply_in_order():
    bars = _bars_4h(60)
    btc = _btc_daily(20, trend="up")  # не блокирует
    strat = WithOverlays(_AlwaysLongStub(), [BtcRegimeFilter(sma_days=5), VolTarget(sigma_star=10.0, W=5)])
    ctx = _ctx(btc_bars=btc)
    prepared = strat.prepare(bars, ctx)
    pos = PositionState(symbol="X")
    row = prepared.iloc[-1]
    d = strat.on_bar(prepared.index[-1], row, pos, ctx)
    # BTC бычий (не блокирует), цена абсолютно плоская (vol=0 -> sigma_star/0 клипается к 1.0)
    assert d.target == 1.0
    assert "O1" not in d.tag
    assert "O2_scaled" in d.tag
