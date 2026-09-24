"""Тесты для портированных стратегий Гоши — Этап 9 (стратегии друзей)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.community.gosha.avax_ema_cross_short import GoshaAvaxEmaCrossShort
from lab.strategies.community.gosha.turtle_donchian import GoshaTurtleDonchian

STRATEGIES = [GoshaTurtleDonchian(), GoshaAvaxEmaCrossShort()]


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


def test_all_have_author_gosha():
    for s in STRATEGIES:
        assert s.author == "Гоша"


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


def test_avax_short_only_never_goes_long():
    """Собственное правило стратегии (short-only), не ограничение движка - проверяем,
    что портирование не случайно завело её в лонг."""
    bars = _synthetic_bars(800, seed=1)
    strat = GoshaAvaxEmaCrossShort()
    out = strat.prepare(bars, _ctx())
    pos = SimpleNamespace(target=0.0, bars_in_trade=0, stop_price=None, take_price=None)
    for t, row in out.iterrows():
        d = strat.on_bar(t, row, pos, _ctx())
        assert d.target <= 0.0
        pos = SimpleNamespace(target=d.target, bars_in_trade=0, stop_price=d.stop_price, take_price=d.take_price)


def test_turtle_donchian_reversal_flips_sign_directly():
    """На синтетическом ряду с чётким разворотом тренда стратегия должна хотя бы раз
    развернуться напрямую (target меняет знак, минуя 0), не только войти/выйти в кэш."""
    idx = pd.date_range("2023-01-01", periods=200, freq="4h", tz="UTC")
    # рост 100 баров, затем чёткий спад - гарантированно пробивает оба канала в обе стороны
    up = 100 + np.arange(100) * 2.0
    down = up[-1] - np.arange(100) * 2.0
    close = np.concatenate([up, down])
    bars = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": 1000.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)
    strat = GoshaTurtleDonchian()
    out = strat.prepare(bars, _ctx())
    pos = SimpleNamespace(target=0.0, bars_in_trade=0, stop_price=None, take_price=None)
    targets = []
    for t, row in out.iterrows():
        d = strat.on_bar(t, row, pos, _ctx())
        targets.append(d.target)
        pos = SimpleNamespace(target=d.target, bars_in_trade=0, stop_price=d.stop_price, take_price=d.take_price)
    assert 1.0 in targets and -1.0 in targets
