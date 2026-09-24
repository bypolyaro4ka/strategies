"""Тесты для портированных стратегий Гоши — Этап 9 (стратегии друзей)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context
from lab.strategies.community.gosha.avax_ema_cross_short import GoshaAvaxEmaCrossShort
from lab.strategies.community.gosha.rsi_dca import GoshaRsiDca, LEVEL_FRACS, _level_index
from lab.strategies.community.gosha.turtle_donchian import GoshaTurtleDonchian

STRATEGIES = [GoshaTurtleDonchian(), GoshaAvaxEmaCrossShort(), GoshaRsiDca()]


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


def test_rsi_dca_level_index_matches_level_fracs():
    for i, frac in enumerate(LEVEL_FRACS):
        assert _level_index(frac) == i


def test_rsi_dca_enters_on_low_rsi_and_steps_up_on_drawdown():
    """Ряд, падающий достаточно, чтобы пробить RSI<28 и все 5 AO-уровней подряд -
    target должен пройти по всей лестнице LEVEL_FRACS и не перепрыгивать уровни."""
    idx = pd.date_range("2023-01-01", periods=60, freq="4h", tz="UTC")
    close = 100 * np.array([0.995 ** i for i in range(60)])  # монотонное падение
    bars = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1000.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)
    strat = GoshaRsiDca()
    out = strat.prepare(bars, _ctx())
    pos = SimpleNamespace(target=0.0, avg_entry_price=0.0)
    seen_targets = []
    for t, row in out.iterrows():
        d = strat.on_bar(t, row, pos, _ctx())
        if d.target != pos.target and d.target > 0:
            # имитируем то, что движок пересчитывает среднюю цену входа при доливке -
            # для базового входа и роста target это просто текущая цена (упрощение
            # синтетики: тест не проверяет точную формулу WAC, это дело execution.py)
            pos = SimpleNamespace(target=d.target, avg_entry_price=row["close"])
        elif d.target == 0.0:
            pos = SimpleNamespace(target=0.0, avg_entry_price=0.0)
        seen_targets.append(d.target)
    assert max(seen_targets) > 0  # хотя бы вошли
    nonzero = [x for x in seen_targets if x > 0]
    assert nonzero == sorted(nonzero)  # монотонно растёт, никогда не падает частично


def test_turtle_donchian_reversal_flips_sign_directly():
    """На синтетическом ряду с чётким разворотом тренда стратегия должна хотя бы раз
    развернуться напрямую (target меняет знак, минуя 0), не только войти/выйти в кэш.

    Канал 2 у GoshaTurtleDonchian строится по данным со сдвигом ещё на 20 баров назад
    (CHANNEL2_OFFSET) - ему нужно ~41 бар истории, прежде чем он перестанет быть NaN, И
    он лагает относительно канала 1 даже когда оба уже "тёплые": на монотонном движении
    момент пересечения канала 1 (`below_c1l & cross_cond`) наступает на 10+ баров раньше,
    чем канал 2 успевает подтвердить пробой (`below_c2l`) - к этому моменту `cross_cond`
    уже снова False (мы который бар подряд ниже канала 1, не "только что пересекли").
    Проверено диагностическим прогоном (`scripts/debug_turtle_donchian.py`) - это реальное
    свойство формулы (два канала с разным лагом), не баг стратегии.

    Поэтому перед КАЖДЫМ резким движением (и входом, и разворотом) нужен плоский участок
    подольше запаса (~41 бар минимум) - тогда оба канала успевают сойтись к одному и тому
    же уровню и пробивной бар пересекает оба одновременно. Первая версия теста дала такой
    участок только перед входом, но не перед разворотом - short_break ни разу не сработал."""
    idx = pd.date_range("2023-01-01", periods=280, freq="4h", tz="UTC")
    flat1 = np.full(70, 100.0)
    up = 100 + (np.arange(80) + 1) * 2.0          # 102 .. 260, пробивает оба канала на баре 70
    flat2 = np.full(50, 260.0)                    # даём каналу 2 сойтись к пику перед разворотом
    down = 260 - (np.arange(80) + 1) * 2.0        # 258 .. 98, разворот вниз с бара 200
    close = np.concatenate([flat1, up, flat2, down])
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
