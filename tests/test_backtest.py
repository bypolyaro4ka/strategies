"""Интеграционные тесты для src/lab/engine/backtest.py — Этап 3.

Benchmark sanity, Determinism, Truncation, Future-poison из 02_ENGINE_SPEC.md, раздел 7.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from lab.data.resample import resample_ohlcv
from lab.engine.backtest import run_backtest
from lab.engine.leak_tests import assert_future_poison_safe, assert_truncation_safe
from lab.strategies.base import Context

from stubs import (
    AlwaysLongStub, EvenDayFlipStub, EvenDayLongStub, FlatStub, PortfolioTopBottomStub, RollingMeanStub,
)


def _cfg(fee_taker=0.0005, slippage=0.0002, slot_fraction=0.10, min_target_change=0.05):
    return SimpleNamespace(
        account=SimpleNamespace(slot_fraction=slot_fraction, min_target_change=min_target_change),
        exchange=SimpleNamespace(fee_taker=fee_taker, slippage=slippage),
    )


def _flat_bars(n_hours: int, start="2023-01-01", price0=100.0) -> pd.DataFrame:
    """Синтетические 1h-бары с open=high=low=close, растущие на 1 за час - удобно
    считать вручную. volume/trades - произвольные ненулевые значения."""
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    price = [price0 + i for i in range(n_hours)]
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price,
        "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)


def _signal_bars(bars_1h: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv(bars_1h, "1d")


def _ctx(cfg):
    return Context(params={}, protocol=cfg)


# --- Benchmark sanity: Flat = 0 ---

def test_flat_benchmark_equity_never_moves():
    bars = _flat_bars(72)
    cfg = _cfg()
    result = run_backtest(
        strategy=FlatStub(), bars_1h={"X": bars}, signal_bars={"X": _signal_bars(bars)},
        funding={}, tf="1d", protocol_cfg=cfg, equity0=10_000, max_slots=1,
    )
    assert (result.equity == 10_000).all()
    assert result.trades == []


# --- Benchmark sanity: Buy & Hold без издержек совпадает с ценой ---

def test_buy_hold_no_cost_matches_price_ratio():
    bars = _flat_bars(72)  # 3 полных дня
    cfg = _cfg(fee_taker=0.0, slippage=0.0, slot_fraction=1.0)  # без издержек, вся equity в позиции
    result = run_backtest(
        strategy=AlwaysLongStub(), bars_1h={"X": bars}, signal_bars={"X": _signal_bars(bars)},
        funding={}, tf="1d", protocol_cfg=cfg, equity0=10_000, max_slots=1,
    )
    # первый сигнал - по закрытию дня 1 (2023-01-01 23:00), исполнение на open дня 2
    entry_price = bars.loc["2023-01-02 00:00:00+00:00", "open"]
    final_price = bars["close"].iloc[-1]
    expected_final_equity = 10_000 * (final_price / entry_price)
    assert result.equity.iloc[-1] == pytest.approx(expected_final_equity)
    # до входа (весь день 1) equity не менялась - позиции ещё не было
    assert (result.equity.loc[:"2023-01-01 23:00:00+00:00"] == 10_000).all()


def test_buy_hold_with_costs_is_worse_than_without():
    bars = _flat_bars(72)
    cfg_free = _cfg(fee_taker=0.0, slippage=0.0, slot_fraction=1.0)
    cfg_paid = _cfg(fee_taker=0.0005, slippage=0.0002, slot_fraction=1.0)
    r_free = run_backtest(AlwaysLongStub(), {"X": bars}, {"X": _signal_bars(bars)}, {}, "1d", cfg_free, 10_000, 1)
    r_paid = run_backtest(AlwaysLongStub(), {"X": bars}, {"X": _signal_bars(bars)}, {}, "1d", cfg_paid, 10_000, 1)
    assert r_paid.equity.iloc[-1] < r_free.equity.iloc[-1]


# --- Determinism: два прогона с одинаковым входом дают идентичный результат ---

def test_determinism_two_identical_runs():
    bars = _flat_bars(96)
    cfg = _cfg()
    args = (EvenDayLongStub(), {"X": bars}, {"X": _signal_bars(bars)}, {}, "1d", cfg, 10_000, 1)
    r1 = run_backtest(*args)
    r2 = run_backtest(*args)
    pd.testing.assert_series_equal(r1.equity, r2.equity)
    assert len(r1.trades) == len(r2.trades)
    assert [t.net_pnl for t in r1.trades] == [t.net_pnl for t in r2.trades]


# --- Truncation / future-poison (на стратегии с реальным индикатором - rolling SMA) ---

def test_truncation_safe_on_rolling_mean_stub():
    bars = _flat_bars(200)
    strategy = RollingMeanStub()
    assert_truncation_safe(strategy, bars, _ctx(_cfg()), n_checks=30)


def test_future_poison_safe_on_rolling_mean_stub():
    bars = _flat_bars(200)
    strategy = RollingMeanStub()
    assert_future_poison_safe(strategy, bars, _ctx(_cfg()), n_checks=30)


# --- Позиция, ещё открытая на конец периода, попадает в trades снимком ---

def test_open_position_at_end_is_captured_as_snapshot():
    bars = _flat_bars(72)  # AlwaysLong открывает на день 2 и держит до конца данных
    cfg = _cfg()
    result = run_backtest(
        AlwaysLongStub(), {"X": bars}, {"X": _signal_bars(bars)}, {}, "1d", cfg, 10_000, 1,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_time is None
    assert t.exit_reason == "open_at_end"
    assert t.gross_pnl > 0  # цена росла, позиция в плюсе


# --- Slot contention через весь цикл (не только изолированная allocate_slots) ---

def test_slot_contention_through_full_loop():
    cfg = _cfg()
    symbols = ["ZZZ", "AAA", "BBB"]  # намеренно не по алфавиту - проверяем сортировку
    bars = {s: _flat_bars(72) for s in symbols}
    signal_bars = {s: _signal_bars(b) for s, b in bars.items()}
    result = run_backtest(
        AlwaysLongStub(), bars, signal_bars, {}, "1d", cfg, equity0=10_000, max_slots=2,
    )
    entered = {o.symbol for o in result.orders if o.reason == "signal" and o.delta_qty > 0}
    assert entered == {"AAA", "BBB"}  # первые два по алфавиту из трёх кандидатов, ZZZ - нет


# --- Reversal: разворот target через 0 закрывает старую сделку и открывает новую ---

def test_reversal_splits_into_two_closed_trades():
    bars = _flat_bars(96)  # 4 дня: нечётный/чётный/нечётный/чётный -> 3 разворота после входа
    cfg = _cfg()
    result = run_backtest(
        EvenDayFlipStub(), {"X": bars}, {"X": _signal_bars(bars)}, {}, "1d", cfg, 10_000, 1,
    )
    # ни одна сделка не должна повиснуть "открытой без exit_time" из-за разворота -
    # только сама последняя (ещё не закрытая на конец периода) вправе иметь exit_time=None
    closed = [t for t in result.trades if t.exit_time is not None]
    assert len(closed) >= 2
    for t in closed:
        assert t.exit_reason in ("reversal", "exit_signal")
        assert t.fees > 0
    # сторона сделки должна чередоваться long/short, а не оставаться одной и той же
    sides = [t.side for t in result.trades]
    assert len(set(sides)) == 2


# --- PortfolioStrategy (S13) - движок должен уметь звать on_bar() со всеми монетами сразу ---

def test_portfolio_strategy_ranks_across_all_symbols():
    n = 72
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    flat = pd.Series([100.0] * n, index=idx)
    up = pd.Series([100.0 + i * 0.2 for i in range(n)], index=idx)  # растёт - должен стать лонгом
    down = pd.Series([200.0 - i * 0.2 for i in range(n)], index=idx)  # падает - должен стать шортом
    bars = {}
    for sym, series in (("AAA", flat), ("BBB", up), ("CCC", down)):
        bars[sym] = pd.DataFrame({
            "open": series, "high": series, "low": series, "close": series,
            "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True,
        }, index=idx)
    cfg = _cfg(slot_fraction=0.30)
    signal_bars = {s: _signal_bars(b) for s, b in bars.items()}
    result = run_backtest(
        PortfolioTopBottomStub(), bars, signal_bars, {}, "1d", cfg, equity0=10_000, max_slots=3,
    )
    long_entries = {o.symbol for o in result.orders if o.reason == "signal" and o.delta_qty > 0 and o.target_after > 0}
    short_entries = {o.symbol for o in result.orders if o.reason == "signal" and o.delta_qty < 0 and o.target_after < 0}
    assert "BBB" in long_entries  # растущая монета когда-то получила лонг
    assert "CCC" in short_entries  # падающая монета когда-то получила шорт


# --- Sanity по сделкам: even-day стратегия открывает и закрывает позиции ---

def test_even_day_strategy_produces_trades_with_costs():
    bars = _flat_bars(96)  # 4 дня
    cfg = _cfg()
    result = run_backtest(
        EvenDayLongStub(), {"X": bars}, {"X": _signal_bars(bars)}, {}, "1d", cfg, 10_000, 1,
    )
    assert len(result.trades) >= 1
    for trade in result.trades:
        assert trade.fees > 0  # издержки реально списались
        assert trade.slippage_cost > 0  # и проскальзывание учтено отдельной строкой
        assert trade.exit_reason == "exit_signal"
