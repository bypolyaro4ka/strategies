"""Тесты для src/lab/benchmarks/ — Этап 4."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.benchmarks.buy_hold import BuyHoldStrategy, run_buy_hold_portfolio
from lab.benchmarks.flat import FlatStrategy
from lab.benchmarks.random_entry import RandomEntryStrategy, random_benchmark_sharpes, trade_profile
from lab.data.resample import resample_ohlcv
from lab.engine.backtest import Trade, run_backtest


def _cfg(fee_taker=0.0005, slippage=0.0002, slot_fraction=0.10, min_target_change=0.05, max_slots=5, ann_days=365):
    return SimpleNamespace(
        account=SimpleNamespace(
            slot_fraction=slot_fraction, min_target_change=min_target_change, max_slots_portfolio=max_slots,
        ),
        exchange=SimpleNamespace(fee_taker=fee_taker, slippage=slippage),
        metrics=SimpleNamespace(annualization_days=ann_days),
    )


def _flat_bars(n_hours, start="2023-01-01", price0=100.0):
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    price = [price0 + i for i in range(n_hours)]
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price,
        "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)


# --- Flat ---

def test_flat_strategy_equity_never_moves():
    bars = _flat_bars(72)
    cfg = _cfg()
    result = run_backtest(FlatStrategy(), {"X": bars}, {"X": resample_ohlcv(bars, "1d")}, {}, "1d", cfg, 10_000, 1)
    assert (result.equity == 10_000).all()


# --- Buy & Hold, режим "пара" ---

def test_buy_hold_pair_matches_price_growth_without_costs():
    bars = _flat_bars(72)
    cfg = _cfg(fee_taker=0.0, slippage=0.0, slot_fraction=1.0)
    result = run_backtest(BuyHoldStrategy(), {"X": bars}, {"X": resample_ohlcv(bars, "1d")}, {}, "1d", cfg, 10_000, 1)
    entry_price = bars.loc["2023-01-02 00:00:00+00:00", "open"]
    expected = 10_000 * (bars["close"].iloc[-1] / entry_price)
    assert result.equity.iloc[-1] == pytest.approx(expected)


# --- Buy & Hold, режим "портфель": весь пул, равный вес, без слотов ---

def test_buy_hold_portfolio_weights_and_no_slot_limit():
    cfg = _cfg(max_slots=5)  # 5 слотов, но в пуле 10 "монет" - B&H всё равно берёт все
    symbols = [f"C{i}" for i in range(10)]
    bars = {s: _flat_bars(48) for s in symbols}
    result = run_buy_hold_portfolio(bars, cfg, equity0=10_000)

    assert len(result.trades) == 10  # все 10, а не только 5 (слот-кап тут не действует)
    # вес каждой монеты = slot_fraction * max_slots / n_pool = 0.10*5/10 = 5%
    expected_weight = cfg.account.slot_fraction * cfg.account.max_slots_portfolio / len(symbols)
    for t in result.trades:
        notional = t.qty * t.entry_price
        assert notional == pytest.approx(expected_weight * 10_000, rel=1e-6)


def test_buy_hold_portfolio_handles_symbols_with_different_start_dates():
    cfg = _cfg()
    early = _flat_bars(72, start="2023-01-01")
    late = _flat_bars(24, start="2023-01-04")  # "HYPE" - появился на 4 дня позже
    result = run_buy_hold_portfolio({"EARLY": early, "LATE": late}, cfg, equity0=10_000)
    late_trade = next(t for t in result.trades if t.symbol == "LATE")
    assert late_trade.entry_time == late.index[0]  # куплена на СВОЁМ первом баре, не общем


# --- Random entry ---

def test_random_entry_prepare_creates_roughly_requested_number_of_trades():
    bars = _flat_bars(24 * 60)  # 60 дней
    signal_bars = resample_ohlcv(bars, "1d")
    strat = RandomEntryStrategy(n_trades=10, avg_holding_bars=3, long_fraction=1.0, seed=42, tf="1d")
    prepared = strat.prepare(signal_bars, None)
    entries = [v for v in strat._schedule.values() if v != 0.0]
    assert 1 <= len(entries) <= 10  # может быть меньше, если не хватило места в ряду


def test_random_entry_deterministic_by_seed():
    bars = _flat_bars(24 * 30)
    signal_bars = resample_ohlcv(bars, "1d")
    s1 = RandomEntryStrategy(5, 3, 0.5, seed=7, tf="1d")
    s2 = RandomEntryStrategy(5, 3, 0.5, seed=7, tf="1d")
    s1.prepare(signal_bars, None)
    s2.prepare(signal_bars, None)
    assert s1._schedule == s2._schedule


def test_trade_profile_extracts_stats():
    trades = [
        Trade("X", "long", pd.Timestamp("2023-01-01", tz="UTC"), 100, 1,
              exit_time=pd.Timestamp("2023-01-03", tz="UTC")),
        Trade("X", "short", pd.Timestamp("2023-01-05", tz="UTC"), 100, 1,
              exit_time=pd.Timestamp("2023-01-06", tz="UTC")),
    ]
    profile = trade_profile(trades)
    assert profile["n_trades"] == 2
    assert profile["long_fraction"] == pytest.approx(0.5)
    assert profile["avg_holding_hours"] == pytest.approx((48 + 24) / 2)


def test_trade_profile_empty_trades():
    assert trade_profile([]) == {"n_trades": 0, "avg_holding_hours": 0.0, "long_fraction": 0.5}


def test_random_benchmark_sharpes_runs_and_returns_array():
    bars = _flat_bars(24 * 40)
    signal_bars = resample_ohlcv(bars, "1d")
    cfg = _cfg()
    sharpes = random_benchmark_sharpes(
        {"X": bars}, {"X": signal_bars}, {}, "1d", cfg, 10_000, max_slots=1,
        n_trades=5, avg_holding_bars=3, long_fraction=0.5, n_seeds=5,
    )
    assert len(sharpes) == 5
    assert isinstance(sharpes, np.ndarray)
