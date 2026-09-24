"""Тесты для src/lab/metrics/performance.py — Этап 4."""

import pandas as pd
import pytest

from lab.engine.backtest import Trade
from lab.metrics import performance as perf


def _trade(**kwargs) -> Trade:
    defaults = dict(
        symbol="X", side="long", entry_time=pd.Timestamp("2023-01-01", tz="UTC"),
        entry_price=100.0, qty=10.0, exit_time=pd.Timestamp("2023-01-02", tz="UTC"),
        exit_price=110.0, gross_pnl=100.0, fees=1.0, slippage_cost=0.5, funding=-0.2,
        net_pnl=98.8, exit_reason="exit_signal",
    )
    defaults.update(kwargs)
    return Trade(**defaults)


def test_total_return():
    equity = pd.Series([10_000.0, 11_000.0], index=pd.date_range("2023-01-01", periods=2, freq="1D", tz="UTC"))
    assert perf.total_return(equity) == pytest.approx(0.10)


def test_cagr_over_exactly_one_year():
    idx = pd.date_range("2023-01-01", periods=2, freq="365D", tz="UTC")
    equity = pd.Series([10_000.0, 11_000.0], index=idx)
    assert perf.cagr(equity, annualization_days=365) == pytest.approx(0.10, abs=1e-6)


def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=pd.date_range("2023-01-01", periods=4, freq="1D", tz="UTC"))
    assert perf.max_drawdown(equity) == pytest.approx(-0.25)


def test_calmar():
    assert perf.calmar(cagr_value=0.2, maxdd_value=-0.1) == pytest.approx(2.0)
    assert perf.calmar(cagr_value=0.2, maxdd_value=0.0) != perf.calmar(cagr_value=0.2, maxdd_value=0.0)  # nan != nan


def test_sharpe_manual_example():
    rets = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    expected = rets.mean() / rets.std(ddof=0) * (365**0.5)
    assert perf.sharpe(rets, annualization_days=365) == pytest.approx(expected)


def test_sortino_only_penalizes_downside():
    rets = pd.Series([0.05, 0.05, 0.05, -0.01])  # почти всегда растёт
    sharpe_v = perf.sharpe(rets, 365)
    sortino_v = perf.sortino(rets, 365)
    assert sortino_v > sharpe_v  # downside deviation меньше полного std - Sortino выше


def test_trade_stats_win_rate_and_profit_factor():
    trades = [
        _trade(net_pnl=100.0),
        _trade(net_pnl=-50.0),
        _trade(net_pnl=30.0),
    ]
    stats = perf.trade_stats(trades)
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["profit_factor"] == pytest.approx(130 / 50)


def test_trade_stats_ignores_still_open_position():
    trades = [_trade(net_pnl=100.0), _trade(exit_time=None, exit_reason="open_at_end")]
    stats = perf.trade_stats(trades)
    assert stats["trades"] == 1


def test_trade_stats_empty():
    stats = perf.trade_stats([])
    assert stats["trades"] == 0
    assert stats["profit_factor"] != stats["profit_factor"]  # nan


def test_cost_share_formula():
    trades = [_trade(gross_pnl=100.0, fees=1.0, slippage_cost=0.5, funding=-0.2)]
    # (fees + slippage - funding) / gross_profit = (1 + 0.5 - (-0.2)) / 100
    assert perf.cost_share(trades) == pytest.approx(1.7 / 100)


def test_cost_share_no_profit_is_nan():
    trades = [_trade(gross_pnl=-100.0)]
    assert perf.cost_share(trades) != perf.cost_share(trades) or True  # nan-safe
    assert pd.isna(perf.cost_share(trades))


def test_exposure_half_the_period():
    index = pd.date_range("2023-01-01", periods=5, freq="1D", tz="UTC")  # 4 дня диапазона
    trades = [_trade(
        entry_time=pd.Timestamp("2023-01-01", tz="UTC"),
        exit_time=pd.Timestamp("2023-01-03", tz="UTC"),
    )]
    assert perf.exposure(trades, index) == pytest.approx(0.5)


def test_funding_paid_sums_signed_values():
    trades = [_trade(funding=-1.0), _trade(funding=2.0)]
    assert perf.funding_paid(trades) == pytest.approx(1.0)


def test_breadth_fraction_positive():
    assert perf.breadth({"A": 10.0, "B": -5.0, "C": 3.0}) == pytest.approx(2 / 3)


def test_percentile_rank():
    dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert perf.percentile_rank(5.5, dist) == pytest.approx(50.0)
    assert perf.percentile_rank(0.0, dist) == pytest.approx(0.0)
    assert perf.percentile_rank(100.0, dist) == pytest.approx(100.0)


def test_monthly_returns_two_months():
    idx = pd.date_range("2023-01-01", "2023-02-28", freq="1D", tz="UTC")  # ровно 2 полных месяца
    equity = pd.Series(10_000.0, index=idx)
    equity.loc["2023-02-01":] = 11_000.0
    monthly = perf.monthly_returns(equity)
    assert len(monthly) == 1  # доходность января "с нуля" не считается (dropna), остался только февраль
    assert monthly.iloc[-1] == pytest.approx(0.10, abs=1e-9)
