"""Метрики производительности (01_PROTOCOL.md, раздел 8).

Считаются по почасовой кривой equity, агрегированной до дневных доходностей (Sharpe,
Sortino, Vol) — MaxDD берём по самой почасовой кривой, точнее, чем по дневной."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_equity(equity: pd.Series) -> pd.Series:
    """Почасовая equity -> дневная (последнее значение за день)."""
    return equity.resample("1D").last().ffill()


def daily_returns(equity: pd.Series) -> pd.Series:
    return daily_equity(equity).pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return float("nan")
    return equity.iloc[-1] / equity.iloc[0] - 1.0


def cagr(equity: pd.Series, annualization_days: int) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    n_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if n_days <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (annualization_days / n_days) - 1.0


def volatility(daily_rets: pd.Series, annualization_days: int) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    return float(daily_rets.std(ddof=0) * np.sqrt(annualization_days))


def sharpe(daily_rets: pd.Series, annualization_days: int, risk_free: float = 0.0) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    excess = daily_rets - risk_free / annualization_days
    std = excess.std(ddof=0)
    if std < 1e-12:
        # "== 0", а не допуск, здесь бы иногда не срабатывал на почти-константных рядах
        # из-за погрешности плавающей точки (см. lab/metrics/dsr.py) - тот же риск и тут
        return float("nan")
    return float(excess.mean() / std * np.sqrt(annualization_days))


def sortino(daily_rets: pd.Series, annualization_days: int, risk_free: float = 0.0) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    excess = daily_rets - risk_free / annualization_days
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev < 1e-12:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(annualization_days))


def max_drawdown(equity: pd.Series) -> float:
    """Отрицательное число (например, -0.25 = просадка 25%)."""
    if len(equity) < 2:
        return float("nan")
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calmar(cagr_value: float, maxdd_value: float) -> float:
    if maxdd_value == 0 or np.isnan(maxdd_value):
        return float("nan")
    return cagr_value / abs(maxdd_value)


def _closed_trades(trades: list) -> list:
    return [t for t in trades if t.exit_time is not None]


def trade_stats(trades: list) -> dict:
    """Trades, Win rate, Profit factor, Avg trade gross/net (б.п.) — 01_PROTOCOL.md §8.
    Только реально закрытые сделки (позиция, открытая на конец периода — не "сделка")."""
    closed = _closed_trades(trades)
    n = len(closed)
    if n == 0:
        return {
            "trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
            "avg_trade_gross_bp": float("nan"), "avg_trade_net_bp": float("nan"),
        }
    wins = [t for t in closed if t.net_pnl > 0]
    losses = [t for t in closed if t.net_pnl < 0]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)

    def bps(t, pnl_attr):
        base = t.qty * t.entry_price
        return (getattr(t, pnl_attr) / base) * 10_000 if base else float("nan")

    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_trade_gross_bp": float(np.mean([bps(t, "gross_pnl") for t in closed])),
        "avg_trade_net_bp": float(np.mean([bps(t, "net_pnl") for t in closed])),
    }


def cost_share(trades: list) -> float:
    """(комиссии + проскальзывание + фандинг) / валовая прибыль (01_PROTOCOL.md §8).
    Фандинг вычитается: полученный фандинг (funding > 0) уменьшает издержки, уплаченный —
    увеличивает. Учитывает и ещё не закрытые на конец периода позиции — издержки по ним
    уже реально понесены."""
    gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    if gross_profit <= 0:
        return float("nan")
    total_costs = sum(t.fees + t.slippage_cost - t.funding for t in trades)
    return total_costs / gross_profit


def exposure(trades: list, index: pd.DatetimeIndex) -> float:
    """Доля времени с ненулевой позицией (01_PROTOCOL.md §8)."""
    if len(index) < 2:
        return 0.0
    total_span = (index[-1] - index[0]).total_seconds()
    if total_span <= 0:
        return 0.0
    covered = sum(
        max(((t.exit_time or index[-1]) - t.entry_time).total_seconds(), 0.0)
        for t in trades
    )
    return min(covered / total_span, 1.0)


def funding_paid(trades: list) -> float:
    """Суммарный фандинг (отдельно от других издержек, 01_PROTOCOL.md §8).
    Отрицательное значение — в среднем платили, положительное — получали."""
    return sum(t.funding for t in trades)


def breadth(pnl_by_symbol: dict[str, float]) -> float:
    """Доля монет в плюсе — только для режима «пара» (01_PROTOCOL.md §8)."""
    if not pnl_by_symbol:
        return float("nan")
    return sum(1 for v in pnl_by_symbol.values() if v > 0) / len(pnl_by_symbol)


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Доходность по месяцам (01_PROTOCOL.md §8, "Monthly table")."""
    monthly = equity.resample("ME").last().ffill()
    return monthly.pct_change().dropna()


def percentile_rank(value: float, distribution: list[float] | np.ndarray) -> float:
    """Перцентиль value внутри distribution, 0-100 (для Random percentile,
    01_PROTOCOL.md §8)."""
    arr = np.asarray(distribution)
    if len(arr) == 0:
        return float("nan")
    return float((arr < value).mean() * 100)


def compute_all(
    equity: pd.Series,
    trades: list,
    annualization_days: int,
    risk_free: float = 0.0,
    pnl_by_symbol: dict[str, float] | None = None,
) -> dict:
    """Собирает все метрики раздела 8 в один dict — то, что пишется в metrics.json
    (02_ENGINE_SPEC.md, раздел 5)."""
    d_rets = daily_returns(equity)
    cagr_v = cagr(equity, annualization_days)
    mdd_v = max_drawdown(equity)
    metrics = {
        "total_return": total_return(equity),
        "cagr": cagr_v,
        "vol": volatility(d_rets, annualization_days),
        "sharpe": sharpe(d_rets, annualization_days, risk_free),
        "sortino": sortino(d_rets, annualization_days, risk_free),
        "maxdd": mdd_v,
        "calmar": calmar(cagr_v, mdd_v),
        "cost_share": cost_share(trades),
        "exposure": exposure(trades, equity.index),
        "funding_paid": funding_paid(trades),
    }
    metrics.update(trade_stats(trades))
    if pnl_by_symbol is not None:
        metrics["breadth"] = breadth(pnl_by_symbol)
    return metrics
