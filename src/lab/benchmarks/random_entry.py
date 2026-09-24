"""Random — случайные входы с тем же числом сделок, средним удержанием и долей
лонгов/шортов, что у сравниваемой стратегии (01_PROTOCOL.md, раздел 7).

Реализовано как обычная BaseStrategy с заранее сгенерированным (в prepare(), по сиду)
расписанием входов/выходов — так весь прогон идёт через тот же run_backtest(), с теми
же издержками/фандингом/слотами, что и у настоящей стратегии. Это и есть смысл фразы
"посчитанными в тех же условиях" из протокола — не отдельная урезанная симуляция.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class RandomEntryStrategy(BaseStrategy):
    id = "BENCH_RANDOM"
    name = "random_entry"
    version = "1.0.0"
    direction = "long_short"

    def __init__(self, n_trades: int, avg_holding_bars: float, long_fraction: float, seed: int, tf: str = "1d"):
        super().__init__()
        self.n_trades = n_trades
        self.avg_holding_bars = max(avg_holding_bars, 1.0)
        self.long_fraction = min(max(long_fraction, 0.0), 1.0)
        self.seed = seed
        self.timeframes = [tf]
        self._schedule: dict[int, float] = {}

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = len(bars)
        schedule: dict[int, float] = {}
        pos_idx = 0
        for _ in range(self.n_trades):
            if pos_idx >= n - 1:
                break
            # экспоненциальное распределение удержания - как и у настоящих сделок,
            # большинство коротких, редкие длинные "тянут" среднее
            hold = max(1, int(round(rng.exponential(self.avg_holding_bars))))
            is_long = rng.random() < self.long_fraction
            exit_idx = min(pos_idx + hold, n - 1)
            schedule[pos_idx] = 1.0 if is_long else -1.0
            schedule[exit_idx] = 0.0
            pos_idx = exit_idx + 1
        self._schedule = schedule

        out = bars.copy()
        out["_bar_idx"] = np.arange(n)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        target = self._schedule.get(int(row["_bar_idx"]), pos.target)
        return Decision(target=target, tag="random")


def random_benchmark_sharpes(
    bars_1h: dict[str, pd.DataFrame],
    signal_bars_raw: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
    tf: str,
    protocol_cfg,
    equity0: float,
    max_slots: int,
    n_trades: int,
    avg_holding_bars: float,
    long_fraction: float,
    n_seeds: int = 1000,
) -> np.ndarray:
    """Прогоняет n_seeds случайных стратегий с тем же профилем сделок, что у реальной
    стратегии, и возвращает массив их Шарпов — распределение для Random percentile
    (01_PROTOCOL.md, раздел 8). signal_bars_raw — бары на сигнальном ТФ БЕЗ prepare()
    (расписание входов у каждого сида своё, готовится здесь)."""
    from lab.engine.backtest import run_backtest
    from lab.metrics.performance import daily_returns
    from lab.metrics.performance import sharpe as sharpe_fn

    annualization_days = protocol_cfg.metrics.annualization_days
    results = np.empty(n_seeds)
    for seed in range(n_seeds):
        strat = RandomEntryStrategy(n_trades, avg_holding_bars, long_fraction, seed, tf=tf)
        prepared = {s: strat.prepare(df, None) for s, df in signal_bars_raw.items()}
        result = run_backtest(strat, bars_1h, prepared, funding, tf, protocol_cfg, equity0, max_slots)
        results[seed] = sharpe_fn(daily_returns(result.equity), annualization_days)
    return results


def trade_profile(trades: list) -> dict:
    """Достаёт из реальных сделок стратегии то, что нужно для генерации сравнимого
    случайного бенчмарка: число сделок, среднее удержание (в барах... точнее в часах,
    пересчёт в бары сигнального ТФ - на вызывающей стороне), доля лонгов."""
    closed = [t for t in trades if t.exit_time is not None]
    if not closed:
        return {"n_trades": 0, "avg_holding_hours": 0.0, "long_fraction": 0.5}
    holdings = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in closed]
    long_count = sum(1 for t in closed if t.side == "long")
    return {
        "n_trades": len(closed),
        "avg_holding_hours": float(np.mean(holdings)),
        "long_fraction": long_count / len(closed),
    }
