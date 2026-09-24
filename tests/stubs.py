"""Синтетические стратегии-заглушки для тестов движка (05_PLAN.md, Этап 3).
Не настоящие стратегии — только чтобы проверять сам движок, независимо от логики S01..S13."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision, PortfolioStrategy


class AlwaysLongStub(BaseStrategy):
    """Всегда target=1. Для проверки Buy & Hold без издержек и детерминизма."""
    id = "STUB_ALWAYS_LONG"
    name = "always_long"
    version = "1.0.0"
    timeframes = ["1d"]
    direction = "long_only"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return bars

    def on_bar(self, t, row, pos, ctx) -> Decision:
        return Decision(target=1.0, tag="entry_long")


class FlatStub(BaseStrategy):
    """Всегда target=0. Для проверки Flat-бенчмарка (equity не меняется)."""
    id = "STUB_FLAT"
    name = "flat"
    version = "1.0.0"
    timeframes = ["1d"]
    direction = "long_only"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return bars

    def on_bar(self, t, row, pos, ctx) -> Decision:
        return Decision(target=0.0, tag="flat")


class EvenDayLongStub(BaseStrategy):
    """Лонг по чётным дням месяца, иначе вне рынка. Простой пример стратегии, которая
    меняет решение день ото дня, но без индикаторов - удобно для проверки цикла движка."""
    id = "STUB_EVEN_DAY"
    name = "even_day_long"
    version = "1.0.0"
    timeframes = ["1d"]
    direction = "long_only"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return bars

    def on_bar(self, t, row, pos, ctx) -> Decision:
        target = 1.0 if t.day % 2 == 0 else 0.0
        return Decision(target=target, tag="even_day")


class EvenDayFlipStub(BaseStrategy):
    """Лонг по чётным дням месяца, шорт по нечётным - никогда не уходит в кэш.
    Для проверки разворота позиции (Reversal): движок не должен терять сделку,
    когда target меняет знак, минуя 0."""
    id = "STUB_EVEN_DAY_FLIP"
    name = "even_day_flip"
    version = "1.0.0"
    timeframes = ["1d"]
    direction = "long_short"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return bars

    def on_bar(self, t, row, pos, ctx) -> Decision:
        target = 1.0 if t.day % 2 == 0 else -1.0
        return Decision(target=target, tag="even_day_flip")


class PortfolioTopBottomStub(PortfolioStrategy):
    """Заглушка PortfolioStrategy (для проверки движка на S13) - лонг символ с
    наибольшим close, шорт символ с наименьшим, остальные в 0. Видит все монеты сразу
    в одном вызове on_bar() - именно то, чего не может обычная BaseStrategy."""
    id = "STUB_PORTFOLIO_TOP_BOTTOM"
    name = "portfolio_top_bottom"
    version = "1.0.0"
    timeframes = ["1d"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return 0

    def prepare(self, bars_by_symbol: dict[str, pd.DataFrame], ctx) -> dict[str, pd.DataFrame]:
        return dict(bars_by_symbol)

    def on_bar(self, t, rows: dict[str, pd.Series], positions: dict, ctx) -> dict[str, Decision]:
        if len(rows) < 2:
            return {s: Decision(target=0.0, tag="too_few_symbols") for s in rows}
        by_close = sorted(rows.items(), key=lambda kv: kv[1]["close"])
        winner, loser = by_close[-1][0], by_close[0][0]
        return {
            s: Decision(target=(1.0 if s == winner else (-1.0 if s == loser else 0.0)), tag="rank")
            for s in rows
        }


class RollingMeanStub(BaseStrategy):
    """close выше скользящей средней за window баров -> лонг, иначе вне рынка.
    Простейший пример стратегии с прогревом и rolling-индикатором - для проверки
    truncation/future-poison тестов не на тривиальной, а хоть немного содержательной логике."""
    id = "STUB_SMA_CROSS"
    name = "sma_cross"
    version = "1.0.0"
    timeframes = ["1d"]
    direction = "long_only"
    default_params = {"window": 3}

    def required_history(self, tf: str) -> int:
        return self.params["window"]

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        # rolling() - скользящее окно: для каждой строки берёт последние `window` строк
        # (включая саму строку) и считает по ним среднее. Строго причинно: строка t
        # использует только строки <= t, ничего из будущего.
        out["sma"] = out["close"].rolling(self.params["window"]).mean()
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["sma"]):
            return Decision(target=0.0, tag="warmup")
        target = 1.0 if row["close"] > row["sma"] else 0.0
        return Decision(target=target, tag="sma_cross")
