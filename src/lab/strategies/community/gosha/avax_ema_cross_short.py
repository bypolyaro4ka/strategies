"""Портирован из s08_avax_ema_cross_short.py (Гоша), 1:1 логика, см. DECLARATION.md рядом.
Оригинал - реконструкция публичного правила Coinrule (TradingView): EMA20 пересекает EMA50
сверху вниз -> шорт, тейк -8%, стоп +16%. Стратегия только шортит, в лонг никогда не идёт
(это её собственное правило, не ограничение движка).

Стоп/тейк у нас исполняет сам движок через `Decision.stop_price/take_price` (внутрибарно,
с приоритетом стопа при касании обоих в одном баре) - отдельно их обрабатывать в on_bar не
нужно, только выставить один раз при входе и держать при hold."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

EMA_FAST, EMA_SLOW = 20, 50
TAKE_PROFIT, STOP_LOSS = 0.08, 0.16


class GoshaAvaxEmaCrossShort(BaseStrategy):
    id = "COMM_GOSHA_AVAX_EMA_CROSS_SHORT"
    name = "avax_ema_cross_short"
    version = "1.0.0"
    author = "Гоша"
    timeframes = ["1h"]  # у автора 30m - в нашем пуле такого ТФ нет, ближайший - 1h, см. SPEC.md
    direction = "long_short"  # по правилу стратегии target никогда не положителен

    def required_history(self, tf: str) -> int:
        return EMA_SLOW + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        ema_fast = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
        ema_slow = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()
        out["crossunder"] = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pos.target < 0:
            return Decision(target=pos.target, stop_price=pos.stop_price, take_price=pos.take_price, tag="hold_short")

        if pd.isna(row["crossunder"]):
            return Decision(target=0.0, tag="warmup")

        if bool(row["crossunder"]):
            close = row["close"]
            return Decision(
                target=-1.0,
                stop_price=close * (1 + STOP_LOSS),
                take_price=close * (1 - TAKE_PROFIT),
                tag="entry_short",
            )
        return Decision(target=0.0, tag="flat")
