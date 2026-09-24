"""S13 — Кросс-секционный моментум, Liu/Tsyvinski/Wu 2022 (03_STRATEGIES.md).

Единственная стратегия проекта на интерфейсе `PortfolioStrategy` (`base.py`) — ей нужно
видеть все монеты пула одновременно, чтобы ранжировать. Движок научился её вызывать
только сейчас (см. JOURNAL, "S13 заблокирован архитектурой" → доработка
`engine/backtest.py`).

"Исполнение в понедельник 00:00 UTC" отдельно не реализуется — это уже даёт сам движок
(решение на закрытии сигнального бара исполняется на open следующего), тот же приём, что
у S02 в режиме `weekly` (см. `s02_tsmom.py`)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import Decision, PortfolioStrategy


class S13(PortfolioStrategy):
    id = "S13"
    name = "cross_sectional_momentum"
    version = "1.0.0"
    default_params = {"L": 21, "k": 3, "skip": 0}
    param_grid = {"L": [7, 14, 21, 28], "k": [2, 3], "skip": [0, 1]}
    timeframes = ["1d"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return self.params.get("L", 21) + self.params.get("skip", 0) + 2

    def prepare(self, bars_by_symbol: dict[str, pd.DataFrame], ctx) -> dict[str, pd.DataFrame]:
        L, skip = self.params.get("L", 21), self.params.get("skip", 0)
        out = {}
        for s, bars in bars_by_symbol.items():
            b = bars.copy()
            close = b["close"]
            b["score"] = close.shift(skip) / close.shift(L) - 1
            out[s] = b
        return out

    def on_bar(self, t, rows: dict[str, pd.Series], positions: dict, ctx) -> dict[str, Decision]:
        k = self.params.get("k", 3)

        if t.dayofweek != 6:  # не воскресенье - держим позиции прошлой недели без изменений
            return {s: Decision(target=positions[s].target, tag="hold") for s in rows}

        scored = {s: row["score"] for s, row in rows.items() if pd.notna(row.get("score"))}
        if len(scored) < 2 * k:
            return {s: Decision(target=0.0, tag="not_enough_history") for s in rows}

        ranked = sorted(scored.items(), key=lambda kv: kv[1])  # по возрастанию score
        shorts = {s for s, _ in ranked[:k]}
        longs = {s for s, _ in ranked[-k:]}

        decisions = {}
        for s in rows:
            if s in longs:
                decisions[s] = Decision(target=1.0, tag="rank_long")
            elif s in shorts:
                decisions[s] = Decision(target=-1.0, tag="rank_short")
            else:
                decisions[s] = Decision(target=0.0, tag="rank_flat")
        return decisions
