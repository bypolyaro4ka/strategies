"""S02 — Time-series momentum (Liu, Tsyvinski, 2021; 03_STRATEGIES.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

TF_MULT = {"1d": 1, "12h": 2}


class S02(BaseStrategy):
    id = "S02"
    name = "tsmom"
    version = "1.0.0"
    default_params = {"lookback_days": (7, 14, 28), "rebalance": "every_bar", "tf": "1d"}
    param_grid = {
        "lookback_days": [(3, 7, 14), (7, 14, 28), (14, 28, 56), (7,), (14,), (28,)],
        "rebalance": ["every_bar", "weekly"],
        "tf": ["1d", "12h"],
    }
    timeframes = ["1d", "12h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        mult = TF_MULT.get(tf, 1)
        return max(self.params.get("lookback_days", (7, 14, 28))) * mult

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        mult = TF_MULT.get(self.params.get("tf", "1d"), 1)
        signs = []
        for L_days in self.params.get("lookback_days", (7, 14, 28)):
            L = L_days * mult
            r = out["close"] / out["close"].shift(L) - 1
            signs.append(np.sign(r))
        target = pd.concat(signs, axis=1).mean(axis=1)

        if self.params.get("rebalance", "every_bar") == "weekly":
            # решение только по закрытию воскресного дневного бара (dayofweek: Monday=0..Sunday=6),
            # между воскресеньями держим прошлое значение - forward-fill.
            # исполнение всё равно на open следующего бара (т.е. понедельника) - это уже
            # даёт движок сам, без специальной обработки здесь.
            is_sunday = out.index.dayofweek == 6
            target = target.where(is_sunday).ffill()

        out["target"] = target
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        target = row["target"]
        return Decision(target=0.0 if pd.isna(target) else float(target), tag="tsmom")
