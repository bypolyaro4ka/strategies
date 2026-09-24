"""S05 — Пробой Боллинджера (03_STRATEGIES.md). Окна в барах."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class S05(BaseStrategy):
    id = "S05"
    name = "bollinger_breakout"
    version = "1.0.0"
    default_params = {"n": 20, "k": 2, "exit_rule": "mid", "tf": "1d"}
    param_grid = {"n": [20, 50], "k": [1.5, 2, 2.5], "exit_rule": ["mid", "opposite"], "tf": ["1d", "12h", "4h"]}
    timeframes = ["1d", "12h", "4h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return self.params.get("n", 20)

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        n = self.params.get("n", 20)
        k = self.params.get("k", 2)
        mid = out["close"].rolling(n).mean()
        std = out["close"].rolling(n).std(ddof=0)
        out["mid"], out["up"], out["dn"] = mid, mid + k * std, mid - k * std
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["mid"]):
            return Decision(target=pos.target, tag="warmup")

        close, mid, up, dn = row["close"], row["mid"], row["up"], row["dn"]
        exit_rule = self.params.get("exit_rule", "mid")

        if pos.target == 0:
            if close > up:
                return Decision(target=1.0, tag="entry_long")
            if close < dn:
                return Decision(target=-1.0, tag="entry_short")
            return Decision(target=0.0, tag="flat")

        if pos.target > 0:
            exit_level = mid if exit_rule == "mid" else dn
            if close < exit_level:
                return Decision(target=0.0, tag="exit_long")
            return Decision(target=pos.target, tag="hold_long")

        exit_level = mid if exit_rule == "mid" else up
        if close > exit_level:
            return Decision(target=0.0, tag="exit_short")
        return Decision(target=pos.target, tag="hold_short")
