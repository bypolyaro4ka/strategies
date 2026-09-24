"""S04 — Turtle System 1 (03_STRATEGIES.md). Окна в барах (bar-native), не в днях."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class S04(BaseStrategy):
    id = "S04"
    name = "turtle_system1"
    version = "1.0.0"
    default_params = {"n_in": 20, "n_out": 10, "k": 2, "atr_n": 20, "tf": "4h"}
    param_grid = {"n_in_out": [(20, 10), (55, 20)], "k": [2, 3], "tf": ["4h", "1d"]}
    timeframes = ["4h", "1d"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return max(self.params.get("n_in", 20), self.params.get("atr_n", 20)) + 1

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        n_in = self.params.get("n_in", 20)
        n_out = self.params.get("n_out", 10)
        atr_n = self.params.get("atr_n", 20)

        prev_close = out["close"].shift(1)
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.ewm(alpha=1 / atr_n, adjust=False).mean()  # ATR по Уайлдеру

        out["entry_high"] = out["high"].shift(1).rolling(n_in).max()
        out["entry_low"] = out["low"].shift(1).rolling(n_in).min()
        out["exit_high"] = out["high"].shift(1).rolling(n_out).max()
        out["exit_low"] = out["low"].shift(1).rolling(n_out).min()
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["entry_high"]) or pd.isna(row["entry_low"]) or pd.isna(row["atr"]):
            return Decision(target=pos.target, tag="warmup")

        k = self.params.get("k", 2)
        close = row["close"]

        if pos.target == 0:
            if close > row["entry_high"]:
                return Decision(target=1.0, stop_price=close - k * row["atr"], tag="entry_long")
            if close < row["entry_low"]:
                return Decision(target=-1.0, stop_price=close + k * row["atr"], tag="entry_short")
            return Decision(target=0.0, tag="flat")

        if pos.target > 0:
            if close < row["exit_low"]:
                return Decision(target=0.0, tag="exit_long")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold_long")

        if close > row["exit_high"]:
            return Decision(target=0.0, tag="exit_short")
        return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold_short")
