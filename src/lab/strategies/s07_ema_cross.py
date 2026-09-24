"""S07 — EMA-кросс golden/death cross (03_STRATEGIES.md)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class S07(BaseStrategy):
    id = "S07"
    name = "ema_cross"
    version = "1.0.0"
    default_params = {"fast": 50, "slow": 200, "mode": "LS", "tf": "4h"}
    param_grid = {
        "fast_slow": [(20, 50), (20, 100), (50, 200)],
        "mode": ["LS", "LF"],
        "tf": ["4h", "1d"],
    }
    timeframes = ["4h", "1d"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return self.params.get("slow", 200) + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        fast = self.params.get("fast", 50)
        slow = self.params.get("slow", 200)
        ema_fast = out["close"].ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = out["close"].ewm(span=slow, min_periods=slow, adjust=False).mean()
        out["ema_fast"], out["ema_slow"] = ema_fast, ema_slow
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["ema_slow"]):
            return Decision(target=0.0, tag="warmup")
        mode = self.params.get("mode", "LS")
        if row["ema_fast"] > row["ema_slow"]:
            return Decision(target=1.0, tag="golden_cross")
        return Decision(target=(-1.0 if mode == "LS" else 0.0), tag="death_cross")
