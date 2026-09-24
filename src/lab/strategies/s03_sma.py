"""S03 — Цена против SMA (Grobys, Ahmed, Sapkota, 2020; 03_STRATEGIES.md)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

TF_MULT = {"1d": 1, "12h": 2, "4h": 6}


class S03(BaseStrategy):
    id = "S03"
    name = "price_vs_sma"
    version = "1.0.0"
    default_params = {"n_days": 20, "band": 0.0, "mode": "LS", "tf": "1d"}
    param_grid = {
        "n_days": [10, 20, 50, 100],
        "band": [0.0, 0.01, 0.02],
        "mode": ["LS", "LF"],
        "tf": ["1d", "12h", "4h"],
    }
    timeframes = ["1d", "12h", "4h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return self.params.get("n_days", 20) * TF_MULT.get(tf, 1)

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        n = self.params.get("n_days", 20) * TF_MULT.get(self.params.get("tf", "1d"), 1)
        out["sma"] = out["close"].rolling(n).mean()  # текущее закрытие включено - стандарт для SMA-фильтра
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        sma = row["sma"]
        if pd.isna(sma):
            return Decision(target=0.0, tag="warmup")
        band = self.params.get("band", 0.0)
        mode = self.params.get("mode", "LS")
        close = row["close"]
        if close > sma * (1 + band):
            return Decision(target=1.0, tag="above_sma")
        if close < sma * (1 - band):
            return Decision(target=(-1.0 if mode == "LS" else 0.0), tag="below_sma")
        return Decision(target=pos.target, tag="inside_band")  # внутри полосы - сохранить прошлый target
