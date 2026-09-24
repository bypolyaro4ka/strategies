"""Flat — всегда вне рынка (01_PROTOCOL.md, раздел 7)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class FlatStrategy(BaseStrategy):
    id = "BENCH_FLAT"
    name = "flat"
    version = "1.0.0"
    timeframes = ["1d", "4h", "12h", "1h"]
    direction = "long_only"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return bars

    def on_bar(self, t, row, pos, ctx) -> Decision:
        return Decision(target=0.0, tag="flat")
