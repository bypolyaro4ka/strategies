"""S10 — RSI(2) Коннорса (03_STRATEGIES.md).

Тайм-стоп через `pos.bars_in_trade` (ведёт сам движок) - как решено в JOURNAL
2026-09-24 (Этап 3, часть 2): движок не обрабатывает `Decision.time_stop_bars`
автоматически, стратегия сама читает `pos.bars_in_trade` в `on_bar()`."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, float("nan")))


class S10(BaseStrategy):
    id = "S10"
    name = "rsi2_connors"
    version = "1.0.0"
    default_params = {"lo": 10, "hi": 90, "filter_sma200": True, "T": 10, "tf": "4h"}
    param_grid = {
        "lo_hi": [(5, 95), (10, 90), (15, 85)],
        "filter_sma200": [True, False],
        "T": [5, 10, 20],
        "tf": ["4h", "1h"],
    }
    timeframes = ["4h", "1h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return 202 if self.params.get("filter_sma200", True) else 7

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        close = out["close"]
        out["rsi2"] = _rsi(close, 2)
        out["sma5"] = close.rolling(5).mean()
        if self.params.get("filter_sma200", True):
            out["sma200"] = close.rolling(200).mean()
        else:
            out["sma200"] = float("nan")
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        rsi2, sma5, close = row["rsi2"], row["sma5"], row["close"]
        use_filter = self.params.get("filter_sma200", True)
        if pd.isna(rsi2) or pd.isna(sma5) or (use_filter and pd.isna(row["sma200"])):
            return Decision(target=0.0, tag="warmup")

        lo, hi, T = self.params.get("lo", 10), self.params.get("hi", 90), self.params.get("T", 10)
        sma200 = row["sma200"]

        if pos.target > 0:
            if close > sma5 or pos.bars_in_trade >= T:
                return Decision(target=0.0, tag="exit_long")
            return Decision(target=pos.target, tag="hold_long")
        if pos.target < 0:
            if close < sma5 or pos.bars_in_trade >= T:
                return Decision(target=0.0, tag="exit_short")
            return Decision(target=pos.target, tag="hold_short")

        long_ok = rsi2 < lo and (not use_filter or close > sma200)
        short_ok = rsi2 > hi and (not use_filter or close < sma200)
        if long_ok:
            return Decision(target=1.0, tag="entry_long")
        if short_ok:
            return Decision(target=-1.0, tag="entry_short")
        return Decision(target=0.0, tag="flat")
