"""S08 — MACD с фильтром EMA (03_STRATEGIES.md)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class S08(BaseStrategy):
    id = "S08"
    name = "macd_ema_filter"
    version = "1.0.0"
    default_params = {"fast": 12, "slow": 26, "signal": 9, "filter_n": 200, "tf": "4h"}
    param_grid = {
        "macd": [(12, 26, 9), (8, 21, 5)],
        "filter_n": [200, 100, None],
        "tf": ["4h", "1h"],
    }
    timeframes = ["4h", "1h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        p = self.params
        filter_n = p.get("filter_n") or 0
        return max(p.get("slow", 26) + p.get("signal", 9), filter_n) + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        close = out["close"]
        ema_fast = close.ewm(span=p.get("fast", 12), min_periods=p.get("fast", 12), adjust=False).mean()
        ema_slow = close.ewm(span=p.get("slow", 26), min_periods=p.get("slow", 26), adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_n = p.get("signal", 9)
        signal = macd.ewm(span=signal_n, min_periods=signal_n, adjust=False).mean()

        out["cross_up"] = (macd.shift(1) <= signal.shift(1)) & (macd > signal)
        out["cross_down"] = (macd.shift(1) >= signal.shift(1)) & (macd < signal)
        out["_warmup"] = signal.isna()

        filter_n = p.get("filter_n")
        if filter_n:
            ema_filter = close.ewm(span=filter_n, min_periods=filter_n, adjust=False).mean()
            out["ema_filter"] = ema_filter
            out["_warmup"] = out["_warmup"] | ema_filter.isna()
        else:
            out["ema_filter"] = float("nan")
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if bool(row["_warmup"]):
            return Decision(target=0.0, tag="warmup")

        cross_up, cross_down = bool(row["cross_up"]), bool(row["cross_down"])
        close = row["close"]
        filter_n = self.params.get("filter_n")
        filter_val = row["ema_filter"] if filter_n else None

        long_ok = cross_up and (filter_val is None or close > filter_val)
        short_ok = cross_down and (filter_val is None or close < filter_val)

        if pos.target > 0:
            if cross_down:
                return Decision(target=(-1.0 if short_ok else 0.0), tag="exit_long")
            return Decision(target=pos.target, tag="hold_long")
        if pos.target < 0:
            if cross_up:
                return Decision(target=(1.0 if long_ok else 0.0), tag="exit_short")
            return Decision(target=pos.target, tag="hold_short")

        if long_ok:
            return Decision(target=1.0, tag="entry_long")
        if short_ok:
            return Decision(target=-1.0, tag="entry_short")
        return Decision(target=0.0, tag="flat")
