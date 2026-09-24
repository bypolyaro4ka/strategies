"""Портирован из Freqtrade IStrategy (Слава). Оригинал: EthStrongBreakoutV1Emergency.py
(пробой 168ч максимума на >=1.25 ATR, EMA50>EMA200, фикс. выход 72ч, emergency SL -15%)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class SlavaEthStrongBreakout(BaseStrategy):
    id = "COMM_SLAVA_ETH_STRONG_BREAKOUT_V1"
    name = "eth_strong_breakout_v1_emergency"
    version = "1.0.0"
    author = "Слава"
    timeframes = ["1h"]
    direction = "long_only"

    BREAKOUT_LOOKBACK = 168
    BREAKOUT_STRENGTH_ATR = 1.25
    ATR_PERIOD = 14
    EMA_FAST, EMA_SLOW = 50, 200
    HOLD_HOURS = 72
    STOPLOSS = -0.15

    def required_history(self, tf: str) -> int:
        return 240

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        prev_close = out["close"].shift(1)
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        out["atr14"] = tr.ewm(alpha=1 / self.ATR_PERIOD, adjust=False).mean()
        out["ema50"] = out["close"].ewm(span=self.EMA_FAST, adjust=False).mean()
        out["ema200"] = out["close"].ewm(span=self.EMA_SLOW, adjust=False).mean()
        out["prev_high_168"] = out["high"].shift(1).rolling(self.BREAKOUT_LOOKBACK).max()
        out["breakout_strength_atr"] = (out["close"] - out["prev_high_168"]) / out["atr14"]
        raw = (out["close"] > out["prev_high_168"]).fillna(False)
        out["breakout_event"] = raw & ~raw.shift(1, fill_value=False)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pos.target > 0:
            if pos.bars_in_trade >= self.HOLD_HOURS:
                return Decision(target=0.0, tag="fixed_72h")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold")

        if pd.isna(row["prev_high_168"]) or pd.isna(row["ema200"]):
            return Decision(target=0.0, tag="warmup")

        cond = (
            bool(row["breakout_event"])
            and row["breakout_strength_atr"] >= self.BREAKOUT_STRENGTH_ATR
            and row["ema50"] > row["ema200"]
            and row["volume"] > 0
        )
        if cond:
            return Decision(target=1.0, stop_price=row["close"] * (1 + self.STOPLOSS), tag="entry_long")
        return Decision(target=0.0, tag="flat")
