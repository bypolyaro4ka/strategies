"""Портирован из Freqtrade IStrategy (Слава), 1:1 логика, см. DECLARATION.md рядом.
Оригинал: SolFrozenV2.py (EMA5/30 cross + объём + regime, фиксированный SL -25%,
структурный выход по остановке роста EMA30 после 64ч, максимум 144ч)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class SlavaSolFrozenV2(BaseStrategy):
    id = "COMM_SLAVA_SOL_FROZEN_V2"
    name = "sol_frozen_v2"
    version = "1.0.0"
    author = "Слава"
    timeframes = ["1h"]
    direction = "long_only"

    EMA_FAST, EMA_SLOW, EMA_REGIME = 5, 30, 200
    EMA30_SLOPE_LOOKBACK = 6
    VOLUME_SMA = 30
    BODY_RATIO_MIN = 0.65
    RET24_MAX = 0.03
    EMA200_SLOPE72_MIN = -0.02
    RET72_MAX = 0.08
    MIN_HOLD_HOURS = 64
    MAX_HOLD_HOURS = 144
    STOPLOSS = -0.25

    def required_history(self, tf: str) -> int:
        return 300  # startup_candle_count у автора

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        out["ema5"] = out["close"].ewm(span=self.EMA_FAST, adjust=False).mean()
        out["ema30"] = out["close"].ewm(span=self.EMA_SLOW, adjust=False).mean()
        out["ema200"] = out["close"].ewm(span=self.EMA_REGIME, adjust=False).mean()
        out["ema30_slope6"] = out["ema30"] / out["ema30"].shift(self.EMA30_SLOPE_LOOKBACK) - 1
        out["ema200_slope72"] = out["ema200"] / out["ema200"].shift(72) - 1
        out["ret24"] = out["close"] / out["close"].shift(24) - 1
        out["ret72"] = out["close"] / out["close"].shift(72) - 1
        out["volume_sma30"] = out["volume"].rolling(self.VOLUME_SMA).mean()
        rng = out["high"] - out["low"]
        body = (out["close"] - out["open"]).abs()
        out["body_ratio"] = body / rng.replace(0, float("nan"))
        out["ema_cross_up"] = (out["ema5"] > out["ema30"]) & (out["ema5"].shift(1) <= out["ema30"].shift(1))
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pos.target > 0:
            age = pos.bars_in_trade
            if age >= self.MAX_HOLD_HOURS:
                return Decision(target=0.0, tag="max_hold_144h")
            if age < self.MIN_HOLD_HOURS:
                return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold_min")
            if pd.notna(row["ema30_slope6"]) and row["ema30_slope6"] <= 0:
                return Decision(target=0.0, tag="ema30_slope_lost_after_64h")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold")

        if pd.isna(row["ema200"]) or pd.isna(row["volume_sma30"]) or pd.isna(row["body_ratio"]):
            return Decision(target=0.0, tag="warmup")

        cond = (
            bool(row["ema_cross_up"]) and row["ema30_slope6"] > 0
            and row["volume"] > row["volume_sma30"]
            and row["body_ratio"] >= self.BODY_RATIO_MIN
            and row["ret24"] < self.RET24_MAX
            and row["ema200_slope72"] > self.EMA200_SLOPE72_MIN
            and row["ret72"] < self.RET72_MAX
            and row["volume"] > 0
        )
        if cond:
            return Decision(target=1.0, stop_price=row["close"] * (1 + self.STOPLOSS), tag="entry_long")
        return Decision(target=0.0, tag="flat")
