"""Портирован из Freqtrade IStrategy (Слава). Оригинал: C2LocalCompressionStrategy.py
(сжатие ATR/диапазона + пробой локального максимума внутри макро-контейнера,
подтверждение по прошлым 72ч, фикс. выход 60ч, ранний выход при просадке после 36ч,
SL -12%). Дефолты автора: CONFIRM_MODE=CONFIRMED, REGIME_MODE=OFF (без фильтра режима)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class SlavaC2LocalCompression(BaseStrategy):
    id = "COMM_SLAVA_C2_LOCAL_COMPRESSION_V1"
    name = "c2_local_compression"
    version = "1.0.0"
    author = "Слава"
    timeframes = ["1h"]
    direction = "long_only"

    ATR_PERIOD = 14
    LOCAL_WINDOW = 12
    MACRO_WINDOW = 72
    RANGE_COMPRESSION_LIMIT = 0.40
    EXPANSION_ATR_RATIO = 1.50
    BODY_LIMIT = 0.60
    CONFIRM_LOOKBACK = 72
    REGIME_MODE = "OFF"
    REGIME_SLOPE_THRESHOLD = 0.0
    EMA_SLOPE_LOOKBACK = 72
    HOLD_HOURS = 60
    AGE_EMERGENCY_HOURS = 36
    AGE_EMERGENCY_LOSS = -0.04
    STOPLOSS = -0.12

    def required_history(self, tf: str) -> int:
        return 400

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        prev_close = out["close"].shift(1)
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.ewm(alpha=1 / self.ATR_PERIOD, adjust=False).mean()
        out["atr_pct"] = out["atr"] / out["close"]
        out["atr_pct_median"] = out["atr_pct"].shift(1).rolling(self.MACRO_WINDOW).median()

        out["local_high"] = out["high"].shift(1).rolling(self.LOCAL_WINDOW).max()
        out["local_low"] = out["low"].shift(1).rolling(self.LOCAL_WINDOW).min()
        out["local_range"] = out["local_high"] - out["local_low"]

        out["macro_high"] = out["high"].shift(1).rolling(self.MACRO_WINDOW).max()
        out["macro_low"] = out["low"].shift(1).rolling(self.MACRO_WINDOW).min()
        out["macro_range"] = out["macro_high"] - out["macro_low"]

        out["range_compression"] = out["local_range"] / out["macro_range"].replace(0, float("nan"))

        candle_range = out["high"] - out["low"]
        body = (out["close"] - out["open"]).abs()
        out["body_ratio"] = body / candle_range.replace(0, float("nan"))
        out["range_atr_ratio"] = candle_range / out["atr"].replace(0, float("nan"))

        out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
        out["ema200_slope72"] = out["ema200"] / out["ema200"].shift(self.EMA_SLOPE_LOOKBACK) - 1

        raw = (
            (out["atr_pct"] < out["atr_pct_median"])
            & (out["range_compression"] < self.RANGE_COMPRESSION_LIMIT)
            & (out["close"] > out["open"])
            & (out["range_atr_ratio"] > self.EXPANSION_ATR_RATIO)
            & (out["body_ratio"] > self.BODY_LIMIT)
            & (out["close"] > out["local_high"])
            & (out["close"] <= out["macro_high"])
        )
        out["c2_raw_signal"] = raw.fillna(False)
        prior_count = out["c2_raw_signal"].shift(1).rolling(self.CONFIRM_LOOKBACK).sum()
        out["c2_confirmed"] = out["c2_raw_signal"] & (prior_count >= 1)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pos.target > 0:
            age = pos.bars_in_trade
            profit = row["close"] / pos.avg_entry_price - 1 if pos.avg_entry_price else 0.0
            if age >= self.AGE_EMERGENCY_HOURS and profit <= self.AGE_EMERGENCY_LOSS:
                return Decision(target=0.0, tag="c2_age_failure")
            if age >= self.HOLD_HOURS:
                return Decision(target=0.0, tag="c2_hold_exit")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold")

        if pd.isna(row.get("c2_confirmed")):
            return Decision(target=0.0, tag="warmup")

        regime_ok = True
        if self.REGIME_MODE == "EMA200_UP":
            regime_ok = pd.notna(row["ema200_slope72"]) and row["ema200_slope72"] > self.REGIME_SLOPE_THRESHOLD

        if bool(row["c2_confirmed"]) and regime_ok and row["volume"] > 0:
            return Decision(target=1.0, stop_price=row["close"] * (1 + self.STOPLOSS), tag="c2_confirmed")
        return Decision(target=0.0, tag="flat")
