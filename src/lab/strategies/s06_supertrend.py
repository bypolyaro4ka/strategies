"""S06 — Supertrend (03_STRATEGIES.md). Окна в барах (bar-native)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class S06(BaseStrategy):
    id = "S06"
    name = "supertrend"
    version = "1.0.0"
    default_params = {"n": 10, "m": 3, "tf": "4h"}
    param_grid = {"n": [7, 10, 14], "m": [2, 3, 4], "tf": ["4h", "1h", "12h"]}
    timeframes = ["4h", "1h", "12h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return self.params.get("n", 10) + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        n = self.params.get("n", 10)
        m = self.params.get("m", 3)

        prev_close = out["close"].shift(1)
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()  # ATR по Уайлдеру

        hl2 = (out["high"] + out["low"]) / 2
        bu = hl2 + m * atr
        bl = hl2 - m * atr
        close = out["close"]

        # fu/fl - рекуррентные "ратчет"-полосы (каждая зависит от своего значения на
        # предыдущем баре) - векторизовать нельзя, считаем циклом по индексу, как ATR
        # Уайлдера, только с двусторонней зависимостью от предыдущего close и своего
        # предыдущего значения (03_STRATEGIES.md, формулы S06).
        n_bars = len(out)
        fu = [float("nan")] * n_bars
        fl = [float("nan")] * n_bars
        trend = [0] * n_bars  # +1 вверх, -1 вниз, 0 - ещё нет тренда (прогрев)

        bu_arr, bl_arr, close_arr, atr_arr = bu.to_numpy(), bl.to_numpy(), close.to_numpy(), atr.to_numpy()
        prev_fu = prev_fl = float("nan")
        prev_trend = -1  # первое валидное состояние считаем "было вниз", как принято в оригинале
        started = False
        for i in range(n_bars):
            if pd.isna(atr_arr[i]):
                continue
            if not started:
                fu[i], fl[i] = bu_arr[i], bl_arr[i]
                trend[i] = prev_trend
                prev_fu, prev_fl = fu[i], fl[i]
                started = True
                continue
            cur_fu = bu_arr[i] if (bu_arr[i] < prev_fu or close_arr[i - 1] > prev_fu) else prev_fu
            cur_fl = bl_arr[i] if (bl_arr[i] > prev_fl or close_arr[i - 1] < prev_fl) else prev_fl
            cur_trend = prev_trend
            if prev_trend < 0 and close_arr[i] > cur_fu:
                cur_trend = 1
            elif prev_trend > 0 and close_arr[i] < cur_fl:
                cur_trend = -1
            fu[i], fl[i], trend[i] = cur_fu, cur_fl, cur_trend
            prev_fu, prev_fl, prev_trend = cur_fu, cur_fl, cur_trend

        out["trend"] = trend
        out.loc[atr.isna(), "trend"] = 0  # прогрев - явный маркер, не 0 из формулы
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        trend = row["trend"]
        if trend == 0:
            return Decision(target=0.0, tag="warmup")
        return Decision(target=float(trend), tag="supertrend")
