"""S09 — Volatility breakout, Larry Williams (03_STRATEGIES.md).

Сигнальный ТФ формально 1h (движок проверяет условие на закрытии каждого часового
бара), но пороги (`up`/`dn`) и фильтр тренда считаются на дневном уровне и транслируются
на все часовые бары внутри дня - см. `prepare()`. "Не больше одного входа в день" не
требует отдельного состояния между барами: как только позиция открыта, она держится до
принудительного выхода на баре 23:00-00:00 (`pos.target != 0` уже кодирует "сегодня уже
входили"), это ровно то же самое разрешённое движком чтение состояния через `pos`, что
используют другие наши стратегии (S04, S05).

Отличие от оригинала (Larry Williams): вход по закрытию часового бара, а не по касанию
уровня внутри бара (только рыночные ордера, как того требует наш движок) - вход чуть хуже
оригинала, честнее для нашей песочницы (см. 03_STRATEGIES.md)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


class S09(BaseStrategy):
    id = "S09"
    name = "volatility_breakout"
    version = "1.0.0"
    default_params = {"k": 0.5, "mode": "LS", "trend_filter": "none", "tf": "1h"}
    param_grid = {
        "k": [0.3, 0.5, 0.7],
        "mode": ["LS", "long_only"],
        "trend_filter": ["none", "sma5d"],
    }
    timeframes = ["1h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return 6 * 24  # 6 полных дней с запасом (5-дневная SMA + сдвиг на день)

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        k = self.params.get("k", 0.5)
        trend_filter = self.params.get("trend_filter", "none")

        day = out.index.normalize()
        daily = out.groupby(day).agg(d_open=("open", "first"), d_high=("high", "max"),
                                      d_low=("low", "min"), d_close=("close", "last"))
        daily["R"] = daily["d_high"].shift(1) - daily["d_low"].shift(1)
        daily["up"] = daily["d_open"] + k * daily["R"]
        daily["dn"] = daily["d_open"] - k * daily["R"]

        out["up"] = day.map(daily["up"])
        out["dn"] = day.map(daily["dn"])
        warmup = out["up"].isna() | out["dn"].isna()

        if trend_filter == "sma5d":
            daily["sma5d"] = daily["d_close"].rolling(5).mean().shift(1)
            daily["trend_ok_long"] = daily["d_close"].shift(1) > daily["sma5d"]
            daily["trend_ok_short"] = daily["d_close"].shift(1) < daily["sma5d"]
            out["trend_ok_long"] = day.map(daily["trend_ok_long"]).astype(float)
            out["trend_ok_short"] = day.map(daily["trend_ok_short"]).astype(float)
            warmup = warmup | day.map(daily["sma5d"]).isna()
        else:
            out["trend_ok_long"] = 1.0
            out["trend_ok_short"] = 1.0

        out["_warmup"] = warmup
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if bool(row["_warmup"]):
            return Decision(target=0.0, tag="warmup")

        if t.hour == 23:  # последний часовой бар дня - принудительный выход
            return Decision(target=0.0, tag="day_end_exit")

        if pos.target != 0:  # сегодня уже входили - держим до конца дня
            return Decision(target=pos.target, tag="hold")

        close = row["close"]
        mode = self.params.get("mode", "LS")
        long_allowed = bool(row["trend_ok_long"])
        short_allowed = (mode == "LS") and bool(row["trend_ok_short"])

        if close > row["up"] and long_allowed:
            return Decision(target=1.0, tag="entry_long")
        if close < row["dn"] and short_allowed:
            return Decision(target=-1.0, tag="entry_short")
        return Decision(target=0.0, tag="flat")
