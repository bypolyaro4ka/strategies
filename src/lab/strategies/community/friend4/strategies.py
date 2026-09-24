"""Портировано из `research_tv_portfolios.py` (4-й друг, имя пока не названо пользователем -
см. DECLARATION.md), функция `signal(d, family, params)` - 6 семейств правил.

ВАЖНО про то, что именно перенесено: в присланном пакете 12 строк `strategies_12.json` -
это уже ОТОБРАННЫЕ по результатам на validation/test комбинации (entry-семейство + entry-ТФ
+ отдельное "дневное" exit-семейство на 1d). Мы переносим не эти 12 строк, а САМИ 6 семейств
правил (`signal()`) - каждое семейство уже само по себе законченная система входа/выхода
(state-machine `stateful()` в оригинале), без чужого postfactum-отбора комбинаций тайм-
фреймов и без кросс-таймфреймовой "дневной exit-надстройки" - см. SPEC.md, почему.

Оригинальная `stateful(le, se, lx, sx)` - это цикл по барам с состоянием side (0/1/-1):
экспозиция входа `le`/`se`, выхода `lx`/`sx`; выход и вход в этом же баре возможны (после
exit тут же проверяется entry). Здесь она переписана как ОДИН шаг `_stateful_step()`,
принимающий предыдущее состояние (наш `pos.target`) и булевы условия текущего бара -
воспроизводит ровно ту же логику, но на баре за баром, как того требует наш `on_bar()`."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


def _wilder(x: pd.Series, period: int) -> pd.Series:
    return x.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = _wilder(delta.clip(lower=0), period)
    down = _wilder((-delta).clip(lower=0), period)
    return 100.0 * up / (up + down)


def _atr(d: pd.DataFrame, period: int) -> pd.Series:
    c, h, l = d["close"], d["high"], d["low"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return _wilder(tr, period)


def _dmi(d: pd.DataFrame, period: int):
    up = d["high"].diff()
    down = -d["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    av = _atr(d, period)
    plus = 100 * _wilder(plus_dm, period) / av
    minus = 100 * _wilder(minus_dm, period) / av
    dx = 100 * (plus - minus).abs() / (plus + minus)
    return plus, minus, _wilder(dx, period)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _stateful_step(prev_target: float, le: bool, se: bool, lx: bool, sx: bool) -> float:
    """Один шаг оригинальной stateful() - см. докстринг модуля."""
    side = 1 if prev_target > 0 else (-1 if prev_target < 0 else 0)
    if side == 1 and lx:
        side = 0
    elif side == -1 and sx:
        side = 0
    if side == 0:
        if le and not se:
            side = 1
        elif se and not le:
            side = -1
    return float(side)


def _supertrend_direction(d: pd.DataFrame, period: int, mult: float) -> pd.Series:
    """Не-numba версия supertrend_direction оригинала - тот же алгоритм (bar-by-bar с
    состоянием upper/lower/direction), но без @njit (нам не нужна скорость на прогоне
    сотен монет x сеток, как в исследовательском скрипте автора)."""
    high, low, close = d["high"], d["low"], d["close"]
    av = _atr(d, period)
    mid = (high + low) / 2.0
    upper_raw = mid + mult * av
    lower_raw = mid - mult * av

    n = len(d)
    out = [0] * n
    upper_prev = lower_prev = 0.0
    prev_valid = False
    direction = -1
    for i in range(n):
        a = av.iloc[i]
        if pd.isna(a):
            out[i] = 0
            continue
        upper, lower = upper_raw.iloc[i], lower_raw.iloc[i]
        if not prev_valid:
            direction = -1
        else:
            if upper >= upper_prev and close.iloc[i - 1] <= upper_prev:
                upper = upper_prev
            if lower <= lower_prev and close.iloc[i - 1] >= lower_prev:
                lower = lower_prev
            if direction < 0:
                direction = 1 if close.iloc[i] > upper else -1
            else:
                direction = -1 if close.iloc[i] < lower else 1
        out[i] = direction
        upper_prev, lower_prev, prev_valid = upper, lower, True
    return pd.Series(out, index=d.index, dtype="int8")


class _Friend4Base(BaseStrategy):
    author = "Друг4"  # имя не названо пользователем - см. DECLARATION.md
    direction = "long_short"
    timeframes = ["1h"]


class Friend4Supertrend(_Friend4Base):
    """family="supertrend": направление супертренда, без стейта входа/выхода - target
    это сам направленный сигнал каждый бар (мемори-лесс, ре-вычисляется заново)."""
    id = "COMM_FRIEND4_SUPERTREND"
    name = "supertrend"
    version = "1.0.0"
    timeframes = ["1d"]  # у автора использовался как "дневное" семейство
    default_params = {"period": 7, "mult": 2.0}
    param_grid = {"period_mult": [(7, 2.0), (10, 3.0), (14, 3.0), (20, 4.0)]}

    def required_history(self, tf: str) -> int:
        return self.params["period"] + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        out["direction"] = _supertrend_direction(out, self.params["period"], self.params["mult"])
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["direction"]) or row["direction"] == 0:
            return Decision(target=0.0, tag="warmup")
        return Decision(target=float(row["direction"]), tag="supertrend")


class Friend4EmaAdxTrend(_Friend4Base):
    """family="ema_adx_trend": EMA fast/slow + ADX-фильтр силы тренда, мемори-лесс."""
    id = "COMM_FRIEND4_EMA_ADX_TREND"
    name = "ema_adx_trend"
    version = "1.0.0"
    timeframes = ["1d"]
    default_params = {"fast": 5, "slow": 20, "period": 14, "threshold": 20}
    param_grid = {"fast_slow_period_threshold": [(5, 20, 14, 20), (9, 30, 14, 25), (12, 50, 20, 20)]}

    def required_history(self, tf: str) -> int:
        return max(self.params["slow"], self.params["period"]) + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        ef = out["close"].ewm(span=p["fast"], min_periods=p["fast"], adjust=False).mean()
        es = out["close"].ewm(span=p["slow"], min_periods=p["slow"], adjust=False).mean()
        plus, minus, adx = _dmi(out, p["period"])
        up = (ef > es) & (adx >= p["threshold"]) & (plus > minus)
        down = (ef < es) & (adx >= p["threshold"]) & (minus > plus)
        out["direction"] = pd.Series(0.0, index=out.index)
        out.loc[up, "direction"] = 1.0
        out.loc[down, "direction"] = -1.0
        out.loc[adx.isna(), "direction"] = float("nan")
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["direction"]):
            return Decision(target=0.0, tag="warmup")
        return Decision(target=float(row["direction"]), tag="ema_adx_trend")


class _Friend4StatefulBase(_Friend4Base):
    """Общий on_bar для семейств с собственной state-machine входа/выхода (le/se/lx/sx,
    см. _stateful_step). prepare() каждого потомка обязан оставить эти 4 булевы колонки."""

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["le"]) or pd.isna(row["se"]) or pd.isna(row["lx"]) or pd.isna(row["sx"]):
            return Decision(target=0.0, tag="warmup")
        target = _stateful_step(pos.target, bool(row["le"]), bool(row["se"]), bool(row["lx"]), bool(row["sx"]))
        return Decision(target=target, tag=self.name)


class Friend4BbSqueezeBreakout(_Friend4StatefulBase):
    """family="bb_squeeze_breakout": Bollinger внутри Keltner (сжатие) -> пробой при
    выходе из сжатия в сторону границы Боллинджера."""
    id = "COMM_FRIEND4_BB_SQUEEZE_BREAKOUT"
    name = "bb_squeeze_breakout"
    version = "1.0.0"
    default_params = {"n": 20, "bb_mult": 2.0, "kc_mult": 1.0}
    param_grid = {"n_bbmult_kcmult": [(20, 2.0, 1.0), (20, 2.0, 1.5), (40, 2.0, 1.5)]}

    def required_history(self, tf: str) -> int:
        return self.params["n"] + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        c = out["close"]
        mean = c.rolling(p["n"], min_periods=p["n"]).mean()
        sd = c.rolling(p["n"], min_periods=p["n"]).std(ddof=0)
        upper, lower = mean + p["bb_mult"] * sd, mean - p["bb_mult"] * sd
        center = c.ewm(span=p["n"], min_periods=p["n"], adjust=False).mean()
        av = _atr(out, p["n"])
        squeeze = (upper < center + p["kc_mult"] * av) & (lower > center - p["kc_mult"] * av)
        released = squeeze.shift(1).fillna(False) & ~squeeze
        out["le"] = released & (c > upper)
        out["se"] = released & (c < lower)
        out["lx"] = c < center
        out["sx"] = c > center
        out.loc[mean.isna() | av.isna(), ["le", "se", "lx", "sx"]] = float("nan")
        return out


class Friend4RsiPullback(_Friend4StatefulBase):
    """family="rsi2_pullback": короткий RSI (по умолчанию период 2) на откате против тренда
    EMA, выход - возврат к быстрой EMA(5) или RSI пересекает 50."""
    id = "COMM_FRIEND4_RSI2_PULLBACK"
    name = "rsi2_pullback"
    version = "1.0.0"
    default_params = {"rsi_n": 2, "edge": 5, "trend_n": 100}
    param_grid = {"rsi_edge_trend": [(2, 5, 100), (2, 10, 100), (3, 10, 150)]}

    def required_history(self, tf: str) -> int:
        return self.params["trend_n"] + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        c = out["close"]
        r = _rsi(c, p["rsi_n"])
        trend = c.ewm(span=p["trend_n"], min_periods=p["trend_n"], adjust=False).mean()
        fast = c.ewm(span=5, min_periods=5, adjust=False).mean()
        out["le"] = (r < p["edge"]) & (c > trend)
        out["se"] = (r > 100 - p["edge"]) & (c < trend)
        out["lx"] = (c >= fast) | (r > 50)
        out["sx"] = (c <= fast) | (r < 50)
        out.loc[trend.isna() | r.isna(), ["le", "se", "lx", "sx"]] = float("nan")
        return out


class Friend4BbRegimeReversion(_Friend4StatefulBase):
    """family="bb_regime_reversion": возврат к среднему Боллинджера, но только в сторону
    долгосрочного режима (EMA(regime_n)) - против режима сигнал не берётся."""
    id = "COMM_FRIEND4_BB_REGIME_REVERSION"
    name = "bb_regime_reversion"
    version = "1.0.0"
    default_params = {"n": 20, "mult": 1.8, "regime_n": 100}
    param_grid = {"n_mult_regime": [(20, 1.8, 100), (20, 2.0, 100), (40, 2.0, 100), (40, 2.5, 100)]}

    def required_history(self, tf: str) -> int:
        return max(self.params["n"], self.params["regime_n"]) + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        c = out["close"]
        mean = c.rolling(p["n"], min_periods=p["n"]).mean()
        sd = c.rolling(p["n"], min_periods=p["n"]).std(ddof=0)
        upper, lower = mean + p["mult"] * sd, mean - p["mult"] * sd
        regime = c.ewm(span=p["regime_n"], min_periods=p["regime_n"], adjust=False).mean()
        out["le"] = (c < lower) & (c > regime)
        out["se"] = (c > upper) & (c < regime)
        out["lx"] = c >= mean
        out["sx"] = c <= mean
        out.loc[mean.isna() | regime.isna(), ["le", "se", "lx", "sx"]] = float("nan")
        return out


class Friend4IchimokuRsi(_Friend4StatefulBase):
    """family="ichimoku_rsi": пересечение Tenkan/Kijun, подтверждённое облаком и RSI."""
    id = "COMM_FRIEND4_ICHIMOKU_RSI"
    name = "ichimoku_rsi"
    version = "1.0.0"
    timeframes = ["1d"]  # у автора использовался как "дневное" семейство
    default_params = {"conversion_n": 7, "base_n": 22, "span_b_n": 44, "rsi_n": 14}
    param_grid = {"conv_base_spanb_rsi": [(7, 22, 44, 14), (9, 26, 52, 14), (12, 30, 60, 21)]}

    def required_history(self, tf: str) -> int:
        p = self.params
        return p["span_b_n"] + p["base_n"] + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        h, l, c = out["high"], out["low"], out["close"]
        tenkan = (h.rolling(p["conversion_n"], min_periods=p["conversion_n"]).max()
                  + l.rolling(p["conversion_n"], min_periods=p["conversion_n"]).min()) / 2
        kijun = (h.rolling(p["base_n"], min_periods=p["base_n"]).max()
                 + l.rolling(p["base_n"], min_periods=p["base_n"]).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(p["base_n"])
        span_b = ((h.rolling(p["span_b_n"], min_periods=p["span_b_n"]).max()
                   + l.rolling(p["span_b_n"], min_periods=p["span_b_n"]).min()) / 2).shift(p["base_n"])
        top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        r = _rsi(c, p["rsi_n"])
        out["le"] = _cross_up(tenkan, kijun) & (c > top) & (r > 50)
        out["se"] = _cross_down(tenkan, kijun) & (c < bottom) & (r < 50)
        out["lx"] = (tenkan < kijun) | (c < bottom)
        out["sx"] = (tenkan > kijun) | (c > top)
        out.loc[kijun.isna() | top.isna() | r.isna(), ["le", "se", "lx", "sx"]] = float("nan")
        return out
