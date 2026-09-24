"""S11 — Возврат к Боллинджеру в боковике + ADX (03_STRATEGIES.md).

Стоп при входе считается от текущего close как приближение к цене исполнения (тот же
приём, что у S04: настоящая цена fill известна только на open следующего бара, а
Decision выставляется на закрытии сигнального). Тайм-стоп через `pos.bars_in_trade`,
как у S10 (движок не обрабатывает `time_stop_bars` сам)."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


def _atr(bars: pd.DataFrame, n: int) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _adx(bars: pd.DataFrame, n: int) -> pd.Series:
    up = bars["high"].diff()
    down = -bars["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(bars, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


class S11(BaseStrategy):
    id = "S11"
    name = "bb_reversion_adx"
    version = "1.0.0"
    default_params = {"adx_thr": 20, "k": 2, "s": 2, "n_bb": 20, "n_adx": 14, "time_stop": 20, "tf": "4h"}
    param_grid = {
        "adx_thr": [15, 20, 25],
        "k": [2, 2.5],
        "s": [1.5, 2, 3],
        "tf": ["4h", "1h"],
    }
    timeframes = ["4h", "1h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return max(self.params.get("n_bb", 20), self.params.get("n_adx", 14)) + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        n_bb, n_adx = self.params.get("n_bb", 20), self.params.get("n_adx", 14)
        k = self.params.get("k", 2)
        close = out["close"]
        mid = close.rolling(n_bb).mean()
        std = close.rolling(n_bb).std(ddof=0)
        out["mid"], out["bb_up"], out["bb_dn"] = mid, mid + k * std, mid - k * std
        out["atr"] = _atr(out, n_adx)
        out["adx"] = _adx(out, n_adx)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["mid"]) or pd.isna(row["adx"]) or pd.isna(row["atr"]):
            return Decision(target=0.0, tag="warmup")

        close, mid = row["close"], row["mid"]
        adx_thr = self.params.get("adx_thr", 20)
        s = self.params.get("s", 2)
        time_stop = self.params.get("time_stop", 20)

        if pos.target > 0:
            if close >= mid or pos.bars_in_trade >= time_stop:
                return Decision(target=0.0, tag="exit_long")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold_long")
        if pos.target < 0:
            if close <= mid or pos.bars_in_trade >= time_stop:
                return Decision(target=0.0, tag="exit_short")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold_short")

        if row["adx"] >= adx_thr:
            return Decision(target=0.0, tag="trending_no_trade")
        if close < row["bb_dn"]:
            return Decision(target=1.0, stop_price=close - s * row["atr"], tag="entry_long")
        if close > row["bb_up"]:
            return Decision(target=-1.0, stop_price=close + s * row["atr"], tag="entry_short")
        return Decision(target=0.0, tag="flat")
