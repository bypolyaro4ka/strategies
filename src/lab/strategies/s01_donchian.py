"""S01 / S01b — Donchian-ансамбль (03_STRATEGIES.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

WINDOWS_DAYS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
LOOKBACK_SETS = {
    "all9": WINDOWS_DAYS,
    "short": [5, 10, 20, 30, 60],
    "long": [20, 30, 60, 90, 150, 250, 360],
}
TF_MULT = {"1d": 1, "12h": 2}


def _track_long(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    n = len(close)
    on_col = np.zeros(n, dtype=bool)
    on, stop = False, np.nan
    for i in range(n):
        u, l = upper[i], lower[i]
        if np.isnan(u) or np.isnan(l):
            on, stop = False, np.nan
            continue
        mid = (u + l) / 2
        c = close[i]
        if not on:
            if c > u:
                on, stop = True, mid
        else:
            stop = max(stop, mid)
            if c < stop:
                on, stop = False, np.nan
        on_col[i] = on
    return on_col


def _track_short(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    n = len(close)
    on_col = np.zeros(n, dtype=bool)
    on, stop = False, np.nan
    for i in range(n):
        u, l = upper[i], lower[i]
        if np.isnan(u) or np.isnan(l):
            on, stop = False, np.nan
            continue
        mid = (u + l) / 2
        c = close[i]
        if not on:
            if c < l:
                on, stop = True, mid
        else:
            stop = min(stop, mid)
            if c > stop:
                on, stop = False, np.nan
        on_col[i] = on
    return on_col


class S01(BaseStrategy):
    id = "S01"
    name = "donchian_ensemble"
    version = "1.0.0"
    default_params = {"lookback_set": "all9", "vol_target": 0.25, "vol_window": 90, "tf": "1d"}
    param_grid = {
        "lookback_set": ["all9", "short", "long"],
        "vol_target": [0.25, 0.5, None],
        "tf": ["1d", "12h"],
    }
    timeframes = ["1d", "12h"]
    direction = "long_only"

    def _windows(self) -> list[int]:
        return LOOKBACK_SETS[self.params.get("lookback_set", "all9")]

    def _mult(self) -> int:
        return TF_MULT[self.params.get("tf", "1d")]

    def required_history(self, tf: str) -> int:
        return max(self._windows()) * TF_MULT.get(tf, 1)

    def _levels(self, bars: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        mult = self._mult()
        prev_close = bars["close"].shift(1)
        levels = {}
        for L_days in self._windows():
            L = L_days * mult
            levels[L_days] = (
                prev_close.rolling(L).max().to_numpy(),
                prev_close.rolling(L).min().to_numpy(),
            )
        return levels

    def _sigma(self, bars: pd.DataFrame) -> np.ndarray:
        mult = self._mult()
        tf = self.params.get("tf", "1d")
        tf_hours = {"1d": 24, "12h": 12}[tf]
        bars_per_year = 365 * 24 / tf_hours
        window_bars = self.params.get("vol_window", 90) * mult
        log_ret = np.log(bars["close"] / bars["close"].shift(1))
        return (log_ret.rolling(window_bars).std() * np.sqrt(bars_per_year)).to_numpy()

    def _scale(self, raw: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        vol_target = self.params.get("vol_target", 0.25)
        if vol_target is None:
            return raw
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(sigma > 0, np.minimum(1.0, vol_target / sigma), 1.0)
        return raw * scale

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        close = out["close"].to_numpy()
        levels = self._levels(out)
        n_on = np.zeros(len(out), dtype=int)
        for L_days, (upper, lower) in levels.items():
            n_on += _track_long(close, upper, lower).astype(int)
        raw = n_on / 9.0
        sigma = self._sigma(out)
        out["target"] = self._scale(raw, sigma)
        out["n_long"] = n_on
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        target = row["target"]
        return Decision(target=0.0 if pd.isna(target) else float(target), tag="donchian_ensemble")


class S01b(S01):
    """S01 + зеркальная шортовая часть (03_STRATEGIES.md). Шортовая часть — гипотеза
    авторов протокола, не проверялась в источнике."""

    id = "S01b"
    name = "donchian_ensemble_mirror_short"
    version = "1.0.0"
    direction = "long_short"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        close = out["close"].to_numpy()
        levels = self._levels(out)
        n_long = np.zeros(len(out), dtype=int)
        n_short = np.zeros(len(out), dtype=int)
        for L_days, (upper, lower) in levels.items():
            n_long += _track_long(close, upper, lower).astype(int)
            n_short += _track_short(close, upper, lower).astype(int)
        raw = (n_long - n_short) / 9.0
        sigma = self._sigma(out)
        out["target"] = self._scale(raw, sigma)
        out["n_long"] = n_long
        out["n_short"] = n_short
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        target = row["target"]
        return Decision(target=0.0 if pd.isna(target) else float(target), tag="donchian_ensemble_mirror")
