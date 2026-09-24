"""Портированы из проекта Коли (src/altcoin_strategies/strategies.py + backtest.py).
См. DECLARATION.md — ВСЕ стратегии здесь идут с пометкой ⚠ (автор развивал и отбирал
по результатам, частично перекрывающим holdout, см. artifacts/active_strategy_shortlist.md).

Общий каркас исполнения (в _KolyaBase) — перевод логики run_backtest() из Kolya:
- exit_mode="trail": стоп подтягивается по ATR от high/low_water с момента входа
  (наш движок хранит текущий стоп в pos.stop_price - подтягиваем через него, без
  дополнительного состояния в самой стратегии).
- exit_mode="mean_reversion": выход при возврате цены к средней полосы Боллинджера.
- exit_mode="fixed": фиксированные стоп и тейк по ATR (или custom_stop), без трейлинга.
- Во всех режимах: принудительный выход через max_bars сигнальных баров
  (pos.bars_in_trade - движок считает сам).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategies.base import BaseStrategy, Decision


def _atr(frame: pd.DataFrame, length: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - 100 / (1 + gains / losses.replace(0, np.nan))


class _KolyaBase(BaseStrategy):
    author = "Коля"
    direction = "long_short"
    exit_mode = "fixed"  # "trail" | "mean_reversion" | "fixed"
    timeframes = ["1h"]

    def required_history(self, tf: str) -> int:
        return self.WARMUP

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pos.target != 0:
            return self._manage_open(row, pos)
        return self._maybe_enter(row)

    def _manage_open(self, row, pos) -> Decision:
        if pos.bars_in_trade >= self.params["max_bars"]:
            return Decision(target=0.0, tag="time_exit")

        if self.exit_mode == "mean_reversion":
            mean = row.get("mean")
            if pd.notna(mean):
                if (pos.target > 0 and row["high"] >= mean) or (pos.target < 0 and row["low"] <= mean):
                    return Decision(target=0.0, tag="mean_reversion")
            return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold")

        if self.exit_mode == "trail":
            atr = row.get("atr")
            if pd.isna(atr):
                return Decision(target=pos.target, stop_price=pos.stop_price, tag="hold")
            trail = atr * self.params["atr_trail_multiple"]
            new_stop = (
                max(pos.stop_price, row["high"] - trail) if pos.target > 0
                else min(pos.stop_price, row["low"] + trail)
            )
            return Decision(target=pos.target, stop_price=new_stop, tag="hold_trail")

        return Decision(target=pos.target, stop_price=pos.stop_price, take_price=pos.take_price, tag="hold")

    def _maybe_enter(self, row) -> Decision:
        signal = row.get("signal", 0)
        atr = row.get("atr")
        if signal == 0 or pd.isna(atr):
            return Decision(target=0.0, tag="flat")

        side = 1.0 if signal > 0 else -1.0
        close = row["close"]
        take = None
        if self.exit_mode == "trail":
            stop = close - side * atr * self.params["atr_trail_multiple"]
        elif self.exit_mode == "mean_reversion":
            stop = close - side * atr * self.params["stop_atr_multiple"]
        else:
            custom_stop = row.get("custom_stop")
            stop = float(custom_stop) if pd.notna(custom_stop) else close - side * atr * self.params["stop_atr_multiple"]
            take = close + side * atr * self.params["target_atr_multiple"]
        return Decision(target=side, stop_price=stop, take_price=take, tag="entry")


class KolyaBreakout(_KolyaBase):
    id = "COMM_KOLYA_BREAKOUT"
    name = "donchian_breakout_volume"
    version = "1.0.0"
    exit_mode = "trail"
    default_params = {
        "donchian_length": 20, "volume_length": 20, "volume_multiple": 1.3,
        "atr_length": 14, "atr_trail_multiple": 3.0, "max_bars": 240,
    }
    WARMUP = 20

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        upper = out["high"].rolling(p["donchian_length"]).max().shift(1)
        lower = out["low"].rolling(p["donchian_length"]).min().shift(1)
        out["atr"] = _atr(out, p["atr_length"])
        vol_ok = out["volume"] > out["volume"].rolling(p["volume_length"]).mean().shift(1) * p["volume_multiple"]
        out["signal"] = np.select(
            [(out["close"] > upper) & vol_ok, (out["close"] < lower) & vol_ok], [1, -1], default=0,
        )
        return out


class KolyaMeanReversion(_KolyaBase):
    id = "COMM_KOLYA_MEAN_REVERSION"
    name = "bollinger_rsi_mean_reversion"
    version = "1.0.0"
    exit_mode = "mean_reversion"
    default_params = {
        "bb_length": 20, "bb_std": 2.0, "rsi_length": 14, "z_entry": 2.0,
        "volatility_lookback": 10, "atr_length": 14, "stop_atr_multiple": 1.5, "max_bars": 96,
    }
    WARMUP = 20

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        mean = out["close"].rolling(p["bb_length"]).mean()
        std = out["close"].rolling(p["bb_length"]).std(ddof=0)
        out["mean"] = mean
        zscore = (out["close"] - mean) / std.replace(0, np.nan)
        out["rsi"] = _rsi(out["close"], p["rsi_length"])
        out["atr"] = _atr(out, p["atr_length"])
        width = (2 * p["bb_std"] * std) / mean
        falling_vol = width < width.shift(p["volatility_lookback"])
        out["signal"] = np.select(
            [(zscore <= -p["z_entry"]) & (out["rsi"] < 35) & falling_vol,
             (zscore >= p["z_entry"]) & (out["rsi"] > 65) & falling_vol],
            [1, -1], default=0,
        )
        return out


class KolyaEmaPullback(_KolyaBase):
    id = "COMM_KOLYA_EMA_PULLBACK"
    name = "ema_trend_pullback"
    version = "1.0.0"
    exit_mode = "fixed"
    default_params = {
        "ema_fast": 20, "ema_trend_fast": 50, "ema_trend_slow": 200, "atr_length": 14,
        "pullback_atr_tolerance": 0.35, "stop_atr_multiple": 2.0, "target_atr_multiple": 3.0, "max_bars": 120,
    }
    WARMUP = 200

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["ema20"] = out["close"].ewm(span=p["ema_fast"], adjust=False, min_periods=p["ema_fast"]).mean()
        ema50 = out["close"].ewm(span=p["ema_trend_fast"], adjust=False, min_periods=p["ema_trend_fast"]).mean()
        ema200 = out["close"].ewm(span=p["ema_trend_slow"], adjust=False, min_periods=p["ema_trend_slow"]).mean()
        out["atr"] = _atr(out, p["atr_length"])
        tolerance = out["atr"] * p["pullback_atr_tolerance"]
        near_ema = (out["low"] <= out["ema20"] + tolerance) & (out["high"] >= out["ema20"] - tolerance)
        bullish = (out["close"] > out["open"]) & (out["close"] > out["close"].shift(1))
        bearish = (out["close"] < out["open"]) & (out["close"] < out["close"].shift(1))
        out["signal"] = np.select(
            [(ema50 > ema200) & near_ema & bullish, (ema50 < ema200) & near_ema & bearish], [1, -1], default=0,
        )
        return out


class KolyaVolatilityBreakout(_KolyaBase):
    id = "COMM_KOLYA_VOLATILITY_BREAKOUT"
    name = "volatility_squeeze_breakout"
    version = "1.0.0"
    exit_mode = "trail"
    default_params = {
        "atr_length": 14, "bb_length": 20, "bb_std": 2.0, "squeeze_lookback": 100, "squeeze_quantile": 0.20,
        "breakout_length": 20, "volume_length": 20, "volume_multiple": 1.4, "atr_trail_multiple": 2.5, "max_bars": 192,
    }
    WARMUP = 100

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["atr"] = _atr(out, p["atr_length"])
        basis = out["close"].rolling(p["bb_length"]).mean()
        deviation = out["close"].rolling(p["bb_length"]).std(ddof=0)
        bb_width = (2 * p["bb_std"] * deviation) / basis
        atr_pct = out["atr"] / out["close"]
        squeeze = (
            (bb_width <= bb_width.rolling(p["squeeze_lookback"]).quantile(p["squeeze_quantile"]))
            & (atr_pct <= atr_pct.rolling(p["squeeze_lookback"]).quantile(p["squeeze_quantile"]))
        )
        vol_ok = out["volume"] > out["volume"].rolling(p["volume_length"]).mean().shift(1) * p["volume_multiple"]
        upper = out["high"].rolling(p["breakout_length"]).max().shift(1)
        lower = out["low"].rolling(p["breakout_length"]).min().shift(1)
        out["signal"] = np.select(
            [squeeze.shift(1) & vol_ok & (out["close"] > upper), squeeze.shift(1) & vol_ok & (out["close"] < lower)],
            [1, -1], default=0,
        )
        return out


class KolyaFundingContrarian(_KolyaBase):
    """Требует колонку funding_rate в bars (в отличие от остальных) - вызывающий код
    должен смёрджить историю фандинга в bars ПЕРЕД prepare(). Если колонки нет -
    сигналов не будет (не падает, просто target=0 всегда)."""
    id = "COMM_KOLYA_FUNDING_CONTRARIAN"
    name = "funding_rate_contrarian"
    version = "1.0.0"
    exit_mode = "fixed"
    default_params = {
        "ema_length": 20, "atr_length": 14, "funding_lookback_events": 60,
        "funding_high_quantile": 0.90, "funding_low_quantile": 0.10,
        "stop_atr_multiple": 2.0, "target_atr_multiple": 3.0, "max_bars": 96,
    }
    WARMUP = 60

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["ema"] = out["close"].ewm(span=p["ema_length"], adjust=False, min_periods=p["ema_length"]).mean()
        out["atr"] = _atr(out, p["atr_length"])
        if "funding_rate" not in out.columns:
            out["signal"] = 0
            return out
        rate = out["funding_rate"]
        prior = rate.dropna().shift(1)
        hi = prior.rolling(p["funding_lookback_events"], min_periods=p["funding_lookback_events"]).quantile(p["funding_high_quantile"])
        lo = prior.rolling(p["funding_lookback_events"], min_periods=p["funding_lookback_events"]).quantile(p["funding_low_quantile"])
        hi = hi.reindex(out.index).ffill()
        lo = lo.reindex(out.index).ffill()
        extreme_pos = (rate > hi).ffill().fillna(False)
        extreme_neg = (rate < lo).ffill().fillna(False)
        bearish_rev = (out["close"] < out["open"]) & (out["close"] < out["ema"]) & (out["close"].shift(1) >= out["ema"].shift(1))
        bullish_rev = (out["close"] > out["open"]) & (out["close"] > out["ema"]) & (out["close"].shift(1) <= out["ema"].shift(1))
        out["signal"] = np.select([extreme_pos & bearish_rev, extreme_neg & bullish_rev], [-1, 1], default=0)
        return out


class KolyaLiquidationSweep(_KolyaBase):
    id = "COMM_KOLYA_LIQUIDATION_SWEEP"
    name = "liquidation_sweep_proxy"
    version = "1.0.0"
    exit_mode = "fixed"
    default_params = {
        "atr_length": 14, "range_length": 20, "volume_length": 20, "volume_multiple": 2.0,
        "range_atr_multiple": 2.0, "wick_fraction": 0.45, "target_atr_multiple": 2.0, "max_bars": 48,
    }
    WARMUP = 20

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["atr"] = _atr(out, p["atr_length"])
        prior_low = out["low"].rolling(p["range_length"]).min().shift(1)
        prior_high = out["high"].rolling(p["range_length"]).max().shift(1)
        total = (out["high"] - out["low"]).replace(0, np.nan)
        lower_wick = (out[["open", "close"]].min(axis=1) - out["low"]) / total
        upper_wick = (out["high"] - out[["open", "close"]].max(axis=1)) / total
        vol_ok = out["volume"] > out["volume"].rolling(p["volume_length"]).mean().shift(1) * p["volume_multiple"]
        large = total > out["atr"] * p["range_atr_multiple"]
        long_ = (out["low"] < prior_low) & (out["close"] > prior_low) & (lower_wick >= p["wick_fraction"]) & vol_ok & large
        short_ = (out["high"] > prior_high) & (out["close"] < prior_high) & (upper_wick >= p["wick_fraction"]) & vol_ok & large
        out["signal"] = np.select([long_, short_], [1, -1], default=0)
        out["custom_stop"] = np.where(long_, out["low"], np.where(short_, out["high"], np.nan))
        return out


class KolyaKeltnerSqueeze(_KolyaBase):
    id = "COMM_KOLYA_KELTNER_SQUEEZE"
    name = "keltner_squeeze_breakout"
    version = "1.0.0"
    exit_mode = "trail"
    default_params = {
        "atr_length": 14, "length": 20, "bb_std": 2.0, "kc_atr": 1.5, "breakout_length": 20,
        "volume_length": 20, "volume_multiple": 1.0, "atr_trail_multiple": 3.0, "max_bars": 240,
    }
    WARMUP = 20

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["atr"] = _atr(out, p["atr_length"])
        basis = out["close"].ewm(span=p["length"], adjust=False, min_periods=p["length"]).mean()
        bb_std = out["close"].rolling(p["length"]).std(ddof=0)
        bb_upper, bb_lower = basis + p["bb_std"] * bb_std, basis - p["bb_std"] * bb_std
        kc_upper, kc_lower = basis + p["kc_atr"] * out["atr"], basis - p["kc_atr"] * out["atr"]
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        upper = out["high"].rolling(p["breakout_length"]).max().shift(1)
        lower = out["low"].rolling(p["breakout_length"]).min().shift(1)
        vol_ok = out["volume"] > out["volume"].rolling(p["volume_length"]).mean().shift(1) * p["volume_multiple"]
        out["signal"] = np.select(
            [squeeze.shift(1) & vol_ok & (out["close"] > upper), squeeze.shift(1) & vol_ok & (out["close"] < lower)],
            [1, -1], default=0,
        )
        return out


class KolyaRsiTrendContinuation(_KolyaBase):
    id = "COMM_KOLYA_RSI_TREND_CONTINUATION"
    name = "rsi_trend_continuation"
    version = "1.0.0"
    exit_mode = "fixed"
    default_params = {"atr_length": 14, "rsi_trigger": 50, "stop_atr_multiple": 2.0, "target_atr_multiple": 3.0, "max_bars": 120}
    WARMUP = 200

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["atr"] = _atr(out, p["atr_length"])
        ema50 = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
        rsi = _rsi(out["close"], 14)
        long_ = (ema50 > ema200) & (rsi.shift(1) < p["rsi_trigger"]) & (rsi >= p["rsi_trigger"])
        short_ = (ema50 < ema200) & (rsi.shift(1) > 100 - p["rsi_trigger"]) & (rsi <= 100 - p["rsi_trigger"])
        out["signal"] = np.select([long_, short_], [1, -1], default=0)
        return out


class KolyaMacdTrend(_KolyaBase):
    id = "COMM_KOLYA_MACD_TREND"
    name = "macd_trend_following"
    version = "1.0.0"
    exit_mode = "fixed"
    default_params = {"atr_length": 14, "stop_atr_multiple": 2.0, "target_atr_multiple": 3.0, "max_bars": 120}
    WARMUP = 200

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        p = self.params
        out["atr"] = _atr(out, p["atr_length"])
        ema_fast = out["close"].ewm(span=12, adjust=False).mean()
        ema_slow = out["close"].ewm(span=26, adjust=False).mean()
        hist = ema_fast - ema_slow - (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
        ema50 = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
        long_ = (ema50 > ema200) & (hist.shift(1) <= 0) & (hist > 0)
        short_ = (ema50 < ema200) & (hist.shift(1) >= 0) & (hist < 0)
        out["signal"] = np.select([long_, short_], [1, -1], default=0)
        return out


ALL_STRATEGIES = [
    KolyaBreakout, KolyaMeanReversion, KolyaEmaPullback, KolyaVolatilityBreakout,
    KolyaFundingContrarian, KolyaLiquidationSweep, KolyaKeltnerSqueeze,
    KolyaRsiTrendContinuation, KolyaMacdTrend,
]
