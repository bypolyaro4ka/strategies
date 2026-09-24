"""S12 — Контртренд по экстремальному фандингу (03_STRATEGIES.md).

Фандинг читается из `ctx.funding` (полная история за весь бэктест — тот же принцип
"дай всё, причинность обеспечивает сама стратегия", что и с барами; `Context`
это и обещает в своём докстринге в `base.py`). Раньше ни одна из S01-S11 в этом не
нуждалась, `ctx.funding` нигде не заполнялся - `scripts/run_dev_batch_d.py` первым
передаёт его явно.

Причинность самого мержа: `merge_asof(..., allow_exact_matches=False)` - отметка
фандинга ровно в момент закрытия бара НЕ используется на этом баре (переносится на
следующий), как того требует правило "Подготовка фандинга" в 03_STRATEGIES.md."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

BAR_HOURS = 4  # ТФ этой стратегии фиксирован (03_STRATEGIES.md не даёт сетки по tf)


class S12(BaseStrategy):
    id = "S12"
    name = "funding_contrarian"
    version = "1.0.0"
    default_params = {"z_in": 2.0, "hold": 18, "confirm": "none", "lookback_days": 30, "tf": "4h"}
    param_grid = {
        "z_in": [1.5, 2.0, 2.5],
        "hold": [6, 18, 30],
        "confirm": ["none", "ema20"],
    }
    timeframes = ["4h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return self.params.get("lookback_days", 30) * (24 // BAR_HOURS) + 5

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()

        funding = ctx.funding
        if funding is None or funding.empty:
            out["f8"], out["z"] = float("nan"), float("nan")
            return out

        f = funding.sort_index().copy()
        interval_hours = f.index.to_series().diff().dt.total_seconds() / 3600
        f["f8"] = f["funding_rate"] * (8.0 / interval_hours)
        lookback = f"{self.params.get('lookback_days', 30)}D"
        f8_mean = f["f8"].rolling(lookback).mean()
        f8_std = f["f8"].rolling(lookback).std(ddof=0).clip(lower=0.00005)
        f["z"] = (f["f8"] - f8_mean) / f8_std

        close_times = out.index + pd.Timedelta(hours=BAR_HOURS)
        # merge_asof требует одинаковую точность datetime64 у обеих сторон - bars и
        # funding приходят из разных parquet-файлов и могут не совпадать по unit
        # (us/ms/ns), даже если оба tz-aware UTC (реальная причина падения на pandas 3.0).
        left = pd.DataFrame(index=pd.DatetimeIndex(close_times).as_unit("us"))
        right = f[["f8", "z"]].set_axis(f.index.as_unit("us"))
        merged = pd.merge_asof(
            left, right,
            left_index=True, right_index=True, direction="backward", allow_exact_matches=False,
        )
        out["f8"] = merged["f8"].to_numpy()
        out["z"] = merged["z"].to_numpy()
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        z, f8 = row["z"], row["f8"]
        if pd.isna(z) or pd.isna(f8):
            return Decision(target=0.0, tag="warmup")

        z_in = self.params.get("z_in", 2.0)
        hold = self.params.get("hold", 18)
        confirm = self.params.get("confirm", "none")
        close, ema20 = row["close"], row["ema20"]

        if pos.target != 0:
            if abs(z) < 0.5 or pos.bars_in_trade >= hold:
                return Decision(target=0.0, tag="exit")
            return Decision(target=pos.target, tag="hold")

        short_ok = z > z_in and f8 >= 0.0002 and (confirm != "ema20" or (pd.notna(ema20) and close < ema20))
        long_ok = z < -z_in and f8 < 0 and (confirm != "ema20" or (pd.notna(ema20) and close > ema20))
        if short_ok:
            return Decision(target=-1.0, tag="entry_short")
        if long_ok:
            return Decision(target=1.0, tag="entry_long")
        return Decision(target=0.0, tag="flat")
