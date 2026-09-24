"""O1/O2 — надстройки (03_STRATEGIES.md). Оборачиваются через `WithOverlays` из
`base.py`: `WithOverlays(S02(), [BtcRegimeFilter(), VolTarget()])`.

`Overlay` не обязан реализовывать `prepare()`/`required_history()` (только `apply()` —
контракт в `base.py`), но обе надстройки здесь их используют: причинные величины
(SMA режима BTC, скользящая волатильность монеты) нужно посчитать один раз векторно,
а не пересчитывать заново на каждом баре внутри `apply()`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategies.base import Decision


class BtcRegimeFilter:
    """O1 — фильтр режима BTC. Блокирует лонг, если BTC ниже своей SMA (медвежий
    режим), и шорт, если BTC выше (бычий режим) — заявленное правило работает "против"
    входа, не переворачивает его. Применяется к S01b, S02, S06 (03_STRATEGIES.md,
    список фиксирован)."""

    def __init__(self, sma_days: int = 50):
        self.sma_days = sma_days

    def required_history(self, tf: str) -> int:
        return 0  # тепло/прогрев зависит от ctx.btc_bars, не от bars текущей стратегии

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        btc = ctx.btc_bars
        if btc is None or btc.empty:
            out["btc_regime"] = float("nan")
            return out

        btc_close = btc["close"]
        btc_sma = btc_close.rolling(self.sma_days, min_periods=self.sma_days).mean()
        regime = pd.Series(float("nan"), index=btc.index)
        regime[btc_sma.notna() & (btc_close > btc_sma)] = 1.0
        regime[btc_sma.notna() & (btc_close <= btc_sma)] = -1.0

        # причинный мёрж: "последнее ПОЛНОЕ дневное закрытие BTC" на момент закрытия
        # бара текущей стратегии - дневной бар BTC с открытием d закрывается в d+1 день.
        bar_duration = bars.index[1] - bars.index[0] if len(bars.index) > 1 else pd.Timedelta(hours=1)
        close_times = (out.index + bar_duration).as_unit("us")
        btc_close_times = (btc.index + pd.Timedelta(days=1)).as_unit("us")
        merged = pd.merge_asof(
            pd.DataFrame(index=close_times),
            pd.DataFrame({"btc_regime": regime.to_numpy()}, index=btc_close_times),
            left_index=True, right_index=True, direction="backward",
        )
        out["btc_regime"] = merged["btc_regime"].to_numpy()
        return out

    def apply(self, t: pd.Timestamp, symbol: str, row: pd.Series, decision: Decision, ctx) -> Decision:
        regime = row.get("btc_regime")
        if regime is None or pd.isna(regime):
            return decision
        if decision.target > 0 and regime < 0:
            return Decision(0.0, decision.stop_price, decision.take_price, decision.time_stop_bars,
                             tag=decision.tag + "+O1_blocked_bear")
        if decision.target < 0 and regime > 0:
            return Decision(0.0, decision.stop_price, decision.take_price, decision.time_stop_bars,
                             tag=decision.tag + "+O1_blocked_bull")
        return decision


class VolTarget:
    """O2 — таргетирование волатильности: масштабирует |target| вниз (никогда вверх,
    `scale <= 1`), если реализованная волатильность монеты выше целевой `sigma_star`.
    Применяется к S02, S03 (у S01 таргетирование волатильности уже встроено в саму
    стратегию — 03_STRATEGIES.md).

    Считает волатильность по `bars["close"]` В ТОМ ЖЕ ТФ, в котором работает обёрнутая
    стратегия (у S02/S03 дефолт — 1d, тогда это буквально "дневные лог-доходности" из
    спецификации) — не тянет отдельно дневные бары для не-дневных вариантов сетки
    (`tf=12h/4h`), это была бы MTF-конструкция того же рода, что уже отклонена для
    4-го друга (нет generic межтаймфреймового чтения в движке). Честное упрощение,
    не тихая замена — сетка O2 по ТФ не тестируется, только дефолтный 1d."""

    def __init__(self, sigma_star: float = 0.5, W: int = 90):
        self.sigma_star = sigma_star
        self.W = W

    def required_history(self, tf: str) -> int:
        return self.W + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        log_ret = np.log(out["close"] / out["close"].shift(1))
        annualization = 365.0 ** 0.5
        sigma = log_ret.rolling(self.W, min_periods=self.W).std(ddof=0) * annualization
        out["vol_scale"] = (self.sigma_star / sigma).clip(upper=1.0)
        return out

    def apply(self, t: pd.Timestamp, symbol: str, row: pd.Series, decision: Decision, ctx) -> Decision:
        scale = row.get("vol_scale")
        if scale is None or pd.isna(scale):
            return decision
        return Decision(decision.target * float(scale), decision.stop_price, decision.take_price,
                         decision.time_stop_bars, tag=decision.tag + "+O2_scaled")
