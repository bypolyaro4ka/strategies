"""Портирован из s02_btc_rsi_dca.py/s03_eth_rsi_dca.py/s09_sol_rsi_dca.py (Гоша) -
идентичная логика под тремя тикерами в оригинале, у нас одна стратегия на весь пул
(как остальные наши S01-S05), см. DECLARATION.md/SPEC.md рядом.

Оригинал - мартингейл-DCA: RSI(14) < 28 -> базовый вход, затем до 5 Safety Orders на
фиксированной лестнице просадки от цены ПЕРВОГО входа, тейк +3% от средней цены, без
стопа. У автора размер лестницы - фиксированные доллары ($500 -> $900 -> ... -> $9448).

ПЕРЕИНТЕРПРЕТАЦИЯ (согласована с пользователем, см. JOURNAL 2026-09-24):
1. Доллары -> доли слота: target_i = cumsum(notional[:i+1]) / max_total_notional -
   то же соотношение объёмов между уровнями, что и у автора, просто в долях, не в $.
2. Триггеры AO у автора считаются от цены ПЕРВОГО входа (`base_price`), которую нужно
   было бы помнить между барами. Наш движок вызывает on_bar() как чистую функцию
   (t, row, pos, ctx) - без собственного состояния стратегии между барами (это
   проверяют truncation/future-poison тесты: on_bar должен давать одинаковый результат
   независимо от того, сколько раз он вызывался раньше). Поэтому триггеры считаются от
   `pos.avg_entry_price` (его ведёт сам движок) - это ЧЕСТНОЕ отличие от оригинала, не
   тихая замена: требует немного БОЛЬШЕЙ просадки для срабатывания каждого следующего
   уровня, чем у автора (средняя цена после долива всегда меньше исходной цены первого
   входа), то есть консервативнее оригинала, а не агрессивнее.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

RSI_LENGTH = 14
RSI_ENTRY = 28.0
AO_DEVIATIONS = (0.02, 0.05, 0.095, 0.16, 0.25)
NOTIONALS = (500.0, 900.0, 1620.0, 2916.0, 5249.0, 9448.0)  # база + 5 AO
TAKE_PROFIT = 0.03

_CUM = np.cumsum(NOTIONALS)
LEVEL_FRACS = tuple(_CUM / _CUM[-1])  # доля слота на каждом уровне: (0.0242, 0.0678, ..., 1.0)


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - 100 / (1 + gains / losses.replace(0, np.nan))


def _level_index(target: float) -> int:
    """Индекс текущего уровня (0 = только база, 5 = все AO залиты) по значению target."""
    diffs = [abs(target - f) for f in LEVEL_FRACS]
    return diffs.index(min(diffs))


class GoshaRsiDca(BaseStrategy):
    id = "COMM_GOSHA_RSI_DCA"
    name = "rsi_dca"
    version = "1.0.0"
    author = "Гоша"
    timeframes = ["4h"]
    direction = "long_only"  # правило автора - только лонг, усреднение вниз

    def required_history(self, tf: str) -> int:
        return RSI_LENGTH + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        out["rsi"] = _rsi(out["close"], RSI_LENGTH)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        close = row["close"]

        if pos.target == 0:
            rsi = row["rsi"]
            if pd.notna(rsi) and rsi < RSI_ENTRY:
                return Decision(target=LEVEL_FRACS[0], tag="entry_base")
            return Decision(target=0.0, tag="flat")

        # тейк +3% от средней цены входа - выход целиком, без частичного закрытия
        if close >= pos.avg_entry_price * (1 + TAKE_PROFIT):
            return Decision(target=0.0, tag="tp_3pct_over_average")

        idx = _level_index(pos.target)
        if idx >= len(AO_DEVIATIONS):
            return Decision(target=pos.target, tag="hold_full")  # все AO уже залиты

        trigger = pos.avg_entry_price * (1 - AO_DEVIATIONS[idx])
        if close <= trigger:
            return Decision(target=LEVEL_FRACS[idx + 1], tag=f"ao_{idx + 1}")
        return Decision(target=pos.target, tag="hold")
