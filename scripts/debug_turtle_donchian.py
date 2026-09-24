"""Временный диагностический скрипт для разбора теста
test_turtle_donchian_reversal_flips_sign_directly - печатает подробности short_break
(канал 1 него минимум, канал 2, условие пересечения) на участке спада. Удалить после разбора."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.strategies.base import Context
from lab.strategies.community.gosha.turtle_donchian import GoshaTurtleDonchian


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


idx = pd.date_range("2023-01-01", periods=250, freq="4h", tz="UTC")
flat = np.full(70, 100.0)
up = 100 + (np.arange(80) + 1) * 2.0
down = up[-1] - (np.arange(100) + 1) * 2.0
close = np.concatenate([flat, up, down])
bars = pd.DataFrame({
    "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
    "volume": 1000.0, "quote_volume": 100.0, "trades": 1, "complete": True,
}, index=idx)

strat = GoshaTurtleDonchian()
out = strat.prepare(bars, _ctx())

l = out["low"]
c1l = l.shift(1).rolling(20).min()
c2l = l.shift(21).rolling(20).min()
c = out["close"]

df = pd.DataFrame({
    "close": c, "c1l": c1l, "c2l": c2l,
    "below_c1l": c < c1l,
    "cross_cond": c.shift(1) >= c1l.shift(1),
    "below_c2l": c < c2l,
    "short_break": out["short_break"],
})
pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 140)
print(df.iloc[145:220])
print()
print("short_break sum overall:", out["short_break"].sum())
print("below_c1l sum in decline (150+):", df["below_c1l"].iloc[150:].sum())
print("cross_cond True count in decline (150+):", df["cross_cond"].iloc[150:].sum())
