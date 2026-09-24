"""Временный диагностический скрипт - проверка исправленной синтетики теста
test_turtle_donchian_reversal_flips_sign_directly (плоский участок и перед входом,
и перед разворотом). Удалить после подтверждения, что short_break срабатывает."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lab.strategies.base import Context
from lab.strategies.community.gosha.turtle_donchian import GoshaTurtleDonchian


def _ctx():
    cfg = SimpleNamespace(account=SimpleNamespace(), exchange=SimpleNamespace(), metrics=SimpleNamespace())
    return Context(params={}, protocol=cfg)


idx = pd.date_range("2023-01-01", periods=280, freq="4h", tz="UTC")
flat1 = np.full(70, 100.0)
up = 100 + (np.arange(80) + 1) * 2.0
flat2 = np.full(50, 260.0)
down = 260 - (np.arange(80) + 1) * 2.0
close = np.concatenate([flat1, up, flat2, down])
bars = pd.DataFrame({
    "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
    "volume": 1000.0, "quote_volume": 100.0, "trades": 1, "complete": True,
}, index=idx)

strat = GoshaTurtleDonchian()
out = strat.prepare(bars, _ctx())

print("long_break True count:", out["long_break"].sum(), "at bars:",
      list(out.index[out["long_break"]].map(lambda t: out.index.get_loc(t))))
print("short_break True count:", out["short_break"].sum(), "at bars:",
      list(out.index[out["short_break"]].map(lambda t: out.index.get_loc(t))))

pos = SimpleNamespace(target=0.0, bars_in_trade=0, stop_price=None, take_price=None)
targets = []
for t, row in out.iterrows():
    d = strat.on_bar(t, row, pos, _ctx())
    targets.append(d.target)
    pos = SimpleNamespace(target=d.target, bars_in_trade=0, stop_price=d.stop_price, take_price=d.take_price)

print()
print("target value counts:", pd.Series(targets).value_counts().to_dict())
print("first bar where target==1.0:", targets.index(1.0) if 1.0 in targets else "NEVER")
print("first bar where target==-1.0:", targets.index(-1.0) if -1.0 in targets else "NEVER")
