"""Этап 7: walk-forward для S08 (macd_ema_filter). ТФ зафиксирован на 4h.
`filter_n` (200/100/None) - категориальный, естественного числового шага между
"фильтр EMA100" и "без фильтра" нет.

Запуск: python scripts/run_wf_s08.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s08_macd import S08

TF = "4h"
MACD_GRID = [(12, 26, 9), (8, 21, 5)]
FILTER_N_GRID = [200, 100, None]

if __name__ == "__main__":
    grid_points = [
        {"macd": macd, "fast": macd[0], "slow": macd[1], "signal": macd[2], "filter_n": fn}
        for macd, fn in product(MACD_GRID, FILTER_N_GRID)
    ]
    run_and_report(
        strategy_id="S08", strategy_cls=S08, tf=TF,
        default_params={"macd": (12, 26, 9), "fast": 12, "slow": 26, "signal": 9, "filter_n": 200},
        grid_points=grid_points,
        ordinal_grids={"macd": MACD_GRID},
        categorical_params=("filter_n",),
        fixed_extra_params={"tf": TF},
    )
