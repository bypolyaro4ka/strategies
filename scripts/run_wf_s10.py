"""Этап 7: walk-forward для S10 (rsi2_connors). ТФ зафиксирован на 4h.
Запуск: python scripts/run_wf_s10.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s10_rsi2 import S10

TF = "4h"
LO_HI_GRID = [(5, 95), (10, 90), (15, 85)]
FILTER_SMA200_GRID = [True, False]
T_GRID = [5, 10, 20]

if __name__ == "__main__":
    grid_points = [
        {"lo_hi": lh, "lo": lh[0], "hi": lh[1], "filter_sma200": f, "T": t}
        for lh, f, t in product(LO_HI_GRID, FILTER_SMA200_GRID, T_GRID)
    ]
    run_and_report(
        strategy_id="S10", strategy_cls=S10, tf=TF,
        default_params={"lo_hi": (10, 90), "lo": 10, "hi": 90, "filter_sma200": True, "T": 10},
        grid_points=grid_points,
        ordinal_grids={"lo_hi": LO_HI_GRID, "T": T_GRID},
        categorical_params=("filter_sma200",),
        fixed_extra_params={"tf": TF},
    )
