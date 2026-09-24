"""Этап 7: walk-forward для S06 (supertrend). ТФ зафиксирован на 4h.
Запуск: python scripts/run_wf_s06.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s06_supertrend import S06

TF = "4h"
N_GRID = [7, 10, 14]
M_GRID = [2, 3, 4]

if __name__ == "__main__":
    grid_points = [{"n": n, "m": m} for n, m in product(N_GRID, M_GRID)]
    run_and_report(
        strategy_id="S06", strategy_cls=S06, tf=TF,
        default_params={"n": 10, "m": 3},
        grid_points=grid_points,
        ordinal_grids={"n": N_GRID, "m": M_GRID},
        categorical_params=(),
        fixed_extra_params={"tf": TF},
    )
