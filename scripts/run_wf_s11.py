"""Этап 7: walk-forward для S11 (bb_reversion_adx). ТФ зафиксирован на 4h.
Запуск: python scripts/run_wf_s11.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s11_bb_adx import S11

TF = "4h"
ADX_THR_GRID = [15, 20, 25]
K_GRID = [2, 2.5]
S_GRID = [1.5, 2, 3]

if __name__ == "__main__":
    grid_points = [
        {"adx_thr": a, "k": k, "s": s}
        for a, k, s in product(ADX_THR_GRID, K_GRID, S_GRID)
    ]
    run_and_report(
        strategy_id="S11", strategy_cls=S11, tf=TF,
        default_params={"adx_thr": 20, "k": 2, "s": 2},
        grid_points=grid_points,
        ordinal_grids={"adx_thr": ADX_THR_GRID, "k": K_GRID, "s": S_GRID},
        categorical_params=(),
        fixed_extra_params={"tf": TF, "n_bb": 20, "n_adx": 14, "time_stop": 20},
    )
