"""Этап 7: walk-forward для S04 (turtle_system1). ТФ зафиксирован на 4h.
Сетка группирует (n_in, n_out) парой в 03_STRATEGIES.md - держим кортеж как отдельный
ключ `n_in_out` только для соседства в плато, реальные `n_in`/`n_out` разворачиваются
рядом (стратегия читает их по отдельности, лишний ключ `n_in_out` в params безвреден).

Запуск: python scripts/run_wf_s04.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s04_turtle import S04

TF = "4h"
N_IN_OUT_GRID = [(20, 10), (55, 20)]
K_GRID = [2, 3]

if __name__ == "__main__":
    grid_points = [
        {"n_in_out": nio, "n_in": nio[0], "n_out": nio[1], "k": k}
        for nio, k in product(N_IN_OUT_GRID, K_GRID)
    ]
    run_and_report(
        strategy_id="S04", strategy_cls=S04, tf=TF,
        default_params={"n_in_out": (20, 10), "n_in": 20, "n_out": 10, "k": 2},
        grid_points=grid_points,
        ordinal_grids={"n_in_out": N_IN_OUT_GRID, "k": K_GRID},
        categorical_params=(),
        fixed_extra_params={"tf": TF, "atr_n": 20},
    )
