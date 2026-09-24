"""Этап 7: walk-forward для S05 (bollinger_breakout). ТФ зафиксирован на 1d.
Запуск: python scripts/run_wf_s05.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s05_bollinger import S05

TF = "1d"
N_GRID = [20, 50]
K_GRID = [1.5, 2, 2.5]
EXIT_RULE_GRID = ["mid", "opposite"]

if __name__ == "__main__":
    grid_points = [
        {"n": n, "k": k, "exit_rule": e}
        for n, k, e in product(N_GRID, K_GRID, EXIT_RULE_GRID)
    ]
    run_and_report(
        strategy_id="S05", strategy_cls=S05, tf=TF,
        default_params={"n": 20, "k": 2, "exit_rule": "mid"},
        grid_points=grid_points,
        ordinal_grids={"n": N_GRID, "k": K_GRID},
        categorical_params=("exit_rule",),
        fixed_extra_params={"tf": TF},
    )
