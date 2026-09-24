"""Этап 7: walk-forward для S13 (cross_sectional_momentum, PortfolioStrategy).
"Отбор монет (V2) не применяется" (03_STRATEGIES.md) - apply_coin_selection=False,
V2 совпадает с V1 (тот же пул, те же параметры). prepare_fn=prepare_fn_portfolio -
один вызов prepare() со всем словарём сразу, не цикл по монетам.

Запуск: python scripts/run_wf_s13.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import prepare_fn_portfolio, run_and_report
from lab.strategies.s13_cross_sectional_momentum import S13

TF = "1d"
L_GRID = [7, 14, 21, 28]
K_GRID = [2, 3]
SKIP_GRID = [0, 1]

if __name__ == "__main__":
    grid_points = [
        {"L": lb, "k": k, "skip": sk}
        for lb, k, sk in product(L_GRID, K_GRID, SKIP_GRID)
    ]
    run_and_report(
        strategy_id="S13", strategy_cls=S13, tf=TF,
        default_params={"L": 21, "k": 3, "skip": 0},
        grid_points=grid_points,
        ordinal_grids={"L": L_GRID, "k": K_GRID, "skip": SKIP_GRID},
        categorical_params=(),
        apply_coin_selection=False,
        prepare_fn=prepare_fn_portfolio,
    )
