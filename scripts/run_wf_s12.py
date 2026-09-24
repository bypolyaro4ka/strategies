"""Этап 7: walk-forward для S12 (funding_contrarian). ТФ зафиксирован на 4h.
Нужен ctx.funding по каждой монете отдельно - используем make_prepare_fn_with_funding()
(см. _wf_common.py), а не общий prepare_fn_base.

Запуск: python scripts/run_wf_s12.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import make_prepare_fn_with_funding, run_and_report
from lab.strategies.s12_funding_contrarian import S12

TF = "4h"
Z_IN_GRID = [1.5, 2.0, 2.5]
HOLD_GRID = [6, 18, 30]
CONFIRM_GRID = ["none", "ema20"]

if __name__ == "__main__":
    grid_points = [
        {"z_in": z, "hold": h, "confirm": c}
        for z, h, c in product(Z_IN_GRID, HOLD_GRID, CONFIRM_GRID)
    ]
    run_and_report(
        strategy_id="S12", strategy_cls=S12, tf=TF,
        default_params={"z_in": 2.0, "hold": 18, "confirm": "none"},
        grid_points=grid_points,
        ordinal_grids={"z_in": Z_IN_GRID, "hold": HOLD_GRID},
        categorical_params=("confirm",),
        fixed_extra_params={"tf": TF, "lookback_days": 30},
        prepare_fn_factory=make_prepare_fn_with_funding,
    )
