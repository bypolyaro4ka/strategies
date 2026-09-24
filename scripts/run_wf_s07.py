"""Этап 7: walk-forward для S07 (ema_cross). ТФ зафиксирован на 4h.
Запуск: python scripts/run_wf_s07.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s07_ema_cross import S07

TF = "4h"
FAST_SLOW_GRID = [(20, 50), (20, 100), (50, 200)]
MODE_GRID = ["LS", "LF"]

if __name__ == "__main__":
    grid_points = [
        {"fast_slow": fs, "fast": fs[0], "slow": fs[1], "mode": m}
        for fs, m in product(FAST_SLOW_GRID, MODE_GRID)
    ]
    run_and_report(
        strategy_id="S07", strategy_cls=S07, tf=TF,
        default_params={"fast_slow": (50, 200), "fast": 50, "slow": 200, "mode": "LS"},
        grid_points=grid_points,
        ordinal_grids={"fast_slow": FAST_SLOW_GRID},
        categorical_params=("mode",),
        fixed_extra_params={"tf": TF},
    )
