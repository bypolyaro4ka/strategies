"""Этап 7: walk-forward для S03 (price_vs_sma). ТФ зафиксирован на 1d.
Запуск: python scripts/run_wf_s03.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s03_sma import S03

TF = "1d"
N_DAYS_GRID = [10, 20, 50, 100]
BAND_GRID = [0.0, 0.01, 0.02]
MODE_GRID = ["LS", "LF"]

if __name__ == "__main__":
    grid_points = [
        {"n_days": n, "band": b, "mode": m}
        for n, b, m in product(N_DAYS_GRID, BAND_GRID, MODE_GRID)
    ]
    run_and_report(
        strategy_id="S03", strategy_cls=S03, tf=TF,
        default_params={"n_days": 20, "band": 0.0, "mode": "LS"},
        grid_points=grid_points,
        ordinal_grids={"n_days": N_DAYS_GRID, "band": BAND_GRID},
        categorical_params=("mode",),
        fixed_extra_params={"tf": TF},
    )
