"""Этап 7: walk-forward для S09 (volatility_breakout). Единственный ТФ у стратегии - 1h
(нет отдельного пункта в сетке 03_STRATEGIES.md, тф не варьируется).

Запуск: python scripts/run_wf_s09.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s09_volatility_breakout import S09

TF = "1h"
K_GRID = [0.3, 0.5, 0.7]
MODE_GRID = ["LS", "long_only"]
TREND_FILTER_GRID = ["none", "sma5d"]

if __name__ == "__main__":
    grid_points = [
        {"k": k, "mode": m, "trend_filter": tf_}
        for k, m, tf_ in product(K_GRID, MODE_GRID, TREND_FILTER_GRID)
    ]
    run_and_report(
        strategy_id="S09", strategy_cls=S09, tf=TF,
        default_params={"k": 0.5, "mode": "LS", "trend_filter": "none"},
        grid_points=grid_points,
        ordinal_grids={"k": K_GRID},
        categorical_params=("mode", "trend_filter"),
        fixed_extra_params={"tf": TF},
    )
