"""Этап 7: walk-forward для S01 и S01b (donchian_ensemble). Оба используют один и тот же
param_grid (S01b наследует его от S01, отличие только в direction) — один скрипт,
два независимых прогона walk-forward (V1/V2 могут получиться разными).

ТФ зафиксирован на дефолтном (1d) - см. _wf_common.py. `lookback_set`/`vol_target` -
оба категориальные (нет естественного числового шага, `vol_target` вдобавок включает
`None`), плато вырождается в выбор лучшей комбинации напрямую (без усреднения по
соседям) - честно для этой сетки, не искусственная подгонка под алгоритм плато.

Запуск: python scripts/run_wf_s01.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s01_donchian import S01, S01b

TF = "1d"
LOOKBACK_SET_GRID = ["all9", "short", "long"]
VOL_TARGET_GRID = [0.25, 0.5, None]

if __name__ == "__main__":
    grid_points = [
        {"lookback_set": ls, "vol_target": vt}
        for ls, vt in product(LOOKBACK_SET_GRID, VOL_TARGET_GRID)
    ]
    for strategy_id, cls in (("S01", S01), ("S01b", S01b)):
        run_and_report(
            strategy_id=strategy_id, strategy_cls=cls, tf=TF,
            default_params={"lookback_set": "all9", "vol_target": 0.25},
            grid_points=grid_points,
            ordinal_grids={},
            categorical_params=("lookback_set", "vol_target"),
            fixed_extra_params={"tf": TF, "vol_window": 90},
        )
