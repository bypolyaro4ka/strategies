"""Этап 7: walk-forward для S02 (tsmom). Первый прогон, обкатавший optimize/walk_forward.py
целиком (train-сетка, плато, отбор монет, test "с нуля", склейка WF-OOS, выбор основного
варианта) — см. JOURNAL. Дальше используется общая обвязка `_wf_common.py`.

Упрощение первого прохода: сетка НЕ ищет по `tf` (у S02 в 03_STRATEGIES.md tf тоже часть
сетки: {1d, 12h}) - тестируется только дефолтный tf=1d. lookback_days трактуется как
порядковый параметр в том порядке, в котором перечислен в 03_STRATEGIES.md.

Запуск: python scripts/run_wf_s02.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wf_common import run_and_report
from lab.strategies.s02_tsmom import S02

TF = "1d"
LOOKBACK_GRID = [(3, 7, 14), (7, 14, 28), (14, 28, 56), (7,), (14,), (28,)]
REBALANCE_GRID = ["every_bar", "weekly"]

if __name__ == "__main__":
    grid_points = [{"lookback_days": lb, "rebalance": rb} for lb, rb in product(LOOKBACK_GRID, REBALANCE_GRID)]
    run_and_report(
        strategy_id="S02", strategy_cls=S02, tf=TF,
        default_params={"lookback_days": (7, 14, 28), "rebalance": "every_bar"},
        grid_points=grid_points,
        ordinal_grids={"lookback_days": LOOKBACK_GRID},
        categorical_params=("rebalance",),
        fixed_extra_params={"tf": TF},
    )
