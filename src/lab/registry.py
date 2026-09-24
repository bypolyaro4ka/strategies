"""Реестр испытаний - append-only CSV (02_ENGINE_SPEC.md, раздел 6; заменяет MLflow,
см. docs/JOURNAL.md, 2026-09-24 "Отказ от MLflow").

`log_run()` только дописывает файл (режим "a", "append") — никогда не читает его целиком,
чтобы переписать, не сортирует и не дедуплицирует существующие строки. Это гарантия
"испытания нельзя удалить или скрыть" (CLAUDE.md, правило 3) на уровне кода, а не только
договорённости.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "reports" / "registry.csv"

RUN_COLUMNS = [
    "run_id", "timestamp_utc", "strategy_id", "strategy_version", "variant", "mode", "tf",
    "symbols", "period_start", "period_end", "fold", "stage", "debug", "git_commit",
    "data_hash", "author", "params_json", "artifacts_path",
]
# Фиксированный набор метрик из 01_PROTOCOL.md, раздел 8 - одинаковый для каждого прогона,
# поэтому колонки можно безопасно фиксировать в CSV-реестре (append-only это допускает,
# только если схема не меняется от строки к строке).
METRIC_COLUMNS = [
    "total_return", "cagr", "vol", "sharpe", "sortino", "maxdd", "calmar", "trades",
    "win_rate", "profit_factor", "avg_trade_net_bp", "cost_share", "exposure",
    "funding_paid", "random_pct", "dsr", "breadth",
]
ALL_COLUMNS = RUN_COLUMNS + METRIC_COLUMNS


def log_run(run_meta: dict, metrics: dict, path: Path = REGISTRY_PATH) -> None:
    """run_meta и metrics - обычные dict, отсутствующие ключи записываются пустой строкой
    (например, `fold` не нужен вне walk-forward). Дописывает ровно одну строку."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**{c: run_meta.get(c, "") for c in RUN_COLUMNS}, **{c: metrics.get(c, "") for c in METRIC_COLUMNS}}
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def count_trials(strategy_id: str, stage: str, path: Path = REGISTRY_PATH) -> int:
    """Число недебажных испытаний данной стратегии на данной стадии - вход для DSR
    (01_PROTOCOL.md, раздел 9)."""
    if not path.exists():
        return 0
    df = pd.read_csv(path, dtype=str)
    if len(df) == 0:
        return 0
    mask = (df["strategy_id"] == strategy_id) & (df["stage"] == stage) & (df["debug"] != "True")
    return int(mask.sum())
