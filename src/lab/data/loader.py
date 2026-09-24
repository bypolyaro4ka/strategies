"""Единственная точка, через которую стратегии и отчёты читают данные
(02_ENGINE_SPEC.md, 2.4). Второй, файловый барьер — holdout_lock.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from lab.config import ProtocolConfig, load_protocol
from lab.data.download import PROCESSED_DIR


class HoldoutLockedError(RuntimeError):
    """Запрошены данные из holdout-периода без разблокировки (см. 01_PROTOCOL.md, раздел 10)."""


def holdout_unlocked(cfg: ProtocolConfig) -> bool:
    """Все три условия одновременно — намеренно строго, разблокировка не должна
    случиться "по ошибке" от одной забытой переменной окружения."""
    if not cfg.frozen:
        return False
    if os.environ.get("HOLDOUT_UNLOCK") != "1":
        return False
    try:
        tags = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
    except (subprocess.SubprocessError, OSError):
        return False
    return "holdout-freeze-v1" in tags


def _check_holdout(end: pd.Timestamp, cfg: ProtocolConfig) -> None:
    holdout_start = pd.Timestamp(cfg.periods.holdout_start, tz="UTC")
    if end >= holdout_start and not holdout_unlocked(cfg):
        raise HoldoutLockedError(
            "Запрошены данные holdout (end >= holdout_start). Разблокировка — только "
            "на Этапе 8, все три условия сразу: protocol.yaml: frozen=true, "
            "переменная окружения HOLDOUT_UNLOCK=1, git tag holdout-freeze-v1 на текущем "
            "коммите (см. 01_PROTOCOL.md, раздел 10)."
        )


def load_bars(symbol: str, tf: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Бары symbol на таймфрейме tf в [start, end] (обе границы включительно).
    start уже должен учитывать прогрев (required_history стратегии) — загрузчик
    сам warmup не добавляет, просто отдаёт то, что попросили."""
    cfg = load_protocol()
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
    _check_holdout(end, cfg)

    path = PROCESSED_DIR / f"{symbol}_{tf}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет данных {path} — запустите data/download.py и data/resample.py")
    df = pd.read_parquet(path)
    return df.loc[(df.index >= start) & (df.index <= end)]


def load_funding(symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """История фандинга symbol в [start, end]. Индекс funding_time (UTC)."""
    cfg = load_protocol()
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
    _check_holdout(end, cfg)

    path = PROCESSED_DIR / f"{symbol}_funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет данных {path} — запустите data/download.py")
    df = pd.read_parquet(path)
    return df.loc[(df.index >= start) & (df.index <= end)]
