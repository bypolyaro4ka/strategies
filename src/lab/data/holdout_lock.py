"""Физическая изоляция holdout-данных — второй барьер поверх loader.py (05_PLAN.md, Этап 2).

Код в loader.py уже отказывается отдавать данные на/после holdout_start без явной
разблокировки (HoldoutLockedError). Этот модуль — дополнительная защита на уровне
файловой системы: после сборки ТФ бары и фандинг с 2026-01-01 физически переносятся
в отдельную подпапку. Если кто-то (агент или сам пользователь) в спешке напишет
`pd.read_parquet("data/processed/BNBUSDT_1d.parquet")` в блокноте для отладки —
он увидит только dev-период, будущее там просто не лежит.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lab.config import load_protocol
from lab.data.download import PROCESSED_DIR

LOCKED_DIR = PROCESSED_DIR / "_holdout_locked"

LOCKED_README = """\
# Не открывать до Этапа 8

Здесь лежат бары и фандинг с holdout_start (2026-01-01) — ровно то, что протокол
(docs/01_PROTOCOL.md, раздел 10) запрещает трогать до заморозки и явного разрешения
пользователя. Обычный код (loader.py) сюда не заглядывает вообще. Если вы открываете
эти файлы руками — вы нарушаете протокол, даже если "просто посмотреть график".
"""


def split_holdout(df: pd.DataFrame, holdout_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Индекс df — open_time или funding_time, UTC. Возвращает (dev, locked)."""
    is_holdout = df.index >= holdout_start
    return df.loc[~is_holdout], df.loc[is_holdout]


def apply_holdout_lock() -> None:
    cfg = load_protocol()
    holdout_start = pd.Timestamp(cfg.periods.holdout_start, tz="UTC")

    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    (LOCKED_DIR / "README.md").write_text(LOCKED_README, encoding="utf-8")

    for path in sorted(PROCESSED_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        dev, locked = split_holdout(df, holdout_start)
        dev.to_parquet(path)
        if len(locked) > 0:
            locked.to_parquet(LOCKED_DIR / path.name)
        print(f"  {path.name}: dev={len(dev)} строк, holdout={len(locked)} строк перенесено")


if __name__ == "__main__":
    apply_holdout_lock()
