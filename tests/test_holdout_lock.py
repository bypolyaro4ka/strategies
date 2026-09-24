"""Тесты для src/lab/data/holdout_lock.py — Этап 2."""

import pandas as pd

from lab.data.holdout_lock import split_holdout


def test_split_holdout_partitions_by_index():
    idx = pd.date_range("2025-12-30", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)
    holdout_start = pd.Timestamp("2026-01-01", tz="UTC")

    dev, locked = split_holdout(df, holdout_start)

    assert list(dev.index) == [pd.Timestamp("2025-12-30", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")]
    assert list(locked.index) == list(idx[2:])
    assert len(dev) + len(locked) == len(df)


def test_split_holdout_all_dev():
    idx = pd.date_range("2023-01-01", periods=3, freq="1D", tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    dev, locked = split_holdout(df, pd.Timestamp("2026-01-01", tz="UTC"))
    assert len(dev) == 3
    assert len(locked) == 0
