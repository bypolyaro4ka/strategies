"""Тесты для src/lab/data/loader.py — Этап 2 (защита holdout — самое важное здесь)."""

import pandas as pd
import pytest

import lab.data.loader as loader_module
from lab.data.loader import HoldoutLockedError, load_bars


def _write_bars(dir_, symbol, tf, idx):
    df = pd.DataFrame({
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": 1.0, "quote_volume": 1.0, "trades": 1, "complete": True,
    }, index=idx)
    df.index.name = "open_time"
    df.to_parquet(dir_ / f"{symbol}_{tf}.parquet")
    return df


def test_load_bars_returns_requested_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "PROCESSED_DIR", tmp_path)
    idx = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    _write_bars(tmp_path, "BNBUSDT", "1d", idx)

    out = load_bars("BNBUSDT", "1d", "2023-01-03", "2023-01-05")
    assert len(out) == 3
    assert out.index.min() == pd.Timestamp("2023-01-03", tz="UTC")


def test_load_bars_raises_on_holdout_without_unlock(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "PROCESSED_DIR", tmp_path)
    monkeypatch.delenv("HOLDOUT_UNLOCK", raising=False)
    idx = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    _write_bars(tmp_path, "BNBUSDT", "1d", idx)  # файл есть, но запрошенный диапазон - holdout

    with pytest.raises(HoldoutLockedError):
        load_bars("BNBUSDT", "1d", "2025-12-01", "2026-01-15")


def test_load_bars_missing_file_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "PROCESSED_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_bars("BNBUSDT", "1d", "2023-01-01", "2023-01-05")
