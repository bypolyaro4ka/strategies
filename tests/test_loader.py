"""Тесты для src/lab/data/loader.py — Этап 2 (защита holdout — самое важное здесь)."""

import pandas as pd
import pytest

import lab.data.loader as loader_module
from lab.data.loader import HoldoutLockedError, holdout_unlocked, load_bars, load_funding


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


def test_load_funding_returns_requested_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "PROCESSED_DIR", tmp_path)
    idx = pd.date_range("2023-01-01", periods=10, freq="8h", tz="UTC")
    df = pd.DataFrame({"funding_rate": [0.0001] * 10}, index=idx)
    df.index.name = "funding_time"
    df.to_parquet(tmp_path / "BNBUSDT_funding.parquet")

    out = load_funding("BNBUSDT", "2023-01-01", "2023-01-02")
    assert len(out) < 10
    assert out.index.max() <= pd.Timestamp("2023-01-02", tz="UTC")


def test_load_funding_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "PROCESSED_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_funding("BNBUSDT", "2023-01-01", "2023-01-05")


def test_load_funding_raises_on_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "PROCESSED_DIR", tmp_path)
    with pytest.raises(HoldoutLockedError):
        load_funding("BNBUSDT", "2025-12-01", "2026-02-01")


# --- holdout_unlocked: все три условия разом ---

def test_holdout_unlocked_false_when_not_frozen():
    from lab.config import load_protocol
    assert holdout_unlocked(load_protocol()) is False  # protocol.yaml: frozen=false


def test_holdout_unlocked_false_without_env_var(monkeypatch):
    import dataclasses
    cfg = dataclasses.replace(load_protocol_for_test(), frozen=True)
    monkeypatch.delenv("HOLDOUT_UNLOCK", raising=False)
    assert holdout_unlocked(cfg) is False


def test_holdout_unlocked_false_without_git_tag(monkeypatch):
    cfg = load_protocol_for_test()
    import dataclasses
    cfg = dataclasses.replace(cfg, frozen=True)
    monkeypatch.setenv("HOLDOUT_UNLOCK", "1")
    # в реальном репозитории на текущем коммите нет тега holdout-freeze-v1
    assert holdout_unlocked(cfg) is False


def test_holdout_unlocked_false_when_git_command_fails(monkeypatch):
    """Если git недоступен (не репозиторий, не установлен...) - по умолчанию заблокировано,
    а не наоборот. "Fail closed", а не "fail open" - это осознанный выбор безопасности."""
    import subprocess
    import dataclasses

    cfg = dataclasses.replace(load_protocol_for_test(), frozen=True)
    monkeypatch.setenv("HOLDOUT_UNLOCK", "1")

    def _boom(*args, **kwargs):
        raise subprocess.SubprocessError("git недоступен")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert holdout_unlocked(cfg) is False


def load_protocol_for_test():
    from lab.config import load_protocol
    return load_protocol()
