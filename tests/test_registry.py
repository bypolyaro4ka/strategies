"""Тесты для src/lab/registry.py — Этап 3 ("Registry append-only" из 02_ENGINE_SPEC.md §7)."""

from lab.registry import count_trials, log_run


def test_log_run_creates_file_with_header(tmp_path):
    path = tmp_path / "registry.csv"
    log_run({"run_id": "r1", "strategy_id": "S01", "stage": "dev", "debug": "False"}, {"sharpe": "1.2"}, path=path)
    text = path.read_text(encoding="utf-8")
    assert "run_id" in text.splitlines()[0]
    assert "r1" in text


def test_log_run_appends_without_touching_previous_rows():
    """Главная гарантия: вторая запись не трогает байты первой строки."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "registry.csv"
        log_run({"run_id": "r1", "strategy_id": "S01", "stage": "dev", "debug": "False"}, {}, path=path)
        after_first = path.read_text(encoding="utf-8")

        log_run({"run_id": "r2", "strategy_id": "S02", "stage": "dev", "debug": "False"}, {}, path=path)
        after_second = path.read_text(encoding="utf-8")

        assert after_second.startswith(after_first)  # первая запись побайтово не изменилась
        assert "r2" in after_second and "r2" not in after_first


def test_count_trials_excludes_debug_and_other_stage(tmp_path):
    path = tmp_path / "registry.csv"
    log_run({"run_id": "r1", "strategy_id": "S01", "stage": "dev", "debug": "False"}, {}, path=path)
    log_run({"run_id": "r2", "strategy_id": "S01", "stage": "dev", "debug": "True"}, {}, path=path)  # debug
    log_run({"run_id": "r3", "strategy_id": "S01", "stage": "wf", "debug": "False"}, {}, path=path)  # другая стадия
    log_run({"run_id": "r4", "strategy_id": "S02", "stage": "dev", "debug": "False"}, {}, path=path)  # другая стратегия

    assert count_trials("S01", "dev", path=path) == 1


def test_count_trials_missing_file_returns_zero(tmp_path):
    assert count_trials("S01", "dev", path=tmp_path / "does_not_exist.csv") == 0
