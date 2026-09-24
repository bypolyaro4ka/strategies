"""Тесты для src/lab/metrics/dsr.py — Этап 4.

Формула DSR сложная и её проверка "полными руками" на нашем этапе рискует стать
переписыванием самой формулы. Вместо точных чисел проверяем свойства, которые обязаны
выполняться независимо от деталей реализации (монотонность по числу испытаний, по силе
сигнала, границы [0, 1]) — если хоть одно из них нарушится, это будет сигналом ошибки.
"""

import numpy as np
import pandas as pd
import pytest

from lab.metrics.dsr import deflated_sharpe_ratio, expected_max_sharpe


def _returns(seed: int, n: int, mean: float, std: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_dsr_in_valid_range():
    r = _returns(0, 500, mean=0.001, std=0.01)
    dsr = deflated_sharpe_ratio(r, n_trials=50)
    assert 0.0 <= dsr <= 1.0


def test_dsr_decreases_with_more_trials():
    r = _returns(1, 500, mean=0.0008, std=0.01)
    dsr_few = deflated_sharpe_ratio(r, n_trials=1)
    dsr_many = deflated_sharpe_ratio(r, n_trials=10_000)
    assert dsr_many < dsr_few  # больше испытаний - выше случайная планка, DSR ниже


def test_dsr_increases_with_stronger_signal():
    weak = _returns(2, 500, mean=0.0002, std=0.01)
    strong = _returns(2, 500, mean=0.003, std=0.01)  # тот же сид - тот же шум, выше снос
    assert deflated_sharpe_ratio(strong, n_trials=100) > deflated_sharpe_ratio(weak, n_trials=100)


def test_dsr_too_few_observations_is_nan():
    r = pd.Series([0.01, -0.01])
    assert pd.isna(deflated_sharpe_ratio(r, n_trials=10))


def test_dsr_zero_variance_is_nan():
    r = pd.Series([0.01] * 100)  # константа - std=0
    assert pd.isna(deflated_sharpe_ratio(r, n_trials=10))


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(sr_std=0.1, n_trials=10) < expected_max_sharpe(sr_std=0.1, n_trials=1000)


def test_expected_max_sharpe_zero_when_std_zero():
    assert expected_max_sharpe(sr_std=0.0, n_trials=100) == pytest.approx(0.0)
