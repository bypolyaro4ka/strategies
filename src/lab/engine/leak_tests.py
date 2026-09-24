"""Переиспользуемые truncation- и future-poison-тесты (02_ENGINE_SPEC.md, раздел 7).

Обязательны для каждой стратегии, включая стратегии друзей (06_LEADERBOARD.md, §5.3).
Вместо того чтобы каждая стратегия писала эту проверку заново, она просто вызывает две
функции отсюда в своём тестовом файле — логика проверки одна и та же для всех 13+ стратегий.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from lab.strategies.base import PositionState


def assert_truncation_safe(strategy, bars: pd.DataFrame, ctx, n_checks: int = 50, seed: int = 0) -> None:
    """Решения, посчитанные на данных до t, должны совпадать с решениями на полных
    данных в точке t. bars.loc[:t] — данные включительно по t (то, что реально доступно
    на момент закрытия бара t)."""
    full_prepared = strategy.prepare(bars, ctx)
    tf = strategy.timeframes[0]
    warmup = strategy.required_history(tf)
    candidates = list(full_prepared.index[warmup:])
    if not candidates:
        return
    rng = random.Random(seed)
    checks = rng.sample(candidates, min(n_checks, len(candidates)))
    pos = PositionState(symbol="TRUNCATION_TEST")

    for t in checks:
        truncated_prepared = strategy.prepare(bars.loc[:t], ctx)
        full_decision = strategy.on_bar(t, full_prepared.loc[t], pos, ctx)
        trunc_decision = strategy.on_bar(t, truncated_prepared.loc[t], pos, ctx)
        assert full_decision.target == trunc_decision.target, (
            f"truncation: решение в t={t} отличается на полных ({full_decision.target}) "
            f"и усечённых ({trunc_decision.target}) данных — возможна утечка будущего"
        )


def assert_future_poison_safe(strategy, bars: pd.DataFrame, ctx, n_checks: int = 50, seed: int = 1) -> None:
    """Замена всех данных после t на мусор (NaN) не должна менять решение в t."""
    full_prepared = strategy.prepare(bars, ctx)
    tf = strategy.timeframes[0]
    warmup = strategy.required_history(tf)
    candidates = list(full_prepared.index[warmup:-1]) if len(full_prepared.index) > warmup else []
    if not candidates:
        return
    rng = random.Random(seed)
    checks = rng.sample(candidates, min(n_checks, len(candidates)))
    pos = PositionState(symbol="FUTURE_POISON_TEST")
    price_cols = [c for c in ("open", "high", "low", "close", "volume", "quote_volume") if c in bars.columns]

    for t in checks:
        poisoned = bars.copy()
        poisoned.loc[poisoned.index > t, price_cols] = np.nan
        poisoned_prepared = strategy.prepare(poisoned, ctx)
        full_decision = strategy.on_bar(t, full_prepared.loc[t], pos, ctx)
        poisoned_decision = strategy.on_bar(t, poisoned_prepared.loc[t], pos, ctx)
        assert full_decision.target == poisoned_decision.target, (
            f"future-poison: решение в t={t} изменилось после порчи будущих данных "
            f"({full_decision.target} -> {poisoned_decision.target}) — стратегия смотрит вперёд"
        )
