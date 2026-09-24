"""Правило плато (04_OPTIMIZATION.md §4.1) и отбор монет (§5)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


def _point_key(point: dict) -> tuple:
    return tuple(sorted(point.items(), key=lambda kv: kv[0]))


@dataclass(frozen=True)
class PlateauResult:
    params: dict
    score: float  # median(Sharpe(p), Sharpe(соседей(p)))
    raw_sharpe: float  # Sharpe(p) самой точки (не median) - для отчёта


def plateau_select(
    points: list[dict],
    sharpes: list[float],
    ordinal_grids: dict[str, list],
    categorical_params: tuple[str, ...] = (),
) -> PlateauResult:
    """Выбирает не точку с максимальным Шарпом, а точку с устойчиво хорошими соседями
    (04_OPTIMIZATION.md §4.1). `points[i]`/`sharpes[i]` — параллельные списки: испытанная
    комбинация параметров и её Sharpe на train. `ordinal_grids` — полный упорядоченный
    список значений сетки для каждого "степного" параметра (по нему ищутся соседи —
    точки, отличающиеся ровно на один шаг по ОДНОМУ параметру). `categorical_params` —
    параметры, для которых соседство не считается (напр. режим LS/LF): плато ищется
    отдельно внутри каждого сочетания категориальных значений, затем берётся лучшее
    из этих локальных плато.
    """
    if not points:
        raise ValueError("plateau_select: пустая сетка результатов")

    index_by_key = {_point_key(p): i for i, p in enumerate(points)}

    def neighbor_indices(p: dict) -> list[int]:
        found = []
        for name, grid in ordinal_grids.items():
            if name not in p or p[name] not in grid:
                continue
            i = grid.index(p[name])
            for j in (i - 1, i + 1):
                if 0 <= j < len(grid):
                    neighbor = dict(p)
                    neighbor[name] = grid[j]
                    k = _point_key(neighbor)
                    if k in index_by_key:
                        found.append(index_by_key[k])
        return found

    def cat_key(p: dict) -> tuple:
        return tuple(p.get(c) for c in categorical_params)

    partitions: dict[tuple, list[int]] = {}
    for i, p in enumerate(points):
        partitions.setdefault(cat_key(p), []).append(i)

    best: PlateauResult | None = None
    for part_indices in partitions.values():
        part_set = set(part_indices)
        for i in part_indices:
            neighbors = [j for j in neighbor_indices(points[i]) if j in part_set]
            score = median([sharpes[i]] + [sharpes[j] for j in neighbors])
            if best is None or score > best.score:
                best = PlateauResult(params=points[i], score=score, raw_sharpe=sharpes[i])
    return best


@dataclass(frozen=True)
class CoinSelectionResult:
    symbols: list[str]
    applied: bool  # False, если отбор не применился (недостаточно монет прошло правило)


def select_coins(
    pair_metrics: dict[str, dict],
    coin_min_sharpe_train: float,
    coin_min_trades_train: int,
    min_coins_after_selection: int,
    full_pool: list[str],
) -> CoinSelectionResult:
    """04_OPTIMIZATION.md §5. `pair_metrics[symbol] = {"sharpe": ..., "trades": ...}` —
    метрики монеты в режиме "пара" на train-окне с уже выбранными параметрами V1.
    Если после правила осталось меньше `min_coins_after_selection` монет — отбор не
    применяется (V2 = V1, весь пул), это возвращается как `applied=False`."""
    kept = [
        s for s in full_pool
        if s in pair_metrics
        and pair_metrics[s]["sharpe"] == pair_metrics[s]["sharpe"]  # не NaN
        and pair_metrics[s]["sharpe"] > coin_min_sharpe_train
        and pair_metrics[s]["trades"] >= coin_min_trades_train
    ]
    if len(kept) < min_coins_after_selection:
        return CoinSelectionResult(symbols=list(full_pool), applied=False)
    return CoinSelectionResult(symbols=kept, applied=True)
