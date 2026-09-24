"""Тесты для optimize/selection.py — Этап 7 (правило плато, отбор монет)."""

from lab.optimize.selection import plateau_select, select_coins


def test_plateau_prefers_stable_neighbor_over_lone_spike():
    # n=30 - изолированный спайк (лучший ОДИНОЧНЫЙ Sharpe в сетке), но оба соседа низкие;
    # n=60 - середина плато (50/60/70 все стабильно высокие) - должна победить, хотя её
    # собственный Sharpe ниже, чем у спайка
    grid = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    sharpes_by_n = {10: 0.3, 20: 0.2, 30: 3.0, 40: 0.15, 50: 0.85, 60: 0.9, 70: 0.88, 80: 0.2, 90: 0.1}
    points = [{"n": n} for n in grid]
    sharpes = [sharpes_by_n[n] for n in grid]
    result = plateau_select(points, sharpes, ordinal_grids={"n": grid})
    assert result.params == {"n": 60}


def test_plateau_single_point_no_neighbors():
    points = [{"n": 20}]
    sharpes = [1.5]
    result = plateau_select(points, sharpes, ordinal_grids={"n": [20]})
    assert result.params == {"n": 20}
    assert result.score == 1.5


def test_plateau_categorical_partitioned_separately():
    # LS - настоящее плато (n=30/40 стабильно высокие), LF - равномерно посредственная
    # (без спайков) - соседи никогда не берутся из другой категории, LS должна победить
    grid = [10, 20, 30, 40, 50]
    ls_sharpes = {10: 0.3, 20: 0.7, 30: 0.9, 40: 0.85, 50: 0.4}
    lf_sharpes = {10: 0.2, 20: 0.3, 30: 0.25, 40: 0.3, 50: 0.2}
    points = [{"n": n, "mode": "LS"} for n in grid] + [{"n": n, "mode": "LF"} for n in grid]
    sharpes = [ls_sharpes[n] for n in grid] + [lf_sharpes[n] for n in grid]
    result = plateau_select(
        points, sharpes, ordinal_grids={"n": grid}, categorical_params=("mode",),
    )
    assert result.params == {"n": 30, "mode": "LS"}
    assert result.score == 0.85


def test_plateau_two_ordinal_dims_neighbors_from_both_params():
    points = [
        {"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 2},
    ]
    sharpes = [0.5, 0.6, 0.55, 1.0, 0.9]
    result = plateau_select(points, sharpes, ordinal_grids={"a": [1, 2, 3], "b": [1, 2]})
    # (a=2,b=2): соседи (a=1,b=2)=0.6, (a=3,b=2)=0.9, (a=2,b=1)=0.55 -> median([1.0,0.9,0.6,0.55])=0.75
    # (a=3,b=2): единственный сосед (a=2,b=2)=1.0 -> median([0.9,1.0])=0.95 - выше, хоть и на
    # одном соседе (у него нет соседа со стороны b=1, такая точка не тестировалась)
    assert result.params == {"a": 3, "b": 2}
    assert result.score == 0.95


def test_select_coins_keeps_only_passing_rule():
    pair_metrics = {
        "AAA": {"sharpe": 0.5, "trades": 10},
        "BBB": {"sharpe": -0.2, "trades": 20},   # sharpe <= 0 - выбывает
        "CCC": {"sharpe": 0.8, "trades": 3},     # trades < 5 - выбывает
        "DDD": {"sharpe": 1.2, "trades": 8},
    }
    result = select_coins(
        pair_metrics, coin_min_sharpe_train=0.0, coin_min_trades_train=5,
        min_coins_after_selection=2, full_pool=["AAA", "BBB", "CCC", "DDD"],
    )
    assert result.applied is True
    assert set(result.symbols) == {"AAA", "DDD"}


def test_select_coins_falls_back_to_full_pool_when_too_few():
    pair_metrics = {
        "AAA": {"sharpe": 0.5, "trades": 10},
        "BBB": {"sharpe": -0.2, "trades": 20},
        "CCC": {"sharpe": -0.8, "trades": 3},
    }
    result = select_coins(
        pair_metrics, coin_min_sharpe_train=0.0, coin_min_trades_train=5,
        min_coins_after_selection=3, full_pool=["AAA", "BBB", "CCC"],
    )
    assert result.applied is False
    assert result.symbols == ["AAA", "BBB", "CCC"]


def test_select_coins_ignores_nan_sharpe():
    pair_metrics = {"AAA": {"sharpe": float("nan"), "trades": 10}, "BBB": {"sharpe": 0.5, "trades": 10}}
    result = select_coins(
        pair_metrics, coin_min_sharpe_train=0.0, coin_min_trades_train=5,
        min_coins_after_selection=1, full_pool=["AAA", "BBB"],
    )
    assert result.applied is True
    assert result.symbols == ["BBB"]
