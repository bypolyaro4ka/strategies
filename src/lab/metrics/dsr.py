"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — 01_PROTOCOL.md, раздел 9.

DSR отвечает на вопрос "какова вероятность, что наблюдаемый Шарп на самом деле выше нуля,
если учесть, что мы перебрали N вариантов, и любой из них мог оказаться лучшим просто
по случайности". Чем больше N (испытаний), тем выше "по умолчанию" ожидаемый максимум
Шарпа среди случайных стратегий — DSR вычитает эту случайную планку перед тем, как
оценивать значимость.

ВАЖНО: формула нетривиальная и чувствительна к деталям (аннуализированный или "сырой"
Шарп, excess или полный куртозис). Перед тем как полагаться на DSR в итоговых выводах
Этапа 8, стоит свериться с оригинальной статьёй (07_SOURCES.md) на паре примеров вручную.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def _period_sharpe_std_error(sr: float, n_obs: int, skew: float, kurt: float) -> float:
    """Стандартная ошибка Шарпа за период наблюдения (не аннуализированного) —
    формула Bailey & López de Prado, учитывает асимметрию и "толстые хвосты" доходностей.
    kurt здесь — "сырой" (не excess) куртозис, для нормального распределения kurt=3."""
    if n_obs <= 1:
        return float("nan")
    variance = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    return float(np.sqrt(max(variance, 0.0) / (n_obs - 1)))


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """Ожидаемый максимум Шарпа среди n_trials независимых испытаний с истинным Шарпом 0 —
    это и есть "случайная планка", с которой сравнивается наблюдаемый результат."""
    n_trials = max(n_trials, 2)  # при n=1 формула не определена (Z^-1(0) = -inf)
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> float:
    """returns — доходности ЗА ПЕРИОД наблюдения (например, дневные) — функция сама
    считает Шарп за период и его дисперсию, аннуализация тут не нужна (что аннуализированный,
    что нет Шарп даёт один и тот же DSR — множитель √N сокращается что в числителе,
    что в знаменателе).

    Возвращает вероятность [0, 1], что истинный Шарп положителен с учётом множественного
    тестирования (n_trials испытаний)."""
    n_obs = len(returns)
    if n_obs < 3:
        return float("nan")
    std = returns.std(ddof=0)
    if std < 1e-12:
        # "== 0" здесь не работает: для почти-константных рядов std получается не ровно 0,
        # а исчезающе малым числом (~1e-18) из-за погрешности плавающей точки, и Шарп
        # улетает в аномально большое число вместо честного nan
        return float("nan")
    sr = float(returns.mean() / std)
    skew = float(returns.skew())
    kurt = float(returns.kurtosis()) + 3  # pandas.kurtosis() - excess; формуле нужен "сырой"

    sr_std = _period_sharpe_std_error(sr, n_obs, skew, kurt)
    if sr_std == 0 or np.isnan(sr_std):
        return float("nan")
    sr0 = expected_max_sharpe(sr_std, n_trials)
    return float(norm.cdf((sr - sr0) / sr_std))
