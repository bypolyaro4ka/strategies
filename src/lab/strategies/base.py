"""Интерфейс стратегии — одинаковый для всех 13 стратегий (02_ENGINE_SPEC.md, раздел 3).

`@dataclass` (стандартная библиотека) — это способ описать класс, у которого основная
работа — хранить данные с типами, без ручного написания `__init__`. `ABC`/`@abstractmethod`
(тоже стандартная библиотека) — способ сказать "это контракт, а не готовый класс": у любой
конкретной стратегии (S01, S02, ...) обязаны быть свои `prepare()` и `on_bar()`, иначе
Python не даст её создать.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Decision:
    """Что стратегия хочет сделать на этом баре.

    target — целевая позиция в долях слота, от -1 (полный шорт) до +1 (полный лонг).
    Остальное — необязательные инструкции по сопровождению уже открытой/новой позиции.
    """
    target: float
    stop_price: float | None = None
    take_price: float | None = None
    time_stop_bars: int | None = None
    tag: str = ""


@dataclass
class PositionState:
    """Текущее состояние позиции по одной монете. Движок передаёт его стратегии на
    каждом баре (только для чтения по смыслу) и сам обновляет после исполнения ордеров."""
    symbol: str
    target: float = 0.0
    qty: float = 0.0            # знаковое количество базового актива
    notional: float = 0.0       # знаковый notional на момент последнего изменения
    avg_entry_price: float = 0.0
    stop_price: float | None = None
    take_price: float | None = None
    time_stop_bars: int | None = None
    bars_in_trade: int = 0

    @property
    def is_open(self) -> bool:
        return self.qty != 0.0

    @property
    def is_long(self) -> bool:
        return self.qty > 0.0


@dataclass
class Context:
    """То, что стратегия может использовать помимо своих же данных: параметры, бары BTC
    (для надстройки O1), фандинг (загрузчик уже гарантирует funding_time <= t) и конфиг
    протокола. Больше ничего — само по себе является защитой от подглядывания: если
    страгегии это не передано, она физически не может этим воспользоваться."""
    params: dict
    protocol: object  # ProtocolConfig; object, чтобы не тянуть цикл импортов
    btc_bars: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None


class BaseStrategy(ABC):
    id: str
    name: str
    version: str
    author: str = "Алексей"  # для лидерборда (06_LEADERBOARD.md) - чья стратегия
    default_params: dict = {}
    param_grid: dict = {}
    timeframes: list[str] = []
    direction: str = "long_short"  # "long_only" | "long_short"

    def __init__(self, params: dict | None = None):
        self.params = {**self.default_params, **(params or {})}

    def required_history(self, tf: str) -> int:
        """Сколько баров прогрева нужно. По умолчанию 0 — переопределяется в стратегии
        (03_STRATEGIES.md, раздел 0: не меньше 3×самое длинное окно для EMA/Wilder,
        самое длинное окно для SMA/max/min)."""
        return 0

    @abstractmethod
    def prepare(self, bars: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        """Считает индикаторы векторно. ТОЛЬКО причинные операции: rolling/ewm/shift
        вперёд во времени. Никаких center=True, bfill, shift(-1), нормализаций по всему
        периоду (02_ENGINE_SPEC.md, раздел 3.1) — это и есть то, что проверяет
        truncation-тест."""

    @abstractmethod
    def on_bar(self, t: pd.Timestamp, row: pd.Series, pos: PositionState, ctx: Context) -> Decision:
        """Вызывается по закрытию каждого полного сигнального бара. row — строка из
        prepare() на момент t. pos — текущая позиция (только читать)."""


class PortfolioStrategy(ABC):
    """Для стратегий, которым нужны все монеты сразу (S13 — кросс-секционный моментум)."""
    id: str
    name: str
    version: str

    def __init__(self, params: dict | None = None):
        self.params = {**getattr(self, "default_params", {}), **(params or {})}

    @abstractmethod
    def prepare(self, bars_by_symbol: dict[str, pd.DataFrame], ctx: Context) -> dict[str, pd.DataFrame]:
        ...

    @abstractmethod
    def on_bar(
        self, t: pd.Timestamp, rows: dict[str, pd.Series], positions: dict[str, PositionState], ctx: Context
    ) -> dict[str, Decision]:
        ...


class Overlay(ABC):
    """Надстройка — обёртка над стратегией, которая модифицирует Decision.target
    (03_STRATEGIES.md, раздел O1/O2). `row` — строка prepare() базовой стратегии на
    момент t (та же, что видит on_bar()) - нужна надстройкам вроде O2 (таргетирование
    волатильности), которым нужна СОБСТВЕННАЯ история цены монеты, не только ctx."""

    @abstractmethod
    def apply(self, t: pd.Timestamp, symbol: str, row: pd.Series, decision: Decision, ctx: Context) -> Decision:
        ...


class WithOverlays:
    """strategy = WithOverlays(S02(), [BtcRegimeFilter(), VolTarget()]).
    Не наследует BaseStrategy напрямую (не стратегия в чистом виде), но даёт тот же
    интерфейс prepare()/on_bar(), которым пользуется движок — с точки зрения движка
    разницы нет."""

    def __init__(self, strategy: BaseStrategy, overlays: list[Overlay]):
        self.strategy = strategy
        self.overlays = overlays
        self.id = strategy.id + "".join(f"+{type(o).__name__}" for o in overlays)
        self.version = strategy.version
        self.timeframes = strategy.timeframes

    def required_history(self, tf: str) -> int:
        return max([self.strategy.required_history(tf)] + [o.required_history(tf) for o in self.overlays
                                                             if hasattr(o, "required_history")])

    def prepare(self, bars: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        out = self.strategy.prepare(bars, ctx)
        for overlay in self.overlays:
            if hasattr(overlay, "prepare"):
                out = overlay.prepare(out, ctx)
        return out

    def on_bar(self, t: pd.Timestamp, row: pd.Series, pos: PositionState, ctx: Context) -> Decision:
        decision = self.strategy.on_bar(t, row, pos, ctx)
        for overlay in self.overlays:
            decision = overlay.apply(t, pos.symbol, row, decision, ctx)
        return decision
