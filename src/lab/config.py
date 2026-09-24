"""Загрузка и валидация config/protocol.yaml.

Это единственная точка, откуда весь остальной код должен узнавать константы
протокола (комиссии, даты периодов, размер позиции и т. п.) — см. CLAUDE.md,
правило 4 ("константы — только из config/protocol.yaml").
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Путь к файлу по умолчанию: .../strategies/config/protocol.yaml
# (Path(__file__) — этот файл, .parents[2] — три уровня вверх от src/lab/config.py до корня проекта)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "protocol.yaml"


class ProtocolConfigError(RuntimeError):
    """Конфиг отсутствует, повреждён или не проходит валидацию."""


@dataclass(frozen=True)
class ExchangeConfig:
    venue: str
    quote: str
    order_type: str
    fee_taker: float
    fee_maker: float
    slippage: float


@dataclass(frozen=True)
class AccountConfig:
    initial_equity: float
    slot_fraction: float
    leverage: float
    max_slots_portfolio: int
    min_target_change: float
    resize_on_equity_change: bool


@dataclass(frozen=True)
class PeriodsConfig:
    timezone: str
    data_start: str
    dev_start: str
    dev_end: str
    holdout_start: str
    holdout_end: str | None


@dataclass(frozen=True)
class WalkForwardFold:
    id: str
    train: tuple[str, str]
    test: tuple[str, str]


@dataclass(frozen=True)
class SelectionConfig:
    objective: str
    plateau_rule: str
    min_trades_train_portfolio: int
    coin_min_trades_train: int
    coin_min_sharpe_train: float
    min_coins_after_selection: int
    min_wf_sharpe_gain_vs_v0: float
    min_folds_better: int


@dataclass(frozen=True)
class MetricsConfig:
    annualization_days: int
    risk_free: float
    random_baseline_seeds: int


@dataclass(frozen=True)
class BtcFilterConfig:
    symbol: str
    sma_days: int


@dataclass(frozen=True)
class RegistryConfig:
    backend: str
    experiment_prefix: str


@dataclass(frozen=True)
class ProtocolConfig:
    """Типизированное представление всего protocol.yaml.

    frozen=True на всех дата-классах здесь — это защита от случайности:
    если где-то в коде попытаться сделать `cfg.account.leverage = 5`,
    Python бросит исключение вместо тихой порчи общего конфига.
    """

    protocol_version: int
    frozen: bool
    exchange: ExchangeConfig
    account: AccountConfig
    periods: PeriodsConfig
    timeframe_base: str
    timeframes_derived: tuple[str, ...]
    execution_timeframe: str
    walk_forward_scheme: str
    folds: tuple[WalkForwardFold, ...]
    selection: SelectionConfig
    metrics: MetricsConfig
    btc_filter: BtcFilterConfig
    registry: RegistryConfig
    raw: dict  # необработанный dict — запасной выход для полей, которые сюда ещё не завели


def _require(d: dict, key: str, path: str):
    if key not in d:
        raise ProtocolConfigError(f"В protocol.yaml отсутствует обязательный ключ: {path}.{key}")
    return d[key]


def _parse(raw: dict) -> ProtocolConfig:
    exch = _require(raw, "exchange", "$")
    acc = _require(raw, "account", "$")
    per = _require(raw, "periods", "$")
    tf = _require(raw, "timeframes", "$")
    wf = _require(raw, "walk_forward", "$")
    sel = _require(raw, "selection", "$")
    met = _require(raw, "metrics", "$")
    btc = _require(raw, "btc_filter", "$")
    reg = _require(raw, "registry", "$")

    folds = tuple(
        WalkForwardFold(id=f["id"], train=tuple(f["train"]), test=tuple(f["test"]))
        for f in _require(wf, "folds", "walk_forward")
    )

    cfg = ProtocolConfig(
        protocol_version=_require(raw, "protocol_version", "$"),
        frozen=bool(_require(raw, "frozen", "$")),
        exchange=ExchangeConfig(
            venue=exch["venue"],
            quote=exch["quote"],
            order_type=exch["order_type"],
            fee_taker=float(exch["fee_taker"]),
            fee_maker=float(exch["fee_maker"]),
            slippage=float(exch["slippage"]),
        ),
        account=AccountConfig(
            initial_equity=float(acc["initial_equity"]),
            slot_fraction=float(acc["slot_fraction"]),
            leverage=float(acc["leverage"]),
            max_slots_portfolio=int(acc["max_slots_portfolio"]),
            min_target_change=float(acc["min_target_change"]),
            resize_on_equity_change=bool(acc["resize_on_equity_change"]),
        ),
        periods=PeriodsConfig(
            timezone=per["timezone"],
            data_start=per["data_start"],
            dev_start=per["dev_start"],
            dev_end=per["dev_end"],
            holdout_start=per["holdout_start"],
            holdout_end=per.get("holdout_end"),
        ),
        timeframe_base=tf["base"],
        timeframes_derived=tuple(tf["derived"]),
        execution_timeframe=tf["execution"],
        walk_forward_scheme=wf["scheme"],
        folds=folds,
        selection=SelectionConfig(
            objective=sel["objective"],
            plateau_rule=sel["plateau_rule"],
            min_trades_train_portfolio=int(sel["min_trades_train_portfolio"]),
            coin_min_trades_train=int(sel["coin_min_trades_train"]),
            coin_min_sharpe_train=float(sel["coin_min_sharpe_train"]),
            min_coins_after_selection=int(sel["min_coins_after_selection"]),
            min_wf_sharpe_gain_vs_v0=float(sel["primary_rule"]["min_wf_sharpe_gain_vs_v0"]),
            min_folds_better=int(sel["primary_rule"]["min_folds_better"]),
        ),
        metrics=MetricsConfig(
            annualization_days=int(met["annualization_days"]),
            risk_free=float(met["risk_free"]),
            random_baseline_seeds=int(met["random_baseline_seeds"]),
        ),
        btc_filter=BtcFilterConfig(symbol=btc["symbol"], sma_days=int(btc["sma_days"])),
        registry=RegistryConfig(backend=reg["backend"], experiment_prefix=reg["experiment_prefix"]),
        raw=raw,
    )
    _sanity_check(cfg)
    return cfg


def _sanity_check(cfg: ProtocolConfig) -> None:
    """Простые проверки на здравый смысл, чтобы опечатка в YAML не молчала."""
    if not (0 <= cfg.exchange.fee_taker <= 0.01):
        raise ProtocolConfigError(f"fee_taker вне разумного диапазона: {cfg.exchange.fee_taker}")
    if not (0 <= cfg.exchange.slippage <= 0.01):
        raise ProtocolConfigError(f"slippage вне разумного диапазона: {cfg.exchange.slippage}")
    if not (0 < cfg.account.slot_fraction <= 1):
        raise ProtocolConfigError(f"slot_fraction вне (0, 1]: {cfg.account.slot_fraction}")
    if cfg.account.leverage != 1:
        raise ProtocolConfigError(
            f"leverage={cfg.account.leverage}: протокол (01_PROTOCOL.md, 5.5) рассчитан на плечо 1x, "
            "остальной движок это предполагает"
        )
    if len(cfg.folds) == 0:
        raise ProtocolConfigError("walk_forward.folds пуст")


@lru_cache(maxsize=1)
def load_protocol(path: str | Path | None = None) -> ProtocolConfig:
    """Читает и валидирует config/protocol.yaml. Результат кэшируется (lru_cache) —
    в рамках одного процесса файл читается и парсится один раз, дальше отдаётся
    тот же объект. Если нужно перечитать (например, после смены frozen в тестах),
    вызвать load_protocol.cache_clear().
    """
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise ProtocolConfigError(f"Файл конфига не найден: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ProtocolConfigError(f"{p} не распарсился в словарь верхнего уровня")
    return _parse(raw)
