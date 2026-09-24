# 02. Спецификация движка

Как устроены данные, интерфейс стратегии, цикл бэктеста, логирование и тесты.
Правила исполнения — в `01_PROTOCOL.md`, раздел 5; здесь — как их реализовать.

## 1. Стек

- Python 3.11+, pandas, numpy, pyarrow (parquet), pyyaml, matplotlib, pytest.
- Опционально для скорости: numba (цикл движка), joblib (параллельные прогоны оптимизации).
- Локально достаточно MacBook; оптимизацию (Этап 7) можно гонять на Linux-сервере — GPU не нужен.

## 2. Слой данных

### 2.1 Файлы

```
data/processed/{SYMBOL}_1h.parquet       # базовые бары
data/processed/{SYMBOL}_4h.parquet       # собранные из 1h
data/processed/{SYMBOL}_12h.parquet
data/processed/{SYMBOL}_1d.parquet
data/processed/{SYMBOL}_funding.parquet  # история фандинга
```

### 2.2 Схемы

Бары (индекс `open_time`, UTC, tz-aware):

| Колонка | Тип | Описание |
|---|---|---|
| open, high, low, close | float64 | цены |
| volume | float64 | объём в базовой монете |
| quote_volume | float64 | объём в USDT |
| trades | int64 | число сделок |
| complete | bool | все 1h-бары внутри есть (для 1h всегда True) |

Фандинг (индекс `funding_time`, UTC): `funding_rate: float64`.

### 2.3 Сборка старших ТФ из 1h

- Границы: 4h — 00/04/08/12/16/20 UTC; 12h — 00/12 UTC; 1d — 00:00 UTC.
  Совпадает с биржевыми свечами Binance.
- `open = first`, `high = max`, `low = min`, `close = last`,
  `volume/quote_volume/trades = sum`.
- `complete = (число 1h-баров == ожидаемому)`.
- Тест: собранные 1d/4h сверить с биржевыми 1d/4h на случайной выборке дат —
  расхождение цен должно быть 0 (допуск на float).

### 2.4 Загрузчик с защитой holdout

Все стратегии и отчёты получают данные **только** через `loader.load(...)`.

```python
# src/lab/data/loader.py — эскиз
class HoldoutLockedError(RuntimeError):
    pass

def load_bars(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    cfg = load_protocol()                      # читает config/protocol.yaml
    holdout_start = pd.Timestamp(cfg["periods"]["holdout_start"], tz="UTC")
    if pd.Timestamp(end, tz="UTC") >= holdout_start and not holdout_unlocked(cfg):
        raise HoldoutLockedError(
            "Запрошены данные holdout. Разблокировка — только на Этапе 8 (см. 01_PROTOCOL, раздел 10)."
        )
    ...

def holdout_unlocked(cfg) -> bool:
    # Все три условия одновременно:
    # 1) cfg["frozen"] is True
    # 2) переменная окружения HOLDOUT_UNLOCK == "1"
    # 3) текущий коммит помечен git tag holdout-freeze-v1
    ...
```

Прогрев: для прогона с `start = S` загрузчик отдаёт данные с `S − warmup`, где warmup
берётся из `strategy.required_history()`. Метрики считаются только с `S`.

## 3. Интерфейс стратегии

### 3.1 Решение стратегии

```python
@dataclass
class Decision:
    target: float                  # целевая позиция в долях слота, [-1, +1]
    stop_price: float | None = None   # стоп для текущей/новой позиции
    take_price: float | None = None   # тейк (если есть)
    time_stop_bars: int | None = None # выход через N сигнальных баров
    tag: str = ""                  # причина: "entry_long", "exit_signal", ...
```

### 3.2 Базовые классы

```python
class BaseStrategy(ABC):
    id: str                 # "S01"
    name: str               # "donchian_ensemble"
    version: str            # "1.0.0" — меняется при любом изменении логики
    default_params: dict
    param_grid: dict        # сетка для walk-forward (03_STRATEGIES)
    timeframes: list[str]   # допустимые сигнальные ТФ, первый — основной
    direction: str          # "long_only" | "long_short"

    def required_history(self, tf: str) -> int:
        """Сколько баров прогрева нужно."""

    @abstractmethod
    def prepare(self, bars: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        """Считает индикаторы векторно. ТОЛЬКО причинные операции:
        rolling/ewm/shift вперёд во времени. Никаких center=True, bfill, shift(-1),
        нормализаций по всему периоду."""

    @abstractmethod
    def on_bar(self, t: pd.Timestamp, row: pd.Series, pos: PositionState, ctx: Context) -> Decision:
        """Вызывается по закрытию каждого полного сигнального бара.
        row — строка из prepare() на момент t. pos — текущая позиция."""


class PortfolioStrategy(ABC):
    """Для стратегий, которым нужны все монеты сразу (S13)."""
    def on_bar(self, t, rows: dict[str, pd.Series], positions: dict[str, PositionState],
               ctx: Context) -> dict[str, Decision]: ...
```

`Context` даёт доступ к параметрам, данным BTC (для надстройки O1), фандингу
(только с `funding_time <= t_close`) и конфигу. Больше ничего.

### 3.3 Надстройки (overlays)

Надстройка — обёртка над стратегией, которая модифицирует `Decision.target`:

```python
class Overlay(ABC):
    def apply(self, t, symbol, decision: Decision, ctx) -> Decision: ...

strategy = WithOverlays(S02(), [BtcRegimeFilter(), VolTarget(sigma=0.5)])
```

Вариант «стратегия + надстройка» — отдельное испытание со своим ID, например `S02+O1`.

## 4. Цикл движка

```
для каждого 1h-бара h (в порядке времени):
    1. если у позиции есть стоп/тейк — проверить по high/low бара h (стоп приоритетнее),
       при срабатывании исполнить (01_PROTOCOL 5.4)
    2. если h — момент фандинга: начислить/списать фандинг
    3. если на открытии h есть отложенный ордер (от сигнала прошлого закрытия) —
       исполнить по open(h) с проскальзыванием и комиссией
    4. отметить equity по close(h)
    5. если h закрывает полный сигнальный бар стратегии:
         decision = strategy.on_bar(...)
         decision = overlays.apply(...)
         если |decision.target - pos.target| >= min_target_change:
             поставить отложенный ордер на open следующего 1h-бара
         обновить стоп/тейк/тайм-стоп
```

Портфельный режим: тот же цикл, но решения по всем монетам собираются на одном закрытии,
затем проверяется лимит плеча 1x, затем ставятся ордера.

### 4.1 Формулы

```
fill_buy  = open * (1 + slippage)
fill_sell = open * (1 - slippage)
fee       = abs(delta_notional) * fee_taker
funding   = -position_qty * mark_price * funding_rate   # для лонга при rate>0 — минус
notional  = target * slot_fraction * equity_at_decision
```

## 5. Выходные артефакты прогона

| Файл | Содержимое |
|---|---|
| `trades.csv` | symbol, side, entry_time, entry_price, exit_time, exit_price, qty, gross_pnl, fees, slippage_cost, funding, net_pnl, exit_reason |
| `orders.csv` | каждое изменение позиции, включая частичные |
| `equity.parquet` | почасовая equity, cash, exposure по монетам |
| `metrics.json` | метрики из `01_PROTOCOL`, раздел 8 |
| `run_meta.json` | strategy id/version, params, tf, symbols, period, fold, variant, git commit, hash данных |

## 6. Реестр испытаний (`reports/registry.csv`)

Без MLflow: не нужен сервер, БД или UI — только неизменяемый журнал прогонов для подсчёта
числа испытаний (это то, что защищает DSR и лидерборд от подгонки задним числом, см.
`01_PROTOCOL.md` §9), плюс путь к артефактам конкретного прогона.

`src/lab/registry.py`, функция `log_run(...)`: **только дописывает** одну строку в
`reports/registry.csv` (создаёт файл с заголовком при первом запуске). Никогда не читает файл
целиком, чтобы переписать, не сортирует и не дедуплицирует существующие строки — это и есть
гарантия «испытания нельзя удалить или скрыть» (`CLAUDE.md`, правило 3) на уровне кода,
а не только на уровне договорённости.

Колонки:

- `run_id, timestamp_utc` — уникальный идентификатор и момент запуска;
- `strategy_id, strategy_version, variant` (V0/V1/V2/community), `mode` (pair/portfolio),
  `tf, symbols` (через `;`), `period_start, period_end, fold`;
- `stage` (dev/wf/holdout), `debug` (true/false), `git_commit, data_hash, author`
  (для стратегий друзей);
- `params_json` — параметры стратегии одной JSON-строкой;
- все метрики из `metrics.json` отдельными колонками (`sharpe`, `dsr`, `maxdd`, `trades`, ...);
- `artifacts_path` — куда сложены `trades.csv`, `orders.csv`, `equity.parquet`, график
  (`reports/<stage>/<run_id>/`, см. раздел 5).

Функция `count_trials(strategy_id, stage)` в `metrics/dsr.py` — просто
`registry[(registry.strategy_id == strategy_id) & (registry.stage == stage) & ~registry.debug]`,
без похода во внешнюю систему.

## 7. Тесты (обязательны до первого реального прогона)

| Тест | Что проверяет | Тип |
|---|---|---|
| **Truncation (анти-подглядывание)** | Для каждой стратегии: решения, посчитанные на данных до t, совпадают с решениями на полных данных в точке t (≥ 200 случайных t) | интеграционный, обязателен для каждой стратегии, включая стратегии друзей |
| Future-poison | Заменить все данные после t на мусор (NaN/случайные цены) — решения до t не меняются | интеграционный |
| Resample | Собранные 4h/12h/1d совпадают с биржевыми | данные |
| Holdout guard | `load_bars(..., end >= holdout_start)` без разблокировки бросает `HoldoutLockedError` | юнит |
| Costs toy | Ручной пример: 1 сделка, известные цены → точные fee/slippage/net_pnl | юнит |
| Funding sign | Лонг при rate > 0 платит, шорт получает; позиция, закрытая до отметки, не платит | юнит |
| Stop priority | Бар задевает и стоп, и тейк → исполнен стоп | юнит |
| Gap через стоп | Open бара хуже стопа → исполнение по open | юнит |
| Leverage cap | Сумма \|notional\| не превышает equity; урезание логируется | юнит |
| Min change | Изменение target < порога не создаёт ордер | юнит |
| Determinism | Два прогона с одним сидом дают идентичные результаты | интеграционный |
| Benchmark sanity | Buy&Hold без издержек совпадает с ценой; Flat = 0 | интеграционный |
| Slot contention | Сигналов на вход больше, чем свободных слотов → открытые позиции сохраняются, новые заполняют слоты по алфавиту символа, остальные пропускаются (01_PROTOCOL 5.5) | юнит |
| Date-leak grep | Ни в одном файле `src/lab/strategies/**/*.py` (включая `community/`) нет литералов дат `2026-*` и `holdout` в захардкоженном виде | статический, обязателен для каждой стратегии, включая стратегии друзей |
| Registry append-only | `log_run()` только добавляет строку в `registry.csv`; существующие строки после вызова побайтово не изменились | юнит |

Покрытие: 100% для `engine/costs.py`, `engine/execution.py`, `data/loader.py`.
Для стратегий — truncation + future-poison + date-leak grep + 2–3 ручных сценария на
синтетических данных (например, для Donchian: ряд, который пробивает максимум на известном баре).

## 8. Производительность

- Индикаторы — векторно в `prepare()`, цикл — только по событиям.
- С 2022 года по сентябрь 2026 ≈ 41 тыс. 1h-баров на монету, ≈ 410 тыс. на портфель из 10 монет.
  На один прогон чистый Python справится, но сетка оптимизации (Этап 7) — это тысячи прогонов,
  поэтому цикл стоит ускорить numba или распараллелить прогоны через joblib.
- Кэшировать `prepare()` по ключу (стратегия, версия, параметры, символ, ТФ) — **на диске**
  (например, `.cache/prepare/<hash>.parquet`), не только в памяти процесса: повторные локальные
  прогоны (отладка отчётов, перезапуск после сбоя, повторный просмотр walk-forward) не должны
  пересчитывать одно и то же на 410 тыс. баров. `.cache/` — в `.gitignore`, как `data/`.
