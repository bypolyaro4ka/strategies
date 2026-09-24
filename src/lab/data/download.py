"""Скачивание 1h-баров и истории фандинга с data.binance.vision (Binance USDT-M futures).

Идемпотентно: то, что уже лежит в data/raw/, не перекачивается — повторный запуск
докачивает только недостающие месяцы/дни. Источник — архивы data.binance.vision
(01_PROTOCOL.md, раздел 4); REST API (fapi.binance.com) — запасной вариант для
последних дней текущего месяца, если для них ещё нет архива.

ВАЖНО: этот модуль честно качает данные **до вчерашнего дня включительно**, то есть
и часть 2026 года тоже (она нужна позже, см. CLAUDE.md, holdout). Сам скрипт её не
анализирует и не печатает — только сохраняет на диск. Барьер, который не даёт коду
стратегий её "увидеть", — в loader.py и holdout_lock.py, а не здесь.
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

VISION_BASE = "https://data.binance.vision/data/futures/um"
REST_BASE = "https://fapi.binance.com"

# Порядок колонок в архивах Binance (data.binance.vision/.../klines/...).
KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UNIVERSE_PATH = PROJECT_ROOT / "config" / "universe.yaml"


def load_universe_symbols() -> list[str]:
    """Binance-символы пула + служебный BTCUSDT (config/universe.yaml)."""
    with UNIVERSE_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = [coin["binance_symbol"] for coin in cfg["pool"]]
    btc = cfg["btc_filter_symbol"]
    if btc not in symbols:
        symbols.append(btc)
    return symbols


def _request_with_retry(method: str, url: str, *, attempts: int = 4, **kwargs) -> requests.Response:
    """requests.get/post с повторными попытками — сеть на 4.5 годах истории иногда
    отваливается на секунду-две, из-за одного такого сбоя терять весь прогон (и всё,
    что он успел скачать до этого) не стоит. Экспоненциальная пауза: 1с, 2с, 4с, 8с."""
    last_error: requests.RequestException | None = None
    for attempt in range(attempts):
        try:
            return requests.request(method, url, **kwargs)
        except requests.RequestException as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    raise last_error  # после всех попыток - отдаём вызывающему коду, пусть решает


def _period_of(ts: pd.Timestamp) -> pd.Period:
    """pd.Period не умеет хранить часовой пояс — берём только 'YYYY-MM' из tz-aware
    метки, не через ts.to_period() (тот при конвертации сам ругается предупреждением)."""
    return pd.Period(ts.strftime("%Y-%m"), freq="M")


def _looks_like_header(first_column_name: object) -> bool:
    """Заголовок Binance vision состоит из букв ("open_time"). Если вместо этого
    там число — значит заголовка в файле не было, и первая строка данных случайно
    "съедена" как имя колонки при pd.read_csv(..., header=0)."""
    try:
        float(first_column_name)
        return False
    except ValueError:
        return True


def _download_zip_csv(url: str, columns: list[str]) -> pd.DataFrame | None:
    """Скачивает .zip с одним .csv внутри и возвращает DataFrame с колонками columns.
    None, если файла с таким именем нет на сервере (404) — например, месяц до листинга
    монеты или ещё не наступил."""
    resp = _request_with_retry("get", url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw_bytes = zf.read(zf.namelist()[0])

    df = pd.read_csv(io.BytesIO(raw_bytes))
    if len(df.columns) != len(columns) or not _looks_like_header(df.columns[0]):
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None)
    df.columns = columns[: len(df.columns)]
    return df


def klines_raw_to_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Сырые колонки архива -> схема из 02_ENGINE_SPEC.md, 2.2, индекс = open_time (UTC)."""
    df = pd.DataFrame({
        "open_time": pd.to_datetime(raw["open_time"].astype("int64"), unit="ms", utc=True),
        "open": raw["open"].astype("float64"),
        "high": raw["high"].astype("float64"),
        "low": raw["low"].astype("float64"),
        "close": raw["close"].astype("float64"),
        "volume": raw["volume"].astype("float64"),
        "quote_volume": raw["quote_volume"].astype("float64"),
        "trades": raw["count"].astype("int64"),
    })
    df["complete"] = True
    return df.set_index("open_time").sort_index()


def funding_raw_to_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "funding_time": pd.to_datetime(raw["calc_time"].astype("int64"), unit="ms", utc=True),
        "funding_rate": raw["last_funding_rate"].astype("float64"),
    })
    return df.set_index("funding_time").sort_index()


def _month_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Period]:
    return list(pd.period_range(_period_of(start), _period_of(end), freq="M"))


def fetch_klines_rest(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Запасной вариант через REST (данные за хвост текущего месяца, если для них
    ещё нет ни месячного, ни дневного архива). Может быть недоступен из некоторых
    регионов/облачных IP — тогда просто логируем предупреждение и едем дальше,
    следующий запуск докачает при повторной попытке."""
    rows: list[list] = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor_ms < end_ms:
        resp = _request_with_retry(
            "get", f"{REST_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "1h", "startTime": cursor_ms, "limit": 1500},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + 3_600_000
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
        time.sleep(0.2)  # не долбить API чаще необходимого
    if not rows:
        return klines_raw_to_df(pd.DataFrame(columns=KLINES_COLUMNS))
    raw = pd.DataFrame(rows, columns=KLINES_COLUMNS)
    return klines_raw_to_df(raw)


def download_symbol_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Идемпотентно скачивает 1h-бары в data/raw/{symbol}/1h/{YYYY-MM}.parquet —
    один файл на месяц. Уже существующие файлы не трогает."""
    out_dir = RAW_DIR / symbol / "1h"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    current_month = _period_of(now)

    for period in _month_periods(start, end):
        dest = out_dir / f"{period}.parquet"
        if dest.exists():
            continue

        try:
            if period < current_month:
                raw = _download_zip_csv(
                    f"{VISION_BASE}/monthly/klines/{symbol}/1h/{symbol}-1h-{period}.zip", KLINES_COLUMNS
                )
                if raw is None:
                    print(f"  [klines] {symbol} {period}: архива нет (до листинга?), пропуск")
                    continue
                df = klines_raw_to_df(raw)
            else:
                # текущий месяц не выложен целиком - собираем из дневных архивов + REST-хвост
                month_start = max(pd.Timestamp(f"{period}-01", tz="UTC"), start)
                frames = []
                for day in pd.date_range(month_start, min(end, now), freq="D"):
                    raw = _download_zip_csv(
                        f"{VISION_BASE}/daily/klines/{symbol}/1h/{symbol}-1h-{day:%Y-%m-%d}.zip",
                        KLINES_COLUMNS,
                    )
                    if raw is not None:
                        frames.append(klines_raw_to_df(raw))
                covered_until = max(
                    (f.index.max() for f in frames), default=month_start - pd.Timedelta(hours=1)
                )
                if covered_until < min(end, now):
                    try:
                        frames.append(
                            fetch_klines_rest(symbol, covered_until + pd.Timedelta(hours=1), min(end, now))
                        )
                    except requests.RequestException as e:
                        print(f"  [klines] {symbol} {period}: REST-хвост недоступен ({e}), докачаем в следующий раз")
                if not frames:
                    print(f"  [klines] {symbol} {period}: данных пока нет, пропуск")
                    continue
                df = pd.concat(frames).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        except requests.RequestException as e:
            print(f"  [klines] {symbol} {period}: сеть не ответила ({e}), пропуск - докачается при повторном запуске")
            continue

        df.to_parquet(dest)
        print(f"  [klines] {symbol} {period}: {len(df)} баров")


def download_symbol_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Идемпотентно скачивает историю фандинга в data/raw/{symbol}/funding/{YYYY-MM}.parquet."""
    out_dir = RAW_DIR / symbol / "funding"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    current_month = _period_of(now)

    for period in _month_periods(start, end):
        dest = out_dir / f"{period}.parquet"
        if dest.exists():
            continue

        try:
            if period < current_month:
                raw = _download_zip_csv(
                    f"{VISION_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{period}.zip",
                    FUNDING_COLUMNS,
                )
                if raw is None:
                    print(f"  [funding] {symbol} {period}: архива нет, пропуск")
                    continue
                df = funding_raw_to_df(raw)
            else:
                month_start = max(pd.Timestamp(f"{period}-01", tz="UTC"), start)
                rows = []
                cursor_ms = int(month_start.timestamp() * 1000)
                end_ms = int(min(end, now).timestamp() * 1000)
                while cursor_ms < end_ms:
                    resp = _request_with_retry(
                        "get", f"{REST_BASE}/fapi/v1/fundingRate",
                        params={"symbol": symbol, "startTime": cursor_ms, "limit": 1000},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    batch = resp.json()
                    if not batch:
                        break
                    rows.extend(batch)
                    next_cursor = batch[-1]["fundingTime"] + 1
                    if next_cursor <= cursor_ms:
                        break
                    cursor_ms = next_cursor
                    time.sleep(0.2)
                if not rows:
                    print(f"  [funding] {symbol} {period}: данных пока нет, пропуск")
                    continue
                raw = pd.DataFrame(
                    {"calc_time": [r["fundingTime"] for r in rows],
                     "last_funding_rate": [r["fundingRate"] for r in rows]}
                )
                df = funding_raw_to_df(raw)
        except requests.RequestException as e:
            print(f"  [funding] {symbol} {period}: сеть не ответила ({e}), пропуск - докачается при повторном запуске")
            continue

        df.to_parquet(dest)
        print(f"  [funding] {symbol} {period}: {len(df)} отметок")


def assemble_symbol(symbol: str) -> None:
    """Склеивает помесячные .parquet из data/raw/{symbol}/... в единые файлы
    data/processed/{symbol}_1h.parquet и data/processed/{symbol}_funding.parquet
    (полный скачанный диапазон, ещё БЕЗ разделения на dev/holdout — этим занимается
    holdout_lock.py следующим шагом)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    kline_files = sorted((RAW_DIR / symbol / "1h").glob("*.parquet"))
    if kline_files:
        klines = pd.concat([pd.read_parquet(f) for f in kline_files]).sort_index()
        klines = klines[~klines.index.duplicated(keep="last")]
        klines.to_parquet(PROCESSED_DIR / f"{symbol}_1h.parquet")

    funding_files = sorted((RAW_DIR / symbol / "funding").glob("*.parquet"))
    if funding_files:
        funding = pd.concat([pd.read_parquet(f) for f in funding_files]).sort_index()
        funding = funding[~funding.index.duplicated(keep="last")]
        funding.to_parquet(PROCESSED_DIR / f"{symbol}_funding.parquet")


def main() -> None:
    from lab.config import load_protocol

    cfg = load_protocol()
    start = pd.Timestamp(cfg.periods.data_start, tz="UTC")
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)  # последний закрытый час

    symbols = load_universe_symbols()
    print(f"Монеты: {', '.join(symbols)}")
    print(f"Диапазон: {start.date()} .. {end.date()} (включая holdout — так и задумано, см. docstring)")

    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        download_symbol_klines(symbol, start, end)
        download_symbol_funding(symbol, start, end)
        assemble_symbol(symbol)

    print(
        "\nГотово (месяцы с сетевыми сбоями, если были, отмечены выше как 'пропуск' — "
        "запустите эту же команду ещё раз, идемпотентность докачает только их)."
        "\nДальше: python -m lab.data.resample"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
