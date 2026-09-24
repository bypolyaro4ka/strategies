"""Сборка 4h/12h/1d баров из 1h (02_ENGINE_SPEC.md, 2.3).

pandas.DataFrame.resample() режет временной ряд на равные интервалы и группирует
строки внутри каждого интервала — то же самое, что groupby, только по времени, а не
по значению колонки. `label="left", closed="left"` — имя бина = его открытие,
и сам открывающий бар входит в бин (а не следующий). Поскольку индекс — это Unix-эпоха
в UTC, а 1970-01-01 00:00 UTC делится и на 4 часа, и на 12, и на сутки без остатка,
границы бинов автоматически совпадают с биржевыми (00/04/08/... UTC для 4h и т. д.) —
объяснять/настраивать смещение вручную не нужно.
"""

from __future__ import annotations

import pandas as pd

from lab.data.download import PROCESSED_DIR, load_universe_symbols
from lab.data.holdout_lock import apply_holdout_lock

RESAMPLE_RULE = {"4h": "4h", "12h": "12h", "1d": "1D"}
EXPECTED_BARS = {"4h": 4, "12h": 12, "1d": 24}


def resample_ohlcv(df_1h: pd.DataFrame, tf: str) -> pd.DataFrame:
    """df_1h — бары с колонками open/high/low/close/volume/quote_volume/trades,
    индекс open_time (UTC, полные часы). Возвращает бары того же вида на ТФ tf."""
    rule = RESAMPLE_RULE[tf]
    g = df_1h.resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "quote_volume": g["quote_volume"].sum(),
        "trades": g["trades"].sum().astype("int64"),
    })
    # complete = все 1h-бары внутри бина есть, ни один не пропущен (01_PROTOCOL.md, 4)
    out["complete"] = g["open"].count() == EXPECTED_BARS[tf]
    out.index.name = "open_time"
    return out


def build_all_timeframes() -> None:
    for symbol in load_universe_symbols():
        src = PROCESSED_DIR / f"{symbol}_1h.parquet"
        if not src.exists():
            print(f"  {symbol}: нет {src.name}, сначала запустите download.py")
            continue
        df_1h = pd.read_parquet(src)
        for tf in ("4h", "12h", "1d"):
            df_tf = resample_ohlcv(df_1h, tf)
            df_tf.to_parquet(PROCESSED_DIR / f"{symbol}_{tf}.parquet")
        print(f"  {symbol}: 4h/12h/1d собраны")


def main() -> None:
    print("Сборка старших ТФ из 1h...")
    build_all_timeframes()
    print("\nИзоляция holdout-данных (data/processed/_holdout_locked/)...")
    apply_holdout_lock()
    print("\nГотово. Дальше: python -m lab.data.validate")


if __name__ == "__main__":
    main()
