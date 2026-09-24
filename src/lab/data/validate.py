"""Проверки качества данных (01_PROTOCOL.md, раздел 4) -> reports/data_quality/report.md.

Работает только с data/processed/ (dev-диапазон, после holdout_lock.py) — отчёт о качестве
не должен ничего говорить про 2026 год, это уже была бы форма подглядывания.
"""

from __future__ import annotations

import pandas as pd

from lab.data.download import PROCESSED_DIR, load_universe_symbols

PROJECT_ROOT = PROCESSED_DIR.parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality" / "report.md"


def check_bars(df: pd.DataFrame, tf: str) -> dict:
    """Возвращает счётчики проблем для одного файла {symbol}_{tf}.parquet."""
    freq = {"1h": "1h", "4h": "4h", "12h": "12h", "1d": "1D"}[tf]
    expected_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    gaps = len(expected_index.difference(df.index))

    duplicates = int(df.index.duplicated().sum())
    zero_volume = int((df["volume"] == 0).sum())

    ohlc_bad = int((
        (df["low"] > df["open"]) | (df["open"] > df["high"]) |
        (df["low"] > df["close"]) | (df["close"] > df["high"]) |
        (df["low"] > df["high"])
    ).sum())

    hourly_return_outliers = None
    if tf == "1h":
        ret = df["close"].pct_change().abs()
        hourly_return_outliers = int((ret > 0.5).sum())

    incomplete = int((~df["complete"]).sum()) if tf != "1h" else 0

    return {
        "rows": len(df),
        "gaps": gaps,
        "duplicates": duplicates,
        "zero_volume": zero_volume,
        "ohlc_invariant_violations": ohlc_bad,
        "hourly_return_outliers_gt_50pct": hourly_return_outliers,
        "incomplete_bars": incomplete,
    }


def check_funding(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"rows": len(df), "intervals_hours_seen": []}
    diffs = df.index.to_series().diff().dropna()
    hours = sorted(set((diffs.dt.total_seconds() / 3600).round(2)))
    return {"rows": len(df), "intervals_hours_seen": hours}


def main() -> None:
    lines = ["# Отчёт о качестве данных (dev-диапазон)\n"]
    lines.append("Проверки по `01_PROTOCOL.md`, раздел 4. Только `data/processed/` (dev), "
                  "holdout физически изолирован (`_holdout_locked/`) и здесь не участвует.\n")

    for symbol in load_universe_symbols():
        lines.append(f"\n## {symbol}\n")
        lines.append("| ТФ | Строк | Пропуски | Дубли | Нулевой объём | OHLC-нарушения | "
                      "Выбросы >50%/ч | Неполные бары |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tf in ("1h", "4h", "12h", "1d"):
            path = PROCESSED_DIR / f"{symbol}_{tf}.parquet"
            if not path.exists():
                lines.append(f"| {tf} | нет файла | | | | | | |")
                continue
            df = pd.read_parquet(path)
            if df.empty:
                lines.append(f"| {tf} | 0 | — | — | — | — | — | — |")
                continue
            r = check_bars(df, tf)
            lines.append(
                f"| {tf} | {r['rows']} | {r['gaps']} | {r['duplicates']} | {r['zero_volume']} | "
                f"{r['ohlc_invariant_violations']} | "
                f"{r['hourly_return_outliers_gt_50pct'] if r['hourly_return_outliers_gt_50pct'] is not None else '—'} | "
                f"{r['incomplete_bars']} |"
            )

        funding_path = PROCESSED_DIR / f"{symbol}_funding.parquet"
        if funding_path.exists():
            fdf = pd.read_parquet(funding_path)
            fr = check_funding(fdf)
            lines.append(f"\nФандинг: {fr['rows']} отметок, интервалы (часы): {fr['intervals_hours_seen']}\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт записан в {REPORT_PATH}")


if __name__ == "__main__":
    main()
