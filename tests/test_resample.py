"""Тесты для src/lab/data/resample.py — Этап 2."""

import pandas as pd

from lab.data.resample import resample_ohlcv


def _make_1h(n_hours: int, start="2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": range(n_hours),
        "high": [v + 1 for v in range(n_hours)],
        "low": [v - 1 for v in range(n_hours)],
        "close": [v + 0.5 for v in range(n_hours)],
        "volume": [10.0] * n_hours,
        "quote_volume": [100.0] * n_hours,
        "trades": [5] * n_hours,
        "complete": [True] * n_hours,
    }, index=idx)


def test_resample_4h_boundaries_and_ohlc():
    df = _make_1h(8)  # два полных 4h-бара
    out = resample_ohlcv(df, "4h")

    assert list(out.index) == [
        pd.Timestamp("2023-01-01 00:00", tz="UTC"),
        pd.Timestamp("2023-01-01 04:00", tz="UTC"),
    ]
    first = out.iloc[0]
    assert first["open"] == 0          # open первого часа бина
    assert first["close"] == 3.5       # close последнего часа бина (индекс 3)
    assert first["high"] == 4          # max(high) среди баров 0..3 -> high=v+1, max при v=3 -> 4
    assert first["low"] == -1          # min(low) среди баров 0..3 -> low=v-1, min при v=0 -> -1
    assert first["volume"] == 40.0     # sum(10.0 * 4)
    assert first["trades"] == 20
    assert bool(first["complete"]) is True


def test_resample_marks_incomplete_when_bars_missing():
    df = _make_1h(4).drop(pd.Timestamp("2023-01-01 02:00", tz="UTC"))  # пропущен один час
    out = resample_ohlcv(df, "4h")
    assert len(out) == 1
    assert bool(out.iloc[0]["complete"]) is False


def test_resample_1d_sums_24_hours():
    df = _make_1h(24)
    out = resample_ohlcv(df, "1d")
    assert len(out) == 1
    assert out.iloc[0]["volume"] == 240.0
    assert bool(out.iloc[0]["complete"]) is True
