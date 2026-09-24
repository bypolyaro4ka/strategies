"""Тесты для src/lab/report/make_report.py — Этап 4 (смоук-тест: отчёт строится без ошибок
и содержит ожидаемые разделы; точный markdown не проверяем - хрупко и малоценно)."""

from types import SimpleNamespace

import pytest

from lab.benchmarks.flat import FlatStrategy
from lab.data.resample import resample_ohlcv
from lab.engine.backtest import run_backtest
from lab.report.make_report import make_report

from stubs import AlwaysLongStub


def _cfg():
    return SimpleNamespace(
        account=SimpleNamespace(slot_fraction=0.10, min_target_change=0.05, max_slots_portfolio=5),
        exchange=SimpleNamespace(fee_taker=0.0005, slippage=0.0002),
        metrics=SimpleNamespace(annualization_days=365, risk_free=0.0),
    )


def _flat_bars(n_hours, start="2023-01-01", price0=100.0):
    import pandas as pd
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    price = [price0 + i * 0.1 for i in range(n_hours)]
    return pd.DataFrame({
        "open": price, "high": price, "low": price, "close": price,
        "volume": 1.0, "quote_volume": 100.0, "trades": 1, "complete": True,
    }, index=idx)


def test_make_report_produces_markdown_and_png(tmp_path):
    bars = _flat_bars(24 * 65)  # >2 месяцев - иначе таблица помесячной доходности пуста
    cfg = _cfg()
    signal_bars = {"X": resample_ohlcv(bars, "1d")}
    strat_result = run_backtest(AlwaysLongStub(), {"X": bars}, signal_bars, {}, "1d", cfg, 10_000, 1)
    flat_result = run_backtest(FlatStrategy(), {"X": bars}, signal_bars, {}, "1d", cfg, 10_000, 1)

    report_path = make_report(
        title="Тестовый отчёт",
        strategy_result=strat_result,
        protocol_cfg=cfg,
        out_dir=tmp_path,
        benchmark_results={"Flat": flat_result},
        n_trials=10,
        verdict_note="работает / неотличимо от случайного - подставляется вызывающим кодом",
    )
    assert report_path.exists()
    assert (tmp_path / "equity.png").exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Sharpe" in text
    assert "DSR" in text
    assert "Flat" in text
    assert "Помесячная доходность" in text
    assert "Статистическая оговорка" in text
