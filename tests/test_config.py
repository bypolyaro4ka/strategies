"""Тесты для src/lab/config.py — Этап 0."""

import pytest

from lab.config import ProtocolConfigError, load_protocol


def test_load_protocol_reads_real_config():
    cfg = load_protocol()
    assert cfg.protocol_version == 1
    assert cfg.frozen is False
    assert cfg.exchange.fee_taker == pytest.approx(0.0005)
    assert cfg.account.initial_equity == 10000
    assert cfg.periods.holdout_start == "2026-01-01"
    assert len(cfg.folds) == 4
    assert cfg.folds[0].id == "F1"


def test_load_protocol_missing_file(tmp_path):
    load_protocol.cache_clear()
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ProtocolConfigError):
        load_protocol(missing)
    load_protocol.cache_clear()


def test_load_protocol_rejects_bad_fee(tmp_path):
    load_protocol.cache_clear()
    bad = tmp_path / "protocol.yaml"
    bad.write_text(
        """
protocol_version: 1
frozen: false
exchange: {venue: x, quote: USDT, order_type: market, fee_taker: 5.0, fee_maker: 0.0002, slippage: 0.0002}
account: {initial_equity: 10000, slot_fraction: 0.1, leverage: 1, max_slots_portfolio: 10, min_target_change: 0.05, resize_on_equity_change: false}
periods: {timezone: UTC, data_start: "2022-01-01", dev_start: "2023-01-01", dev_end: "2025-12-31", holdout_start: "2026-01-01", holdout_end: null}
timeframes: {base: 1h, derived: ["4h"], execution: 1h}
walk_forward: {scheme: anchored, folds: [{id: F1, train: ["2023-01-01","2023-12-31"], test: ["2024-01-01","2024-06-30"]}]}
selection: {objective: sharpe_daily, plateau_rule: neighbor_median, min_trades_train_portfolio: 30, coin_min_trades_train: 5, coin_min_sharpe_train: 0.0, min_coins_after_selection: 3, primary_rule: {min_wf_sharpe_gain_vs_v0: 0.3, min_folds_better: 3}}
metrics: {annualization_days: 365, risk_free: 0.0, random_baseline_seeds: 1000}
btc_filter: {symbol: BTCUSDT, sma_days: 50}
registry: {backend: mlflow, experiment_prefix: strategy_lab}
""",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolConfigError):
        load_protocol(bad)
    load_protocol.cache_clear()
