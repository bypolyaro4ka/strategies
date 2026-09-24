"""Тесты для src/lab/engine/costs.py — Этап 3.

"Costs toy" и "Funding sign" из 02_ENGINE_SPEC.md, раздел 7.
"""

import pytest

from lab.engine.costs import fee_amount, fill_price, funding_payment, slippage_cost


def test_fill_price_buy_and_sell():
    assert fill_price(100.0, is_buy=True, slippage=0.0002) == pytest.approx(100.02)
    assert fill_price(100.0, is_buy=False, slippage=0.0002) == pytest.approx(99.98)


def test_fee_amount():
    assert fee_amount(delta_notional=1000.0, fee_taker=0.0005) == pytest.approx(0.5)
    assert fee_amount(delta_notional=-1000.0, fee_taker=0.0005) == pytest.approx(0.5)  # знак не важен


def test_costs_toy_round_trip_flat_price():
    """Ручной пример при неизменной цене - изолирует именно издержки от движения цены:
    круг туда-обратно должен стоить ровно 2×(fee_taker + slippage) = 0.14% (01_PROTOCOL.md, 5.2)."""
    fee_taker = 0.0005
    slippage = 0.0002
    notional = 1000.0
    price = 100.0

    entry_fill = fill_price(price, is_buy=True, slippage=slippage)
    assert entry_fill == pytest.approx(100.02)
    entry_fee = fee_amount(notional, fee_taker)
    assert entry_fee == pytest.approx(0.5)

    exit_fill = fill_price(price, is_buy=False, slippage=slippage)  # цена не изменилась
    assert exit_fill == pytest.approx(99.98)
    qty = notional / entry_fill
    exit_notional = qty * exit_fill
    exit_fee = fee_amount(exit_notional, fee_taker)

    gross_pnl = exit_notional - notional
    net_pnl = gross_pnl - entry_fee - exit_fee
    assert gross_pnl < 0  # даже без движения цены проскальзывание уже дало убыток
    assert net_pnl < gross_pnl  # а комиссии съели ещё сверху

    round_trip_cost_pct = -net_pnl / notional
    assert round_trip_cost_pct == pytest.approx(0.0014, abs=1e-4)


def test_costs_toy_manual_numbers():
    """Совсем ручной пример без объекта позиции: одна сделка, известные цены -> точные
    fee/slippage по отдельности."""
    entry_fill = fill_price(100.0, is_buy=True, slippage=0.0002)
    exit_fill = fill_price(110.0, is_buy=False, slippage=0.0002)
    assert entry_fill == pytest.approx(100.02)
    assert exit_fill == pytest.approx(109.978)

    qty = 1000.0 / entry_fill  # notional=1000 на входе
    entry_fee = fee_amount(1000.0, 0.0005)
    exit_notional = qty * exit_fill
    exit_fee = fee_amount(exit_notional, 0.0005)

    assert entry_fee == pytest.approx(0.5)
    assert exit_fee == pytest.approx(qty * exit_fill * 0.0005)
    gross_pnl = exit_notional - 1000.0
    net_pnl = gross_pnl - entry_fee - exit_fee
    assert net_pnl == pytest.approx(gross_pnl - entry_fee - exit_fee)


def test_slippage_cost_formula():
    assert slippage_cost(abs_qty=10, fill_price=100, slippage=0.0002) == pytest.approx(0.2)
    assert slippage_cost(abs_qty=-10, fill_price=100, slippage=0.0002) == pytest.approx(0.2)  # знак не важен


def test_funding_sign_long_pays_when_rate_positive():
    payment = funding_payment(position_qty=10.0, mark_price=100.0, funding_rate=0.0001)
    assert payment < 0  # лонг платит


def test_funding_sign_short_receives_when_rate_positive():
    payment = funding_payment(position_qty=-10.0, mark_price=100.0, funding_rate=0.0001)
    assert payment > 0  # шорт получает


def test_funding_sign_long_receives_when_rate_negative():
    payment = funding_payment(position_qty=10.0, mark_price=100.0, funding_rate=-0.0001)
    assert payment > 0  # лонг получает при отрицательной ставке


def test_funding_zero_position_no_payment():
    assert funding_payment(position_qty=0.0, mark_price=100.0, funding_rate=0.0005) == 0.0
