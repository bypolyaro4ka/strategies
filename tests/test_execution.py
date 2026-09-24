"""Тесты для src/lab/engine/execution.py — Этап 3.

"Stop priority", "Gap через стоп", "Leverage cap", "Min change" из 02_ENGINE_SPEC.md, раздел 7.
"""

import pytest

from lab.engine.execution import (
    adjust_position,
    apply_leverage_cap,
    check_stop_take,
    is_short_liquidated,
    passes_min_change,
    target_notional,
)


# --- Stop priority: бар задевает и стоп, и тейк -> исполнен стоп ---

def test_stop_priority_over_take_long():
    result = check_stop_take(
        is_long=True, open_=100, high=110, low=90,
        stop_price=95, take_price=105, slippage=0.0002,
    )
    assert result.triggered
    assert result.reason == "stop"


def test_stop_priority_over_take_short():
    result = check_stop_take(
        is_long=False, open_=100, high=110, low=90,
        stop_price=105, take_price=95, slippage=0.0002,
    )
    assert result.triggered
    assert result.reason == "stop"


# --- Gap через стоп: open хуже стопа -> исполнение по open ---

def test_gap_through_stop_long_executes_at_open():
    result = check_stop_take(
        is_long=True, open_=90, high=92, low=88,
        stop_price=95, take_price=None, slippage=0.0002,
    )
    assert result.reason == "stop"
    assert result.exit_price == pytest.approx(90 * (1 - 0.0002))


def test_no_gap_stop_executes_at_stop_price():
    result = check_stop_take(
        is_long=True, open_=100, high=101, low=94,
        stop_price=95, take_price=None, slippage=0.0002,
    )
    assert result.reason == "stop"
    assert result.exit_price == pytest.approx(95 * (1 - 0.0002))


def test_gap_through_stop_short_executes_at_open():
    result = check_stop_take(
        is_long=False, open_=115, high=118, low=112,
        stop_price=110, take_price=None, slippage=0.0002,
    )
    assert result.reason == "stop"
    assert result.exit_price == pytest.approx(115 * (1 + 0.0002))


def test_take_hit_without_stop():
    result = check_stop_take(
        is_long=True, open_=100, high=110, low=99,
        stop_price=None, take_price=105, slippage=0.0002,
    )
    assert result.reason == "take"
    assert result.exit_price == pytest.approx(105 * (1 - 0.0002))


def test_no_trigger_when_price_stays_between():
    result = check_stop_take(
        is_long=True, open_=100, high=102, low=98,
        stop_price=95, take_price=105, slippage=0.0002,
    )
    assert not result.triggered


# --- Ликвидация шорта ---

def test_short_liquidated_at_100pct_gain():
    assert is_short_liquidated(entry_price=100, mark_price=200) is True
    assert is_short_liquidated(entry_price=100, mark_price=199.99) is False


# --- Лимит плеча: сумма |notional| не превышает equity ---

def test_leverage_cap_no_cut_when_within_limit():
    notional, cut = apply_leverage_cap(desired_notional=300, other_symbols_notional_abs_sum=500, equity=1000)
    assert notional == 300
    assert cut is False


def test_leverage_cap_cuts_to_available_room():
    notional, cut = apply_leverage_cap(desired_notional=800, other_symbols_notional_abs_sum=500, equity=1000)
    assert notional == pytest.approx(500)
    assert cut is True


def test_leverage_cap_preserves_sign_for_short():
    notional, cut = apply_leverage_cap(desired_notional=-900, other_symbols_notional_abs_sum=300, equity=1000)
    assert notional == pytest.approx(-700)
    assert cut is True


def test_leverage_cap_no_room_left():
    notional, cut = apply_leverage_cap(desired_notional=100, other_symbols_notional_abs_sum=1000, equity=1000)
    assert notional == 0
    assert cut is True


# --- adjust_position: доливки/сокращения/разворот ---

def test_adjust_position_noop_when_flat_and_zero_delta():
    qty, avg_entry, realized = adjust_position(qty_old=0, avg_entry_old=0, delta_qty=0, fill_price=100)
    assert (qty, avg_entry, realized) == (0.0, 0.0, 0.0)


def test_adjust_position_opens_from_flat():
    qty, avg_entry, realized = adjust_position(qty_old=0, avg_entry_old=0, delta_qty=10, fill_price=100)
    assert qty == 10
    assert avg_entry == 100
    assert realized == 0


def test_adjust_position_adds_same_direction_weighted_average():
    qty, avg_entry, realized = adjust_position(qty_old=10, avg_entry_old=100, delta_qty=10, fill_price=120)
    assert qty == 20
    assert avg_entry == pytest.approx(110)  # (10*100 + 10*120) / 20
    assert realized == 0


def test_adjust_position_partial_reduce_keeps_avg_entry():
    qty, avg_entry, realized = adjust_position(qty_old=10, avg_entry_old=100, delta_qty=-4, fill_price=120)
    assert qty == 6
    assert avg_entry == 100  # у оставшейся части цена входа не меняется
    assert realized == pytest.approx(4 * (120 - 100))  # реализован PnL по закрытой части


def test_adjust_position_full_close():
    qty, avg_entry, realized = adjust_position(qty_old=10, avg_entry_old=100, delta_qty=-10, fill_price=90)
    assert qty == 0
    assert realized == pytest.approx(10 * (90 - 100))


def test_adjust_position_flip_long_to_short():
    qty, avg_entry, realized = adjust_position(qty_old=10, avg_entry_old=100, delta_qty=-15, fill_price=90)
    assert qty == -5
    assert avg_entry == 90  # новая (короткая) позиция открыта по цене исполнения
    assert realized == pytest.approx(10 * (90 - 100))  # PnL реализован только по закрытой части лонга


def test_adjust_position_short_side_pnl_sign():
    # шорт: закрытие (покупка) дешевле входа - прибыль
    qty, avg_entry, realized = adjust_position(qty_old=-10, avg_entry_old=100, delta_qty=10, fill_price=80)
    assert qty == 0
    assert realized == pytest.approx(200)  # (100 - 80) * 10


# --- target_notional ---

def test_target_notional_formula():
    assert target_notional(target=0.5, slot_fraction=0.10, equity=10_000) == pytest.approx(500.0)


# --- Min change: изменение меньше порога не создаёт ордер ---

def test_min_change_below_threshold_ignored():
    assert passes_min_change(new_target=0.5, old_target=0.47, min_target_change=0.05) is False


def test_min_change_at_threshold_passes():
    assert passes_min_change(new_target=0.5, old_target=0.45, min_target_change=0.05) is True


def test_min_change_above_threshold_passes():
    assert passes_min_change(new_target=1.0, old_target=0.0, min_target_change=0.05) is True
