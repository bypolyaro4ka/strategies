"""Стопы/тейки, ликвидация, размер позиции, лимит плеча (01_PROTOCOL.md, разделы 5.4-5.6).

Все функции здесь чистые (вход -> выход, без состояния) — так их можно проверить по
отдельности, не поднимая весь цикл движка. `@dataclass` используется как лёгкая "коробочка"
для нескольких связанных значений результата, вместо кортежа — так у полей есть имена.
"""

from __future__ import annotations

from dataclasses import dataclass

from lab.engine.costs import fill_price


@dataclass
class StopResult:
    triggered: bool
    exit_price: float | None = None
    reason: str = ""  # "stop" | "take" | ""


def check_stop_take(
    *,
    is_long: bool,
    open_: float,
    high: float,
    low: float,
    stop_price: float | None,
    take_price: float | None,
    slippage: float,
) -> StopResult:
    """Проверка стопа/тейка внутри бара, по high/low (01_PROTOCOL.md, 5.4):
    - если задет и стоп, и тейк на одном баре — считаем, что первым сработал стоп;
    - при гэпе через стоп (open уже хуже уровня) — исполнение по open, если он хуже стопа;
    - исполнение всегда со стандартным проскальзыванием, в невыгодную сторону.
    """
    stop_hit = stop_price is not None and ((low <= stop_price) if is_long else (high >= stop_price))

    if stop_hit:
        if is_long:
            gapped_through = open_ < stop_price
            raw_price = open_ if gapped_through else stop_price
            exit_price = fill_price(raw_price, is_buy=False, slippage=slippage)  # продаём, закрывая лонг
        else:
            gapped_through = open_ > stop_price
            raw_price = open_ if gapped_through else stop_price
            exit_price = fill_price(raw_price, is_buy=True, slippage=slippage)  # откупаем, закрывая шорт
        return StopResult(True, exit_price, "stop")

    take_hit = take_price is not None and ((high >= take_price) if is_long else (low <= take_price))
    if take_hit:
        exit_price = fill_price(take_price, is_buy=not is_long, slippage=slippage)
        return StopResult(True, exit_price, "take")

    return StopResult(False)


def is_short_liquidated(entry_price: float, mark_price: float) -> bool:
    """Шорт при плече 1x ликвидируется при росте цены на 100% от входа (01_PROTOCOL.md, 5.6).
    Лонг при 1x не ликвидируется вообще — для него эта функция не вызывается."""
    return mark_price >= entry_price * 2


def target_notional(target: float, slot_fraction: float, equity: float) -> float:
    """01_PROTOCOL.md, 5.5: notional = target * slot_fraction * equity, на момент решения."""
    return target * slot_fraction * equity


def apply_leverage_cap(desired_notional: float, other_symbols_notional_abs_sum: float, equity: float) -> tuple[float, bool]:
    """Урезает desired_notional ОДНОЙ конкретной позиции так, чтобы суммарный |notional|
    по всем позициям не превышал equity (плечо 1x) — не трогая уже открытые позиции по
    другим монетам (01_PROTOCOL.md, 5.5: "новая позиция... урезается до доступного объёма").
    Возвращает (урезанный notional, было ли урезание)."""
    available = max(equity - other_symbols_notional_abs_sum, 0.0)
    if abs(desired_notional) <= available:
        return desired_notional, False
    sign = 1.0 if desired_notional >= 0 else -1.0
    return sign * available, True


def passes_min_change(new_target: float, old_target: float, min_target_change: float) -> bool:
    """Изменения |Δtarget| < min_target_change игнорируются — защита от микросделок
    (01_PROTOCOL.md, 5.5). Маленький допуск (1e-9) — граница часто попадает на "круглые"
    десятичные значения (0.5 - 0.45 в float — это 0.04999999999999999, а не 0.05), без
    допуска такие случаи то проходили бы порог, то нет, в зависимости от порядка вычислений."""
    return abs(new_target - old_target) >= min_target_change - 1e-9
