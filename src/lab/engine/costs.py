"""Издержки исполнения: комиссия, проскальзывание, фандинг (01_PROTOCOL.md, разделы 5.2-5.3).

Три маленькие чистые функции — без состояния, без побочных эффектов, поэтому их легко
проверить один в один по формулам из протокола (это и есть тест "Costs toy" /
"Funding sign" из 02_ENGINE_SPEC.md, раздел 7).
"""

from __future__ import annotations


def fill_price(open_price: float, is_buy: bool, slippage: float) -> float:
    """Цена исполнения рыночного ордера на открытии бара.

    is_buy=True — позиция растёт в сторону лонга (открываем/наращиваем лонг ИЛИ
    закрываем/сокращаем шорт), False — наоборот. Проскальзывание всегда против нас:
    покупаем дороже (open × (1 + slippage)), продаём дешевле (open × (1 - slippage))."""
    return open_price * (1 + slippage) if is_buy else open_price * (1 - slippage)


def fee_amount(delta_notional: float, fee_taker: float) -> float:
    """delta_notional — изменение notional по конкретному ордеру (вход, выход, доливка,
    сокращение — комиссия берётся с каждого изменения отдельно, 01_PROTOCOL.md, 5.2)."""
    return abs(delta_notional) * fee_taker


def slippage_cost(abs_qty: float, fill_price: float, slippage: float) -> float:
    """Сколько именно "стоило" проскальзывание на этом исполнении — отдельно от комиссии,
    для разбивки издержек в trades.csv (02_ENGINE_SPEC.md, раздел 5) и Cost share
    (01_PROTOCOL.md, раздел 8). fill_price уже включает проскальзывание (см. fill_price()
    выше); берём его как базу для расчёта — при slippage=0.02% разница с "чистой" ценой
    открытия на порядок меньше самого проскальзывания, ей пренебрегаем."""
    return abs(abs_qty) * fill_price * slippage


def funding_payment(position_qty: float, mark_price: float, funding_rate: float) -> float:
    """Знак результата: отрицательный — мы платим, положительный — мы получаем.

    Лонг (position_qty > 0) при funding_rate > 0 платит; шорт (position_qty < 0) при
    funding_rate > 0 получает; при funding_rate < 0 — наоборот (01_PROTOCOL.md, 5.3).
    mark_price — цена закрытия 1h-бара, заканчивающегося в момент фандинга."""
    return -position_qty * mark_price * funding_rate
