"""Главный цикл бэктеста (02_ENGINE_SPEC.md, раздел 4).

Один и тот же цикл обслуживает и режим «пара» (один символ, свой счёт), и «портфель»
(несколько символов, общий счёт, лимит слотов) — пара это просто портфель из одного
символа. Разница — только в переданных equity0/max_slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from lab.engine.costs import fee_amount, fill_price, funding_payment, slippage_cost
from lab.engine.execution import (
    adjust_position,
    apply_leverage_cap,
    check_stop_take,
    is_short_liquidated,
    passes_min_change,
    target_notional,
)
from lab.engine.portfolio import allocate_slots
from lab.strategies.base import Context, PortfolioStrategy, PositionState

TF_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24}


def is_signal_close(bar_open: pd.Timestamp, tf: str) -> bool:
    """True, если 1h-бар с открытием bar_open закрывает полный бар сигнального ТФ tf.
    Работает через остаток от деления, потому что 00:00 UTC 1970-01-01 (эпоха, от которой
    считает pandas) делится без остатка и на 4 часа, и на 12, и на сутки — совпадать с
    биржевыми границами (00/04/08... UTC) это будет само по себе, без ручных смещений."""
    dur = TF_HOURS[tf]
    close = bar_open + pd.Timedelta(hours=1)
    elapsed = close - close.normalize()
    return elapsed % pd.Timedelta(hours=dur) == pd.Timedelta(0)


def signal_open_for_close(bar_open: pd.Timestamp, tf: str) -> pd.Timestamp:
    """Время открытия сигнального бара, который закрывается вместе с 1h-баром bar_open."""
    dur = TF_HOURS[tf]
    close = bar_open + pd.Timedelta(hours=1)
    return close - pd.Timedelta(hours=dur)


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage_cost: float = 0.0
    funding: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class Order:
    symbol: str
    time: pd.Timestamp
    target_before: float
    target_after: float
    fill_price: float
    delta_qty: float
    fee: float
    slippage_cost: float
    reason: str  # "signal" | "stop" | "take" | "liquidation"


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)


def _close_trade(trade: Trade, t: pd.Timestamp, price: float, realized: float, fee: float, slip: float, reason: str) -> None:
    """gross_pnl считается по РЕАЛЬНЫМ ценам исполнения (то есть уже с учётом
    проскальзывания — оно ценовой эффект, а не отдельно вычитаемая статья, как комиссия).
    slippage_cost хранится отдельно только как диагностика для Cost share
    (01_PROTOCOL.md §8: (комиссии+проскальзывание+фандинг)/валовая прибыль) — его не нужно
    вычитать из net_pnl ещё раз, иначе посчитали бы дважды."""
    trade.exit_time = t
    trade.exit_price = price
    trade.gross_pnl = realized
    trade.fees += fee
    trade.slippage_cost += slip
    trade.net_pnl = trade.gross_pnl - trade.fees + trade.funding
    trade.exit_reason = reason


def run_backtest(
    strategy,
    bars_1h: dict[str, pd.DataFrame],
    signal_bars: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
    tf: str,
    protocol_cfg,
    equity0: float,
    max_slots: int,
    btc_bars: pd.DataFrame | None = None,
    signals_start: pd.Timestamp | None = None,
) -> BacktestResult:
    """
    bars_1h — 1h-бары по каждому символу (для исполнения/стопов/фандинга/equity).
    signal_bars — бары на сигнальном ТФ strategy.tf, УЖЕ после strategy.prepare()
        (с посчитанными индикаторами и колонкой complete).
    funding — история фандинга по каждому символу, индекс funding_time.
    max_slots — для режима «пара» передавайте len(bars_1h) (обычно 1) — конкуренции
        просто не возникнет, т. к. кандидат на слот всегда один.
    signals_start — до этого момента стратегия не получает сигнальных решений вообще
        (позиции гарантированно остаются плоскими), хотя `prepare()`/индикаторы уже
        видят все бары с самого начала. Нужно для walk-forward (04_OPTIMIZATION.md §4):
        test-окно должно начинаться "с нуля" (без унаследованных из train позиций), но
        с прогретыми индикаторами (прогрев — это прошлое, не подглядывание). None (по
        умолчанию) — прежнее поведение, решения генерируются с самого первого бара.
    """
    symbols = list(bars_1h.keys())
    slot_fraction = protocol_cfg.account.slot_fraction
    fee_taker = protocol_cfg.exchange.fee_taker
    slippage = protocol_cfg.exchange.slippage
    min_target_change = protocol_cfg.account.min_target_change

    positions = {s: PositionState(symbol=s) for s in symbols}
    pending: dict[str, object] = {s: None for s in symbols}
    locked = {s: False for s in symbols}
    open_trade: dict[str, Trade | None] = {s: None for s in symbols}
    trades: list[Trade] = []
    orders: list[Order] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []

    cash_equity = equity0
    ctx = Context(params=getattr(strategy, "params", {}), protocol=protocol_cfg, btc_bars=btc_bars, funding=None)
    is_portfolio_strategy = isinstance(strategy, PortfolioStrategy)

    all_times = sorted(set().union(*(df.index for df in bars_1h.values())))

    for h in all_times:
        rows = {s: bars_1h[s].loc[h] for s in symbols if h in bars_1h[s].index}

        # --- шаг 1: ликвидация шорта, затем стопы/тейки ---
        for s, row in rows.items():
            pos = positions[s]
            if not pos.is_open:
                continue

            liquidated = (not pos.is_long) and is_short_liquidated(pos.avg_entry_price, row["high"])
            if liquidated:
                exit_price = fill_price(row["high"], is_buy=True, slippage=slippage)
                reason = "liquidation"
            else:
                sr = check_stop_take(
                    is_long=pos.is_long, open_=row["open"], high=row["high"], low=row["low"],
                    stop_price=pos.stop_price, take_price=pos.take_price, slippage=slippage,
                )
                if not sr.triggered:
                    continue
                exit_price, reason = sr.exit_price, sr.reason

            delta_qty = -pos.qty
            fee = fee_amount(delta_qty * exit_price, fee_taker)
            slip = slippage_cost(delta_qty, exit_price, slippage)
            qty_new, avg_new, realized = adjust_position(pos.qty, pos.avg_entry_price, delta_qty, exit_price)
            cash_equity += realized - fee
            if open_trade[s] is not None:
                _close_trade(open_trade[s], h, exit_price, realized, fee, slip, reason)
                trades.append(open_trade[s])
                open_trade[s] = None
            orders.append(Order(s, h, pos.target, 0.0, exit_price, delta_qty, fee, slip, reason))
            pos.qty, pos.avg_entry_price, pos.bars_in_trade = qty_new, avg_new, 0
            pos.target = 0.0
            pos.stop_price = pos.take_price = pos.time_stop_bars = None
            pending[s] = None
            locked[s] = True  # новый вход не раньше следующего сигнального бара (01_PROTOCOL 5.4)

        # --- шаг 2: фандинг ---
        mark_time = h + pd.Timedelta(hours=1)
        for s, row in rows.items():
            fdf = funding.get(s)
            pos = positions[s]
            if fdf is None or not pos.is_open or mark_time not in fdf.index:
                continue
            rate = fdf.loc[mark_time, "funding_rate"]
            payment = funding_payment(pos.qty, row["close"], rate)
            cash_equity += payment
            if open_trade[s] is not None:
                open_trade[s].funding += payment

        # --- шаг 3: исполнение отложенных ордеров на open(h) ---
        for s, row in rows.items():
            decision = pending[s]
            if decision is None:
                continue
            pending[s] = None
            pos = positions[s]
            desired_notional = target_notional(decision.target, slot_fraction, cash_equity)
            other_abs = sum(abs(positions[o].notional) for o in symbols if o != s)
            capped_notional, _ = apply_leverage_cap(desired_notional, other_abs, cash_equity)

            is_buy = capped_notional > pos.notional
            px = fill_price(row["open"], is_buy=is_buy, slippage=slippage)
            qty_target = capped_notional / px if px else 0.0
            delta_qty = qty_target - pos.qty
            fee = fee_amount(delta_qty * px, fee_taker)
            slip = slippage_cost(delta_qty, px, slippage)
            qty_new, avg_new, realized = adjust_position(pos.qty, pos.avg_entry_price, delta_qty, px)
            cash_equity += realized - fee

            was_flat = pos.qty == 0
            crosses_zero = not was_flat and qty_new != 0 and (qty_new > 0) != (pos.qty > 0)

            if was_flat and qty_new != 0:
                open_trade[s] = Trade(
                    symbol=s, side="long" if qty_new > 0 else "short",
                    entry_time=h, entry_price=px, qty=abs(qty_new), slippage_cost=slip,
                )
            elif crosses_zero:
                # Разворот без прохода через 0 (напр. target +1 -> -1): один ордер
                # одновременно закрывает старую позицию и открывает новую в другую
                # сторону. Это ДВЕ сделки для реестра trades, не продолжение одной -
                # иначе Trades/Win rate/Profit factor занижаются (сделка никогда не
                # "закрывается", пока стратегия не уйдёт в кэш). Комиссию и
                # проскальзывание одного ордера делим пропорционально между
                # закрываемой и открываемой частью по объёму.
                close_frac = abs(pos.qty) / abs(delta_qty)
                open_frac = abs(qty_new) / abs(delta_qty)
                if open_trade[s] is not None:
                    _close_trade(
                        open_trade[s], h, px, realized,
                        fee * close_frac, slip * close_frac, "reversal",
                    )
                    trades.append(open_trade[s])
                open_trade[s] = Trade(
                    symbol=s, side="long" if qty_new > 0 else "short",
                    entry_time=h, entry_price=px, qty=abs(qty_new),
                    fees=fee * open_frac, slippage_cost=slip * open_frac,
                )
            elif qty_new == 0 and open_trade[s] is not None:
                _close_trade(open_trade[s], h, px, realized, fee, slip, "exit_signal")
                trades.append(open_trade[s])
                open_trade[s] = None
            elif open_trade[s] is not None:
                open_trade[s].fees += fee  # доливка/сокращение - та же сделка продолжается
                open_trade[s].slippage_cost += slip

            pos.qty, pos.avg_entry_price = qty_new, avg_new
            pos.target = decision.target
            pos.notional = capped_notional if qty_new != 0 else 0.0
            if qty_new == 0 or crosses_zero:
                pos.bars_in_trade = 0
            pos.stop_price = decision.stop_price
            pos.take_price = decision.take_price
            pos.time_stop_bars = decision.time_stop_bars
            orders.append(Order(s, h, pos.target, decision.target, px, delta_qty, fee, slip, "signal"))

        # --- шаг 4: equity по close(h) ---
        unrealized = sum(
            positions[s].qty * (row["close"] - positions[s].avg_entry_price)
            for s, row in rows.items() if positions[s].is_open
        )
        equity_points.append((h, cash_equity + unrealized))

        # --- шаг 5: сигнальные решения ---
        # PortfolioStrategy (S13) видит все монеты сразу в одном вызове on_bar() - нужно
        # ранжирование топ-k/боттом-k по всему пулу, не решить по одной монете независимо.
        # Остальная часть шага 5 (min_target_change, слоты) - буквально та же логика,
        # что и для обычной BaseStrategy, разница только в том, как получен `decisions`.
        decisions = {}
        if signals_start is not None and h < signals_start:
            pass  # решения не генерируются - позиции гарантированно остаются плоскими
        elif is_portfolio_strategy:
            eligible_rows, eligible_positions = {}, {}
            s_open = signal_open_for_close(h, tf)
            for s in rows:
                if not is_signal_close(h, tf):
                    continue
                sdf = signal_bars[s]
                if s_open not in sdf.index or not bool(sdf.loc[s_open, "complete"]):
                    continue
                pos = positions[s]
                if pos.is_open:
                    pos.bars_in_trade += 1
                if locked[s]:
                    locked[s] = False
                    continue
                eligible_rows[s] = sdf.loc[s_open]
                eligible_positions[s] = pos
            if eligible_rows:
                raw_decisions = strategy.on_bar(s_open, eligible_rows, eligible_positions, ctx)
                for s, decision in raw_decisions.items():
                    pos = positions[s]
                    if passes_min_change(decision.target, pos.target, min_target_change):
                        decisions[s] = decision
                    else:
                        pos.stop_price, pos.take_price = decision.stop_price, decision.take_price
        else:
            for s in rows:
                if not is_signal_close(h, tf):
                    continue
                sdf = signal_bars[s]
                s_open = signal_open_for_close(h, tf)
                if s_open not in sdf.index or not bool(sdf.loc[s_open, "complete"]):
                    continue
                pos = positions[s]
                if pos.is_open:
                    pos.bars_in_trade += 1
                if locked[s]:
                    locked[s] = False
                    continue
                decision = strategy.on_bar(s_open, sdf.loc[s_open], pos, ctx)
                if passes_min_change(decision.target, pos.target, min_target_change):
                    decisions[s] = decision
                else:
                    pos.stop_price, pos.take_price = decision.stop_price, decision.take_price

        open_symbols = {s for s in symbols if positions[s].is_open}
        new_entry_candidates = [s for s, d in decisions.items() if not positions[s].is_open and d.target != 0]
        admitted_new = allocate_slots(open_symbols, new_entry_candidates, max_slots)

        for s, decision in decisions.items():
            if not positions[s].is_open and decision.target != 0 and s not in admitted_new:
                continue  # слота не хватило - сигнал в этом баре не исполняется
            pending[s] = decision

    # Позиции, ещё открытые на конец периода, не "закрытая сделка" (Trades в метриках их
    # не считает), но комиссии/фандинг по ним уже реально потрачены/получены - фиксируем
    # снимком по последней цене, иначе метрики издержек их бы просто потеряли.
    for s in symbols:
        if open_trade[s] is not None and positions[s].is_open:
            pos = positions[s]
            last_close = bars_1h[s]["close"].iloc[-1]
            t = open_trade[s]
            t.exit_time = None
            t.exit_price = last_close
            t.gross_pnl = pos.qty * (last_close - pos.avg_entry_price)
            t.net_pnl = t.gross_pnl - t.fees + t.funding
            t.exit_reason = "open_at_end"
            trades.append(t)

    if equity_points:
        equity = pd.Series(dict(equity_points))
    else:
        # bars_1h был пуст на весь запрошенный диапазон (напр. окно walk-forward
        # раньше начала истории молодой монеты, см. HYPE в JOURNAL/universe.yaml) -
        # pd.Series(dict()) даёт RangeIndex, а не DatetimeIndex, и ломает любой код
        # выше, который считает дневные метрики (resample требует DatetimeIndex).
        equity = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    equity.index.name = "time"
    return BacktestResult(equity=equity, trades=trades, orders=orders)
