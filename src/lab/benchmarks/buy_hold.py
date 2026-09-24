"""Buy & Hold (01_PROTOCOL.md, раздел 7).

Режим «пара»: обычная стратегия target=+1 весь период — прогоняется через run_backtest()
как любая другая. Режим «портфель»: отдельная функция (не через run_backtest) — решение
от 2026-09-24 (см. JOURNAL): весь пул, равный вес, без ребалансировки, вне очереди на слот,
это принципиально другая механика, чем у стратегий с дискретными входами/выходами.
"""

from __future__ import annotations

import pandas as pd

from lab.engine.backtest import BacktestResult, Trade
from lab.engine.costs import fee_amount, fill_price
from lab.strategies.base import BaseStrategy, Decision


class BuyHoldStrategy(BaseStrategy):
    """Для режима «пара»: target=+1 весь период."""
    id = "BENCH_BUY_HOLD"
    name = "buy_hold"
    version = "1.0.0"
    timeframes = ["1d", "4h", "12h", "1h"]
    direction = "long_only"

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        return bars

    def on_bar(self, t, row, pos, ctx) -> Decision:
        return Decision(target=1.0, tag="buy_hold")


def run_buy_hold_portfolio(bars_1h: dict[str, pd.DataFrame], protocol_cfg, equity0: float) -> BacktestResult:
    """Покупает КАЖДУЮ монету пула один раз, на её первом доступном 1h-баре (для HYPE это
    2025-05-30, а не начало периода — короче история, но принцип тот же), весом
    `slot_fraction * max_slots_portfolio / n_pool` от equity0, и держит без ребалансировки
    до конца периода. Не участвует в конкуренции за слот (01_PROTOCOL.md, раздел 7) —
    поэтому не переиспользует run_backtest(), а считается напрямую, но теми же формулами
    издержек (costs.py), что и обычные стратегии — сравнение остаётся честным по издержкам."""
    symbols = list(bars_1h.keys())
    slot_fraction = protocol_cfg.account.slot_fraction
    max_slots = protocol_cfg.account.max_slots_portfolio
    fee_taker = protocol_cfg.exchange.fee_taker
    slippage = protocol_cfg.exchange.slippage
    weight = slot_fraction * max_slots / len(symbols)

    entries: dict[str, tuple[pd.Timestamp, float, float]] = {}
    cash = equity0
    trades: list[Trade] = []

    for s in symbols:
        df = bars_1h[s]
        t0 = df.index[0]
        px = fill_price(df.loc[t0, "open"], is_buy=True, slippage=slippage)
        notional = weight * equity0
        qty = notional / px
        fee = fee_amount(notional, fee_taker)
        cash -= fee
        entries[s] = (t0, px, qty)
        trades.append(Trade(symbol=s, side="long", entry_time=t0, entry_price=px, qty=qty, fees=fee))

    all_times = sorted(set().union(*(df.index for df in bars_1h.values())))
    equity_points = []
    for h in all_times:
        unrealized = sum(
            qty * (bars_1h[s].loc[h, "close"] - px)
            for s, (t0, px, qty) in entries.items()
            if h >= t0 and h in bars_1h[s].index
        )
        equity_points.append((h, cash + unrealized))

    for trade in trades:
        t0, px, qty = entries[trade.symbol]
        last_close = bars_1h[trade.symbol]["close"].iloc[-1]
        trade.exit_price = last_close
        trade.gross_pnl = qty * (last_close - px)
        trade.net_pnl = trade.gross_pnl - trade.fees
        trade.exit_reason = "open_at_end"

    equity = pd.Series(dict(equity_points))
    equity.index.name = "time"
    return BacktestResult(equity=equity, trades=trades, orders=[])
