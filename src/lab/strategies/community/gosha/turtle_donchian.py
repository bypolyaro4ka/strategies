"""Портирован из s05_eth_turtle_donchian.py (Гоша), 1:1 логика правил входа/выхода,
см. DECLARATION.md рядом. Оригинал — реконструкция публичного 3Commas TradingView-скрипта
"Turtle Strategy": двойной Donchian (канал 1 = 20 баров, канал 2 = тот же Donchian со сдвигом
ещё на 20 баров назад) + 10-барный канал выхода.

Размер позиции (`equity_fraction=0.01` у автора) НЕ портируется - у нас сайзинг задаёт
движок (`slot_fraction` протокола), стратегия отдаёт только target в [-1, 1]."""

from __future__ import annotations

import pandas as pd

from lab.strategies.base import BaseStrategy, Decision

CHANNEL1 = 20
CHANNEL2 = 20
CHANNEL2_OFFSET = 20
EXIT = 10


class GoshaTurtleDonchian(BaseStrategy):
    id = "COMM_GOSHA_TURTLE_DONCHIAN"
    name = "turtle_donchian"
    version = "1.0.0"
    author = "Гоша"
    timeframes = ["4h"]
    direction = "long_short"

    def required_history(self, tf: str) -> int:
        return CHANNEL1 + CHANNEL2_OFFSET + CHANNEL2 + 2

    def prepare(self, bars: pd.DataFrame, ctx) -> pd.DataFrame:
        out = bars.copy()
        h, l, c = out["high"], out["low"], out["close"]
        c1h = h.shift(1).rolling(CHANNEL1).max()
        c1l = l.shift(1).rolling(CHANNEL1).min()
        # канал 2 - тот же Donchian, но построен по данным, сдвинутым ещё на
        # CHANNEL2_OFFSET баров назад (как в оригинале - "смещённый вправо" канал)
        c2h = h.shift(1 + CHANNEL2_OFFSET).rolling(CHANNEL2).max()
        c2l = l.shift(1 + CHANNEL2_OFFSET).rolling(CHANNEL2).min()
        out["exit_high"] = h.shift(1).rolling(EXIT).max()
        out["exit_low"] = l.shift(1).rolling(EXIT).min()
        # пробой засчитывается только на баре, где close ПЕРЕСЕКАЕТ канал 1 (а не всё
        # время, пока close выше него), и подтверждён каналом 2 - как в оригинале
        out["long_break"] = (c > c1h) & (c.shift(1) <= c1h.shift(1)) & (c > c2h)
        out["short_break"] = (c < c1l) & (c.shift(1) >= c1l.shift(1)) & (c < c2l)
        return out

    def on_bar(self, t, row, pos, ctx) -> Decision:
        if pd.isna(row["exit_high"]) or pd.isna(row["long_break"]) or pd.isna(row["short_break"]):
            return Decision(target=0.0, tag="warmup")

        long_break, short_break = bool(row["long_break"]), bool(row["short_break"])
        close = row["close"]
        side = pos.target

        # 10-барный выход проверяется первым (как у автора: `pos=None` в первом блоке).
        # Важно: у автора это НЕ ранний return - следующий блок читает уже обновлённый
        # pos, то есть выход и немедленный вход в другую сторону МОГУТ произойти в одном
        # баре (через ветку `elif pos is None`). Ранний return здесь был бы багом порта -
        # терял бы эту возможность разворота "выход+вход одним баром".
        exited_this_bar = False
        if side > 0 and close < row["exit_low"]:
            side, exited_this_bar = 0.0, True
        elif side < 0 and close > row["exit_high"]:
            side, exited_this_bar = 0.0, True

        if side > 0:
            if short_break:
                return Decision(target=-1.0, tag="opposite_breakout_reverse")
            return Decision(target=side, tag="hold_long")
        if side < 0:
            if long_break:
                return Decision(target=1.0, tag="opposite_breakout_reverse")
            return Decision(target=side, tag="hold_short")

        # side == 0 здесь: либо уже была плоская позиция, либо только что вышли этим же
        # баром - в обоих случаях автор тут же проверяет вход (третья ветка `elif pos is None`)
        if long_break:
            return Decision(target=1.0, tag="entry_long")
        if short_break:
            return Decision(target=-1.0, tag="entry_short")
        return Decision(target=0.0, tag="10_bar_exit_channel" if exited_this_bar else "flat")
