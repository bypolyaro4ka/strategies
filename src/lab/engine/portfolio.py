"""Конкуренция за слот в портфельном режиме (01_PROTOCOL.md, 5.5,
`slot_contention_rule: existing_first_then_alpha`, см. JOURNAL 2026-09-24)."""

from __future__ import annotations


def allocate_slots(open_symbols: set[str], candidate_symbols: list[str], max_slots: int) -> set[str]:
    """open_symbols — монеты, у которых уже есть открытая позиция (сохраняют слот).
    candidate_symbols — монеты, которые на этом баре хотят открыть НОВУЮ позицию
    (сейчас были плоскими). Возвращает подмножество candidate_symbols, которому хватило
    свободных слотов — остальные сигналы в этом баре просто не исполняются.

    Правило (01_PROTOCOL.md, 5.5): открытые позиции сохраняют слот; свободные слоты
    отдаются новым сигналам по алфавиту символа Binance — детерминированно, без учёта
    силы сигнала (защита от скрытой оптимизации задним числом)."""
    free_slots = max(max_slots - len(open_symbols), 0)
    if free_slots <= 0:
        return set()
    admitted = sorted(s for s in candidate_symbols if s not in open_symbols)[:free_slots]
    return set(admitted)
