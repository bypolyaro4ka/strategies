"""Тесты для src/lab/engine/portfolio.py — Этап 3 (конкуренция за слот)."""

from lab.engine.portfolio import allocate_slots


def test_existing_positions_never_displaced():
    admitted = allocate_slots(open_symbols={"AAA", "BBB"}, candidate_symbols=["CCC"], max_slots=2)
    assert admitted == set()  # свободных слотов нет - CCC не проходит, AAA/BBB не трогаем


def test_new_candidates_admitted_alphabetically_until_slots_run_out():
    admitted = allocate_slots(open_symbols={"AAA"}, candidate_symbols=["ZZZ", "BBB", "CCC"], max_slots=3)
    assert admitted == {"BBB", "CCC"}  # 2 свободных слота, по алфавиту - BBB и CCC, не ZZZ


def test_all_candidates_admitted_when_slots_enough():
    admitted = allocate_slots(open_symbols=set(), candidate_symbols=["BBB", "AAA"], max_slots=5)
    assert admitted == {"AAA", "BBB"}


def test_no_candidates_no_op():
    assert allocate_slots(open_symbols={"AAA"}, candidate_symbols=[], max_slots=5) == set()


def test_full_portfolio_ten_pool_five_slots_scenario():
    """Ровно ситуация, из-за которой правило вообще понадобилось: 10 монет пула,
    5 слотов, 3 уже открыты - остаётся 2 свободных на 7 претендентов."""
    open_symbols = {"BNBUSDT", "XRPUSDT", "SOLUSDT"}
    candidates = ["ZECUSDT", "TRXUSDT", "DOGEUSDT", "ADAUSDT", "BCHUSDT", "LINKUSDT", "HYPEUSDT"]
    admitted = allocate_slots(open_symbols, candidates, max_slots=5)
    assert admitted == {"ADAUSDT", "BCHUSDT"}  # первые два по алфавиту
    assert len(open_symbols | admitted) == 5
