"""Сводная DEV-таблица по ВСЕМ стратегиям (свои 13 + 21 друзей) из reports/registry.csv.

ВАЖНО - это НЕ лидерборд из 06_LEADERBOARD.md. Тот лидерборд:
- ранжирует по Sharpe на HOLDOUT (2026), не на dev;
- требует основной вариант из config/prereg/<ID>.yaml (заморозка параметров ДО holdout);
- holdout всё ещё закрыт (CLAUDE.md, правило 1) - его нельзя трогать до Этапа 8 и явного
  подтверждения пользователя.

Здесь - честная, но промежуточная сверка: по последнему логированному в registry.csv
прогону stage=dev, mode=portfolio, variant=V0 на каждую стратегию (без выбора "лучшего"
из нескольких прогонов - см. count_by ниже, берём последний по timestamp, не max по Sharpe,
иначе это было бы подглядыванием в собственный dev-результат при отборе строки).

Запуск: python scripts/make_dev_leaderboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "reports" / "registry.csv"
OUT_PATH = PROJECT_ROOT / "reports" / "dev" / "leaderboard_dev.md"


def main():
    if not REGISTRY_PATH.exists():
        print(f"Нет {REGISTRY_PATH} - сначала прогоните бэктесты.")
        return

    df = pd.read_csv(REGISTRY_PATH, dtype=str)
    df = df[(df["stage"] == "dev") & (df["mode"] == "portfolio") & (df["variant"] == "V0") & (df["debug"] != "True")]
    if df.empty:
        print("В registry.csv нет ни одного dev/portfolio/V0 прогона.")
        return

    # последний по времени прогон на каждую strategy_id - не max(sharpe), см. докстринг
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").groupby("strategy_id", as_index=False).last()

    numeric_cols = ["sharpe", "total_return", "maxdd", "trades", "cost_share"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["warn"] = df["strategy_id"].str.startswith("COMM_").map({True: "⚠ видела holdout", False: ""})
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    lines = [
        "# DEV-сводка по всем стратегиям (не финальный лидерборд!)\n",
        f"Период: dev 2023-01-01..2025-12-31, портфельный режим, вариант V0 (дефолтные/заявленные "
        "параметры, без оптимизации). Источник — последний прогон каждой стратегии в "
        "`reports/registry.csv`.\n",
        "**Это не лидерборд из `06_LEADERBOARD.md`.** Тот считается по Sharpe на holdout "
        "(2026), в основном варианте из `config/prereg/`, после заморозки параметров. "
        "Holdout ещё закрыт (CLAUDE.md, правило 1) и не будет открыт этим скриптом. "
        "Здесь — просто честная сверка \"что получилось на dev у всех сразу\", по-хорошему "
        "промежуточный шаг, не финал: у своих 13 primary-вариант (V0/V1/V2) по walk-forward "
        "ещё не для всех решён (Этап 7 в процессе), а у всех 21 стратегии друзей — заведомая "
        "пометка ⚠ видела holdout (см. DECLARATION.md каждого автора), их честная проверка "
        "— только форвард-тест (§6), не dev и не holdout число.\n",
        "| Место | ID | Автор | Sharpe | Return | MaxDD | Trades | Cost share | Пометка |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in df.iterrows():
        lines.append(
            f"| {i + 1} | {row['strategy_id']} | {row.get('author', '') or '—'} | "
            f"{row['sharpe']:.2f} | {row['total_return']*100:.1f}% | {row['maxdd']*100:.1f}% | "
            f"{int(row['trades']) if row['trades'] == row['trades'] else '—'} | "
            f"{row['cost_share']*100:.1f}% | {row['warn']} |"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Готово: {OUT_PATH} ({len(df)} строк)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
