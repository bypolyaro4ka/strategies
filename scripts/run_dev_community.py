"""Первый честный dev-прогон стратегий друзей (06_LEADERBOARD.md §5.3, п.2): "Прогон на
dev (2023-2025)... в заявленной конфигурации, портфельный режим" - без walk-forward
оптимизации (Этап 7 - только для наших 13, см. JOURNAL). Один прогон на дефолтных
параметрах автора (не отобранных постфактум) на каждую стратегию, как и было решено
в DECLARATION.md/SPEC.md каждого автора.

holdout всё ещё закрыт (CLAUDE.md, правило 1) - здесь ТОЛЬКО dev-часть. Прогон holdout
для этих стратегий будет на Этапе 8 вместе со всеми остальными, но по п.3 §5.3 их
единственная честная проверка - форвард-тест (все 4 автора отмечены ⚠ видела holdout
в своих DECLARATION.md), поэтому даже после разморозки их holdout-число НЕ идёт в
основной лидерборд.

Запуск: python scripts/run_dev_community.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from lab.benchmarks.buy_hold import run_buy_hold_portfolio
from lab.benchmarks.flat import FlatStrategy
from lab.config import load_protocol
from lab.data.loader import load_bars, load_funding
from lab.engine.backtest import run_backtest
from lab.metrics import performance as perf
from lab.registry import log_run
from lab.report.make_report import make_report
from lab.strategies.base import Context
from lab.strategies.community.friend4.strategies import (
    Friend4BbRegimeReversion,
    Friend4BbSqueezeBreakout,
    Friend4EmaAdxTrend,
    Friend4IchimokuRsi,
    Friend4RsiPullback,
    Friend4Supertrend,
)
from lab.strategies.community.gosha.avax_ema_cross_short import GoshaAvaxEmaCrossShort
from lab.strategies.community.gosha.rsi_dca import GoshaRsiDca
from lab.strategies.community.gosha.turtle_donchian import GoshaTurtleDonchian
from lab.strategies.community.kolya.strategies import (
    KolyaBreakout,
    KolyaEmaPullback,
    KolyaFundingContrarian,
    KolyaKeltnerSqueeze,
    KolyaLiquidationSweep,
    KolyaMacdTrend,
    KolyaMeanReversion,
    KolyaRsiTrendContinuation,
    KolyaVolatilityBreakout,
)
from lab.strategies.community.slava.c2_local_compression import SlavaC2LocalCompression
from lab.strategies.community.slava.eth_strong_breakout import SlavaEthStrongBreakout
from lab.strategies.community.slava.sol_frozen_v2 import SlavaSolFrozenV2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_START, DEV_END = "2023-01-01", "2025-12-31"
TF_HOURS = {"1h": 1, "4h": 4, "12h": 12, "1d": 24}

# KolyaFundingContrarian - единственная стратегия здесь, которой нужна колонка
# funding_rate прямо в bars (не ctx.funding, как у S12, см. её докстринг) - её
# конструктор передаётся отдельно, остальные 20 - обычные BaseStrategy.
STRATEGIES = [
    GoshaTurtleDonchian(), GoshaAvaxEmaCrossShort(), GoshaRsiDca(),
    SlavaSolFrozenV2(), SlavaEthStrongBreakout(), SlavaC2LocalCompression(),
    KolyaBreakout(), KolyaMeanReversion(), KolyaEmaPullback(), KolyaVolatilityBreakout(),
    KolyaFundingContrarian(), KolyaLiquidationSweep(), KolyaKeltnerSqueeze(),
    KolyaRsiTrendContinuation(), KolyaMacdTrend(),
    Friend4Supertrend(), Friend4EmaAdxTrend(), Friend4BbSqueezeBreakout(),
    Friend4RsiPullback(), Friend4BbRegimeReversion(), Friend4IchimokuRsi(),
]
NEEDS_FUNDING_IN_BARS = {"COMM_KOLYA_FUNDING_CONTRARIAN"}


def load_universe():
    with (PROJECT_ROOT / "config" / "universe.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [c["binance_symbol"] for c in cfg["pool"]]


def merge_funding_into_bars(bars_tf: pd.DataFrame, funding: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Причинный мёрдж funding_rate в колонку bars (KolyaFundingContrarian читает её из
    bars, а не из ctx.funding, в отличие от S12) - тот же merge_asof(direction="backward",
    allow_exact_matches=False), что и в s12_funding_contrarian.py, чтобы отметка на
    закрытии бара не использовалась на этом же баре."""
    out = bars_tf.copy()
    if funding is None or funding.empty:
        out["funding_rate"] = float("nan")
        return out
    bar_hours = TF_HOURS[tf]
    close_times = out.index + pd.Timedelta(hours=bar_hours)
    left = pd.DataFrame(index=pd.DatetimeIndex(close_times).as_unit("us"))
    right = funding[["funding_rate"]].set_axis(funding.index.as_unit("us"))
    merged = pd.merge_asof(left, right, left_index=True, right_index=True,
                            direction="backward", allow_exact_matches=False)
    out["funding_rate"] = merged["funding_rate"].to_numpy()
    return out


def load_symbol_data(symbol: str, tf: str):
    bars_1h = load_bars(symbol, "1h", "2022-01-01", DEV_END)
    bars_tf = load_bars(symbol, tf, "2022-01-01", DEV_END)
    funding = load_funding(symbol, "2022-01-01", DEV_END)
    return bars_1h, bars_tf, funding


def run_one(strategy, symbols, cfg):
    tf = strategy.timeframes[0]
    warmup = strategy.required_history(tf)
    print(f"\n=== {strategy.id} ({strategy.author}/{strategy.name}), tf={tf}, warmup={warmup} ===")

    bars_1h_by_symbol, bars_tf_by_symbol, funding_by_symbol, prepared_by_symbol = {}, {}, {}, {}
    for s in symbols:
        try:
            b1h, btf, fnd = load_symbol_data(s, tf)
        except FileNotFoundError:
            print(f"  {s}: нет данных, пропуск")
            continue
        if len(btf) < warmup + 5:
            print(f"  {s}: истории мало ({len(btf)} баров), пропуск")
            continue
        if strategy.id in NEEDS_FUNDING_IN_BARS:
            btf = merge_funding_into_bars(btf, fnd, tf)
        bars_1h_by_symbol[s] = b1h
        bars_tf_by_symbol[s] = btf
        funding_by_symbol[s] = fnd
        ctx = Context(params=strategy.params, protocol=cfg)
        prepared_by_symbol[s] = strategy.prepare(btf, ctx)

    if not bars_1h_by_symbol:
        print("  нет ни одной монеты с данными, пропуск стратегии")
        return None, [], None

    dev_start_ts = pd.Timestamp(DEV_START, tz="UTC")
    max_slots = cfg.account.max_slots_portfolio

    portfolio_result = run_backtest(
        strategy, bars_1h_by_symbol, prepared_by_symbol, funding_by_symbol, tf, cfg, 10_000, max_slots,
    )
    portfolio_result.equity = portfolio_result.equity.loc[dev_start_ts:]

    flat_result = run_backtest(
        FlatStrategy(), bars_1h_by_symbol, bars_tf_by_symbol, funding_by_symbol, tf, cfg, 10_000, max_slots,
    )
    flat_result.equity = flat_result.equity.loc[dev_start_ts:]
    bh_result = run_buy_hold_portfolio(bars_1h_by_symbol, cfg, 10_000)
    bh_result.equity = bh_result.equity.loc[dev_start_ts:]

    out_dir = PROJECT_ROOT / "reports" / "dev" / strategy.id
    report_path = make_report(
        title=f"{strategy.id} — {strategy.author}/{strategy.name} "
              f"(V0 заявленная конфигурация, портфель, dev {DEV_START}..{DEV_END}, ⚠ видела holdout)",
        strategy_result=portfolio_result,
        protocol_cfg=cfg,
        out_dir=out_dir,
        benchmark_results={"Flat": flat_result, "Buy & Hold": bh_result},
        random_sharpes=None,
        n_trials=1,
    )

    metrics = perf.compute_all(portfolio_result.equity, portfolio_result.trades, cfg.metrics.annualization_days)
    log_run(
        {
            "run_id": str(uuid.uuid4())[:8], "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
            "strategy_id": strategy.id, "strategy_version": strategy.version, "variant": "V0",
            "mode": "portfolio", "tf": tf, "symbols": ";".join(bars_1h_by_symbol.keys()),
            "period_start": DEV_START, "period_end": DEV_END, "stage": "dev", "debug": "False",
            "author": strategy.author, "artifacts_path": str(out_dir.relative_to(PROJECT_ROOT)),
        },
        metrics,
    )
    print(f"  портфель: Sharpe={metrics['sharpe']:.2f} return={metrics['total_return']*100:.1f}% "
          f"trades={metrics['trades']} -> {report_path}")

    pair_rows = []
    for s in bars_1h_by_symbol:
        strat_pair = type(strategy)(params=strategy.params)
        ctx = Context(params=strat_pair.params, protocol=cfg)
        prepared = strat_pair.prepare(bars_tf_by_symbol[s], ctx)
        r = run_backtest(strat_pair, {s: bars_1h_by_symbol[s]}, {s: prepared}, {s: funding_by_symbol[s]}, tf, cfg, 10_000, 1)
        r.equity = r.equity.loc[dev_start_ts:]
        m = perf.compute_all(r.equity, r.trades, cfg.metrics.annualization_days)
        pair_rows.append((s, m))
        log_run(
            {
                "run_id": str(uuid.uuid4())[:8], "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
                "strategy_id": strategy.id, "strategy_version": strategy.version, "variant": "V0",
                "mode": "pair", "tf": tf, "symbols": s, "period_start": DEV_START, "period_end": DEV_END,
                "stage": "dev", "debug": "False", "author": strategy.author,
            },
            m,
        )
    print(f"  пара: {len(pair_rows)} монет прогнано")
    return metrics, pair_rows, report_path


def main():
    cfg = load_protocol()
    symbols = load_universe()
    print(f"Монеты: {symbols}")

    summary = []
    t0 = time.time()
    for strategy in STRATEGIES:
        portfolio_metrics, pair_rows, report_path = run_one(strategy, symbols, cfg)
        if portfolio_metrics is None:
            continue
        summary.append((strategy, portfolio_metrics, pair_rows, report_path))

    lines = [
        "# Стратегии друзей — V0 (заявленная конфигурация) на dev 2023-01-01..2025-12-31\n",
        "**Все строки ⚠ видела holdout** (см. DECLARATION.md каждого автора) — по "
        "06_LEADERBOARD.md §5.3/§6 их честная проверка только форвард-тест, в основной "
        "лидерборд эти числа не идут, здесь только dev-диагностика.\n",
        "Режим «портфель» (основной) + «пара» по каждой монете (диагностика). "
        "Один прогон на дефолтах автора, без walk-forward оптимизации (это НЕ Этап 7).\n",
    ]
    lines.append("## Портфель (V0, дефолты автора)\n")
    lines.append("| ID | Автор | Sharpe | Return | MaxDD | Trades | Cost share | Отчёт |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for strategy, m, _, report_path in summary:
        rel = report_path.relative_to(PROJECT_ROOT / "reports" / "dev")
        lines.append(
            f"| {strategy.id} | {strategy.author} | {m['sharpe']:.2f} | {m['total_return']*100:.1f}% | "
            f"{m['maxdd']*100:.1f}% | {m['trades']} | {m['cost_share']*100:.1f}% | [{rel}]({rel}) |"
        )

    lines.append("\n## Пара — Sharpe по монетам\n")
    lines.append("| ID | " + " | ".join(symbols) + " |")
    lines.append("|---|" + "---|" * len(symbols))
    for strategy, _, pair_rows, _ in summary:
        by_symbol = dict(pair_rows)
        cells = [f"{by_symbol[s]['sharpe']:.2f}" if s in by_symbol and by_symbol[s]['sharpe'] == by_symbol[s]['sharpe'] else "—" for s in symbols]
        lines.append(f"| {strategy.id} | " + " | ".join(cells) + " |")

    out_path = PROJECT_ROOT / "reports" / "dev" / "community_batch.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nГотово за {time.time()-t0:.0f}с. Сводка: {out_path}")


if __name__ == "__main__":
    main()
