"""Run historical backtest on Phase-A approved NIFTY 100 universe.

Strictly excludes unapproved instruments (SIEMENS, TMPV, pending anomaly stocks,
and structurally ineligible stocks).
Evaluates baseline strategy across:
- In-Sample period: 2024-01-01 to 2024-12-31
- Out-of-Sample period: 2025-01-01 to 2025-12-31
- Full Historical period: 2024-01-01 to 2025-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swingtrade.backtest.approval_gate import evaluate_backtest_universe
from swingtrade.backtest.costs import CostModel
from swingtrade.backtest.engine import BacktestEngine
from swingtrade.backtest.metrics import calculate_performance_metrics
from swingtrade.risk.position_sizer import PositionSizer
from swingtrade.strategy.baseline import BaselineSwingStrategy
from swingtrade.strategy.indicators import calculate_indicators

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "data_eligibility.toml"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "nifty100_mapped.csv"
DEFAULT_EXCEPTIONS = ROOT / "data" / "raw" / "data_quality_exceptions.csv"
DEFAULT_ANOMALY_REVIEWS = ROOT / "reports" / "anomaly_review_notes.csv"
DEFAULT_REPORT_DIR = ROOT / "reports"


def run_research_backtest(
    *,
    config_path: Path = DEFAULT_CONFIG,
    universe_path: Path = DEFAULT_UNIVERSE,
    exceptions_path: Path = DEFAULT_EXCEPTIONS,
    anomaly_reviews_path: Path = DEFAULT_ANOMALY_REVIEWS,
    report_dir: Path = DEFAULT_REPORT_DIR,
    initial_capital: float = 1_000_000.0,
) -> dict[str, object]:
    """Execute end-to-end historical backtest with strict universe gating."""
    print("=" * 70)
    print("PHASE A: EVALUATING BACKTEST UNIVERSE APPROVAL GATE")
    print("=" * 70)

    gate_report = evaluate_backtest_universe(
        config_path=config_path,
        universe_path=universe_path,
        exceptions_path=exceptions_path,
        anomaly_reviews_path=anomaly_reviews_path,
        project_root=ROOT,
    )

    approved = gate_report.approved_symbols
    rejected = gate_report.rejected_symbols

    print(f"Total Universe Members: {len(gate_report.decisions)}")
    print(f"Approved for Backtest:  {len(approved)}")
    print(f"Barred / Excluded:      {len(rejected)}")
    print(f"Barred constituents:    {', '.join(rejected)}\n")

    # Load dataset directory from config
    dataset_dir = ROOT / "data" / "processed_2024_2025_retry"
    universe_df = pd.read_csv(universe_path)
    sym_to_key = dict(zip(universe_df["symbol"], universe_df["instrument_key"], strict=False))

    print("Loading historical candles for approved symbols...")
    dataset: dict[str, pd.DataFrame] = {}
    benchmark_components: list[pd.Series] = []

    for sym in approved:
        key = sym_to_key[sym]
        encoded = key.replace("|", "_")
        matches = list(dataset_dir.glob(f"{encoded}_*.csv"))
        if not matches:
            continue
        raw_df = pd.read_csv(matches[0])
        # Precompute indicators with point-in-time technical logic
        ind_df = calculate_indicators(raw_df)
        dataset[sym] = ind_df
        # Benchmark series: daily returns of each approved constituent
        bench_df = ind_df.set_index(pd.to_datetime(ind_df["timestamp"]))["close"]
        benchmark_components.append(bench_df.pct_change().fillna(0.0))

    print(f"Loaded and validated {len(dataset)} approved datasets.\n")

    # Calculate equal-weighted benchmark cumulative return curve
    bench_returns = pd.concat(benchmark_components, axis=1).mean(axis=1)
    bench_cumulative = (1.0 + bench_returns).cumprod() * initial_capital

    strategy = BaselineSwingStrategy()
    cost_model = CostModel()
    position_sizer = PositionSizer(risk_fraction=0.005, max_open_positions=3)
    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_model=cost_model,
        position_sizer=position_sizer,
    )

    periods = [
        ("In-Sample (2024)", "2024-01-01", "2024-12-31"),
        ("Out-of-Sample (2025)", "2025-01-01", "2025-12-31"),
        ("Full Historical (2024-2025)", "2024-01-01", "2025-12-31"),
    ]

    results_summary: list[dict[str, object]] = []

    print("=" * 70)
    print("PHASE B & C: EXECUTING HISTORICAL SIMULATION ACROSS PERIODS")
    print("=" * 70)

    for period_name, start_d, end_d in periods:
        print(f"\n--- Running: {period_name} [{start_d} to {end_d}] ---")
        s_bench = pd.to_datetime(start_d).tz_localize("Asia/Kolkata")
        e_bench = pd.to_datetime(end_d).tz_localize("Asia/Kolkata")
        bench_slice = bench_cumulative.loc[
            (bench_cumulative.index >= s_bench) & (bench_cumulative.index <= e_bench)
        ] if not bench_cumulative.empty else None

        res = engine.run(
            strategy=strategy,
            dataset=dataset,
            approved_universe=approved,
            start_date=start_d,
            end_date=end_d,
        )

        metrics = calculate_performance_metrics(
            res,
            benchmark_series=bench_slice if bench_slice is not None and not bench_slice.empty else None,
        )

        metrics_dict = metrics.to_dict()
        metrics_dict["period"] = period_name
        results_summary.append(metrics_dict)

        print(f"  Final Equity:           ₹{metrics.final_equity:,.2f}")
        print(f"  Net Profit:             ₹{metrics.total_net_profit:,.2f} ({metrics.total_return_pct:+.2f}%)")
        print(f"  CAGR:                   {metrics.cagr_pct:.2f}%")
        print(f"  Max Drawdown:           {metrics.max_drawdown_pct:.2f}%")
        print(f"  Total Trades:           {metrics.total_trades}")
        print(f"  Win Rate:               {metrics.win_rate_pct:.1f}%")
        print(f"  Profit Factor:          {metrics.profit_factor:.2f}")
        print(f"  Avg Win / Avg Loss:     ₹{metrics.avg_win:,.2f} / ₹{metrics.avg_loss:,.2f} (Ratio: {metrics.win_loss_ratio:.2f})")
        print(f"  Market Exposure:        {metrics.market_exposure_pct:.1f}%")
        print(f"  Portfolio Turnover:     {metrics.turnover_ratio:.2f}x")
        print(f"  Total Transaction Costs: ₹{metrics.total_transaction_costs:,.2f}")
        if metrics.benchmark_return_pct is not None:
            print(f"  Benchmark Return:       {metrics.benchmark_return_pct:+.2f}% (Alpha: {metrics.alpha_pct:+.2f}%)")

        # Save detailed trade log and equity curve for full period
        if period_name.startswith("Full"):
            report_dir.mkdir(parents=True, exist_ok=True)
            trade_rows = [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": t.exit_date.strftime("%Y-%m-%d"),
                    "shares": t.shares,
                    "nominal_entry_price": round(t.nominal_entry_price, 2),
                    "effective_entry_price": round(t.effective_entry_price, 2),
                    "nominal_exit_price": round(t.nominal_exit_price, 2),
                    "effective_exit_price": round(t.effective_exit_price, 2),
                    "exit_reason": t.exit_reason,
                    "entry_costs": round(t.entry_costs, 2),
                    "exit_costs": round(t.exit_costs, 2),
                    "total_costs": round(t.total_costs, 2),
                    "pnl_gross": round(t.pnl_gross, 2),
                    "pnl_net": round(t.pnl_net, 2),
                    "return_pct": round(t.return_pct * 100.0, 2),
                    "holding_days": t.holding_days,
                }
                for t in res.trades
            ]
            trades_df = pd.DataFrame(trade_rows)
            trades_df.to_csv(report_dir / "research_trades_2024_2025.csv", index=False)
            res.equity_curve.to_csv(report_dir / "research_equity_curve_2024_2025.csv", index=False)

    summary_df = pd.DataFrame(results_summary)
    cols = ["period", "total_return_pct", "cagr_pct", "max_drawdown_pct", "total_trades", "win_rate_pct", "profit_factor", "total_transaction_costs"]
    available_cols = [c for c in cols if c in summary_df.columns]
    summary_df[available_cols].to_csv(report_dir / "research_performance_summary_2024_2025.csv", index=False)
    print("\n" + "=" * 70)
    print("Research artifacts saved successfully to reports/")
    print("=" * 70)

    return {"summary": results_summary, "approved_count": len(approved), "rejected_count": len(rejected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    parser.add_argument("--anomaly-reviews", type=Path, default=DEFAULT_ANOMALY_REVIEWS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    run_research_backtest(
        config_path=args.config,
        universe_path=args.universe,
        exceptions_path=args.exceptions,
        anomaly_reviews_path=args.anomaly_reviews,
        report_dir=args.report_dir,
        initial_capital=args.capital,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
