"""Unit tests for backtest performance metrics calculation."""

import pandas as pd
import pytest

from swingtrade.backtest.engine import BacktestResult, DailySnapshot, Trade
from swingtrade.backtest.metrics import calculate_performance_metrics


def make_dummy_result(
    initial_capital: float = 1_000_000.0,
    trades: list[Trade] | None = None,
    equity_values: list[float] | None = None,
) -> BacktestResult:
    trades_list = trades or []
    eq_vals = equity_values or [initial_capital, initial_capital]
    dates = pd.date_range("2024-01-01", periods=len(eq_vals), freq="D")

    snapshots = [
        DailySnapshot(
            date=d,
            cash=val if not trades_list else val * 0.5,
            holdings_value=0.0 if not trades_list else val * 0.5,
            total_equity=val,
            open_positions_count=0 if not trades_list else 1,
            daily_return=0.0,
        )
        for d, val in zip(dates, eq_vals, strict=True)
    ]

    eq_df = pd.DataFrame(
        {
            "date": dates,
            "cash": [s.cash for s in snapshots],
            "holdings_value": [s.holdings_value for s in snapshots],
            "total_equity": eq_vals,
            "open_positions": [s.open_positions_count for s in snapshots],
            "daily_return": [0.0] * len(eq_vals),
        }
    )
    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown_pct"] = (
        (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
    ) * 100.0

    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=eq_vals[-1],
        trades=trades_list,
        equity_curve=eq_df,
        daily_snapshots=snapshots,
        approved_symbols=["ABB"],
    )


def test_zero_trades_edge_case_handled_safely() -> None:
    res = make_dummy_result(trades=[])
    metrics = calculate_performance_metrics(res)

    assert metrics.total_trades == 0
    assert metrics.win_rate_pct == 0.0
    assert metrics.profit_factor == 0.0
    assert metrics.avg_win == 0.0
    assert metrics.avg_loss == 0.0
    assert metrics.total_transaction_costs == 0.0
    assert metrics.total_return_pct == 0.0


def test_standard_metrics_calculation() -> None:
    t1 = Trade(
        trade_id=1,
        symbol="ABB",
        entry_date=pd.Timestamp("2024-01-02"),
        exit_date=pd.Timestamp("2024-01-10"),
        shares=100,
        nominal_entry_price=100.0,
        effective_entry_price=100.1,
        nominal_exit_price=120.0,
        effective_exit_price=119.9,
        exit_reason="PROFIT_TARGET",
        entry_costs=20.0,
        exit_costs=20.0,
        total_costs=40.0,
        pnl_gross=2000.0,
        pnl_net=1940.0,
        return_pct=0.194,
        holding_days=8,
        is_closed=True,
    )
    t2 = Trade(
        trade_id=2,
        symbol="ABB",
        entry_date=pd.Timestamp("2024-01-15"),
        exit_date=pd.Timestamp("2024-01-20"),
        shares=100,
        nominal_entry_price=100.0,
        effective_entry_price=100.1,
        nominal_exit_price=90.0,
        effective_exit_price=89.9,
        exit_reason="STOP_LOSS",
        entry_costs=20.0,
        exit_costs=20.0,
        total_costs=40.0,
        pnl_gross=-1000.0,
        pnl_net=-1060.0,
        return_pct=-0.106,
        holding_days=5,
        is_closed=True,
    )

    res = make_dummy_result(
        initial_capital=100_000.0,
        trades=[t1, t2],
        equity_values=[100_000.0, 101_940.0, 100_880.0],
    )

    metrics = calculate_performance_metrics(res)

    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate_pct == 50.0
    # Profit factor: 1940 / 1060 ~ 1.83
    assert metrics.profit_factor == pytest.approx(1940.0 / 1060.0, rel=1e-2)
    assert metrics.total_transaction_costs == 80.0
    assert metrics.avg_holding_days == 6.5
    assert metrics.total_return_pct == pytest.approx(0.88, abs=0.01)


def test_benchmark_comparison_and_alpha() -> None:
    res = make_dummy_result(
        initial_capital=100_000.0,
        trades=[],
        equity_values=[100_000.0, 110_000.0],
    )
    bench = pd.Series([1000.0, 1050.0])  # Benchmark return = +5%

    metrics = calculate_performance_metrics(res, benchmark_series=bench)

    assert metrics.total_return_pct == 10.0
    assert metrics.benchmark_return_pct == 5.0
    assert metrics.alpha_pct == 5.0
