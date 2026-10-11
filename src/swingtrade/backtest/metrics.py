"""Performance and risk metrics calculation for historical backtest results."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from swingtrade.backtest.engine import BacktestResult


@dataclass(frozen=True)
class PerformanceReport:
    """Consolidated performance, cost, and risk metrics for a backtest."""

    initial_capital: float
    final_equity: float
    total_net_profit: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    avg_trade_return_pct: float
    avg_holding_days: float
    market_exposure_pct: float
    turnover_ratio: float
    total_transaction_costs: float
    brokerage_and_statutory_costs: float
    slippage_costs: float
    benchmark_return_pct: float | None = None
    alpha_pct: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert report metrics to a dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_net_profit": self.total_net_profit,
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_loss_ratio": self.win_loss_ratio,
            "avg_trade_return_pct": self.avg_trade_return_pct,
            "avg_holding_days": self.avg_holding_days,
            "market_exposure_pct": self.market_exposure_pct,
            "turnover_ratio": self.turnover_ratio,
            "total_transaction_costs": self.total_transaction_costs,
            "brokerage_and_statutory_costs": self.brokerage_and_statutory_costs,
            "slippage_costs": self.slippage_costs,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha_pct": self.alpha_pct,
        }


def calculate_performance_metrics(
    result: BacktestResult,
    benchmark_series: pd.Series | None = None,
) -> PerformanceReport:
    """Calculate comprehensive metrics from simulation output.

    Handles zero-trade and flat-equity edge cases safely.
    """
    initial_cap = float(result.initial_capital)
    final_eq = float(result.final_equity)
    net_profit = final_eq - initial_cap
    total_return_pct = (net_profit / initial_cap) * 100.0 if initial_cap > 0 else 0.0

    eq_df = result.equity_curve
    total_days = len(eq_df)

    # CAGR calculation
    if total_days > 1:
        # 252 trading days per year convention
        years = total_days / 252.0
        if years > 0 and final_eq > 0 and initial_cap > 0:
            cagr_pct = ((final_eq / initial_cap) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr_pct = 0.0
    else:
        cagr_pct = 0.0

    # Drawdown metrics
    max_dd_pct = 0.0
    max_dd_duration = 0
    if not eq_df.empty and "drawdown_pct" in eq_df.columns:
        min_dd = float(eq_df["drawdown_pct"].min())
        max_dd_pct = abs(min_dd)

        # Drawdown duration in days
        current_dd_dur = 0
        for dd in eq_df["drawdown_pct"]:
            if dd < 0:
                current_dd_dur += 1
                max_dd_duration = max(max_dd_duration, current_dd_dur)
            else:
                current_dd_dur = 0

    # Trade-level statistics
    trades = result.trades
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl_net > 0)
    losing_trades = sum(1 for t in trades if t.pnl_net < 0)
    breakeven_trades = sum(1 for t in trades if t.pnl_net == 0)

    win_rate_pct = (
        (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
    )

    gross_gains = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    gross_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))

    if gross_losses > 0:
        profit_factor = gross_gains / gross_losses
    elif gross_gains > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_win = (
        float(gross_gains / winning_trades) if winning_trades > 0 else 0.0
    )
    avg_loss = (
        float(gross_losses / losing_trades) if losing_trades > 0 else 0.0
    )
    win_loss_ratio = (
        float(avg_win / avg_loss) if avg_loss > 0 else (
            float("inf") if avg_win > 0 else 0.0
        )
    )

    avg_trade_ret = (
        float(sum(t.return_pct for t in trades) / total_trades) * 100.0
        if total_trades > 0
        else 0.0
    )
    avg_holding = (
        float(sum(t.holding_days for t in trades) / total_trades)
        if total_trades > 0
        else 0.0
    )

    # Exposure & Turnover
    if not eq_df.empty:
        days_with_positions = (eq_df["open_positions"] > 0).sum()
        market_exposure_pct = (days_with_positions / len(eq_df)) * 100.0
        avg_equity = float(eq_df["total_equity"].mean())
    else:
        market_exposure_pct = 0.0
        avg_equity = initial_cap

    traded_volume = sum(
        (t.shares * t.nominal_entry_price) + (t.shares * t.nominal_exit_price)
        for t in trades
    )
    turnover_ratio = (
        traded_volume / avg_equity if avg_equity > 0 else 0.0
    )

    # Transaction costs breakdown
    total_costs = sum(t.total_costs for t in trades)
    slippage_costs = sum(
        (abs(t.effective_entry_price - t.nominal_entry_price) * t.shares)
        + (abs(t.effective_exit_price - t.nominal_exit_price) * t.shares)
        for t in trades
    )
    fees_and_taxes = max(0.0, total_costs - slippage_costs)

    # Benchmark comparison (if provided)
    bench_ret_pct: float | None = None
    alpha_pct: float | None = None
    if benchmark_series is not None and len(benchmark_series) > 1:
        first_b = float(benchmark_series.iloc[0])
        last_b = float(benchmark_series.iloc[-1])
        if first_b > 0:
            bench_ret_pct = ((last_b - first_b) / first_b) * 100.0
            alpha_pct = total_return_pct - bench_ret_pct

    # Protect inf / NaN for serialization
    clean_profit_factor = 999.99 if math.isinf(profit_factor) else profit_factor
    clean_wl_ratio = 999.99 if math.isinf(win_loss_ratio) else win_loss_ratio

    return PerformanceReport(
        initial_capital=initial_cap,
        final_equity=final_eq,
        total_net_profit=net_profit,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration_days=max_dd_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=clean_profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=clean_wl_ratio,
        avg_trade_return_pct=avg_trade_ret,
        avg_holding_days=avg_holding,
        market_exposure_pct=market_exposure_pct,
        turnover_ratio=turnover_ratio,
        total_transaction_costs=total_costs,
        brokerage_and_statutory_costs=fees_and_taxes,
        slippage_costs=slippage_costs,
        benchmark_return_pct=bench_ret_pct,
        alpha_pct=alpha_pct,
    )
