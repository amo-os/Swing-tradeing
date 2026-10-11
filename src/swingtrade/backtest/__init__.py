"""Backtest framework, universe gating, execution engine, and metrics."""

from swingtrade.backtest.approval_gate import (
    AnomalyReviewStatus,
    ApprovalDecision,
    UniverseApprovalReport,
    evaluate_backtest_universe,
    evaluate_symbol_approval,
    load_gate_config,
    load_review_records,
    load_universe_records,
)
from swingtrade.backtest.costs import (
    CostBreakdown,
    CostModel,
    OrderSide,
)
from swingtrade.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    DailySnapshot,
    ExitReason,
    Position,
    Trade,
)
from swingtrade.backtest.metrics import (
    PerformanceReport,
    calculate_performance_metrics,
)

__all__ = [
    "AnomalyReviewStatus",
    "ApprovalDecision",
    "BacktestEngine",
    "BacktestResult",
    "CostBreakdown",
    "CostModel",
    "DailySnapshot",
    "ExitReason",
    "OrderSide",
    "PerformanceReport",
    "Position",
    "Trade",
    "UniverseApprovalReport",
    "calculate_performance_metrics",
    "evaluate_backtest_universe",
    "evaluate_symbol_approval",
    "load_gate_config",
    "load_review_records",
    "load_universe_records",
]
