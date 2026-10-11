"""Strategy interfaces and implementations."""

from swingtrade.strategy.base import Signal, SignalDirection, Strategy
from swingtrade.strategy.baseline import BaselineSwingStrategy
from swingtrade.strategy.eligibility import check_eligibility, load_quarantined_symbols
from swingtrade.strategy.indicators import calculate_indicators

__all__ = [
    "BaselineSwingStrategy",
    "Signal",
    "SignalDirection",
    "Strategy",
    "calculate_indicators",
    "check_eligibility",
    "load_quarantined_symbols",
]
