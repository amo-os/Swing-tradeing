"""Base abstractions for swing-trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class SignalDirection(StrEnum):
    """Direction of a trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    """A trading signal generated at the close of a trading session."""

    symbol: str
    date: pd.Timestamp
    direction: SignalDirection
    stop_loss: float | None = None
    profit_target: float | None = None
    reason: str = ""


class Strategy(ABC):
    """Abstract base class for swing-trading strategies.

    Strategies receive strictly historical data available up to and including
    the evaluation bar.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def generate_signals(
        self,
        current_date: pd.Timestamp,
        historical_data: dict[str, pd.DataFrame],
        open_positions: dict[str, object],
    ) -> list[Signal]:
        """Generate trading signals at session close for next-session execution.

        Parameters
        ----------
        current_date : pd.Timestamp
            The date of the current daily session close.
        historical_data : dict[str, pd.DataFrame]
            Point-in-time data for approved symbols sliced strictly up to current_date.
        open_positions : dict[str, object]
            Currently active open positions in the portfolio.
        """
        ...
