"""Position sizing and risk management for equity swing trading."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingResult:
    """Result of a position sizing calculation."""

    approved: bool
    shares: int
    risk_amount: float
    allocation_amount: float
    risk_per_share: float
    reason: str


@dataclass(frozen=True)
class PositionSizer:
    """Calculates deterministic order sizes strictly respecting risk limits.

    Defaults:
    - risk_fraction: 0.005 (0.5% of current equity per trade)
    - max_open_positions: 3
    - max_allocation_fraction: 1.0 / max_open_positions (~33.3% of equity)
    """

    risk_fraction: float = 0.005
    max_open_positions: int = 3
    max_allocation_fraction: float = 0.3333333333333333

    def __post_init__(self) -> None:
        if self.risk_fraction <= 0 or self.risk_fraction > 0.1:
            raise ValueError(
                f"risk_fraction must be between 0 and 0.10 (got {self.risk_fraction})"
            )
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")
        if self.max_allocation_fraction <= 0 or self.max_allocation_fraction > 1.0:
            raise ValueError("max_allocation_fraction must be in (0.0, 1.0]")

    def size_position(
        self,
        *,
        symbol: str,
        current_equity: float,
        available_cash: float,
        entry_price: float,
        stop_loss: float,
        open_position_count: int,
    ) -> SizingResult:
        """Compute the allowed number of shares for an entry order.

        Uses only point-in-time portfolio equity and price levels.
        """
        if current_equity <= 0:
            return SizingResult(
                approved=False,
                shares=0,
                risk_amount=0.0,
                allocation_amount=0.0,
                risk_per_share=0.0,
                reason="Non-positive portfolio equity",
            )

        if available_cash <= 0:
            return SizingResult(
                approved=False,
                shares=0,
                risk_amount=0.0,
                allocation_amount=0.0,
                risk_per_share=0.0,
                reason="Insufficient available cash",
            )

        if open_position_count >= self.max_open_positions:
            return SizingResult(
                approved=False,
                shares=0,
                risk_amount=0.0,
                allocation_amount=0.0,
                risk_per_share=0.0,
                reason=(
                    f"Maximum open positions reached ({open_position_count}/"
                    f"{self.max_open_positions})"
                ),
            )

        if entry_price <= 0:
            return SizingResult(
                approved=False,
                shares=0,
                risk_amount=0.0,
                allocation_amount=0.0,
                risk_per_share=0.0,
                reason="Entry price must be positive",
            )

        if stop_loss >= entry_price:
            return SizingResult(
                approved=False,
                shares=0,
                risk_amount=0.0,
                allocation_amount=0.0,
                risk_per_share=0.0,
                reason=(
                    f"Stop loss ({stop_loss:.2f}) must be below entry price "
                    f"({entry_price:.2f}) for long swing trade"
                ),
            )

        risk_per_share = entry_price - stop_loss
        dollar_risk_budget = current_equity * self.risk_fraction

        # Sizing by risk: shares = risk_budget / risk_per_share
        shares_by_risk = math.floor(dollar_risk_budget / risk_per_share)

        # Sizing by capital allocation cap:
        max_position_capital = current_equity * self.max_allocation_fraction
        # Cannot exceed available cash (with slight buffer for costs)
        capital_limit = min(max_position_capital, available_cash * 0.995)
        shares_by_capital = math.floor(capital_limit / entry_price)

        shares = min(shares_by_risk, shares_by_capital)

        if shares < 1:
            return SizingResult(
                approved=False,
                shares=0,
                risk_amount=0.0,
                allocation_amount=0.0,
                risk_per_share=risk_per_share,
                reason=(
                    f"Calculated share count is zero (risk_shares={shares_by_risk}, "
                    f"capital_shares={shares_by_capital})"
                ),
            )

        total_allocation = shares * entry_price
        total_risk = shares * risk_per_share

        return SizingResult(
            approved=True,
            shares=shares,
            risk_amount=total_risk,
            allocation_amount=total_allocation,
            risk_per_share=risk_per_share,
            reason="Position sized successfully",
        )
