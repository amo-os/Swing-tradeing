"""Indian cash-equity transaction cost model.

Calculates realistic brokerage, statutory taxes (STT, GST, Stamp Duty, SEBI fees),
exchange transaction charges, and slippage for equity delivery swing trading.

Default values are documented based on standard Indian retail cash delivery rates.
These defaults represent realistic research assumptions; actual broker contracts
or rate revisions should be configured explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OrderSide(StrEnum):
    """Direction of an execution order."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed breakdown of costs incurred for a trade execution."""

    side: OrderSide
    shares: int
    nominal_price: float
    effective_price: float
    gross_value: float
    slippage_cost: float
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    total_statutory_charges: float
    total_costs: float
    net_cash_impact: float


@dataclass(frozen=True)
class CostModel:
    """Configurable cost model for Indian cash delivery trading.

    Defaults:
    - Brokerage: flat 20 INR per executed order or 0.05%, whichever is lower.
    - STT (Securities Transaction Tax): 0.1% on delivery turnover (Buy and Sell).
    - Exchange txn fee (NSE): 0.00322% on turnover.
    - SEBI fee: 10 INR per crore (0.0001%).
    - Stamp duty: 0.015% on Buy side delivery only.
    - GST: 18% on (brokerage + exchange txn fee + SEBI fee).
    - Slippage: 0.05% (5 bps) half-spread / market impact.
    """

    brokerage_pct: float = 0.0005
    brokerage_flat_max: float = 20.0
    stt_delivery_pct: float = 0.001
    exchange_charge_pct: float = 0.0000322
    sebi_turnover_pct: float = 0.000001
    stamp_duty_buy_pct: float = 0.00015
    gst_pct: float = 0.18
    slippage_pct: float = 0.0005

    def __post_init__(self) -> None:
        if self.brokerage_pct < 0 or self.brokerage_flat_max < 0:
            raise ValueError("Brokerage rates cannot be negative")
        if self.stt_delivery_pct < 0 or self.stamp_duty_buy_pct < 0:
            raise ValueError("Tax rates cannot be negative")
        if self.slippage_pct < 0:
            raise ValueError("Slippage cannot be negative")

    def calculate_costs(
        self, side: OrderSide | str, shares: int, nominal_price: float
    ) -> CostBreakdown:
        """Calculate exact transaction charges and net cash impact."""
        if shares <= 0:
            raise ValueError("Shares must be positive")
        if nominal_price <= 0:
            raise ValueError("Price must be positive")

        order_side = OrderSide(side.upper()) if isinstance(side, str) else side
        gross_value = shares * nominal_price

        # Slippage adjustment to fill price
        slippage_cost = gross_value * self.slippage_pct
        if order_side == OrderSide.BUY:
            effective_price = nominal_price * (1.0 + self.slippage_pct)
        else:
            effective_price = nominal_price * (1.0 - self.slippage_pct)

        effective_turnover = shares * effective_price

        # Brokerage calculation
        pct_brokerage = gross_value * self.brokerage_pct
        brokerage = min(self.brokerage_flat_max, pct_brokerage)

        # STT (Delivery: 0.1% on both Buy and Sell, rounded to nearest rupee per Indian tax rules)
        stt = float(round(gross_value * self.stt_delivery_pct))

        # Exchange transaction charges
        exchange_charges = gross_value * self.exchange_charge_pct

        # SEBI turnover fees
        sebi_charges = gross_value * self.sebi_turnover_pct

        # Stamp duty (Delivery: 0.015% on Buy side only, rounded to nearest rupee)
        if order_side == OrderSide.BUY:
            stamp_duty = float(round(gross_value * self.stamp_duty_buy_pct))
        else:
            stamp_duty = 0.0

        # GST: 18% on Brokerage + Exchange charges + SEBI charges
        gst_base = brokerage + exchange_charges + sebi_charges
        gst = gst_base * self.gst_pct

        statutory_charges = stt + exchange_charges + sebi_charges + stamp_duty + gst
        total_costs = brokerage + statutory_charges + slippage_cost

        # Net cash flow:
        # BUY: outflow of (shares * effective_price) + brokerage + statutory_charges
        # SELL: inflow of (shares * effective_price) - brokerage - statutory_charges
        statutory_and_brokerage = brokerage + statutory_charges
        if order_side == OrderSide.BUY:
            net_cash_impact = -(effective_turnover + statutory_and_brokerage)
        else:
            net_cash_impact = effective_turnover - statutory_and_brokerage

        return CostBreakdown(
            side=order_side,
            shares=shares,
            nominal_price=nominal_price,
            effective_price=effective_price,
            gross_value=gross_value,
            slippage_cost=slippage_cost,
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            gst=gst,
            total_statutory_charges=statutory_charges,
            total_costs=total_costs,
            net_cash_impact=net_cash_impact,
        )
