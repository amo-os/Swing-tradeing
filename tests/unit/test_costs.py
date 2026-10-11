"""Unit tests for transaction cost and statutory tax calculations."""

import pytest

from swingtrade.backtest.costs import CostModel, OrderSide


def test_default_cost_model_rates() -> None:
    model = CostModel()
    assert model.stt_delivery_pct == 0.001
    assert model.stamp_duty_buy_pct == 0.00015
    assert model.gst_pct == 0.18
    assert model.slippage_pct == 0.0005


def test_buy_costs_calculation() -> None:
    model = CostModel(
        brokerage_pct=0.0005,
        brokerage_flat_max=20.0,
        stt_delivery_pct=0.001,
        exchange_charge_pct=0.00003,
        sebi_turnover_pct=0.000001,
        stamp_duty_buy_pct=0.00015,
        gst_pct=0.18,
        slippage_pct=0.001,
    )
    # 100 shares @ nominal 1000 = 100,000 INR
    res = model.calculate_costs(OrderSide.BUY, 100, 1000.0)

    assert res.side == OrderSide.BUY
    assert res.shares == 100
    assert res.nominal_price == 1000.0
    # Buy effective price is higher due to slippage
    assert res.effective_price == 1000.0 * 1.001
    assert res.slippage_cost == 100.0
    # Brokerage capped at 20 INR
    assert res.brokerage == 20.0
    # STT = 0.1% of 100,000 = 100 INR
    assert res.stt == 100.0
    # Stamp duty = 0.015% of 100,000 = 15 INR on BUY
    assert res.stamp_duty == 15.0
    # GST on brokerage + exchange + sebi
    assert res.gst > 0
    # Total costs include brokerage, statutory charges, and slippage
    assert res.total_costs == (
        res.brokerage + res.total_statutory_charges + res.slippage_cost
    )
    # Net cash impact for BUY is negative (cash outflow)
    assert res.net_cash_impact < 0
    assert abs(res.net_cash_impact) > res.gross_value


def test_sell_costs_calculation_has_no_stamp_duty() -> None:
    model = CostModel()
    # 100 shares @ nominal 1000 = 100,000 INR
    res = model.calculate_costs(OrderSide.SELL, 100, 1000.0)

    assert res.side == OrderSide.SELL
    # Stamp duty is 0 on sell side for delivery
    assert res.stamp_duty == 0.0
    # STT is paid on sell side
    assert res.stt > 0
    # Sell effective price is lower due to slippage
    assert res.effective_price < res.nominal_price
    # Net cash impact is positive (cash inflow) but reduced by fees
    assert res.net_cash_impact > 0
    assert res.net_cash_impact < (100 * res.effective_price)


def test_invalid_parameters_raise_value_error() -> None:
    model = CostModel()
    with pytest.raises(ValueError, match="Shares must be positive"):
        model.calculate_costs(OrderSide.BUY, 0, 100.0)

    with pytest.raises(ValueError, match="Price must be positive"):
        model.calculate_costs(OrderSide.BUY, 10, -50.0)

    with pytest.raises(ValueError, match="cannot be negative"):
        CostModel(slippage_pct=-0.01)
