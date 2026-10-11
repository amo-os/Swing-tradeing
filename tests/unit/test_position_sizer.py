"""Unit tests for position sizing and portfolio risk management."""

import pytest

from swingtrade.risk.position_sizer import PositionSizer


def test_default_position_sizer_parameters() -> None:
    sizer = PositionSizer()
    assert sizer.risk_fraction == 0.005
    assert sizer.max_open_positions == 3


def test_sizes_position_strictly_by_risk_fraction() -> None:
    # 1,000,000 equity, risk_fraction = 0.005 -> risk budget = 5,000 INR
    sizer = PositionSizer(risk_fraction=0.005, max_open_positions=3)

    # Entry = 500, Stop = 450 -> risk per share = 50 INR
    # Expected shares = floor(5000 / 50) = 100 shares
    # Allocation = 100 * 500 = 50,000 INR (well within capital limit 333,333)
    res = sizer.size_position(
        symbol="ABB",
        current_equity=1_000_000.0,
        available_cash=800_000.0,
        entry_price=500.0,
        stop_loss=450.0,
        open_position_count=0,
    )

    assert res.approved
    assert res.shares == 100
    assert res.risk_amount == 5_000.0
    assert res.allocation_amount == 50_000.0
    assert res.risk_per_share == 50.0


def test_caps_size_by_max_allocation_limit() -> None:
    # 100,000 equity, max_allocation = 1/3 ~ 33,333 INR
    # Risk budget = 500 INR. Entry = 100, Stop = 99 -> risk per share = 1
    # Risk-based shares = 500 shares (50,000 INR > 33,333 cap)
    # Capital cap shares = floor(33,333.33 / 100) = 333 shares
    sizer = PositionSizer(risk_fraction=0.005, max_open_positions=3)

    res = sizer.size_position(
        symbol="TCS",
        current_equity=100_000.0,
        available_cash=100_000.0,
        entry_price=100.0,
        stop_loss=99.0,
        open_position_count=0,
    )

    assert res.approved
    assert res.shares == 333
    assert res.allocation_amount <= 33_333.34


def test_rejects_when_max_open_positions_reached() -> None:
    sizer = PositionSizer(max_open_positions=3)

    res = sizer.size_position(
        symbol="INFY",
        current_equity=1_000_000.0,
        available_cash=500_000.0,
        entry_price=1500.0,
        stop_loss=1450.0,
        open_position_count=3,
    )

    assert not res.approved
    assert res.shares == 0
    assert "Maximum open positions reached" in res.reason


def test_rejects_invalid_stop_loss_at_or_above_entry() -> None:
    sizer = PositionSizer()

    res = sizer.size_position(
        symbol="RELIANCE",
        current_equity=1_000_000.0,
        available_cash=500_000.0,
        entry_price=2500.0,
        stop_loss=2550.0,
        open_position_count=0,
    )

    assert not res.approved
    assert "must be below entry price" in res.reason


def test_rejects_when_available_cash_is_depleted() -> None:
    sizer = PositionSizer()

    res = sizer.size_position(
        symbol="HDFCBANK",
        current_equity=1_000_000.0,
        available_cash=0.0,
        entry_price=1500.0,
        stop_loss=1400.0,
        open_position_count=1,
    )

    assert not res.approved
    assert "Insufficient available cash" in res.reason


def test_invalid_sizer_initialization() -> None:
    with pytest.raises(ValueError, match="risk_fraction"):
        PositionSizer(risk_fraction=0.0)

    with pytest.raises(ValueError, match="max_open_positions"):
        PositionSizer(max_open_positions=0)
