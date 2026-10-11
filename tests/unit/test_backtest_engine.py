"""Unit tests for the BacktestEngine with explicit synthetic fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from swingtrade.backtest.costs import CostModel
from swingtrade.backtest.engine import BacktestEngine, ExitReason
from swingtrade.risk.position_sizer import PositionSizer
from swingtrade.strategy.base import Signal, SignalDirection, Strategy


def make_synthetic_daily_candles(
    prices: list[tuple[float, float, float, float]],
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Create deterministic synthetic daily candles clearly marked as synthetic.

    Parameters
    ----------
    prices : list[tuple[open, high, low, close]]
    """
    dates = pd.date_range(start_date, periods=len(prices), freq="D", tz="Asia/Kolkata")
    records = []
    for date, (op, hi, lo, cl) in zip(dates, prices, strict=True):
        records.append(
            {
                "timestamp": date,
                "open": float(op),
                "high": float(hi),
                "low": float(lo),
                "close": float(cl),
                "volume": 100_000,
            }
        )
    return pd.DataFrame(records)


class FixedSignalStrategy(Strategy):
    """Synthetic test strategy emitting programmed signals on specific dates."""

    def __init__(self, signals: list[Signal]) -> None:
        self._signals = signals
        self.inspected_history_lengths: list[int] = []

    @property
    def name(self) -> str:
        return "SyntheticFixedSignalStrategy"

    def generate_signals(
        self,
        current_date: pd.Timestamp,
        historical_data: dict[str, pd.DataFrame],
        open_positions: dict[str, object],
    ) -> list[Signal]:
        # Track history length to verify strict point-in-time slicing (no future bars)
        for df in historical_data.values():
            self.inspected_history_lengths.append(len(df))
            assert (df["timestamp"] <= current_date).all(), "Look-ahead bias detected!"

        return [s for s in self._signals if s.date == current_date]


def test_order_timing_executes_on_next_bar_open() -> None:
    # 3 bars: Day 0, Day 1, Day 2
    # Signal emitted on Day 0 close. Executed on Day 1 Open.
    candles = make_synthetic_daily_candles(
        [
            (100.0, 105.0, 99.0, 102.0),   # Day 0: close = 102
            (103.0, 108.0, 101.0, 106.0),  # Day 1: open = 103 (Entry here)
            (106.0, 110.0, 105.0, 108.0),  # Day 2: close = 108 (End of test exit)
        ]
    )
    t0 = candles.iloc[0]["timestamp"]
    t1 = candles.iloc[1]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SYNTH001",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=90.0,
                profit_target=120.0,
            )
        ]
    )

    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=CostModel(slippage_pct=0.001),  # 0.1% slippage
        position_sizer=PositionSizer(risk_fraction=0.01),
    )

    result = engine.run(
        strategy=strategy,
        dataset={"SYNTH001": candles},
        approved_universe=["SYNTH001"],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == t1
    assert trade.nominal_entry_price == 103.0
    # Effective price includes 0.1% buy slippage
    assert trade.effective_entry_price == pytest.approx(103.0 * 1.001)
    assert trade.is_closed


def test_intraday_stop_loss_execution() -> None:
    # Day 0: Signal emitted
    # Day 1: Entry at Open 100. Stop loss is 95. Day 1 stays within range (98 to 102).
    # Day 2: Open 99, High 101, Low 93, Close 94. Low (93) breaches Stop (95).
    # Fill price should be exactly stop loss (95.0) with sell slippage.
    candles = make_synthetic_daily_candles(
        [
            (100.0, 102.0, 99.0, 101.0),   # Day 0
            (100.0, 102.0, 98.0, 100.0),   # Day 1: entry at open 100
            (99.0, 101.0, 93.0, 94.0),     # Day 2: low breaches 95
        ]
    )
    t0 = candles.iloc[0]["timestamp"]
    t2 = candles.iloc[2]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SYNTH001",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=95.0,
            )
        ]
    )

    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=CostModel(slippage_pct=0.0),
    )

    result = engine.run(
        strategy=strategy,
        dataset={"SYNTH001": candles},
        approved_universe=["SYNTH001"],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_date == t2
    assert trade.nominal_exit_price == 95.0
    assert trade.exit_reason == ExitReason.STOP_LOSS.value


def test_gap_down_stop_loss_executes_at_open_not_stop() -> None:
    # Day 0: Signal emitted with stop 95.
    # Day 1: Entry at 100.
    # Day 2: Gaps down at Open 90.0 (< stop 95.0)!
    # Fill price MUST be 90.0, never the unavailable stop price 95.0!
    candles = make_synthetic_daily_candles(
        [
            (100.0, 102.0, 99.0, 101.0),   # Day 0
            (100.0, 102.0, 98.0, 100.0),   # Day 1: entry at 100
            (90.0, 92.0, 88.0, 89.0),      # Day 2: gaps down to 90.0
        ]
    )
    t0 = candles.iloc[0]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SYNTH001",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=95.0,
            )
        ]
    )

    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=CostModel(slippage_pct=0.0),
    )

    result = engine.run(
        strategy=strategy,
        dataset={"SYNTH001": candles},
        approved_universe=["SYNTH001"],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.nominal_exit_price == 90.0
    assert trade.exit_reason == ExitReason.STOP_LOSS_GAP.value


def test_profit_target_execution() -> None:
    # Day 0: Signal emitted with target 110.
    # Day 1: Entry at 100.
    # Day 2: Open 105, High 112, Low 104, Close 111. High breaches target 110.
    candles = make_synthetic_daily_candles(
        [
            (100.0, 102.0, 99.0, 101.0),   # Day 0
            (100.0, 102.0, 98.0, 100.0),   # Day 1: entry
            (105.0, 112.0, 104.0, 111.0),  # Day 2: hits target
        ]
    )
    t0 = candles.iloc[0]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SYNTH001",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=95.0,
                profit_target=110.0,
            )
        ]
    )

    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=CostModel(slippage_pct=0.0),
    )

    result = engine.run(
        strategy=strategy,
        dataset={"SYNTH001": candles},
        approved_universe=["SYNTH001"],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.nominal_exit_price == 110.0
    assert trade.exit_reason == ExitReason.PROFIT_TARGET.value


def test_gap_up_profit_target_executes_at_open() -> None:
    # Target 110. Day 2 gaps up to Open 115.0!
    candles = make_synthetic_daily_candles(
        [
            (100.0, 102.0, 99.0, 101.0),   # Day 0
            (100.0, 102.0, 98.0, 100.0),   # Day 1
            (115.0, 120.0, 114.0, 118.0),  # Day 2: gaps up to 115
        ]
    )
    t0 = candles.iloc[0]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SYNTH001",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=95.0,
                profit_target=110.0,
            )
        ]
    )

    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=CostModel(slippage_pct=0.0),
    )

    result = engine.run(
        strategy=strategy,
        dataset={"SYNTH001": candles},
        approved_universe=["SYNTH001"],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.nominal_exit_price == 115.0
    assert trade.exit_reason == ExitReason.PROFIT_TARGET_GAP.value


def test_strictly_bars_unapproved_symbols() -> None:
    # Even if an unapproved symbol (e.g. SIEMENS) is in dataset or emits signal,
    # it must be barred by the engine if not in approved_universe.
    candles = make_synthetic_daily_candles(
        [
            (100.0, 102.0, 99.0, 101.0),
            (100.0, 102.0, 98.0, 100.0),
        ]
    )
    t0 = candles.iloc[0]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SIEMENS",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=90.0,
            )
        ]
    )

    engine = BacktestEngine(initial_capital=100_000.0)

    # Approved universe does NOT contain SIEMENS
    with pytest.raises(ValueError, match="No valid data available for approved symbols"):
        engine.run(
            strategy=strategy,
            dataset={"SIEMENS": candles},
            approved_universe=["ABB", "TCS"],
        )


def test_end_of_test_open_positions_closed_cleanly() -> None:
    candles = make_synthetic_daily_candles(
        [
            (100.0, 102.0, 99.0, 101.0),   # Day 0: signal
            (100.0, 102.0, 98.0, 100.0),   # Day 1: enter at 100
            (102.0, 104.0, 101.0, 103.0),  # Day 2: ends without hitting stop
        ]
    )
    t0 = candles.iloc[0]["timestamp"]
    t2 = candles.iloc[2]["timestamp"]

    strategy = FixedSignalStrategy(
        [
            Signal(
                symbol="SYNTH001",
                date=t0,
                direction=SignalDirection.BUY,
                stop_loss=90.0,
                profit_target=130.0,
            )
        ]
    )

    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=CostModel(slippage_pct=0.0),
        close_open_positions_at_end=True,
    )

    result = engine.run(
        strategy=strategy,
        dataset={"SYNTH001": candles},
        approved_universe=["SYNTH001"],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_date == t2
    assert trade.nominal_exit_price == 103.0
    assert trade.exit_reason == ExitReason.END_OF_TEST.value
    assert trade.is_closed
