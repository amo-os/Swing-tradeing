"""Unit tests for BaselineSwingStrategy rules and signal generation."""

import numpy as np
import pandas as pd

from swingtrade.strategy.baseline import BaselineSwingStrategy
from swingtrade.strategy.indicators import calculate_indicators


def make_bullish_trend_candles(rows: int = 250) -> pd.DataFrame:
    """Create synthetic ascending candles with healthy pullback momentum."""
    index = np.arange(rows)
    # Steadily increasing close prices to build clean EMA-50 > EMA-200
    close = 100.0 + index * 0.5
    high = close + 1.5
    low = close - 1.5
    open_p = close - 0.2
    volume = 100_000 + (index % 10) * 1000

    dates = pd.date_range("2024-01-01", periods=rows, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return calculate_indicators(df)


def test_insufficient_history_emits_no_signals() -> None:
    strategy = BaselineSwingStrategy()
    short_df = make_bullish_trend_candles(150)
    current_date = short_df.iloc[-1]["timestamp"]

    signals = strategy.generate_signals(
        current_date=current_date,
        historical_data={"STOCK": short_df},
        open_positions={},
    )

    assert signals == []


def test_valid_bullish_pullback_emits_buy_signal() -> None:
    strategy = BaselineSwingStrategy()
    df = make_bullish_trend_candles(250)
    # Simulate a clean pullback bounce on the final bar
    last_idx = df.index[-1]
    df.loc[last_idx, "rsi_14"] = 55.0
    df.loc[last_idx, "macd"] = 2.0
    df.loc[last_idx, "macd_signal"] = 1.0
    current_date = df.iloc[-1]["timestamp"]

    signals = strategy.generate_signals(
        current_date=current_date,
        historical_data={"BULL": df},
        open_positions={},
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.symbol == "BULL"
    assert sig.direction.value == "BUY"
    assert sig.stop_loss is not None
    assert sig.profit_target is not None
    last_close = float(df.iloc[-1]["close"])
    assert sig.stop_loss < last_close < sig.profit_target


def test_exit_signal_emitted_when_trend_breaks() -> None:
    strategy = BaselineSwingStrategy()
    df = make_bullish_trend_candles(250)
    # Artificially force last close below EMA-50
    df.loc[df.index[-1], "close"] = df.loc[df.index[-1], "ema_50"] - 10.0
    current_date = df.iloc[-1]["timestamp"]

    class MockPosition:
        days_held = 3

    signals = strategy.generate_signals(
        current_date=current_date,
        historical_data={"BULL": df},
        open_positions={"BULL": MockPosition()},
    )

    assert len(signals) == 1
    assert signals[0].direction.value == "SELL"
    assert "fell below EMA-50" in signals[0].reason
