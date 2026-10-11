"""Technical indicators for daily-candle swing-trading research."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of OHLCV data with technical indicators added.

    Input candles must be unique and in ascending timestamp order.
    This function does not adjust prices for corporate actions.
    """
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("Cannot calculate indicators on an empty dataset")

    result = frame.copy()

    timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Invalid or missing timestamps")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be in ascending order")

    price_columns = ("open", "high", "low", "close")
    numeric_columns = (*price_columns, "volume")

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result[list(numeric_columns)].isna().any().any():
        raise ValueError("OHLCV columns must contain valid numeric values")

    if (result[list(price_columns)] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")

    if (result["high"] < result["low"]).any():
        raise ValueError("High cannot be below low")
    if (result["high"] < result[["open", "close"]].max(axis=1)).any():
        raise ValueError("High cannot be below open or close")
    if (result["low"] > result[["open", "close"]].min(axis=1)).any():
        raise ValueError("Low cannot be above open or close")

    high = result["high"]
    low = result["low"]
    close = result["close"]
    previous_close = close.shift(1)

    # Exponential moving averages use the full available history.
    result["ema_50"] = close.ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    result["ema_200"] = close.ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    # RSI using Wilder-style exponential smoothing.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    average_loss = loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    result["rsi_14"] = rsi

    # MACD.
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    # True range and Average True Range using Wilder-style smoothing.
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    # Directional movement and ADX.
    upward_move = high.diff()
    downward_move = -low.diff()

    plus_dm = upward_move.where(
        (upward_move > downward_move) & (upward_move > 0), 0.0
    )
    minus_dm = downward_move.where(
        (downward_move > upward_move) & (downward_move > 0), 0.0
    )

    smoothed_tr = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean() / smoothed_tr

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    dx = dx.mask(di_sum == 0, 0.0)

    result["plus_di_14"] = plus_di
    result["minus_di_14"] = minus_di
    result["adx_14"] = dx.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    # The current candle's volume is not included in its prior-volume average.
    result["average_volume_20"] = result["volume"].shift(1).rolling(
        window=20, min_periods=20
    ).mean()

    return result
