"""Unit tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from swingtrade.strategy.indicators import calculate_indicators


def make_ohlcv(rows: int = 260) -> pd.DataFrame:
    """Build deterministic synthetic daily candles."""
    index = np.arange(rows)
    close = 100 + index * 0.15 + 2 * np.sin(index / 5)

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "open": close - 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000 + index * 10,
    })


def test_calculates_all_expected_indicators() -> None:
    result = calculate_indicators(make_ohlcv())

    expected = {
        "ema_50",
        "ema_200",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "average_volume_20",
    }

    assert expected.issubset(result.columns)

    for column in expected:
        assert result[column].notna().any(), f"{column} has no values"


def test_ema_200_requires_200_candles() -> None:
    result = calculate_indicators(make_ohlcv(199))
    assert result["ema_200"].isna().all()

    result = calculate_indicators(make_ohlcv(200))
    assert result["ema_200"].notna().sum() == 1


def test_average_volume_excludes_current_candle() -> None:
    frame = make_ohlcv(25)
    result = calculate_indicators(frame)

    expected = frame["volume"].iloc[4:24].mean()
    assert result.loc[24, "average_volume_20"] == pytest.approx(expected)


def test_does_not_modify_input_dataframe() -> None:
    frame = make_ohlcv()
    original = frame.copy(deep=True)

    calculate_indicators(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_rejects_missing_required_column() -> None:
    frame = make_ohlcv().drop(columns=["close"])

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_indicators(frame)


def test_rejects_duplicate_timestamps() -> None:
    frame = make_ohlcv()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

    with pytest.raises(ValueError, match="Duplicate timestamps"):
        calculate_indicators(frame)


def test_rejects_unsorted_timestamps() -> None:
    frame = make_ohlcv()
    frame.loc[[0, 1], "timestamp"] = frame.loc[
        [1, 0], "timestamp"
    ].to_numpy()

    with pytest.raises(ValueError, match="ascending order"):
        calculate_indicators(frame)


def test_rejects_negative_volume() -> None:
    frame = make_ohlcv()
    frame.loc[10, "volume"] = -1

    with pytest.raises(ValueError, match="Volume cannot be negative"):
        calculate_indicators(frame)


def test_rejects_invalid_ohlc_relationship() -> None:
    frame = make_ohlcv()
    frame.loc[10, "high"] = frame.loc[10, "low"] - 1

    with pytest.raises(ValueError, match="High cannot be below low"):
        calculate_indicators(frame)
