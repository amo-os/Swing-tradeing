import pandas as pd
import pytest

from swingtrade.data.normalization import (
    OHLCV_COLUMNS,
    CandleNormalizationError,
    normalize_upstox_candles,
)


def test_valid_candles_are_normalized() -> None:
    candles = [
        ["2025-01-09T00:00:00+05:30", 100, 105, 98, 103, 1200, 0],
        ["2025-01-10T00:00:00+05:30", 103, 108, 101, 106, 1500, 0],
    ]

    frame = normalize_upstox_candles(candles)

    assert list(frame.columns) == OHLCV_COLUMNS
    assert len(frame) == 2
    assert frame.iloc[0]["open"] == 100.0
    assert frame.iloc[0]["close"] == 103.0
    assert frame.iloc[1]["volume"] == 1500.0
    assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])


def test_empty_candle_list_returns_empty_dataframe() -> None:
    frame = normalize_upstox_candles([])

    assert frame.empty
    assert list(frame.columns) == OHLCV_COLUMNS


def test_candle_with_too_few_fields_is_rejected() -> None:
    candles = [["2025-01-10T00:00:00+05:30", 100, 105]]

    with pytest.raises(CandleNormalizationError, match="invalid structure"):
        normalize_upstox_candles(candles)


def test_non_numeric_candle_value_is_rejected() -> None:
    candles = [["2025-01-10T00:00:00+05:30", "bad", 105, 98, 103, 1200]]

    with pytest.raises(CandleNormalizationError, match="non-numeric"):
        normalize_upstox_candles(candles)


def test_invalid_timestamp_is_rejected() -> None:
    candles = [["not-a-date", 100, 105, 98, 103, 1200]]

    with pytest.raises(CandleNormalizationError, match="timestamps are invalid"):
        normalize_upstox_candles(candles)


def test_missing_timestamp_is_rejected() -> None:
    candles = [[None, 100, 105, 98, 103, 1200]]

    with pytest.raises(CandleNormalizationError, match="no timestamp"):
        normalize_upstox_candles(candles)


def test_extra_upstox_fields_are_ignored() -> None:
    candles = [["2025-01-10T00:00:00+05:30", 100, 105, 98, 103, 1200, 500]]

    frame = normalize_upstox_candles(candles)

    assert list(frame.columns) == OHLCV_COLUMNS
    assert len(frame.columns) == 6
    assert frame.iloc[0]["volume"] == 1200.0
