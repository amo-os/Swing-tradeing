from __future__ import annotations

from typing import Any

import pandas as pd

OHLCV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class CandleNormalizationError(ValueError):
    """Raised when raw candle data cannot be normalized safely."""


def normalize_upstox_candles(candles: list[Any]) -> pd.DataFrame:
    """Convert Upstox candle arrays into a standardized OHLCV DataFrame.

    Upstox candle arrays contain:
    timestamp, open, high, low, close, volume, open_interest.

    The output preserves the timestamps' supplied timezone information.
    It does not adjust prices for splits or other corporate actions.
    """
    if not isinstance(candles, list):
        raise CandleNormalizationError("Candles must be provided as a list")

    if not candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    rows: list[dict[str, Any]] = []

    for index, candle in enumerate(candles):
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            raise CandleNormalizationError(
                f"Candle at index {index} has an invalid structure"
            )

        timestamp, open_price, high, low, close, volume = candle[:6]

        if timestamp is None or not str(timestamp).strip():
            raise CandleNormalizationError(
                f"Candle at index {index} has no timestamp"
            )

        try:
            prices_and_volume = [
                float(open_price),
                float(high),
                float(low),
                float(close),
                float(volume),
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise CandleNormalizationError(
                f"Candle at index {index} contains a non-numeric value"
            ) from exc

        rows.append(
            {
                "timestamp": timestamp,
                "open": prices_and_volume[0],
                "high": prices_and_volume[1],
                "low": prices_and_volume[2],
                "close": prices_and_volume[3],
                "volume": prices_and_volume[4],
            }
        )

    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
    )

    if frame["timestamp"].isna().any():
        raise CandleNormalizationError("One or more timestamps are invalid")

    return frame
