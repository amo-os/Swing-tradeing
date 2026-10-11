from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def validate_ohlcv(frame: pd.DataFrame) -> list[str]:
    """Return validation errors for an OHLCV DataFrame."""
    errors: list[str] = []

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]

    if frame.empty:
        return ["Dataset is empty"]

    data = frame.copy()

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    if timestamps.isna().any():
        errors.append("Invalid or missing timestamps")

    valid_timestamps = timestamps.dropna()
    if valid_timestamps.duplicated().any():
        errors.append("Duplicate timestamps")

    if not valid_timestamps.is_monotonic_increasing:
        errors.append("Timestamps are not in ascending order")

    price_columns = ("open", "high", "low", "close")

    for column in (*price_columns, "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[list(price_columns)].isna().any().any():
        errors.append("Invalid or missing price values")

    if data["volume"].isna().any():
        errors.append("Invalid or missing volume values")

    prices = data[list(price_columns)]
    if (prices <= 0).any().any():
        errors.append("Prices must be positive")

    if (data["volume"].dropna() < 0).any():
        errors.append("Volume cannot be negative")

    if not prices.isna().any().any():
        if (data["high"] < data["low"]).any():
            errors.append("High cannot be below low")

        if (data["high"] < data[["open", "close"]].max(axis=1)).any():
            errors.append("High is below open or close")

        if (data["low"] > data[["open", "close"]].min(axis=1)).any():
            errors.append("Low is above open or close")

    return errors
