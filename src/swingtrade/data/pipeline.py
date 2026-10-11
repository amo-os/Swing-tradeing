from __future__ import annotations

from datetime import date

import pandas as pd

from swingtrade.data.normalization import normalize_upstox_candles
from swingtrade.data.upstox_client import UpstoxClient
from swingtrade.data.validation import validate_ohlcv


class DataPipelineError(RuntimeError):
    """Raised when market data cannot pass the validation pipeline."""


def fetch_validated_daily_ohlcv(
    client: UpstoxClient,
    instrument_key: str,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    """Fetch, normalize, order, and validate daily market data.

    Returns an OHLCV DataFrame in ascending timestamp order.
    Raises DataPipelineError if the data fails validation.
    """
    raw_data = client.get_historical_daily_candles(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
    )

    candles = raw_data["candles"]
    frame = normalize_upstox_candles(candles)

    if frame.empty:
        raise DataPipelineError("No candles were returned for the requested period")

    frame = frame.sort_values("timestamp").reset_index(drop=True)

    errors = validate_ohlcv(frame)
    if errors:
        details = "; ".join(errors)
        raise DataPipelineError(f"OHLCV validation failed: {details}")

    return frame
