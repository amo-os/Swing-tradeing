from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from swingtrade.data.pipeline import (
    DataPipelineError,
    fetch_validated_daily_ohlcv,
)
from swingtrade.data.storage import DataStorageError, save_validated_ohlcv
from swingtrade.data.upstox_client import UpstoxAPIError, UpstoxClient


class FetchAndSaveError(RuntimeError):
    """Raised when historical data cannot be safely fetched and saved."""


def fetch_and_save_daily_ohlcv(
    instrument_key: str,
    from_date: date,
    to_date: date,
    output_dir: Path = Path("data/processed"),
    overwrite: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Fetch, validate, and save daily OHLCV data for one instrument.

    This function only retrieves and stores market data. It does not place
    orders or make trading decisions.
    """
    try:
        with UpstoxClient() as client:
            frame = fetch_validated_daily_ohlcv(
                client=client,
                instrument_key=instrument_key,
                from_date=from_date,
                to_date=to_date,
            )

        saved_path = save_validated_ohlcv(
            frame=frame,
            instrument_key=instrument_key,
            from_date=from_date,
            to_date=to_date,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    except (UpstoxAPIError, DataPipelineError, DataStorageError) as exc:
        raise FetchAndSaveError(str(exc)) from exc

    return frame, saved_path
