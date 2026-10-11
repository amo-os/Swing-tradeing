from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from swingtrade.data.normalization import OHLCV_COLUMNS
from swingtrade.data.validation import validate_ohlcv


class DataStorageError(RuntimeError):
    """Raised when validated market data cannot be saved safely."""


def _safe_instrument_name(instrument_key: str) -> str:
    """Convert an instrument key into a filesystem-safe filename component."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", instrument_key).strip("_")

    if not safe_name:
        raise DataStorageError("Instrument key cannot produce a safe filename")

    return safe_name


def save_validated_ohlcv(
    frame: pd.DataFrame,
    instrument_key: str,
    from_date: date,
    to_date: date,
    output_dir: Path = Path("data/processed"),
    overwrite: bool = False,
) -> Path:
    """Validate and save OHLCV data as CSV.

    Existing files are protected unless overwrite=True.
    Returns the path of the saved CSV file.
    """
    if not instrument_key.strip():
        raise DataStorageError("Instrument key must not be empty")

    if from_date > to_date:
        raise DataStorageError("from_date must not be after to_date")

    missing_columns = [column for column in OHLCV_COLUMNS if column not in frame]
    if missing_columns:
        raise DataStorageError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    if frame.empty:
        raise DataStorageError("Cannot save an empty dataset")

    errors = validate_ohlcv(frame[OHLCV_COLUMNS])
    if errors:
        raise DataStorageError(
            f"OHLCV validation failed: {'; '.join(errors)}"
        )

    ordered = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)

    safe_name = _safe_instrument_name(instrument_key)
    filename = f"{safe_name}_{from_date.isoformat()}_{to_date.isoformat()}.csv"

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename

    if destination.exists() and not overwrite:
        raise DataStorageError(
            f"File already exists; refusing to overwrite: {destination}"
        )

    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        ordered.to_csv(temporary, index=False)

        if destination.exists() and not overwrite:
            raise DataStorageError(
                f"File already exists; refusing to overwrite: {destination}"
            )

        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return destination
