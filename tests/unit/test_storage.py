from datetime import date

import pandas as pd
import pytest

from swingtrade.data.storage import DataStorageError, save_validated_ohlcv

VALID_FRAME = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            ["2025-01-09T00:00:00+05:30", "2025-01-10T00:00:00+05:30"]
        ),
        "open": [100.0, 103.0],
        "high": [105.0, 108.0],
        "low": [98.0, 101.0],
        "close": [103.0, 106.0],
        "volume": [1200.0, 1500.0],
    }
)


def test_valid_data_is_saved_to_csv(tmp_path) -> None:
    path = save_validated_ohlcv(
        VALID_FRAME,
        "NSE_EQ|TEST",
        date(2025, 1, 1),
        date(2025, 1, 10),
        output_dir=tmp_path,
    )

    assert path.exists()
    assert path.suffix == ".csv"

    saved = pd.read_csv(path)
    assert len(saved) == 2
    assert list(saved.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_existing_file_is_not_overwritten_by_default(tmp_path) -> None:
    path = save_validated_ohlcv(
        VALID_FRAME,
        "NSE_EQ|TEST",
        date(2025, 1, 1),
        date(2025, 1, 10),
        output_dir=tmp_path,
    )
    original_contents = path.read_text()

    with pytest.raises(DataStorageError, match="already exists"):
        save_validated_ohlcv(
            VALID_FRAME,
            "NSE_EQ|TEST",
            date(2025, 1, 1),
            date(2025, 1, 10),
            output_dir=tmp_path,
        )

    assert path.read_text() == original_contents


def test_invalid_ohlcv_data_is_rejected(tmp_path) -> None:
    invalid = VALID_FRAME.copy()
    invalid.loc[0, "high"] = 90.0

    with pytest.raises(DataStorageError, match="validation failed"):
        save_validated_ohlcv(
            invalid,
            "NSE_EQ|TEST",
            date(2025, 1, 1),
            date(2025, 1, 10),
            output_dir=tmp_path,
        )


def test_empty_data_is_rejected(tmp_path) -> None:
    with pytest.raises(DataStorageError, match="empty dataset"):
        save_validated_ohlcv(
            VALID_FRAME.iloc[0:0],
            "NSE_EQ|TEST",
            date(2025, 1, 1),
            date(2025, 1, 10),
            output_dir=tmp_path,
        )


def test_missing_required_column_is_rejected(tmp_path) -> None:
    incomplete = VALID_FRAME.drop(columns=["volume"])

    with pytest.raises(DataStorageError, match="Missing required columns"):
        save_validated_ohlcv(
            incomplete,
            "NSE_EQ|TEST",
            date(2025, 1, 1),
            date(2025, 1, 10),
            output_dir=tmp_path,
        )


def test_reversed_date_range_is_rejected(tmp_path) -> None:
    with pytest.raises(DataStorageError, match="from_date"):
        save_validated_ohlcv(
            VALID_FRAME,
            "NSE_EQ|TEST",
            date(2025, 1, 10),
            date(2025, 1, 1),
            output_dir=tmp_path,
        )
