from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from swingtrade.data.fetch_and_save import (
    FetchAndSaveError,
    fetch_and_save_daily_ohlcv,
)
from swingtrade.data.pipeline import DataPipelineError
from swingtrade.data.storage import DataStorageError
from swingtrade.data.upstox_client import UpstoxAPIError

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


@patch("swingtrade.data.fetch_and_save.save_validated_ohlcv")
@patch("swingtrade.data.fetch_and_save.fetch_validated_daily_ohlcv")
@patch("swingtrade.data.fetch_and_save.UpstoxClient")
def test_fetch_and_save_returns_frame_and_path(
    mock_client_class: MagicMock,
    mock_fetch: MagicMock,
    mock_save: MagicMock,
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "NSE_EQ_TEST_2025-01-01_2025-01-10.csv"
    mock_fetch.return_value = VALID_FRAME
    mock_save.return_value = expected_path

    frame, path = fetch_and_save_daily_ohlcv(
        instrument_key="NSE_EQ|TEST",
        from_date=date(2025, 1, 1),
        to_date=date(2025, 1, 10),
        output_dir=tmp_path,
    )

    assert frame.equals(VALID_FRAME)
    assert path == expected_path
    mock_client_class.assert_called_once_with()
    mock_fetch.assert_called_once()
    mock_save.assert_called_once()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (UpstoxAPIError("API request failed"), "API request failed"),
        (DataPipelineError("OHLCV validation failed"), "OHLCV validation failed"),
        (DataStorageError("File already exists"), "File already exists"),
    ],
)
@patch("swingtrade.data.fetch_and_save.save_validated_ohlcv")
@patch("swingtrade.data.fetch_and_save.fetch_validated_daily_ohlcv")
@patch("swingtrade.data.fetch_and_save.UpstoxClient")
def test_expected_errors_are_wrapped(
    mock_client_class: MagicMock,
    mock_fetch: MagicMock,
    mock_save: MagicMock,
    failure: Exception,
    message: str,
    tmp_path: Path,
) -> None:
    mock_fetch.return_value = VALID_FRAME
    mock_save.side_effect = failure

    if isinstance(failure, (UpstoxAPIError, DataPipelineError)):
        mock_fetch.side_effect = failure

    with pytest.raises(FetchAndSaveError, match=message):
        fetch_and_save_daily_ohlcv(
            instrument_key="NSE_EQ|TEST",
            from_date=date(2025, 1, 1),
            to_date=date(2025, 1, 10),
            output_dir=tmp_path,
        )


@patch("swingtrade.data.fetch_and_save.save_validated_ohlcv")
@patch("swingtrade.data.fetch_and_save.fetch_validated_daily_ohlcv")
@patch("swingtrade.data.fetch_and_save.UpstoxClient")
def test_overwrite_option_is_forwarded(
    mock_client_class: MagicMock,
    mock_fetch: MagicMock,
    mock_save: MagicMock,
    tmp_path: Path,
) -> None:
    mock_fetch.return_value = VALID_FRAME
    mock_save.return_value = tmp_path / "data.csv"

    fetch_and_save_daily_ohlcv(
        instrument_key="NSE_EQ|TEST",
        from_date=date(2025, 1, 1),
        to_date=date(2025, 1, 10),
        output_dir=tmp_path,
        overwrite=True,
    )

    assert mock_save.call_args.kwargs["overwrite"] is True
