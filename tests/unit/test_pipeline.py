from datetime import date

import pytest

from swingtrade.data.pipeline import (
    DataPipelineError,
    fetch_validated_daily_ohlcv,
)


class FakeUpstoxClient:
    """Fake client that returns predefined candles without network access."""

    def __init__(self, candles: list) -> None:
        self.candles = candles
        self.last_request: dict | None = None

    def get_historical_daily_candles(
        self,
        instrument_key: str,
        from_date: date,
        to_date: date,
    ) -> dict:
        self.last_request = {
            "instrument_key": instrument_key,
            "from_date": from_date,
            "to_date": to_date,
        }
        return {"candles": self.candles}


def test_pipeline_returns_valid_candles_in_chronological_order() -> None:
    # Deliberately reverse the dates to verify chronological sorting.
    candles = [
        ["2025-01-10T00:00:00+05:30", 103, 108, 101, 106, 1500, 0],
        ["2025-01-09T00:00:00+05:30", 100, 105, 98, 103, 1200, 0],
    ]
    client = FakeUpstoxClient(candles)

    frame = fetch_validated_daily_ohlcv(
        client,  # type: ignore[arg-type]
        "NSE_EQ|TEST",
        date(2025, 1, 1),
        date(2025, 1, 10),
    )

    assert len(frame) == 2
    assert frame["timestamp"].is_monotonic_increasing
    assert frame.iloc[0]["close"] == 103.0
    assert frame.iloc[1]["close"] == 106.0


def test_pipeline_rejects_empty_candle_response() -> None:
    client = FakeUpstoxClient([])

    with pytest.raises(DataPipelineError, match="No candles were returned"):
        fetch_validated_daily_ohlcv(
            client,  # type: ignore[arg-type]
            "NSE_EQ|TEST",
            date(2025, 1, 1),
            date(2025, 1, 10),
        )


def test_pipeline_rejects_invalid_ohlcv_data() -> None:
    # The high price is below the low price, so validation must fail.
    candles = [
        ["2025-01-10T00:00:00+05:30", 100, 95, 98, 103, 1200, 0],
    ]
    client = FakeUpstoxClient(candles)

    with pytest.raises(DataPipelineError, match="OHLCV validation failed"):
        fetch_validated_daily_ohlcv(
            client,  # type: ignore[arg-type]
            "NSE_EQ|TEST",
            date(2025, 1, 1),
            date(2025, 1, 10),
        )


def test_pipeline_passes_requested_instrument_and_dates_to_client() -> None:
    client = FakeUpstoxClient(
        [["2025-01-10T00:00:00+05:30", 100, 105, 98, 103, 1200, 0]]
    )

    fetch_validated_daily_ohlcv(
        client,  # type: ignore[arg-type]
        "NSE_EQ|TEST",
        date(2025, 1, 1),
        date(2025, 1, 10),
    )

    assert client.last_request == {
        "instrument_key": "NSE_EQ|TEST",
        "from_date": date(2025, 1, 1),
        "to_date": date(2025, 1, 10),
    }
