import json
from datetime import date

import httpx
import pytest

from swingtrade.data.upstox_client import (
    MAX_ATTEMPTS,
    UpstoxAPIError,
    UpstoxClient,
)

TEST_FROM_DATE = date(2025, 1, 1)
TEST_TO_DATE = date(2025, 1, 10)
TEST_INSTRUMENT = "NSE_EQ|INE669E01016"


class FakeHTTPClient:
    """Fake HTTP client for offline tests."""

    def __init__(
        self,
        response: httpx.Response | None = None,
        error: Exception | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.errors = list(errors or [])
        self.closed = False
        self.last_url: str | None = None
        self.call_count = 0

    def get(self, url: str) -> httpx.Response:
        self.last_url = url
        self.call_count += 1

        if self.errors:
            raise self.errors.pop(0)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("No fake response was configured")

        return self.response

    def close(self) -> None:
        self.closed = True


def make_response(
    payload: object,
    status_code: int = 200,
) -> httpx.Response:
    """Build a simulated HTTP response without network access."""
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        request=httpx.Request("GET", "https://api.upstox.com/test"),
    )


def make_client(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response | None = None,
    error: Exception | None = None,
    errors: list[Exception] | None = None,
) -> tuple[UpstoxClient, FakeHTTPClient]:
    """Create the Upstox client with a fake HTTP transport."""
    fake_http = FakeHTTPClient(
        response=response,
        error=error,
        errors=errors,
    )

    monkeypatch.setattr(
        "swingtrade.data.upstox_client.httpx.Client",
        lambda **kwargs: fake_http,
    )
    monkeypatch.setattr(
        "swingtrade.data.upstox_client.time.sleep",
        lambda seconds: None,
    )

    client = UpstoxClient(token="test-token")
    return client, fake_http


def test_successful_daily_candle_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = [
        ["2025-01-10", 100.0, 105.0, 98.0, 103.0, 1200, 0],
    ]
    client, fake_http = make_client(
        monkeypatch,
        response=make_response(
            {"status": "success", "data": {"candles": candles}}
        ),
    )

    result = client.get_historical_daily_candles(
        TEST_INSTRUMENT,
        TEST_FROM_DATE,
        TEST_TO_DATE,
    )

    assert result["candles"] == candles
    assert fake_http.last_url is not None
    assert "/days/1/2025-01-10/2025-01-01" in fake_http.last_url
    assert fake_http.call_count == 1

    client.close()
    assert fake_http.closed


def test_missing_token_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "swingtrade.data.upstox_client.load_dotenv",
        lambda: False,
    )

    with pytest.raises(UpstoxAPIError, match="Missing UPSTOX_ACCESS_TOKEN"):
        UpstoxClient()


def test_empty_instrument_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = make_client(monkeypatch)

    with pytest.raises(ValueError, match="instrument_key"):
        client.get_historical_daily_candles(
            " ",
            TEST_FROM_DATE,
            TEST_TO_DATE,
        )

    client.close()


def test_reversed_date_range_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = make_client(monkeypatch)

    with pytest.raises(ValueError, match="from_date"):
        client.get_historical_daily_candles(
            TEST_INSTRUMENT,
            TEST_TO_DATE,
            TEST_FROM_DATE,
        )

    client.close()


def test_http_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake_http = make_client(
        monkeypatch,
        response=make_response({"status": "error"}, status_code=401),
    )

    with pytest.raises(UpstoxAPIError, match="HTTP 401"):
        client.get_historical_daily_candles(
            TEST_INSTRUMENT,
            TEST_FROM_DATE,
            TEST_TO_DATE,
        )

    assert fake_http.call_count == 1
    client.close()


def test_network_error_is_reported_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://api.upstox.com/test")
    client, fake_http = make_client(
        monkeypatch,
        error=httpx.ConnectError("Simulated offline connection", request=request),
    )

    with pytest.raises(
        UpstoxAPIError,
        match=f"Unable to reach Upstox after {MAX_ATTEMPTS} attempts",
    ):
        client.get_historical_daily_candles(
            TEST_INSTRUMENT,
            TEST_FROM_DATE,
            TEST_TO_DATE,
        )

    assert fake_http.call_count == MAX_ATTEMPTS
    client.close()


def test_temporary_protocol_error_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://api.upstox.com/test")
    client, fake_http = make_client(
        monkeypatch,
        response=make_response(
            {"status": "success", "data": {"candles": [["2025-01-10", 100, 105, 98, 103, 1200, 0]]}}
        ),
        errors=[
            httpx.RemoteProtocolError(
                "Simulated temporary connection interruption",
                request=request,
            )
        ],
    )

    result = client.get_historical_daily_candles(
        TEST_INSTRUMENT,
        TEST_FROM_DATE,
        TEST_TO_DATE,
    )

    assert len(result["candles"]) == 1
    assert fake_http.call_count == 2
    client.close()


def test_unsuccessful_api_payload_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = make_client(
        monkeypatch,
        response=make_response({"status": "error", "data": {}}),
    )

    with pytest.raises(UpstoxAPIError, match="unsuccessful request"):
        client.get_historical_daily_candles(
            TEST_INSTRUMENT,
            TEST_FROM_DATE,
            TEST_TO_DATE,
        )

    client.close()


def test_unexpected_response_format_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = make_client(
        monkeypatch,
        response=make_response({"status": "success", "data": {}}),
    )

    with pytest.raises(UpstoxAPIError, match="response format"):
        client.get_historical_daily_candles(
            TEST_INSTRUMENT,
            TEST_FROM_DATE,
            TEST_TO_DATE,
        )

    client.close()


def test_invalid_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        200,
        content=b"not valid json",
        request=httpx.Request("GET", "https://api.upstox.com/test"),
    )
    client, _ = make_client(monkeypatch, response=response)

    with pytest.raises(UpstoxAPIError, match="invalid JSON"):
        client.get_historical_daily_candles(
            TEST_INSTRUMENT,
            TEST_FROM_DATE,
            TEST_TO_DATE,
        )

    client.close()
