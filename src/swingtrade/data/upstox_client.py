
from __future__ import annotations

import os
import time
from datetime import date
from typing import Self

import httpx
from dotenv import load_dotenv

BASE_URL = "https://api.upstox.com"
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class UpstoxAPIError(RuntimeError):
    """Raised when Upstox returns an API or transport error."""


class UpstoxClient:
    """Minimal read-only client for Upstox historical market data."""

    def __init__(self, token: str | None = None) -> None:
        load_dotenv()

        self._token = token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self._token:
            raise UpstoxAPIError(
                "Missing UPSTOX_ACCESS_TOKEN. Check your local .env file."
            )

        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            timeout=20.0,
        )

    def get_historical_daily_candles(
        self,
        instrument_key: str,
        from_date: date,
        to_date: date,
    ) -> dict:
        """Fetch daily candles, retrying temporary transport and HTTP failures."""
        if not instrument_key.strip():
            raise ValueError("instrument_key must not be empty")

        if from_date > to_date:
            raise ValueError("from_date must not be after to_date")

        endpoint = (
            f"/v3/historical-candle/"
            f"{instrument_key}/days/1/{to_date.isoformat()}/{from_date.isoformat()}"
        )

        response: httpx.Response | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                candidate = self._http.get(endpoint)
                candidate.raise_for_status()
                response = candidate
                break

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code

                if status_code not in RETRYABLE_STATUS_CODES:
                    raise UpstoxAPIError(
                        f"Upstox returned HTTP {status_code}"
                    ) from exc

                if attempt == MAX_ATTEMPTS:
                    raise UpstoxAPIError(
                        f"Upstox returned HTTP {status_code} after "
                        f"{MAX_ATTEMPTS} attempts"
                    ) from exc

                retry_after = exc.response.headers.get("Retry-After")
                delay = RETRY_DELAY_SECONDS * attempt

                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass

                time.sleep(min(delay, 30.0))

            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise UpstoxAPIError(
                        "Unable to reach Upstox after "
                        f"{MAX_ATTEMPTS} attempts: {type(exc).__name__}"
                    ) from exc

                time.sleep(RETRY_DELAY_SECONDS * attempt)

            except httpx.HTTPError as exc:
                raise UpstoxAPIError(
                    f"Upstox HTTP request failed: {type(exc).__name__}"
                ) from exc

        if response is None:
            raise UpstoxAPIError("Upstox request did not produce a response")

        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstoxAPIError("Upstox returned invalid JSON") from exc

        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise UpstoxAPIError("Upstox reported an unsuccessful request")

        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("candles"), list):
            raise UpstoxAPIError("Unexpected historical-candle response format")

        return data

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
