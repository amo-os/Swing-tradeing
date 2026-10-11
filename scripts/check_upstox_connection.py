from datetime import date

from swingtrade.data.upstox_client import UpstoxAPIError, UpstoxClient

# A sample instrument key from the official Upstox documentation.
INSTRUMENT_KEY = "NSE_EQ|INE848E01016"

# Use a historical period so this check does not depend on today's session.
FROM_DATE = date(2025, 1, 1)
TO_DATE = date(2025, 1, 10)


def main() -> int:
    try:
        with UpstoxClient() as client:
            data = client.get_historical_daily_candles(
                instrument_key=INSTRUMENT_KEY,
                from_date=FROM_DATE,
                to_date=TO_DATE,
            )

        candles = data["candles"]

        print("Upstox API request successful.")
        print(f"Instrument key: {INSTRUMENT_KEY}")
        print(f"Requested period: {FROM_DATE} to {TO_DATE}")
        print(f"Daily candles returned: {len(candles)}")
        return 0

    except UpstoxAPIError as exc:
        print(f"Upstox API check failed: {exc}")
        print("Check the token, network connection, and API response.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
