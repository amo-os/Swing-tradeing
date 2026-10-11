from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from swingtrade.data.fetch_and_save import (
    FetchAndSaveError,
    fetch_and_save_daily_ohlcv,
)


def parse_date(value: str) -> date:
    """Parse a date supplied in YYYY-MM-DD format."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; use YYYY-MM-DD."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical daily OHLCV candles from Upstox, "
            "validate them, and save them as a CSV file."
        )
    )
    parser.add_argument(
        "--instrument-key",
        required=True,
        help="Upstox instrument key, for example NSE_EQ|INE848E01016",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        type=parse_date,
        help="First requested date, in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        type=parse_date,
        help="Last requested date, in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for the CSV file (default: data/processed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing CSV for the same instrument and dates",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.instrument_key.strip():
        parser.error("--instrument-key must not be empty")

    if args.from_date > args.to_date:
        parser.error("--from-date must not be after --to-date")

    try:
        frame, saved_path = fetch_and_save_daily_ohlcv(
            instrument_key=args.instrument_key,
            from_date=args.from_date,
            to_date=args.to_date,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except FetchAndSaveError as exc:
        print(f"Historical data fetch failed: {exc}")
        return 1

    print("Historical data fetched, validated, and saved.")
    print(f"Instrument key: {args.instrument_key}")
    print(f"Requested period: {args.from_date} to {args.to_date}")
    print(f"Candles saved: {len(frame)}")
    print(f"CSV file: {saved_path}")
    print("Execution mode: data collection only; no orders placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
