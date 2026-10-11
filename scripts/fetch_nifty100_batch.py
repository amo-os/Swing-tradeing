from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from swingtrade.data.pipeline import (
    DataPipelineError,
    fetch_validated_daily_ohlcv,
)
from swingtrade.data.storage import (
    DataStorageError,
    save_validated_ohlcv,
)
from swingtrade.data.upstox_client import UpstoxAPIError, UpstoxClient
from swingtrade.data.validation import REQUIRED_COLUMNS, validate_ohlcv

DEFAULT_UNIVERSE = Path("data/universe/nifty100_mapped.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REPORT_DIR = Path("reports")

REPORT_COLUMNS = [
    "symbol",
    "instrument_key",
    "status",
    "candles",
    "file",
    "message",
]


class BatchDownloadError(RuntimeError):
    """Raised when the batch download cannot be configured safely."""


def parse_date(value: str) -> date:
    """Parse a date supplied in YYYY-MM-DD format."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; use YYYY-MM-DD."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download validated daily OHLCV data for the mapped Nifty 100. "
            "This tool collects data only and never places orders."
        )
    )
    parser.add_argument(
        "--from-date",
        required=True,
        type=parse_date,
        help="First requested date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        type=parse_date,
        help="Last requested date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help=f"Mapped universe CSV (default: {DEFAULT_UNIVERSE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for daily CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Directory for the batch report (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview the plan without contacting Upstox",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing files that pass structural OHLCV validation",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing files for the requested date range",
    )

    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Allow replacing an existing report for the same date range",
    )
    return parser


def load_universe(path: Path) -> list[dict[str, str]]:
    """Load and strictly validate the mapped Nifty 100 CSV."""
    if not path.is_file():
        raise BatchDownloadError(f"Universe file not found: {path}")

    required = {
        "company_name",
        "symbol",
        "isin",
        "instrument_key",
        "instrument_type",
        "segment",
    }

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not required.issubset(
                reader.fieldnames
            ):
                raise BatchDownloadError(
                    f"Universe CSV is missing required columns: "
                    f"{sorted(required)}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BatchDownloadError(f"Could not read universe CSV: {exc}") from exc

    if len(rows) != 100:
        raise BatchDownloadError(
            f"Expected exactly 100 universe rows; found {len(rows)}"
        )

    symbols = [row["symbol"].strip() for row in rows]
    isins = [row["isin"].strip() for row in rows]
    keys = [row["instrument_key"].strip() for row in rows]

    if any(not value for value in symbols + isins + keys):
        raise BatchDownloadError(
            "Universe contains an empty symbol, ISIN, or instrument key"
        )

    if len(set(symbols)) != 100:
        raise BatchDownloadError("Universe contains duplicate symbols")

    if len(set(isins)) != 100:
        raise BatchDownloadError("Universe contains duplicate ISINs")

    if len(set(keys)) != 100:
        raise BatchDownloadError("Universe contains duplicate instrument keys")

    for row in rows:
        if row["segment"].strip() != "NSE_EQ":
            raise BatchDownloadError(
                f"{row['symbol']}: unexpected segment {row['segment']!r}"
            )
        if row["instrument_type"].strip() != "EQ":
            raise BatchDownloadError(
                f"{row['symbol']}: unexpected instrument type "
                f"{row['instrument_type']!r}"
            )
        if not row["instrument_key"].strip().startswith("NSE_EQ|"):
            raise BatchDownloadError(
                f"{row['symbol']}: invalid NSE equity instrument key"
            )

    return sorted(rows, key=lambda row: row["symbol"].strip())


def expected_output_path(
    instrument_key: str,
    from_date: date,
    to_date: date,
    output_dir: Path,
) -> Path:
    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+", "_", instrument_key
    ).strip("_")
    if not safe_name:
        raise BatchDownloadError("Instrument key produced an invalid filename")

    filename = f"{safe_name}_{from_date.isoformat()}_{to_date.isoformat()}.csv"
    return output_dir / filename


def validate_existing_file(
    path: Path,
    from_date: date,
    to_date: date,
) -> tuple[bool, int, str]:
    """Check existing file structure and timestamp range before resuming."""
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return False, 0, f"Cannot read existing CSV: {exc}"

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        return False, 0, f"Missing columns: {sorted(missing)}"

    frame = frame[list(REQUIRED_COLUMNS)]
    errors = validate_ohlcv(frame)
    if errors:
        return False, len(frame), "; ".join(errors)

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    dates = timestamps.dt.date

    if (dates < from_date).any() or (dates > to_date).any():
        return False, len(frame), "Timestamps fall outside the requested range"

    return True, len(frame), "Structurally valid; completeness not guaranteed"


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    overwrite: bool = False,
) -> None:
    """Write a report safely, optionally replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    try:
        with temporary.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()

        if path.exists() and not overwrite:
            raise BatchDownloadError(
                f"Report already exists; refusing to overwrite: {path}"
            )

        if overwrite:
            temporary.replace(path)
        else:
            temporary.rename(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_batch(args: argparse.Namespace) -> int:
    if args.from_date > args.to_date:
        raise BatchDownloadError(
            "--from-date must not be after --to-date"
        )

    if args.delay_seconds < 0:
        raise BatchDownloadError("--delay-seconds cannot be negative")

    universe = load_universe(args.universe)
    report_path = args.report_dir / (
        f"nifty100_download_report_{args.from_date.isoformat()}_"
        f"{args.to_date.isoformat()}.csv"
    )

    if report_path.exists() and not args.overwrite_report and not args.dry_run:
        raise BatchDownloadError(
            f"Report already exists; refusing to overwrite: {report_path}. "
            "Choose another report directory or pass --overwrite-report."
        )

    print(f"Universe: {args.universe}")
    print(f"Stocks: {len(universe)}")
    print(f"Period: {args.from_date} to {args.to_date}")
    print(f"Output directory: {args.output_dir}")
    print(f"Delay between requests: {args.delay_seconds} seconds")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DATA DOWNLOAD ONLY'}")
    print()

    if args.dry_run:
        existing = 0
        for row in universe:
            path = expected_output_path(
                row["instrument_key"],
                args.from_date,
                args.to_date,
                args.output_dir,
            )
            if path.exists():
                existing += 1
        print(f"Existing files for this exact period: {existing}")
        print("No API requests made. No files changed.")
        print("No trading orders placed.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    successes = 0
    skipped = 0
    failures = 0

    with UpstoxClient() as client:
        for index, row in enumerate(universe, start=1):
            symbol = row["symbol"].strip()
            instrument_key = row["instrument_key"].strip()
            destination = expected_output_path(
                instrument_key,
                args.from_date,
                args.to_date,
                args.output_dir,
            )

            print(f"[{index}/100] {symbol}: ", end="", flush=True)

            if destination.exists() and args.resume:
                valid, count, message = validate_existing_file(
                    destination,
                    args.from_date,
                    args.to_date,
                )
                if valid:
                    print(f"SKIPPED ({count} candles; completeness unverified)")
                    results.append(
                        {
                            "symbol": symbol,
                            "instrument_key": instrument_key,
                            "status": "skipped",
                            "candles": count,
                            "file": str(destination),
                            "message": message,
                        }
                    )
                    skipped += 1
                    continue

                print(f"EXISTING FILE INVALID ({message})")
                results.append(
                    {
                        "symbol": symbol,
                        "instrument_key": instrument_key,
                        "status": "failed",
                        "candles": count,
                        "file": str(destination),
                        "message": f"Resume refused: {message}",
                    }
                )
                failures += 1
                continue

            try:
                frame = fetch_validated_daily_ohlcv(
                    client=client,
                    instrument_key=instrument_key,
                    from_date=args.from_date,
                    to_date=args.to_date,
                )
                saved_path = save_validated_ohlcv(
                    frame=frame,
                    instrument_key=instrument_key,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                )
            except (
                UpstoxAPIError,
                DataPipelineError,
                DataStorageError,
            ) as exc:
                print(f"FAILED ({exc})")
                results.append(
                    {
                        "symbol": symbol,
                        "instrument_key": instrument_key,
                        "status": "failed",
                        "candles": 0,
                        "file": str(destination),
                        "message": str(exc),
                    }
                )
                failures += 1
            else:
                print(f"OK ({len(frame)} candles)")
                results.append(
                    {
                        "symbol": symbol,
                        "instrument_key": instrument_key,
                        "status": "success",
                        "candles": len(frame),
                        "file": str(saved_path),
                        "message": "",
                    }
                )
                successes += 1

            if index < len(universe) and args.delay_seconds:
                time.sleep(args.delay_seconds)

    write_report(
        report_path,
        results,
        overwrite=args.overwrite_report,
    )

    print()
    print("Batch finished.")
    print(f"Successful downloads: {successes}")
    print(f"Skipped validated files: {skipped}")
    print(f"Failures: {failures}")
    print(f"Report: {report_path}")
    print("Execution mode: data collection only; no orders placed.")

    return 1 if failures else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return run_batch(args)
    except (BatchDownloadError, UpstoxAPIError, OSError) as exc:
        print(f"Batch download could not run safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
