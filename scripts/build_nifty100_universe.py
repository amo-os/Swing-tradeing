from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

CONSTITUENTS_PATH = Path("data/universe/nifty100_constituents.csv")
INSTRUMENTS_PATH = Path("data/raw/upstox_nse_instruments.json.gz")
OUTPUT_PATH = Path("data/universe/nifty100_mapped.csv")

OUTPUT_COLUMNS = [
    "company_name",
    "industry",
    "symbol",
    "series",
    "isin",
    "instrument_key",
    "instrument_type",
    "segment",
]


class UniverseBuildError(RuntimeError):
    """Raised when the Nifty 100 universe cannot be built safely."""


def load_constituents(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise UniverseBuildError(f"Constituent file not found: {path}")

    try:
        with path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            rows = list(csv.DictReader(file))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise UniverseBuildError(
            f"Could not read constituent CSV: {exc}"
        ) from exc

    required = {"Company Name", "Industry", "Symbol", "Series", "ISIN Code"}
    if not rows or not required.issubset(rows[0]):
        raise UniverseBuildError(
            "Constituent CSV is empty or missing required columns"
        )

    if len(rows) != 100:
        raise UniverseBuildError(
            f"Expected 100 Nifty 100 constituents; found {len(rows)}"
        )

    symbols = [row["Symbol"].strip() for row in rows]
    isins = [row["ISIN Code"].strip() for row in rows]

    if any(not value for value in symbols + isins):
        raise UniverseBuildError("A constituent has an empty symbol or ISIN")

    if len(set(symbols)) != len(symbols):
        raise UniverseBuildError("Duplicate constituent symbols detected")

    if len(set(isins)) != len(isins):
        raise UniverseBuildError("Duplicate constituent ISINs detected")

    if any(row["Series"].strip() != "EQ" for row in rows):
        raise UniverseBuildError("Unexpected non-EQ constituent series found")

    return rows


def load_equity_instruments(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise UniverseBuildError(f"Instrument master not found: {path}")

    try:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            instruments = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UniverseBuildError(
            f"Could not read compressed instrument master: {exc}"
        ) from exc

    if not isinstance(instruments, list):
        raise UniverseBuildError("Instrument master must contain a JSON list")

    by_isin: dict[str, list[dict]] = {}

    for item in instruments:
        if not isinstance(item, dict):
            continue

        if item.get("segment") != "NSE_EQ":
            continue

        isin = item.get("isin")
        if isinstance(isin, str) and isin.strip():
            by_isin.setdefault(isin.strip(), []).append(item)

    if not by_isin:
        raise UniverseBuildError("No NSE_EQ instruments with ISINs found")

    return by_isin


def build_universe(
    constituents: list[dict[str, str]],
    instruments_by_isin: dict[str, list[dict]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    errors: list[str] = []

    for row in constituents:
        symbol = row["Symbol"].strip()
        isin = row["ISIN Code"].strip()
        candidates = instruments_by_isin.get(isin, [])

        if len(candidates) != 1:
            errors.append(
                f"{symbol} ({isin}): expected one NSE_EQ match, "
                f"found {len(candidates)}"
            )
            continue

        instrument = candidates[0]
        master_symbol = str(instrument.get("trading_symbol", "")).strip()
        instrument_key = str(instrument.get("instrument_key", "")).strip()
        instrument_type = str(instrument.get("instrument_type", "")).strip()

        if master_symbol != symbol:
            errors.append(
                f"{symbol} ({isin}): master symbol is {master_symbol!r}"
            )
            continue

        if not instrument_key.startswith("NSE_EQ|"):
            errors.append(
                f"{symbol} ({isin}): invalid instrument key {instrument_key!r}"
            )
            continue

        if instrument_type != "EQ":
            errors.append(
                f"{symbol} ({isin}): unexpected instrument type "
                f"{instrument_type!r}"
            )
            continue

        output.append(
            {
                "company_name": row["Company Name"].strip(),
                "industry": row["Industry"].strip(),
                "symbol": symbol,
                "series": row["Series"].strip(),
                "isin": isin,
                "instrument_key": instrument_key,
                "instrument_type": instrument_type,
                "segment": "NSE_EQ",
            }
        )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise UniverseBuildError(
            f"Universe validation failed ({len(errors)} issue(s)):\n{details}"
        )

    if len(output) != 100:
        raise UniverseBuildError(
            f"Expected 100 mapped stocks; produced {len(output)}"
        )

    return sorted(output, key=lambda item: item["symbol"])


def write_universe(rows: list[dict[str, str]], output_path: Path) -> None:
    if output_path.exists():
        raise UniverseBuildError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with temporary_path.open(
            "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        if output_path.exists():
            raise UniverseBuildError(
                f"Output appeared during writing; refusing to overwrite: "
                f"{output_path}"
            )

        temporary_path.rename(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a validated Nifty 100 to Upstox instrument-key mapping."
        )
    )
    parser.add_argument(
        "--constituents",
        type=Path,
        default=CONSTITUENTS_PATH,
    )
    parser.add_argument(
        "--instruments",
        type=Path,
        default=INSTRUMENTS_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )
    args = parser.parse_args()

    try:
        constituents = load_constituents(args.constituents)
        instruments = load_equity_instruments(args.instruments)
        rows = build_universe(constituents, instruments)
        write_universe(rows, args.output)
    except (UniverseBuildError, OSError) as exc:
        print(f"Universe build failed: {exc}", file=sys.stderr)
        return 1

    print("Nifty 100 universe built and validated.")
    print(f"Constituents mapped: {len(rows)}")
    print(f"Output: {args.output}")
    print("Matching method: ISIN, with symbol and instrument-type checks.")
    print("No trading orders were placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
