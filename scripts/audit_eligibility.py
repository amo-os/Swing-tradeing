"""Read-only eligibility audit for the configured historical equity dataset."""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path

import pandas as pd

from swingtrade.strategy.eligibility import (
    check_eligibility,
    load_quarantined_symbols,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "data_eligibility.toml"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "nifty100_mapped.csv"
DEFAULT_EXCEPTIONS = ROOT / "data" / "raw" / "data_quality_exceptions.csv"
DEFAULT_ANOMALY_REVIEWS = ROOT / "reports" / "anomaly_review_notes.csv"
DEFAULT_REPORT = ROOT / "reports" / "eligibility_audit.csv"

REPORT_COLUMNS = [
    "symbol",
    "instrument_key",
    "candle_file",
    "candle_count",
    "eligible",
    "reasons",
    "anomaly_review_status",
    "anomaly_review_details",
]


def load_config(path: Path) -> dict:
    """Load and minimally validate the TOML eligibility policy."""
    if not path.is_file():
        raise FileNotFoundError(f"Eligibility configuration not found: {path}")

    with path.open("rb") as file:
        config = tomllib.load(file)

    dataset = config.get("dataset")
    exclusions = config.get("corporate_action_exclusions")

    if not isinstance(dataset, dict) or not isinstance(exclusions, dict):
        raise TypeError("Configuration sections must be TOML tables")

    directory = dataset.get("directory")
    minimum = dataset.get("minimum_candles")
    timeframe = dataset.get("timeframe")

    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("dataset.directory must be a non-empty string")

    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("dataset.minimum_candles must be a positive integer")

    if timeframe != "1d":
        raise ValueError("This audit currently supports daily candles only")

    for symbol, reason in exclusions.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Corporate-action exclusion has an invalid symbol")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Missing exclusion reason for {symbol}")

    return config


def load_universe(path: Path) -> list[dict[str, str]]:
    """Load and validate the mapped Nifty 100 universe."""
    if not path.is_file():
        raise FileNotFoundError(f"Universe mapping not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required = {"symbol", "instrument_key"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Universe mapping is empty or missing required columns")

    normalized: list[dict[str, str]] = []
    seen_symbols: set[str] = set()
    seen_keys: set[str] = set()

    for row in rows:
        symbol = row["symbol"].strip().upper()
        key = row["instrument_key"].strip()

        if not symbol or not key:
            raise ValueError("Universe mapping contains an empty symbol or key")
        if symbol in seen_symbols:
            raise ValueError(f"Duplicate universe symbol: {symbol}")
        if key in seen_keys:
            raise ValueError(f"Duplicate instrument key: {key}")
        if not key.startswith("NSE_EQ|"):
            raise ValueError(f"Unexpected instrument key for {symbol}: {key}")

        seen_symbols.add(symbol)
        seen_keys.add(key)
        normalized.append({"symbol": symbol, "instrument_key": key})

    if len(normalized) != 100:
        raise ValueError(
            f"Expected 100 universe constituents; found {len(normalized)}"
        )

    return sorted(normalized, key=lambda row: row["symbol"])



def load_anomaly_reviews(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate anomaly-review decisions keyed by stock symbol."""
    if not path.is_file():
        raise FileNotFoundError(f"Anomaly-review file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"symbol", "review_finding", "backtest_disposition"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Anomaly-review file is missing required columns: "
                + ", ".join(sorted(required))
            )

        reviews: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip().upper()
            finding = (row.get("review_finding") or "").strip()
            disposition = (row.get("backtest_disposition") or "").strip()

            if not symbol or not finding or not disposition:
                raise ValueError(
                    f"Anomaly-review file has an empty required value "
                    f"on line {line_number}"
                )
            if symbol in reviews:
                raise ValueError(
                    f"Duplicate anomaly-review symbol on line {line_number}: "
                    f"{symbol}"
                )

            reviews[symbol] = {
                "review_finding": finding,
                "backtest_disposition": disposition,
            }

    return reviews


def find_candle_file(directory: Path, instrument_key: str) -> Path | None:
    """Find a unique dataset file for an exact mapped instrument key."""
    encoded_key = instrument_key.replace("|", "_")
    matches = list(directory.glob(f"{encoded_key}_*.csv"))

    if len(matches) > 1:
        raise ValueError(
            f"Multiple candle files found for {instrument_key}: "
            f"{[path.name for path in matches]}"
        )

    return matches[0] if matches else None


def audit(
    *,
    config_path: Path = DEFAULT_CONFIG,
    universe_path: Path = DEFAULT_UNIVERSE,
    exceptions_path: Path = DEFAULT_EXCEPTIONS,
    anomaly_reviews_path: Path = DEFAULT_ANOMALY_REVIEWS,
    report_path: Path = DEFAULT_REPORT,
) -> pd.DataFrame:
    """Audit every universe member without modifying input data."""
    config = load_config(config_path)
    dataset = config["dataset"]
    dataset_dir = (ROOT / dataset["directory"]).resolve()
    minimum = dataset["minimum_candles"]
    exclusions = {
        symbol.strip().upper(): reason.strip()
        for symbol, reason in config["corporate_action_exclusions"].items()
    }

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    universe = load_universe(universe_path)
    quarantined = load_quarantined_symbols(exceptions_path)
    anomaly_reviews = load_anomaly_reviews(anomaly_reviews_path)

    records: list[dict[str, object]] = []

    for item in universe:
        symbol = item["symbol"]
        key = item["instrument_key"]
        candle_path = find_candle_file(dataset_dir, key)
        reasons: list[str] = []
        candle_count = 0

        if candle_path is None:
            reasons.append("Missing historical candle file")
        else:
            frame = pd.read_csv(candle_path)
            result = check_eligibility(
                frame,
                symbol,
                min_candles=minimum,
                quarantined_symbols=quarantined,
            )
            candle_count = result.candle_count
            reasons.extend(result.reasons)

        if symbol in quarantined:
            quarantine_reason = (
                f"Unresolved data-quality exception: {quarantined[symbol]}"
            )
            if not any(
                reason.startswith("Unresolved data-quality exception")
                for reason in reasons
            ):
                reasons.append(quarantine_reason)

        if symbol in exclusions:
            reasons.append(f"Corporate-action exclusion: {exclusions[symbol]}")

        review = anomaly_reviews.get(symbol)
        if symbol in exclusions:
            review_status = "corporate_action_exclusion"
        elif review is None:
            review_status = "no_recorded_anomaly"
        else:
            disposition = review["backtest_disposition"].casefold()
            if "exclude" in disposition and "corporate-action" in disposition:
                review_status = "corporate_action_exclusion"
            elif "pending" in disposition or "further review" in disposition:
                review_status = "pending_validation"
            else:
                review_status = "review_recorded"

        review_details = ""
        if review is not None:
            review_details = (
                review["review_finding"]
                + " | Disposition: "
                + review["backtest_disposition"]
            )

        records.append(
            {
                "symbol": symbol,
                "instrument_key": key,
                "candle_file": candle_path.name if candle_path else "",
                "candle_count": candle_count,
                "eligible": not reasons,
                "reasons": " | ".join(reasons),
                "anomaly_review_status": review_status,
                "anomaly_review_details": review_details,
            }
        )

    report = pd.DataFrame(records, columns=REPORT_COLUMNS)

    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {report_path}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to the eligibility TOML configuration",
    )
    parser.add_argument(
        "--universe", type=Path, default=DEFAULT_UNIVERSE,
        help="Path to the mapped universe CSV",
    )
    parser.add_argument(
        "--exceptions", type=Path, default=DEFAULT_EXCEPTIONS,
        help="Path to the data-quality exceptions CSV",
    )
    parser.add_argument(
        "--anomaly-reviews", type=Path, default=DEFAULT_ANOMALY_REVIEWS,
        help="Path to the anomaly-review CSV",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
        help="New report path; existing files are never overwritten",
    )
    args = parser.parse_args()

    try:
        report = audit(
            config_path=args.config,
            universe_path=args.universe,
            exceptions_path=args.exceptions,
            anomaly_reviews_path=args.anomaly_reviews,
            report_path=args.report,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Eligibility audit failed: {exc}\n")

    eligible = int(report["eligible"].sum())
    excluded = len(report) - eligible

    print(f"Universe members audited: {len(report)}")
    print(f"Eligible: {eligible}")
    print(f"Excluded: {excluded}")
    print(f"Report saved: {args.report}")
    print("\nExclusions by symbol:")
    rejected = report.loc[~report["eligible"], ["symbol", "candle_count", "reasons"]]
    if rejected.empty:
        print("None")
    else:
        print(rejected.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
