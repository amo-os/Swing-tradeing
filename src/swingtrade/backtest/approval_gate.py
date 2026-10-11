"""Backtest-universe approval gate.

Determines whether an equity instrument is permitted to enter a historical
backtest universe based on structural eligibility, unresolved data-quality
exceptions, corporate actions, and documented anomaly-review decisions.
"""

from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from swingtrade.strategy.eligibility import (
    check_eligibility,
    load_quarantined_symbols,
)


class AnomalyReviewStatus(StrEnum):
    """Normalized status of anomaly review for an instrument."""

    NO_RECORDED_ANOMALY = "no_recorded_anomaly"
    PENDING_VALIDATION = "pending_validation"
    CORPORATE_ACTION_EXCLUSION = "corporate_action_exclusion"
    REVIEW_APPROVED = "approved"
    REVIEW_REJECTED = "rejected"
    REVIEW_RECORDED = "review_recorded"


@dataclass(frozen=True)
class ApprovalDecision:
    """Explicit decision determining if a symbol may enter the backtest universe."""

    symbol: str
    instrument_key: str
    approved: bool
    eligible: bool
    anomaly_review_status: str
    candle_count: int
    reasons: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class UniverseApprovalReport:
    """Collection of approval decisions for an audited universe."""

    decisions: tuple[ApprovalDecision, ...]

    @property
    def approved_symbols(self) -> list[str]:
        """Return sorted list of symbols permitted for backtesting."""
        return sorted([d.symbol for d in self.decisions if d.approved])

    @property
    def rejected_symbols(self) -> list[str]:
        """Return sorted list of symbols barred from backtesting."""
        return sorted([d.symbol for d in self.decisions if not d.approved])

    def get_decision(self, symbol: str) -> ApprovalDecision | None:
        """Look up decision for a given symbol."""
        symbol_normalized = symbol.strip().upper()
        for d in self.decisions:
            if d.symbol == symbol_normalized:
                return d
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert decisions to a pandas DataFrame."""
        records = [
            {
                "symbol": d.symbol,
                "instrument_key": d.instrument_key,
                "approved": d.approved,
                "eligible": d.eligible,
                "anomaly_review_status": d.anomaly_review_status,
                "candle_count": d.candle_count,
                "reasons": " | ".join(d.reasons),
                "sources": " | ".join(d.sources),
            }
            for d in self.decisions
        ]
        return pd.DataFrame(records)


def load_gate_config(config_path: str | Path) -> dict:
    """Load and strictly validate the TOML eligibility configuration.

    Fails closed if the file is missing, unreadable, or structurally invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Eligibility configuration not found: {path}")

    try:
        with path.open("rb") as file:
            config = tomllib.load(file)
    except Exception as exc:
        raise ValueError(f"Malformed TOML in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a table")

    dataset = config.get("dataset")
    exclusions = config.get("corporate_action_exclusions")

    if not isinstance(dataset, dict):
        raise TypeError("Missing or invalid [dataset] section in config")
    if not isinstance(exclusions, dict):
        raise TypeError(
            "Missing or invalid [corporate_action_exclusions] section in config"
        )

    directory = dataset.get("directory")
    minimum = dataset.get("minimum_candles")
    timeframe = dataset.get("timeframe")

    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("dataset.directory must be a non-empty string")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("dataset.minimum_candles must be a positive integer")
    if timeframe != "1d":
        raise ValueError(
            f"Unsupported timeframe '{timeframe}': only '1d' daily candles supported"
        )

    for sym, reason in exclusions.items():
        if not isinstance(sym, str) or not sym.strip():
            raise ValueError("Corporate-action exclusion contains an empty symbol")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Missing exclusion reason for corporate action: {sym}")

    return config


def load_universe_records(
    universe_path: str | Path, expected_count: int | None = 100
) -> list[dict[str, str]]:
    """Load and validate universe constituents.

    Fails closed on missing file, duplicate symbols/keys, or missing columns.
    """
    path = Path(universe_path)
    if not path.is_file():
        raise FileNotFoundError(f"Universe mapping not found: {path}")

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("Universe mapping file is empty")
            required = {"symbol", "instrument_key"}
            if not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Universe mapping missing required columns: {sorted(required)}"
                )
            rows = list(reader)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to parse universe file {path}: {exc}") from exc

    normalized: list[dict[str, str]] = []
    seen_symbols: set[str] = set()
    seen_keys: set[str] = set()

    for line_idx, row in enumerate(rows, start=2):
        sym = (row.get("symbol") or "").strip().upper()
        key = (row.get("instrument_key") or "").strip()

        if not sym or not key:
            raise ValueError(
                f"Universe mapping contains empty symbol/key on line {line_idx}"
            )
        if sym in seen_symbols:
            raise ValueError(f"Duplicate symbol in universe: {sym}")
        if key in seen_keys:
            raise ValueError(f"Duplicate instrument key in universe: {key}")
        if not key.startswith("NSE_EQ|"):
            raise ValueError(f"Unexpected instrument key format for {sym}: {key}")

        seen_symbols.add(sym)
        seen_keys.add(key)
        normalized.append({"symbol": sym, "instrument_key": key})

    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} universe constituents; found {len(normalized)}"
        )

    return sorted(normalized, key=lambda item: item["symbol"])


def load_review_records(
    reviews_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Load anomaly-review records keyed by stock symbol.

    Fails closed if the file is missing, has duplicate symbols, or missing columns.
    """
    path = Path(reviews_path)
    if not path.is_file():
        raise FileNotFoundError(f"Anomaly-review file not found: {path}")

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("Anomaly-review file is empty")
            required = {"symbol", "review_finding", "backtest_disposition"}
            if not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Anomaly-review file missing required columns: {sorted(required)}"
                )

            reviews: dict[str, dict[str, str]] = {}
            for line_no, row in enumerate(reader, start=2):
                sym = (row.get("symbol") or "").strip().upper()
                finding = (row.get("review_finding") or "").strip()
                disposition = (row.get("backtest_disposition") or "").strip()

                if not sym or not finding or not disposition:
                    raise ValueError(
                        f"Empty required field on line {line_no} of anomaly reviews"
                    )
                if sym in reviews:
                    raise ValueError(
                        f"Duplicate anomaly-review symbol on line {line_no}: {sym}"
                    )

                reviews[sym] = {
                    "review_finding": finding,
                    "backtest_disposition": disposition,
                }
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to parse anomaly reviews in {path}: {exc}") from exc

    return reviews


def classify_anomaly_disposition(
    disposition: str, is_ca_excluded: bool
) -> tuple[AnomalyReviewStatus, str]:
    """Parse disposition text to categorize review status."""
    if is_ca_excluded:
        return (
            AnomalyReviewStatus.CORPORATE_ACTION_EXCLUSION,
            "Corporate-action exclusion configured",
        )

    disp_lower = disposition.casefold().strip()

    if disp_lower.startswith(("approved:", "backtest_approved:")):
        detail = disposition.split(":", 1)[1].strip()
        return AnomalyReviewStatus.REVIEW_APPROVED, detail

    if "exclude" in disp_lower and "corporate-action" in disp_lower:
        return (
            AnomalyReviewStatus.CORPORATE_ACTION_EXCLUSION,
            disposition,
        )

    if (
        "pending" in disp_lower
        or "further review" in disp_lower
        or "retain the event" in disp_lower
    ):
        return (
            AnomalyReviewStatus.PENDING_VALIDATION,
            disposition,
        )

    if (
        disp_lower.startswith(("rejected:", "exclude:"))
        or "rejected" in disp_lower
    ):
        return AnomalyReviewStatus.REVIEW_REJECTED, disposition

    return AnomalyReviewStatus.REVIEW_RECORDED, disposition


def find_candle_file_for_key(directory: Path, instrument_key: str) -> Path | None:
    """Locate unique candle CSV file for instrument key."""
    encoded_key = instrument_key.replace("|", "_")
    matches = sorted(directory.glob(f"{encoded_key}_*.csv"))
    if len(matches) > 1:
        raise ValueError(
            f"Multiple candle files found for key {instrument_key}: "
            f"{[p.name for p in matches]}"
        )
    return matches[0] if matches else None


def evaluate_symbol_approval(
    symbol: str,
    instrument_key: str,
    *,
    dataset_dir: Path,
    minimum_candles: int,
    corporate_action_exclusions: dict[str, str],
    quarantined_symbols: dict[str, str],
    anomaly_reviews: dict[str, dict[str, str]],
    source_refs: dict[str, Path] | None = None,
) -> ApprovalDecision:
    """Evaluate a single instrument for backtest universe approval.

    Enforces all approval gate rules:
    - Rejects structurally ineligible datasets.
    - Rejects unresolved data-quality exceptions.
    - Rejects configured corporate-action exclusions.
    - Rejects pending anomaly reviews.
    - Requires explicit approval for any instrument with a recorded anomaly.
    - Treats 'no_recorded_anomaly' as absence of review, not proof of approval.
    """
    symbol_norm = symbol.strip().upper()
    reasons: list[str] = []
    sources: list[str] = []
    candle_count = 0
    eligible = False

    ca_reason = corporate_action_exclusions.get(symbol_norm)
    is_ca_excluded = ca_reason is not None

    candle_file = find_candle_file_for_key(dataset_dir, instrument_key)
    if candle_file is None:
        reasons.append("Missing historical candle file")
        if source_refs and "dataset_dir" in source_refs:
            sources.append(f"dataset:{source_refs['dataset_dir'].name}")
    else:
        sources.append(f"candle_file:{candle_file.name}")
        try:
            frame = pd.read_csv(candle_file)
            check_res = check_eligibility(
                frame,
                symbol_norm,
                min_candles=minimum_candles,
                quarantined_symbols=quarantined_symbols,
            )
            candle_count = check_res.candle_count
            eligible = check_res.eligible
            if not check_res.eligible:
                reasons.extend(check_res.reasons)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            reasons.append(f"Corrupt or unreadable candle file: {exc}")
            eligible = False

    # Check unresolved data-quality exception
    if symbol_norm in quarantined_symbols:
        q_detail = quarantined_symbols[symbol_norm]
        q_msg = f"Unresolved data-quality exception: {q_detail}"
        if not any("Unresolved data-quality exception" in r for r in reasons):
            reasons.append(q_msg)
        if source_refs and "exceptions" in source_refs:
            sources.append(f"exceptions:{source_refs['exceptions'].name}")

    # Check corporate-action exclusion
    if is_ca_excluded:
        reasons.append(f"Corporate-action exclusion: {ca_reason}")
        if source_refs and "config" in source_refs:
            sources.append(f"config:{source_refs['config'].name}")

    # Check anomaly review
    review = anomaly_reviews.get(symbol_norm)
    if review is not None:
        if source_refs and "anomaly_reviews" in source_refs:
            sources.append(f"anomaly_reviews:{source_refs['anomaly_reviews'].name}")
        status, detail = classify_anomaly_disposition(
            review["backtest_disposition"], is_ca_excluded
        )
    else:
        if is_ca_excluded:
            status = AnomalyReviewStatus.CORPORATE_ACTION_EXCLUSION
            detail = ca_reason or ""
        else:
            status = AnomalyReviewStatus.NO_RECORDED_ANOMALY
            detail = ""

    approved = False

    if is_ca_excluded or symbol_norm in quarantined_symbols or not eligible:
        approved = False
    elif status == AnomalyReviewStatus.PENDING_VALIDATION:
        approved = False
        reasons.append(
            f"Anomaly review is pending validation: {detail}. "
            "Explicit approval required before backtesting."
        )
    elif status == AnomalyReviewStatus.REVIEW_REJECTED:
        approved = False
        reasons.append(f"Anomaly review rejected backtest approval: {detail}")
    elif status == AnomalyReviewStatus.REVIEW_RECORDED:
        approved = False
        reasons.append(
            f"Anomaly review recorded without explicit approval: {detail}. "
            "Requires explicit approval decision."
        )
    elif status == AnomalyReviewStatus.REVIEW_APPROVED:
        # Explicit approval documented and passed structural checks
        approved = True
        reasons.append(f"Explicitly approved following anomaly review: {detail}")
    elif status == AnomalyReviewStatus.NO_RECORDED_ANOMALY:
        # No review record: eligible, no exceptions, no corporate action
        approved = True
        reasons.append("Passed structural eligibility checks with no recorded anomalies")

    return ApprovalDecision(
        symbol=symbol_norm,
        instrument_key=instrument_key,
        approved=approved,
        eligible=eligible,
        anomaly_review_status=status.value,
        candle_count=candle_count,
        reasons=tuple(reasons),
        sources=tuple(sources),
    )


def evaluate_backtest_universe(
    *,
    config_path: str | Path,
    universe_path: str | Path,
    exceptions_path: str | Path,
    anomaly_reviews_path: str | Path,
    project_root: str | Path | None = None,
) -> UniverseApprovalReport:
    """Evaluate entire universe against approval gate rules.

    Fails closed if any required input file or configuration is missing or malformed.
    """
    root = Path(project_root) if project_root else Path.cwd()
    c_path = Path(config_path) if Path(config_path).is_absolute() else root / config_path
    u_path = Path(universe_path) if Path(universe_path).is_absolute() else root / universe_path
    e_path = (
        Path(exceptions_path)
        if Path(exceptions_path).is_absolute()
        else root / exceptions_path
    )
    a_path = (
        Path(anomaly_reviews_path)
        if Path(anomaly_reviews_path).is_absolute()
        else root / anomaly_reviews_path
    )

    config = load_gate_config(c_path)
    dataset_cfg = config["dataset"]
    rel_dataset_dir = dataset_cfg["directory"]
    dataset_dir = (
        Path(rel_dataset_dir)
        if Path(rel_dataset_dir).is_absolute()
        else root / rel_dataset_dir
    ).resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    minimum_candles = dataset_cfg["minimum_candles"]
    corporate_action_exclusions = {
        s.strip().upper(): r.strip()
        for s, r in config["corporate_action_exclusions"].items()
    }

    universe = load_universe_records(u_path)
    quarantined = load_quarantined_symbols(e_path)
    anomaly_reviews = load_review_records(a_path)

    source_refs = {
        "config": c_path,
        "universe": u_path,
        "exceptions": e_path,
        "anomaly_reviews": a_path,
        "dataset_dir": dataset_dir,
    }

    decisions: list[ApprovalDecision] = []
    for item in universe:
        decision = evaluate_symbol_approval(
            symbol=item["symbol"],
            instrument_key=item["instrument_key"],
            dataset_dir=dataset_dir,
            minimum_candles=minimum_candles,
            corporate_action_exclusions=corporate_action_exclusions,
            quarantined_symbols=quarantined,
            anomaly_reviews=anomaly_reviews,
            source_refs=source_refs,
        )
        decisions.append(decision)

    return UniverseApprovalReport(decisions=tuple(decisions))
