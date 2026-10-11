"""Eligibility checks for historical swing-trading research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class EligibilityResult:
    """Eligibility decision and reasons for a single instrument."""

    eligible: bool
    symbol: str
    candle_count: int
    reasons: tuple[str, ...]


def load_quarantined_symbols(
    exceptions_path: str | Path,
) -> dict[str, str]:
    """Load unresolved data-quality exceptions keyed by stock symbol.

    An exception remains blocking unless its disposition explicitly starts
    with 'resolved:' (case-insensitive). Missing exception files are treated
    as an empty list; malformed existing files raise an error.
    """
    path = Path(exceptions_path)

    if not path.exists():
        return {}

    exceptions = pd.read_csv(path)

    required = {"symbol", "disposition"}
    missing = required - set(exceptions.columns)
    if missing:
        raise ValueError(
            f"Exception file is missing columns: {sorted(missing)}"
        )

    quarantined: dict[str, str] = {}

    for _, row in exceptions.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        disposition = str(row["disposition"]).strip()

        if not symbol or symbol == "NAN":
            raise ValueError("Exception file contains an empty symbol")

        if not disposition or disposition.upper() == "NAN":
            disposition = "Unresolved exception"

        if not disposition.lower().startswith("resolved:"):
            quarantined[symbol] = disposition

    return quarantined


def check_eligibility(
    frame: pd.DataFrame,
    symbol: str,
    *,
    min_candles: int = 200,
    quarantined_symbols: dict[str, str] | None = None,
) -> EligibilityResult:
    """Check whether OHLCV history is eligible for strategy research.

    This function does not modify the input frame, adjust prices, calculate
    indicators, or generate trading signals.
    """
    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        raise ValueError("Symbol cannot be empty")

    if min_candles < 1:
        raise ValueError("min_candles must be at least 1")

    reasons: list[str] = []

    if normalized_symbol in (quarantined_symbols or {}):
        detail = (quarantined_symbols or {})[normalized_symbol]
        reasons.append(f"Unresolved data-quality exception: {detail}")

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        reasons.append(f"Missing required columns: {sorted(missing)}")
        return EligibilityResult(
            False, normalized_symbol, len(frame), tuple(reasons)
        )

    if frame.empty:
        reasons.append("Dataset is empty")
        return EligibilityResult(
            False, normalized_symbol, 0, tuple(reasons)
        )

    if len(frame) < min_candles:
        reasons.append(
            f"Insufficient history: {len(frame)} candles; "
            f"at least {min_candles} required"
        )

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        reasons.append("Invalid or missing timestamps")
    else:
        if timestamps.duplicated().any():
            reasons.append("Duplicate timestamps")
        if not timestamps.is_monotonic_increasing:
            reasons.append("Timestamps are not in ascending order")

    numeric_columns = (*PRICE_COLUMNS, "volume")
    numeric = frame.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if numeric.isna().any().any():
        reasons.append("OHLCV contains missing or non-numeric values")
    else:
        if not numeric.apply(
            lambda column: column.map(lambda value: abs(value) != float("inf"))
        ).all().all():
            reasons.append("OHLCV contains infinite values")
        else:
            if (numeric.loc[:, PRICE_COLUMNS] <= 0).any().any():
                reasons.append("OHLC prices must be positive")

            if (numeric["volume"] < 0).any():
                reasons.append("Volume cannot be negative")

            if (numeric["high"] < numeric["low"]).any():
                reasons.append("High cannot be below low")

            if (
                numeric["high"]
                < numeric.loc[:, ["open", "close"]].max(axis=1)
            ).any():
                reasons.append("High cannot be below open or close")

            if (
                numeric["low"]
                > numeric.loc[:, ["open", "close"]].min(axis=1)
            ).any():
                reasons.append("Low cannot be above open or close")

    return EligibilityResult(
        eligible=not reasons,
        symbol=normalized_symbol,
        candle_count=len(frame),
        reasons=tuple(reasons),
    )
