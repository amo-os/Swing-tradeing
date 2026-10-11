"""CLI script to audit and report backtest universe approval gate status.

Preserves existing audit files and outputs a traceable backtest universe gate report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swingtrade.backtest.approval_gate import evaluate_backtest_universe

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "data_eligibility.toml"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "nifty100_mapped.csv"
DEFAULT_EXCEPTIONS = ROOT / "data" / "raw" / "data_quality_exceptions.csv"
DEFAULT_ANOMALY_REVIEWS = ROOT / "reports" / "anomaly_review_notes.csv"
DEFAULT_REPORT = ROOT / "reports" / "backtest_universe_gate_v1.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the eligibility TOML configuration",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help="Path to the mapped universe CSV",
    )
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=DEFAULT_EXCEPTIONS,
        help="Path to the data-quality exceptions CSV",
    )
    parser.add_argument(
        "--anomaly-reviews",
        type=Path,
        default=DEFAULT_ANOMALY_REVIEWS,
        help="Path to the anomaly-review CSV",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Report destination path; never overwrites existing files",
    )
    args = parser.parse_args()

    # Safety: Never overwrite existing audit files
    protected_files = {
        "eligibility_audit.csv",
        "eligibility_audit_v2.csv",
        "eligibility_audit_v3.csv",
    }
    if args.report.name in protected_files or args.report.exists():
        parser.exit(1, f"Safety check: refusing to overwrite existing report {args.report}\n")

    try:
        report = evaluate_backtest_universe(
            config_path=args.config,
            universe_path=args.universe,
            exceptions_path=args.exceptions,
            anomaly_reviews_path=args.anomaly_reviews,
            project_root=ROOT,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Universe approval gate failed: {exc}\n")

    df = report.to_dataframe()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.report, index=False)

    total = len(report.decisions)
    approved = report.approved_symbols
    rejected = report.rejected_symbols

    print(f"Universe constituents audited: {total}")
    print(f"Backtest Approved: {len(approved)}")
    print(f"Backtest Barred/Rejected: {len(rejected)}")
    print(f"Report saved to: {args.report}\n")

    print("Rejected constituents detail:")
    for sym in rejected:
        d = report.get_decision(sym)
        assert d is not None
        print(f"  [{sym}] Status: {d.anomaly_review_status} | Eligible: {d.eligible}")
        print(f"    Reasons: {' | '.join(d.reasons)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
