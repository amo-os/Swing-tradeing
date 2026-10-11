"""Tests for the read-only historical eligibility audit."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_eligibility import audit, load_config, load_universe


def make_candles(rows: int = 220) -> pd.DataFrame:
    """Create structurally valid synthetic daily OHLCV candles."""
    close = [100.0 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=rows, freq="D", tz="Asia/Kolkata"
            ),
            "open": [value - 0.2 for value in close],
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000 + i for i in range(rows)],
        }
    )


def write_universe(path: Path) -> list[dict[str, str]]:
    """Create a valid 100-symbol mapping for isolated audit tests."""
    rows = [
        {
            "symbol": f"STOCK{i:03d}",
            "instrument_key": f"NSE_EQ|TEST{i:03d}",
        }
        for i in range(100)
    ]
    rows[0]["symbol"] = "IDEA"
    rows[0]["instrument_key"] = "NSE_EQ|INE669E01016"
    rows[1]["symbol"] = "SIEMENS"
    rows[1]["instrument_key"] = "NSE_EQ|TEST001"
    rows[2]["symbol"] = "TMPV"
    rows[2]["instrument_key"] = "NSE_EQ|TEST002"
    rows[3]["symbol"] = "ENRIN"
    rows[3]["instrument_key"] = "NSE_EQ|TEST003"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["symbol", "instrument_key"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_test_config(path: Path, dataset_dir: Path) -> None:
    """Write the policy used by the temporary test dataset."""
    path.write_text(
        "[dataset]\n"
        f'directory = "{dataset_dir.as_posix()}"\n'
        "minimum_candles = 200\n"
        'timeframe = "1d"\n'
        "\n"
        "[corporate_action_exclusions]\n"
        'SIEMENS = "Corporate action pending"\n'
        'TMPV = "Corporate action pending"\n',
        encoding="utf-8",
    )


def write_candle_file(
    directory: Path, instrument_key: str, frame: pd.DataFrame
) -> Path:
    """Save synthetic candles using the production filename convention."""
    filename = instrument_key.replace("|", "_") + "_2024-01-01_2025-12-31.csv"
    path = directory / filename
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def audit_setup(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "candles"
    dataset_dir.mkdir()
    universe_path = tmp_path / "universe.csv"
    config_path = tmp_path / "eligibility.toml"
    exceptions_path = tmp_path / "exceptions.csv"
    anomaly_reviews_path = tmp_path / "anomaly_reviews.csv"
    report_path = tmp_path / "report.csv"

    rows = write_universe(universe_path)
    write_test_config(config_path, dataset_dir)

    # Valid histories for all symbols except the intentionally missing one.
    for row in rows:
        if row["symbol"] == "STOCK099":
            continue

        count = 134 if row["symbol"] == "ENRIN" else 220
        frame = make_candles(count)

        if row["symbol"] == "IDEA":
            frame.loc[10, "volume"] = -1

        write_candle_file(dataset_dir, row["instrument_key"], frame)

    exceptions_path.write_text(
        "symbol,instrument_key,date,field,reported_value,issue,disposition\n"
        'IDEA,NSE_EQ|INE669E01016,2024-08-30,volume,-1,Negative volume,'
        '"Quarantined; investigate provider data before backtesting"\n',
        encoding="utf-8",
    )

    with anomaly_reviews_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["symbol", "review_finding", "backtest_disposition"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "symbol": "SIEMENS",
                    "review_finding": "Confirmed demerger",
                    "backtest_disposition": (
                        "Exclude from strategy evaluation until corporate-action "
                        "adjustment and resulting-share treatment are implemented."
                    ),
                },
                {
                    "symbol": "TMPV",
                    "review_finding": "Confirmed demerger",
                    "backtest_disposition": (
                        "Exclude from strategy evaluation until corporate-action "
                        "adjustment and resulting-share treatment are implemented."
                    ),
                },
                {
                    "symbol": "STOCK010",
                    "review_finding": "Potential genuine event",
                    "backtest_disposition": (
                        "Preserve raw data; retain pending broader series validation."
                    ),
                },
            ]
        )

    return {
        "dataset_dir": dataset_dir,
        "universe": universe_path,
        "config": config_path,
        "exceptions": exceptions_path,
        "anomaly_reviews": anomaly_reviews_path,
        "report": report_path,
    }


def run_audit(paths: dict[str, Path]) -> pd.DataFrame:
    return audit(
        config_path=paths["config"],
        universe_path=paths["universe"],
        exceptions_path=paths["exceptions"],
        anomaly_reviews_path=paths["anomaly_reviews"],
        report_path=paths["report"],
    )


def test_audit_checks_all_100_constituents(audit_setup) -> None:
    report = run_audit(audit_setup)

    assert len(report) == 100
    assert report["symbol"].nunique() == 100
    assert "STOCK099" in set(report["symbol"])


def test_missing_file_is_excluded(audit_setup) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "STOCK099"].iloc[0]

    assert not row["eligible"]
    assert "Missing historical candle file" in row["reasons"]


def test_short_history_is_excluded(audit_setup) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "ENRIN"].iloc[0]

    assert not row["eligible"]
    assert row["candle_count"] == 134
    assert "Insufficient history" in row["reasons"]


def test_idea_unresolved_exception_is_excluded(audit_setup) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "IDEA"].iloc[0]

    assert not row["eligible"]
    assert "Unresolved data-quality exception" in row["reasons"]
    assert "Volume cannot be negative" in row["reasons"]


@pytest.mark.parametrize("symbol", ["SIEMENS", "TMPV"])
def test_corporate_action_symbols_are_excluded(audit_setup, symbol) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == symbol].iloc[0]

    assert not row["eligible"]
    assert "Corporate-action exclusion" in row["reasons"]


def test_valid_unexcluded_symbol_is_eligible(audit_setup) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "STOCK010"].iloc[0]

    assert row["eligible"]
    assert row["reasons"] == ""


def test_existing_report_is_never_overwritten(audit_setup) -> None:
    audit_setup["report"].write_text("existing report\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_audit(audit_setup)

    assert audit_setup["report"].read_text(encoding="utf-8") == (
        "existing report\n"
    )


def test_load_config_rejects_non_daily_timeframe(tmp_path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        '[dataset]\n'
        'directory = "candles"\n'
        'minimum_candles = 200\n'
        'timeframe = "1h"\n'
        '\n'
        '[corporate_action_exclusions]\n'
        'SIEMENS = "Pending"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="daily candles"):
        load_config(config_path)


def test_load_universe_rejects_wrong_constituent_count(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    path.write_text(
        "symbol,instrument_key\nABB,NSE_EQ|INE117A01022\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected 100"):
        load_universe(path)


def test_missing_file_still_reports_unresolved_exception(audit_setup) -> None:
    idea_files = list(
        audit_setup["dataset_dir"].glob("NSE_EQ_INE669E01016_*.csv")
    )
    assert len(idea_files) == 1
    idea_files[0].unlink()

    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "IDEA"].iloc[0]

    assert not row["eligible"]
    assert "Missing historical candle file" in row["reasons"]
    assert "Unresolved data-quality exception" in row["reasons"]


def test_pending_anomaly_review_is_reported_without_automatic_exclusion(
    audit_setup,
) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "STOCK010"].iloc[0]

    assert row["eligible"]
    assert row["anomaly_review_status"] == "pending_validation"
    assert "Potential genuine event" in row["anomaly_review_details"]
    assert "pending broader series validation" in row["anomaly_review_details"]


@pytest.mark.parametrize("symbol", ["SIEMENS", "TMPV"])
def test_corporate_action_review_status_is_reported(audit_setup, symbol) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == symbol].iloc[0]

    assert not row["eligible"]
    assert row["anomaly_review_status"] == "corporate_action_exclusion"
    assert "Confirmed demerger" in row["anomaly_review_details"]


def test_no_recorded_anomaly_is_not_marked_as_backtest_approval(
    audit_setup,
) -> None:
    report = run_audit(audit_setup)
    row = report.loc[report["symbol"] == "STOCK011"].iloc[0]

    assert row["anomaly_review_status"] == "no_recorded_anomaly"
    assert "backtest_approved" not in report.columns
