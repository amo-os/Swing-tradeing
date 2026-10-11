"""Unit tests for the backtest-universe approval gate."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from swingtrade.backtest.approval_gate import (
    AnomalyReviewStatus,
    evaluate_backtest_universe,
    evaluate_symbol_approval,
    load_gate_config,
    load_review_records,
    load_universe_records,
)


def make_valid_candles(rows: int = 220) -> pd.DataFrame:
    """Create structurally valid synthetic daily OHLCV candles."""
    close = [100.0 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=rows, freq="D", tz="Asia/Kolkata"
            ),
            "open": [val - 0.2 for val in close],
            "high": [val + 1.0 for val in close],
            "low": [val - 1.0 for val in close],
            "close": close,
            "volume": [1000 + i for i in range(rows)],
        }
    )


def write_test_universe(path: Path, count: int = 100) -> list[dict[str, str]]:
    """Create a synthetic universe CSV."""
    rows = [
        {
            "symbol": f"STOCK{i:03d}",
            "instrument_key": f"NSE_EQ|INE{i:03d}0000000",
        }
        for i in range(count)
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "instrument_key"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


@pytest.fixture
def gate_setup(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "candles"
    dataset_dir.mkdir()
    universe_path = tmp_path / "universe.csv"
    config_path = tmp_path / "gate_config.toml"
    exceptions_path = tmp_path / "exceptions.csv"
    reviews_path = tmp_path / "reviews.csv"

    rows = write_test_universe(universe_path, 100)
    # Define corporate-action exclusion for STOCK001
    config_path.write_text(
        "[dataset]\n"
        f'directory = "{dataset_dir.as_posix()}"\n'
        "minimum_candles = 200\n"
        'timeframe = "1d"\n'
        "\n"
        "[corporate_action_exclusions]\n"
        'STOCK001 = "Corporate action pending resolution"\n',
        encoding="utf-8",
    )

    # Populate candles
    for row in rows:
        sym = row["symbol"]
        key = row["instrument_key"]
        if sym == "STOCK099":
            # Intentionally missing candle file
            continue
        elif sym == "STOCK002":
            # Insufficient candles
            df = make_valid_candles(150)
        elif sym == "STOCK003":
            # Corrupted negative volume
            df = make_valid_candles(220)
            df.loc[5, "volume"] = -100
        else:
            df = make_valid_candles(220)

        filename = key.replace("|", "_") + "_2024-01-01_2025-12-31.csv"
        df.to_csv(dataset_dir / filename, index=False)

    # Exceptions file: STOCK004 quarantined
    exceptions_path.write_text(
        "symbol,instrument_key,date,field,reported_value,issue,disposition\n"
        'STOCK004,NSE_EQ|INE0040000000,2024-05-01,volume,-500,Negative volume,'
        '"Quarantined; investigate provider feed"\n',
        encoding="utf-8",
    )

    # Anomaly reviews
    with reviews_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["symbol", "review_finding", "backtest_disposition"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "symbol": "STOCK005",
                    "review_finding": "High volatility event",
                    "backtest_disposition": "Pending validation; retain for review",
                },
                {
                    "symbol": "STOCK006",
                    "review_finding": "Confirmed bad tick",
                    "backtest_disposition": "Rejected: bad tick distorts pricing",
                },
                {
                    "symbol": "STOCK007",
                    "review_finding": "Investigated price jump",
                    "backtest_disposition": (
                        "Approved: verified against exchange circular"
                    ),
                },
                {
                    "symbol": "STOCK001",
                    "review_finding": "Demerger",
                    "backtest_disposition": (
                        "Exclude from strategy evaluation until corporate-action "
                        "adjustment is implemented"
                    ),
                },
            ]
        )

    return {
        "dataset_dir": dataset_dir,
        "universe_path": universe_path,
        "config_path": config_path,
        "exceptions_path": exceptions_path,
        "reviews_path": reviews_path,
        "tmp_path": tmp_path,
    }


def test_missing_config_fails_closed(gate_setup) -> None:
    with pytest.raises(FileNotFoundError, match="Eligibility configuration not found"):
        load_gate_config(gate_setup["tmp_path"] / "nonexistent.toml")


def test_malformed_config_fails_closed(tmp_path: Path) -> None:
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("invalid [ [ toml syntax", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed TOML"):
        load_gate_config(bad_toml)


def test_config_missing_required_sections_fails_closed(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "cfg.toml"
    bad_cfg.write_text("[dataset]\ndirectory = 'data'\n", encoding="utf-8")
    with pytest.raises(TypeError, match="corporate_action_exclusions"):
        load_gate_config(bad_cfg)


def test_evaluate_symbol_approval_direct(gate_setup) -> None:
    decision = evaluate_symbol_approval(
        symbol="STOCK010",
        instrument_key="NSE_EQ|INE0100000000",
        dataset_dir=gate_setup["dataset_dir"],
        minimum_candles=200,
        corporate_action_exclusions={},
        quarantined_symbols={},
        anomaly_reviews={},
    )
    assert decision.approved
    assert decision.eligible
    assert decision.candle_count == 220



def test_missing_universe_file_fails_closed(gate_setup) -> None:
    with pytest.raises(FileNotFoundError, match="Universe mapping not found"):
        load_universe_records(gate_setup["tmp_path"] / "missing_u.csv")


def test_malformed_universe_missing_columns_fails_closed(tmp_path: Path) -> None:
    bad_u = tmp_path / "bad_u.csv"
    bad_u.write_text("symbol\nSTOCK001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_universe_records(bad_u)


def test_duplicate_symbol_in_universe_fails_closed(tmp_path: Path) -> None:
    bad_u = tmp_path / "dup_u.csv"
    bad_u.write_text(
        "symbol,instrument_key\n"
        "STOCK001,NSE_EQ|KEY1\n"
        "STOCK001,NSE_EQ|KEY2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate symbol"):
        load_universe_records(bad_u, expected_count=None)


def test_duplicate_key_in_universe_fails_closed(tmp_path: Path) -> None:
    bad_u = tmp_path / "dup_k.csv"
    bad_u.write_text(
        "symbol,instrument_key\n"
        "STOCK001,NSE_EQ|KEY1\n"
        "STOCK002,NSE_EQ|KEY1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate instrument key"):
        load_universe_records(bad_u, expected_count=None)


def test_missing_anomaly_reviews_file_fails_closed(gate_setup) -> None:
    with pytest.raises(FileNotFoundError, match="Anomaly-review file not found"):
        load_review_records(gate_setup["tmp_path"] / "missing_r.csv")


def test_duplicate_anomaly_review_symbol_fails_closed(tmp_path: Path) -> None:
    bad_r = tmp_path / "dup_r.csv"
    bad_r.write_text(
        "symbol,review_finding,backtest_disposition\n"
        "STOCK001,finding1,disp1\n"
        "STOCK001,finding2,disp2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate anomaly-review symbol"):
        load_review_records(bad_r)


def test_rejects_corporate_action_exclusion(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK001")
    assert decision is not None
    assert not decision.approved
    assert decision.anomaly_review_status == (
        AnomalyReviewStatus.CORPORATE_ACTION_EXCLUSION.value
    )
    assert any("Corporate-action exclusion" in r for r in decision.reasons)


def test_rejects_insufficient_history(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK002")
    assert decision is not None
    assert not decision.approved
    assert not decision.eligible
    assert any("Insufficient history" in r for r in decision.reasons)


def test_rejects_structural_corruption(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK003")
    assert decision is not None
    assert not decision.approved
    assert not decision.eligible
    assert any("Volume cannot be negative" in r for r in decision.reasons)


def test_rejects_unresolved_data_quality_exception(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK004")
    assert decision is not None
    assert not decision.approved
    assert any("Unresolved data-quality exception" in r for r in decision.reasons)


def test_rejects_pending_anomaly_review(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK005")
    assert decision is not None
    assert not decision.approved
    assert decision.anomaly_review_status == (
        AnomalyReviewStatus.PENDING_VALIDATION.value
    )
    assert any("pending validation" in r.lower() for r in decision.reasons)


def test_rejects_explicit_rejection_in_review(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK006")
    assert decision is not None
    assert not decision.approved
    assert decision.anomaly_review_status == (
        AnomalyReviewStatus.REVIEW_REJECTED.value
    )
    assert any("rejected backtest approval" in r.lower() for r in decision.reasons)


def test_approves_stock_with_explicit_approval_decision(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK007")
    assert decision is not None
    assert decision.approved
    assert decision.eligible
    assert decision.anomaly_review_status == (
        AnomalyReviewStatus.REVIEW_APPROVED.value
    )
    assert any("Explicitly approved" in r for r in decision.reasons)


def test_missing_candle_file_rejected(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK099")
    assert decision is not None
    assert not decision.approved
    assert not decision.eligible
    assert any("Missing historical candle file" in r for r in decision.reasons)


def test_clean_stock_without_anomalies_approved(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK010")
    assert decision is not None
    assert decision.approved
    assert decision.eligible
    assert decision.anomaly_review_status == (
        AnomalyReviewStatus.NO_RECORDED_ANOMALY.value
    )
    assert any("Passed structural eligibility" in r for r in decision.reasons)


def test_no_recorded_anomaly_is_not_proof_of_approval(gate_setup) -> None:
    """A stock with no recorded anomaly but insufficient history must be rejected."""
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK002")
    assert decision is not None
    assert decision.anomaly_review_status == (
        AnomalyReviewStatus.NO_RECORDED_ANOMALY.value
    )
    assert not decision.approved


def test_traceable_sources_recorded(gate_setup) -> None:
    report = evaluate_backtest_universe(
        config_path=gate_setup["config_path"],
        universe_path=gate_setup["universe_path"],
        exceptions_path=gate_setup["exceptions_path"],
        anomaly_reviews_path=gate_setup["reviews_path"],
    )
    decision = report.get_decision("STOCK010")
    assert decision is not None
    assert len(decision.sources) > 0
    assert any("candle_file:" in src for src in decision.sources)


def test_production_universe_evaluation() -> None:
    """Verify that on real project files, approval gate rejects all 12 unapproved stocks."""
    root = Path(__file__).resolve().parents[2]
    config_path = root / "config" / "data_eligibility.toml"
    universe_path = root / "data" / "universe" / "nifty100_mapped.csv"
    exceptions_path = root / "data" / "raw" / "data_quality_exceptions.csv"
    reviews_path = root / "reports" / "anomaly_review_notes.csv"

    report = evaluate_backtest_universe(
        config_path=config_path,
        universe_path=universe_path,
        exceptions_path=exceptions_path,
        anomaly_reviews_path=reviews_path,
        project_root=root,
    )

    assert len(report.decisions) == 100
    approved = report.approved_symbols
    rejected = report.rejected_symbols

    # Exactly 88 approved, 12 rejected
    assert len(approved) == 88
    assert len(rejected) == 12

    # The 7 structural/CA exclusions must be rejected
    for sym in ["ENRIN", "TATACAP", "TMCV", "IDEA", "VAML", "SIEMENS", "TMPV"]:
        assert sym in rejected, f"{sym} should be rejected"
        dec = report.get_decision(sym)
        assert dec is not None and not dec.approved

    # The 5 pending anomaly reviews must be rejected
    for sym in ["ADANIENT", "ADANIGREEN", "ADANIPORTS", "PFC", "POLYCAB"]:
        assert sym in rejected, f"{sym} should be rejected due to pending review"
        dec = report.get_decision(sym)
        assert dec is not None and not dec.approved
        assert dec.anomaly_review_status == "pending_validation"
