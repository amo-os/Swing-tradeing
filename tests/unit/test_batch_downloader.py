from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts import fetch_nifty100_batch as batch

UNIVERSE_COLUMNS = [
    "company_name",
    "industry",
    "symbol",
    "series",
    "isin",
    "instrument_key",
    "instrument_type",
    "segment",
]


def make_universe_rows(count: int = 100) -> list[dict[str, str]]:
    rows = []

    for index in range(count):
        isin = f"INE{index:06d}A01001"
        rows.append(
            {
                "company_name": f"Company {index:03d}",
                "industry": "Technology",
                "symbol": f"STOCK{index:03d}",
                "series": "EQ",
                "isin": isin,
                "instrument_key": f"NSE_EQ|{isin}",
                "instrument_type": "EQ",
                "segment": "NSE_EQ",
            }
        )

    return rows


def write_universe(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-02T00:00:00+05:30",
                 "2025-01-03T00:00:00+05:30"]
            ),
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [103.0, 104.0],
            "volume": [1000, 1200],
        }
    )


def test_load_universe_accepts_100_valid_stocks(tmp_path: Path) -> None:
    path = tmp_path / "universe.csv"
    write_universe(path, make_universe_rows())

    rows = batch.load_universe(path)

    assert len(rows) == 100
    assert [row["symbol"] for row in rows] == sorted(
        row["symbol"] for row in rows
    )


def test_load_universe_rejects_wrong_row_count(tmp_path: Path) -> None:
    path = tmp_path / "universe.csv"
    write_universe(path, make_universe_rows(99))

    with pytest.raises(batch.BatchDownloadError, match="exactly 100"):
        batch.load_universe(path)


def test_load_universe_rejects_duplicate_symbols(tmp_path: Path) -> None:
    rows = make_universe_rows()
    rows[1]["symbol"] = rows[0]["symbol"]

    path = tmp_path / "universe.csv"
    write_universe(path, rows)

    with pytest.raises(batch.BatchDownloadError, match="duplicate symbols"):
        batch.load_universe(path)


def test_load_universe_rejects_wrong_segment(tmp_path: Path) -> None:
    rows = make_universe_rows()
    rows[0]["segment"] = "NSE_FO"

    path = tmp_path / "universe.csv"
    write_universe(path, rows)

    with pytest.raises(batch.BatchDownloadError, match="unexpected segment"):
        batch.load_universe(path)


def test_dry_run_makes_no_api_requests_or_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    universe_path = tmp_path / "universe.csv"
    output_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    write_universe(universe_path, make_universe_rows())

    def unexpected_client(*args: object, **kwargs: object) -> None:
        pytest.fail("Dry run must not create an Upstox client")

    monkeypatch.setattr(batch, "UpstoxClient", unexpected_client)

    args = batch.build_parser().parse_args(
        [
            "--from-date", "2025-01-01",
            "--to-date", "2025-12-31",
            "--universe", str(universe_path),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
            "--dry-run",
        ]
    )

    result = batch.run_batch(args)
    output = capsys.readouterr().out

    assert result == 0
    assert "Stocks: 100" in output
    assert "No API requests made" in output
    assert not output_dir.exists()
    assert not report_dir.exists()


def test_validate_existing_file_accepts_valid_ohlcv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.csv"
    make_ohlcv_frame().to_csv(path, index=False)

    valid, count, message = batch.validate_existing_file(
        path,
        date(2025, 1, 1),
        date(2025, 12, 31),
    )

    assert valid is True
    assert count == 2
    assert "completeness not guaranteed" in message


def test_validate_existing_file_rejects_out_of_range_timestamp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out_of_range.csv"
    frame = make_ohlcv_frame()
    frame["timestamp"] = [
        "2024-12-31T00:00:00+05:30",
        "2025-01-03T00:00:00+05:30",
    ]
    frame.to_csv(path, index=False)

    valid, _, message = batch.validate_existing_file(
        path,
        date(2025, 1, 1),
        date(2025, 12, 31),
    )

    assert valid is False
    assert "outside the requested range" in message


def test_validate_existing_file_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_columns.csv"
    pd.DataFrame({"timestamp": ["2025-01-02"]}).to_csv(path, index=False)

    valid, count, message = batch.validate_existing_file(
        path,
        date(2025, 1, 1),
        date(2025, 12, 31),
    )

    assert valid is False
    assert count == 0
    assert "Missing columns" in message


def test_write_report_refuses_to_overwrite_existing_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.csv"
    original = "existing report\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(batch.BatchDownloadError, match="refusing to overwrite"):
        batch.write_report(path, [])

    assert path.read_text(encoding="utf-8") == original


def test_expected_output_path_contains_requested_dates(
    tmp_path: Path,
) -> None:
    path = batch.expected_output_path(
        "NSE_EQ|INE123456789",
        date(2025, 1, 1),
        date(2025, 12, 31),
        tmp_path,
    )

    assert path.parent == tmp_path
    assert "2025-01-01" in path.name
    assert "2025-12-31" in path.name
    assert "|" not in path.name


def test_write_report_replaces_existing_report_when_enabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.csv"
    path.write_text("old report\n", encoding="utf-8")

    batch.write_report(
        path,
        [
            {
                "symbol": "TEST",
                "instrument_key": "NSE_EQ|INE000A01001",
                "status": "success",
                "candles": 2,
                "file": "test.csv",
                "message": "",
            }
        ],
        overwrite=True,
    )

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["status"] == "success"
    assert not (tmp_path / ".report.csv.tmp").exists()


def test_write_report_preserves_original_if_replacement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.csv"
    original = "original report\n"
    path.write_text(original, encoding="utf-8")

    original_replace = Path.replace

    def fail_temporary_replacement(
        self: Path,
        target: str | Path,
    ) -> Path:
        if self.name == ".report.csv.tmp":
            raise OSError("Simulated replacement failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_temporary_replacement)

    with pytest.raises(OSError, match="Simulated replacement failure"):
        batch.write_report(path, [], overwrite=True)

    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".report.csv.tmp").exists()
