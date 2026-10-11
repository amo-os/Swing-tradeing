from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_nifty100_universe import (
    UniverseBuildError,
    build_universe,
    load_constituents,
    write_universe,
)


def make_constituent(
    symbol: str = "TEST",
    isin: str = "INE000A01001",
) -> dict[str, str]:
    return {
        "Company Name": f"{symbol} Limited",
        "Industry": "Technology",
        "Symbol": symbol,
        "Series": "EQ",
        "ISIN Code": isin,
    }


def make_instrument(
    symbol: str = "TEST",
    isin: str = "INE000A01001",
    instrument_key: str | None = None,
    instrument_type: str = "EQ",
) -> dict:
    return {
        "segment": "NSE_EQ",
        "trading_symbol": symbol,
        "isin": isin,
        "instrument_key": instrument_key or f"NSE_EQ|{isin}",
        "instrument_type": instrument_type,
    }


def make_valid_dataset(count: int = 100):
    constituents = []
    instruments_by_isin = {}

    for index in range(count):
        symbol = f"STOCK{index:03d}"
        isin = f"INE{index:06d}A01001"
        constituents.append(make_constituent(symbol, isin))
        instruments_by_isin[isin] = [make_instrument(symbol, isin)]

    return constituents, instruments_by_isin


def test_build_universe_maps_valid_instrument() -> None:
    constituents, instruments = make_valid_dataset()
    rows = build_universe(constituents, instruments)

    assert len(rows) == 100
    assert rows[0]["symbol"] == "STOCK000"
    assert rows[0]["isin"] == "INE000000A01001"
    assert rows[0]["instrument_key"] == "NSE_EQ|INE000000A01001"
    assert rows[0]["segment"] == "NSE_EQ"
    assert rows[0]["instrument_type"] == "EQ"


def test_build_universe_rejects_missing_instrument() -> None:
    constituents, instruments = make_valid_dataset()
    instruments.pop(constituents[0]["ISIN Code"])

    with pytest.raises(UniverseBuildError, match="found 0"):
        build_universe(constituents, instruments)


def test_build_universe_rejects_ambiguous_isin() -> None:
    constituents, instruments = make_valid_dataset()
    isin = constituents[0]["ISIN Code"]
    instruments[isin].append(make_instrument())

    with pytest.raises(UniverseBuildError, match="found 2"):
        build_universe(constituents, instruments)


def test_build_universe_rejects_symbol_mismatch() -> None:
    constituents, instruments = make_valid_dataset()
    isin = constituents[0]["ISIN Code"]
    instruments[isin] = [make_instrument(symbol="OTHER", isin=isin)]

    with pytest.raises(UniverseBuildError, match="master symbol"):
        build_universe(constituents, instruments)


def test_build_universe_rejects_wrong_instrument_type() -> None:
    constituents, instruments = make_valid_dataset()
    isin = constituents[0]["ISIN Code"]
    symbol = constituents[0]["Symbol"]
    instruments[isin] = [
        make_instrument(symbol=symbol, isin=isin, instrument_type="BE")
    ]

    with pytest.raises(UniverseBuildError, match="instrument type"):
        build_universe(constituents, instruments)


def test_build_universe_rejects_wrong_segment_key() -> None:
    constituents, instruments = make_valid_dataset()
    isin = constituents[0]["ISIN Code"]
    symbol = constituents[0]["Symbol"]
    instruments[isin] = [
        make_instrument(
            symbol=symbol,
            isin=isin,
            instrument_key=f"NSE_FO|{isin}",
        )
    ]

    with pytest.raises(UniverseBuildError, match="invalid instrument key"):
        build_universe(constituents, instruments)


def test_build_universe_sorts_by_symbol() -> None:
    constituents, instruments = make_valid_dataset()
    constituents.reverse()

    rows = build_universe(constituents, instruments)

    assert len(rows) == 100
    assert [row["symbol"] for row in rows] == sorted(
        row["symbol"] for row in rows
    )


def test_load_constituents_rejects_wrong_row_count(tmp_path: Path) -> None:
    path = tmp_path / "constituents.csv"
    columns = ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(make_constituent())

    with pytest.raises(UniverseBuildError, match="Expected 100"):
        load_constituents(path)


def test_load_constituents_rejects_duplicate_symbols(tmp_path: Path) -> None:
    path = tmp_path / "constituents.csv"
    columns = ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for index in range(100):
            row = make_constituent(
                symbol="DUPLICATE" if index < 2 else f"STOCK{index:03d}",
                isin=f"INE{index:06d}A01001",
            )
            writer.writerow(row)

    with pytest.raises(UniverseBuildError, match="Duplicate constituent symbols"):
        load_constituents(path)


def test_write_universe_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "nifty100_mapped.csv"
    path.write_text("existing data\n", encoding="utf-8")

    with pytest.raises(UniverseBuildError, match="refusing to overwrite"):
        write_universe([], path)

    assert path.read_text(encoding="utf-8") == "existing data\n"
