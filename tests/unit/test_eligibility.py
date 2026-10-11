"""Unit tests for strategy dataset eligibility."""

import numpy as np
import pandas as pd
import pytest

from swingtrade.strategy.eligibility import (
    check_eligibility,
    load_quarantined_symbols,
)


def make_ohlcv(rows: int = 220) -> pd.DataFrame:
    """Create deterministic, structurally valid daily candles."""
    index = np.arange(rows)
    close = 100.0 + index * 0.1

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=rows, freq="D"
            ),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000 + index * 10,
        }
    )


def test_accepts_valid_history_with_enough_candles() -> None:
    result = check_eligibility(make_ohlcv(), "ABB")

    assert result.eligible
    assert result.symbol == "ABB"
    assert result.candle_count == 220
    assert result.reasons == ()


def test_rejects_history_below_minimum() -> None:
    result = check_eligibility(make_ohlcv(199), "ENRIN")

    assert not result.eligible
    assert any("Insufficient history" in reason for reason in result.reasons)


def test_supports_custom_minimum_history() -> None:
    result = check_eligibility(
        make_ohlcv(134), "ENRIN", min_candles=100
    )

    assert result.eligible


def test_rejects_negative_volume() -> None:
    frame = make_ohlcv()
    frame.loc[10, "volume"] = -1

    result = check_eligibility(frame, "TEST")

    assert not result.eligible
    assert "Volume cannot be negative" in result.reasons


def test_rejects_infinite_prices() -> None:
    frame = make_ohlcv()
    frame.loc[10, "close"] = float("inf")

    result = check_eligibility(frame, "TEST")

    assert not result.eligible
    assert any("infinite" in reason for reason in result.reasons)


def test_rejects_duplicate_timestamps() -> None:
    frame = make_ohlcv()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

    result = check_eligibility(frame, "TEST")

    assert not result.eligible
    assert "Duplicate timestamps" in result.reasons


def test_rejects_invalid_ohlc_relationship() -> None:
    frame = make_ohlcv()
    frame.loc[10, "high"] = frame.loc[10, "low"] - 1

    result = check_eligibility(frame, "TEST")

    assert not result.eligible
    assert "High cannot be below low" in result.reasons


def test_rejects_unresolved_quarantined_symbol() -> None:
    result = check_eligibility(
        make_ohlcv(),
        "IDEA",
        quarantined_symbols={
            "IDEA": "Negative volume on 2024-08-30"
        },
    )

    assert not result.eligible
    assert any(
        "Unresolved data-quality exception" in reason
        for reason in result.reasons
    )


def test_loads_unresolved_exceptions(tmp_path) -> None:
    path = tmp_path / "exceptions.csv"
    path.write_text(
        "symbol,disposition\n"
        'IDEA,"Quarantined; investigate provider data"\n'
        'TEST,"resolved: verified against official source"\n',
        encoding="utf-8",
    )

    result = load_quarantined_symbols(path)

    assert "IDEA" in result
    assert "TEST" not in result


def test_missing_exception_file_means_no_exceptions(tmp_path) -> None:
    assert load_quarantined_symbols(tmp_path / "missing.csv") == {}


def test_does_not_modify_input_dataframe() -> None:
    frame = make_ohlcv()
    original = frame.copy(deep=True)

    check_eligibility(frame, "TEST")

    pd.testing.assert_frame_equal(frame, original)


def test_rejects_missing_required_columns() -> None:
    frame = make_ohlcv().drop(columns=["close"])

    result = check_eligibility(frame, "TEST")

    assert not result.eligible
    assert any("Missing required columns" in reason for reason in result.reasons)


def test_rejects_empty_dataset() -> None:
    frame = make_ohlcv(0)

    result = check_eligibility(frame, "TEST")

    assert not result.eligible
    assert "Dataset is empty" in result.reasons


def test_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        check_eligibility(make_ohlcv(), "  ")


def test_rejects_invalid_minimum_history() -> None:
    with pytest.raises(ValueError, match="min_candles"):
        check_eligibility(make_ohlcv(), "TEST", min_candles=0)
