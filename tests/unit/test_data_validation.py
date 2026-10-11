import pandas as pd

from swingtrade.data.validation import validate_ohlcv


def valid_data():
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-09-28",
                "open": 100,
                "high": 105,
                "low": 98,
                "close": 103,
                "volume": 1000,
            },
            {
                "timestamp": "2026-09-29",
                "open": 103,
                "high": 108,
                "low": 101,
                "close": 106,
                "volume": 1200,
            },
        ]
    )


def test_valid_data_passes():
    assert validate_ohlcv(valid_data()) == []


def test_invalid_high_is_rejected():
    data = valid_data()
    data.loc[0, "high"] = 99

    assert any("High is below" in e for e in validate_ohlcv(data))


def test_duplicate_timestamps_are_rejected():
    data = valid_data()
    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]

    assert "Duplicate timestamps" in validate_ohlcv(data)


def test_negative_volume_is_rejected():
    data = valid_data()
    data.loc[0, "volume"] = -10

    assert "Volume cannot be negative" in validate_ohlcv(data)


def test_missing_columns_are_rejected():
    data = valid_data().drop(columns=["volume"])

    assert any("Missing required columns" in e for e in validate_ohlcv(data))


def test_empty_data_is_rejected():
    data = valid_data().iloc[0:0]

    assert "Dataset is empty" in validate_ohlcv(data)
