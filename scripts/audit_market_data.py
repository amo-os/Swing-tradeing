from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed_2024_2025_retry")
UNIVERSE_FILE = Path("data/universe/nifty100_mapped.csv")
REPORT_DIR = Path("reports")

MIN_HISTORY = 200
LARGE_MOVE_THRESHOLD = 0.20


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    universe = pd.read_csv(UNIVERSE_FILE)
    names = universe.set_index("instrument_key")[
        ["symbol", "company_name"]
    ]

    records = []

    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path)

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print(f"SKIP: Missing OHLCV columns in {path.name}")
            continue

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        )
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep=False).reset_index(
            drop=True
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        key = "NSE_EQ|" + path.name.split("_")[2]
        stock = names.loc[key] if key in names.index else None
        symbol = stock["symbol"] if stock is not None else path.name
        company = stock["company_name"] if stock is not None else "Unknown"

        if len(df) < MIN_HISTORY:
            records.append({
                "symbol": symbol,
                "company_name": company,
                "date": "",
                "issue": "SHORT_HISTORY",
                "detail": f"Only {len(df)} candles; EMA-200 needs sufficient warm-up",
                "disposition": "Exclude from EMA-200 signals pending more history",
            })

        returns = df["close"].pct_change()

        for i in range(1, len(df)):
            change = returns.iloc[i]

            if pd.notna(change) and abs(change) >= LARGE_MOVE_THRESHOLD:
                event_date = df.iloc[i]["timestamp"].tz_convert(
                    "Asia/Kolkata"
                ).date().isoformat()

                records.append({
                    "symbol": symbol,
                    "company_name": company,
                    "date": event_date,
                    "issue": "LARGE_CLOSE_CHANGE",
                    "detail": (
                        f"Close-to-close change {change * 100:.2f}%; "
                        f"previous close={df.iloc[i-1]['close']}; "
                        f"current close={df.iloc[i]['close']}"
                    ),
                    "disposition": "Investigate; do not automatically adjust",
                })

        if (df["volume"] < 0).any():
            records.append({
                "symbol": symbol,
                "company_name": company,
                "date": "",
                "issue": "NEGATIVE_VOLUME",
                "detail": "One or more candles have negative volume",
                "disposition": "Quarantine pending provider investigation",
            })

    report = pd.DataFrame(
        records,
        columns=[
            "symbol",
            "company_name",
            "date",
            "issue",
            "detail",
            "disposition",
        ],
    )

    output = REPORT_DIR / "market_data_anomalies.csv"

    if output.exists():
        raise SystemExit(
            f"Refusing to overwrite existing report: {output}"
        )

    report.to_csv(output, index=False)

    print(f"Files audited: {len(list(DATA_DIR.glob('*.csv')))}")
    print(f"Findings recorded: {len(report)}")
    print(f"Report saved: {output}")
    print("\nFindings by issue:")
    if report.empty:
        print("No findings recorded.")
    else:
        print(report["issue"].value_counts().to_string())


if __name__ == "__main__":
    main()
