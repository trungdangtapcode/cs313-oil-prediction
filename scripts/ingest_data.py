# ingest_data.py

import yfinance as yf
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

START_DATE = "2015-01-01"
END_DATE = "2026-03-20"

TICKERS = {
    "oil": "BZ=F",      # Brent oil
    "usd": "DX-Y.NYB",  # Dollar Index
    "sp500": "^GSPC",
    "vix": "^VIX"
}

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# DOWNLOAD MARKET DATA
# =========================

def download_market_data():

    dfs = []

    for name, ticker in TICKERS.items():

        print(f"Downloading {name} ({ticker})...")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False
        )

        df = df[["Close"]]
        df.columns = [f"{name}_close"]

        dfs.append(df)

    market_df = pd.concat(dfs, axis=1, sort=False)

    market_df.reset_index(inplace=True)
    market_df.rename(columns={"Date": "date"}, inplace=True)

    return market_df


# =========================
# MAIN
# =========================

def main():

    market = download_market_data()

    market.to_csv(RAW_DIR / "market_data.csv", index=False)

    print("Data ingestion complete.")


if __name__ == "__main__":
    main()