"""
Collect macroeconomic data from FRED (Federal Reserve Economic Data).

Provides: DXY (Dollar Index), Treasury rates, VIX, and other macro indicators
that may correlate with stablecoin stress.

FRED API: https://fred.stlouisfed.org/docs/api/fred/
Free API key required: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR


class FREDCollector:
    """Collect macroeconomic data from FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Key series for crypto/stablecoin analysis
    SERIES = {
        # Dollar strength
        "DXY": "DTWEXBGS",  # Trade Weighted U.S. Dollar Index (Broad)

        # Interest rates
        "FEDFUNDS": "FEDFUNDS",  # Federal Funds Effective Rate
        "T10Y2Y": "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year (yield curve)
        "T3M": "DTB3",  # 3-Month Treasury Bill Rate
        "T1Y": "DGS1",  # 1-Year Treasury Rate
        "T10Y": "DGS10",  # 10-Year Treasury Rate

        # Risk/volatility
        "VIX": "VIXCLS",  # CBOE Volatility Index

        # Liquidity/money supply
        "M2": "M2SL",  # M2 Money Stock
        "WALCL": "WALCL",  # Fed Total Assets (Balance Sheet)

        # Credit spreads
        "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # ICE BofA US High Yield Index Option-Adjusted Spread

        # Economic indicators
        "UNRATE": "UNRATE",  # Unemployment Rate
        "CPIAUCSL": "CPIAUCSL",  # Consumer Price Index
    }

    def __init__(self, api_key: str = None, rate_limit_delay: float = 0.5):
        """
        Initialize collector.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
            rate_limit_delay: Delay between requests
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_series(
        self,
        series_id: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Get a single FRED series.

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and value columns
        """
        if not self.api_key:
            raise ValueError("FRED_API_KEY required. Get one at https://fred.stlouisfed.org/docs/api/api_key.html")

        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df[["date", "value"]].dropna()

    def get_series_info(self, series_id: str) -> dict:
        """Get metadata for a FRED series."""
        if not self.api_key:
            raise ValueError("FRED_API_KEY required")

        url = f"{self.BASE_URL}/series"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        data = response.json()
        series = data.get("seriess", [])
        return series[0] if series else {}

    def collect_macro_data(
        self,
        start_date: str = "2018-01-01",
        end_date: str = None,
        series_list: list = None,
    ) -> pd.DataFrame:
        """
        Collect multiple macro series and merge into single DataFrame.

        Args:
            start_date: Start date
            end_date: End date (defaults to today)
            series_list: List of series names to collect (defaults to all)

        Returns:
            DataFrame with date index and columns for each series
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if series_list is None:
            series_list = list(self.SERIES.keys())

        print(f"Collecting FRED data from {start_date} to {end_date}")
        print(f"Series: {series_list}")

        all_data = {}

        for name in series_list:
            series_id = self.SERIES.get(name, name)
            print(f"  Fetching {name} ({series_id})...", end=" ")

            try:
                df = self.get_series(series_id, start_date, end_date)
                if not df.empty:
                    all_data[name] = df.set_index("date")["value"]
                    print(f"{len(df)} observations")
                else:
                    print("no data")
            except Exception as e:
                print(f"error: {e}")

        if not all_data:
            return pd.DataFrame()

        # Merge all series
        merged = pd.DataFrame(all_data)
        merged.index.name = "date"

        # Forward fill missing values (common for different frequencies)
        merged = merged.sort_index().ffill()

        return merged

    def collect_key_indicators(self, start_date: str = "2018-01-01") -> pd.DataFrame:
        """
        Collect key indicators most relevant for stablecoin analysis.

        Returns DataFrame with:
        - DXY: Dollar strength
        - VIX: Market fear
        - T10Y2Y: Yield curve (recession indicator)
        - FEDFUNDS: Fed policy rate
        """
        key_series = ["DXY", "VIX", "T10Y2Y", "FEDFUNDS", "T3M", "T10Y"]
        return self.collect_macro_data(start_date=start_date, series_list=key_series)

    def save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath)
        print(f"Saved to {filepath}")
        return filepath


def main():
    """Collect FRED macro data."""
    collector = FREDCollector()

    if not collector.api_key:
        print("="*60)
        print("FRED API KEY REQUIRED")
        print("="*60)
        print("\nTo get a free API key:")
        print("1. Go to https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Create an account and request a key")
        print("3. Set the key: export FRED_API_KEY='your_key_here'")
        return None

    # Collect key indicators
    print("\n" + "="*60)
    print("COLLECTING FRED MACRO DATA")
    print("="*60)

    df = collector.collect_key_indicators(start_date="2018-01-01")

    if not df.empty:
        print(f"\nCollected {len(df)} days of data")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Columns: {list(df.columns)}")

        # Summary stats
        print("\nSummary statistics:")
        print(df.describe().round(2))

        # Save
        collector.save_data(df, "fred_macro.csv")

        return df

    return None


if __name__ == "__main__":
    main()
