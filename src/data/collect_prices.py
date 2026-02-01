"""
Collect historical price data for stablecoins from CoinGecko API.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import STABLECOINS, RAW_DATA_DIR


class CoinGeckoCollector:
    """Collect price data from CoinGecko API."""

    BASE_URL_FREE = "https://api.coingecko.com/api/v3"
    BASE_URL_PRO = "https://pro-api.coingecko.com/api/v3"

    def __init__(self, api_key: str = None, rate_limit_delay: float = 1.5):
        """
        Initialize collector.

        Args:
            api_key: CoinGecko API key (optional, or set COINGECKO_API_KEY env var)
            rate_limit_delay: Seconds to wait between API calls (free tier limit)
        """
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

        # Set appropriate URL and header based on API key
        if self.api_key:
            # Demo API uses free URL with demo header
            self.base_url = self.BASE_URL_FREE
            self.session.headers["x-cg-demo-api-key"] = self.api_key
        else:
            self.base_url = self.BASE_URL_FREE

    def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365,
    ) -> dict:
        """
        Get historical market data.

        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency
            days: Number of days (use large number like 3650 for ~10 years)

        Returns:
            Dict with prices, market_caps, total_volumes
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def collect_stablecoin_data(
        self,
        coin_key: str,
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Collect and format stablecoin price data.

        Args:
            coin_key: Key from STABLECOINS config
            days: Number of days of history (default 3650 for ~10 years)

        Returns:
            DataFrame with timestamp, price, market_cap, volume
        """
        coin_config = STABLECOINS.get(coin_key)
        if not coin_config:
            raise ValueError(f"Unknown stablecoin: {coin_key}")

        print(f"Collecting data for {coin_config['name']}...")
        data = self.get_market_chart(
            coin_id=coin_config["coingecko_id"],
            days=days,
        )

        # Convert to DataFrame
        df = pd.DataFrame({
            "timestamp": [x[0] for x in data["prices"]],
            "price": [x[1] for x in data["prices"]],
            "market_cap": [x[1] for x in data["market_caps"]],
            "volume": [x[1] for x in data["total_volumes"]],
        })

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["coin"] = coin_key
        df["peg"] = coin_config["peg"]
        df["type"] = coin_config["type"]

        # Calculate deviation from peg
        df["deviation"] = (df["price"] - df["peg"]) / df["peg"]
        df["abs_deviation"] = df["deviation"].abs()

        return df

    def save_data(self, df: pd.DataFrame, coin_key: str) -> Path:
        """Save DataFrame to CSV."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = RAW_DATA_DIR / f"{coin_key}_prices.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        return filepath


def main():
    """Collect USDT data."""
    collector = CoinGeckoCollector()

    # Collect USDT data (~10 years of history)
    df = collector.collect_stablecoin_data(
        coin_key="usdt",
        days=3650,
    )

    print(f"\nCollected {len(df)} data points")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: {df['price'].min():.4f} to {df['price'].max():.4f}")
    print(f"Max deviation: {df['abs_deviation'].max():.4%}")

    # Save data
    collector.save_data(df, "usdt")

    return df


if __name__ == "__main__":
    main()
