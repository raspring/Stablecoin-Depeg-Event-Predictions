"""
Collect historical price data for stablecoins from CoinGecko API.
"""

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

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, rate_limit_delay: float = 1.5):
        """
        Initialize collector.

        Args:
            rate_limit_delay: Seconds to wait between API calls (free tier limit)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_market_chart_range(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        from_timestamp: int = None,
        to_timestamp: int = None,
    ) -> dict:
        """
        Get historical market data within a time range.

        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency
            from_timestamp: Start time (Unix timestamp)
            to_timestamp: End time (Unix timestamp)

        Returns:
            Dict with prices, market_caps, total_volumes
        """
        if from_timestamp is None:
            # Default to 4 years ago
            from_timestamp = int(datetime(2020, 1, 1).timestamp())
        if to_timestamp is None:
            to_timestamp = int(datetime.now().timestamp())

        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": from_timestamp,
            "to": to_timestamp,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def collect_stablecoin_data(
        self,
        coin_key: str,
        from_date: datetime = None,
        to_date: datetime = None,
    ) -> pd.DataFrame:
        """
        Collect and format stablecoin price data.

        Args:
            coin_key: Key from STABLECOINS config
            from_date: Start date
            to_date: End date

        Returns:
            DataFrame with timestamp, price, market_cap, volume
        """
        coin_config = STABLECOINS.get(coin_key)
        if not coin_config:
            raise ValueError(f"Unknown stablecoin: {coin_key}")

        from_ts = int(from_date.timestamp()) if from_date else None
        to_ts = int(to_date.timestamp()) if to_date else None

        print(f"Collecting data for {coin_config['name']}...")
        data = self.get_market_chart_range(
            coin_id=coin_config["coingecko_id"],
            from_timestamp=from_ts,
            to_timestamp=to_ts,
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

    # Collect USDT data from 2020 to now
    df = collector.collect_stablecoin_data(
        coin_key="usdt",
        from_date=datetime(2020, 1, 1),
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
