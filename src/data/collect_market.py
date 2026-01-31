"""
Collect broader market data and sentiment indicators.

Provides: BTC/ETH prices, Fear & Greed Index, market dominance.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR


class MarketDataCollector:
    """Collect market-wide indicators that may predict stablecoin stress."""

    COINGECKO_URL = "https://api.coingecko.com/api/v3"
    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    def __init__(self, rate_limit_delay: float = 1.5):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_fear_greed_index(self, limit: int = 0) -> pd.DataFrame:
        """
        Get Crypto Fear & Greed Index history.

        Args:
            limit: Number of days (0 = all available)

        Returns:
            DataFrame with fear/greed values
        """
        params = {"limit": limit, "format": "json"}

        response = self.session.get(self.FEAR_GREED_URL, params=params)
        response.raise_for_status()

        data = response.json().get("data", [])

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["fear_greed_value"] = df["value"].astype(int)
        df["fear_greed_class"] = df["value_classification"]

        time.sleep(self.rate_limit_delay)

        return df[["timestamp", "fear_greed_value", "fear_greed_class"]]

    def get_global_market_data(self) -> dict:
        """Get global crypto market statistics."""
        url = f"{self.COINGECKO_URL}/global"

        response = self.session.get(url)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        return response.json().get("data", {})

    def get_btc_eth_prices(
        self,
        from_date: datetime = None,
        to_date: datetime = None,
    ) -> pd.DataFrame:
        """
        Get BTC and ETH historical prices.

        Market leaders often indicate broader crypto stress.
        """
        from_ts = int(from_date.timestamp()) if from_date else int(datetime(2020, 1, 1).timestamp())
        to_ts = int(to_date.timestamp()) if to_date else int(datetime.now().timestamp())

        all_data = []

        for coin_id in ["bitcoin", "ethereum"]:
            url = f"{self.COINGECKO_URL}/coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": from_ts,
                "to": to_ts,
            }

            print(f"Collecting {coin_id} prices...")

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            df = pd.DataFrame({
                "timestamp": [x[0] for x in data["prices"]],
                f"{coin_id}_price": [x[1] for x in data["prices"]],
                f"{coin_id}_volume": [x[1] for x in data["total_volumes"]],
                f"{coin_id}_mcap": [x[1] for x in data["market_caps"]],
            })

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            all_data.append(df)

            time.sleep(self.rate_limit_delay)

        # Merge BTC and ETH data
        merged = all_data[0].merge(all_data[1], on="timestamp", how="outer")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # Calculate additional metrics
        merged["btc_returns"] = merged["bitcoin_price"].pct_change()
        merged["eth_returns"] = merged["ethereum_price"].pct_change()
        merged["btc_volatility_24h"] = merged["btc_returns"].rolling(24).std()
        merged["eth_volatility_24h"] = merged["eth_returns"].rolling(24).std()

        return merged

    def get_stablecoin_market_share(self) -> pd.DataFrame:
        """Get current stablecoin market dominance data."""
        url = f"{self.COINGECKO_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": "stablecoins",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        data = response.json()
        df = pd.DataFrame(data)

        if not df.empty:
            df["timestamp"] = datetime.now()
            df = df[["timestamp", "id", "symbol", "name", "current_price",
                     "market_cap", "total_volume", "price_change_percentage_24h"]]

        return df

    def save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        return filepath


def main():
    """Collect market-wide data."""
    collector = MarketDataCollector()

    # Fear & Greed Index
    print("Collecting Fear & Greed Index...")
    fng_df = collector.get_fear_greed_index(limit=0)
    print(f"  {len(fng_df)} days of data")
    collector.save_data(fng_df, "market_fear_greed.csv")

    # BTC/ETH prices
    print("\nCollecting BTC/ETH prices...")
    prices_df = collector.get_btc_eth_prices(from_date=datetime(2020, 1, 1))
    print(f"  {len(prices_df)} records")
    collector.save_data(prices_df, "market_btc_eth.csv")

    # Current stablecoin market share
    print("\nCollecting stablecoin market share...")
    market_df = collector.get_stablecoin_market_share()
    print(f"  {len(market_df)} stablecoins")
    collector.save_data(market_df, "market_stablecoin_share.csv")

    return fng_df, prices_df, market_df


if __name__ == "__main__":
    main()
