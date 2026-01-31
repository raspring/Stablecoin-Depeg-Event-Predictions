"""
Collect DeFi metrics from DefiLlama API.

Provides: TVL, stablecoin market caps, chain distribution, mcap/TVL ratios.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR


class DefiLlamaCollector:
    """Collect data from DefiLlama API (free, no auth required)."""

    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"

    # DefiLlama stablecoin IDs
    STABLECOIN_IDS = {
        "usdt": 1,
        "usdc": 2,
        "dai": 5,
        "frax": 6,
        "tusd": 4,
        "busd": 3,
    }

    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_stablecoin_charts(self, stablecoin_id: int) -> dict:
        """
        Get historical data for a stablecoin.

        Returns mcap, price data over time.
        """
        url = f"{self.STABLECOINS_URL}/stablecoincharts/all"
        params = {"stablecoin": stablecoin_id}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def get_stablecoin_prices(self, stablecoin_id: int) -> dict:
        """Get historical price data for a stablecoin."""
        url = f"{self.STABLECOINS_URL}/stablecoinprices"
        params = {"stablecoin": stablecoin_id}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def get_stablecoin_history(self, stablecoin_id: int) -> dict:
        """Get detailed stablecoin history including chain breakdown."""
        url = f"{self.STABLECOINS_URL}/stablecoin/{stablecoin_id}"

        response = self.session.get(url)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def get_all_stablecoins(self) -> list:
        """Get list of all tracked stablecoins."""
        url = f"{self.STABLECOINS_URL}/stablecoins"
        params = {"includePrices": "true"}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def get_protocol_tvl_history(self, protocol: str) -> list:
        """Get TVL history for a specific protocol."""
        url = f"{self.BASE_URL}/protocol/{protocol}"

        response = self.session.get(url)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def collect_stablecoin_data(self, coin_key: str) -> pd.DataFrame:
        """
        Collect comprehensive stablecoin data.

        Args:
            coin_key: Stablecoin key (usdt, usdc, etc.)

        Returns:
            DataFrame with historical stablecoin metrics
        """
        stablecoin_id = self.STABLECOIN_IDS.get(coin_key)
        if not stablecoin_id:
            raise ValueError(f"Unknown stablecoin: {coin_key}")

        print(f"Collecting DefiLlama data for {coin_key}...")

        # Get chart data (mcap by chain over time)
        chart_data = self.get_stablecoin_charts(stablecoin_id)

        records = []
        for entry in chart_data:
            timestamp = datetime.fromtimestamp(entry["date"])

            # Total circulating across all chains
            total_circulating = sum(
                chain_data.get("circulating", {}).get("peggedUSD", 0)
                for chain_data in entry.get("totalCirculating", {}).values()
                if isinstance(chain_data, dict)
            )

            # Bridged vs native (if available)
            total_bridged = sum(
                chain_data.get("bridgedTo", {}).get("peggedUSD", 0)
                for chain_data in entry.get("totalCirculating", {}).values()
                if isinstance(chain_data, dict)
            )

            records.append({
                "timestamp": timestamp,
                "total_circulating_usd": total_circulating,
                "total_bridged_usd": total_bridged,
            })

        df = pd.DataFrame(records)
        df["coin"] = coin_key

        # Calculate metrics
        df["circulating_change_pct"] = df["total_circulating_usd"].pct_change()
        df["circulating_change_7d"] = df["total_circulating_usd"].pct_change(periods=7)

        return df

    def collect_chain_distribution(self, coin_key: str) -> pd.DataFrame:
        """
        Collect chain distribution data for a stablecoin.

        Useful for understanding concentration risk.
        """
        stablecoin_id = self.STABLECOIN_IDS.get(coin_key)
        if not stablecoin_id:
            raise ValueError(f"Unknown stablecoin: {coin_key}")

        print(f"Collecting chain distribution for {coin_key}...")

        history = self.get_stablecoin_history(stablecoin_id)
        chain_balances = history.get("chainBalances", {})

        records = []
        for chain, chain_data in chain_balances.items():
            tokens = chain_data.get("tokens", [])
            for entry in tokens:
                records.append({
                    "timestamp": datetime.fromtimestamp(entry["date"]),
                    "chain": chain,
                    "circulating_usd": entry.get("circulating", {}).get("peggedUSD", 0),
                    "bridged_usd": entry.get("bridgedTo", {}).get("peggedUSD", 0),
                })

        df = pd.DataFrame(records)
        df["coin"] = coin_key

        return df

    def save_data(self, df: pd.DataFrame, coin_key: str, suffix: str = "") -> Path:
        """Save DataFrame to CSV."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{coin_key}_defillama{suffix}.csv"
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        return filepath


def main():
    """Collect USDT data from DefiLlama."""
    collector = DefiLlamaCollector()

    # Collect main stablecoin metrics
    df = collector.collect_stablecoin_data("usdt")
    print(f"\nCollected {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Current circulating: ${df['total_circulating_usd'].iloc[-1]:,.0f}")

    collector.save_data(df, "usdt", "_metrics")

    # Collect chain distribution
    chain_df = collector.collect_chain_distribution("usdt")
    print(f"\nChain distribution: {len(chain_df)} records")
    print(f"Chains: {chain_df['chain'].nunique()}")

    collector.save_data(chain_df, "usdt", "_chains")

    return df, chain_df


if __name__ == "__main__":
    main()
