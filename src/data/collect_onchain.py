"""
Collect on-chain metrics from Etherscan and other block explorers.

Provides: Transaction counts, gas prices, whale transactions, contract interactions.
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


class EtherscanCollector:
    """
    Collect on-chain data from Etherscan API V2.

    Requires free API key from https://etherscan.io/apis
    """

    BASE_URL = "https://api.etherscan.io/v2/api"

    # USDT contract on Ethereum
    CONTRACTS = {
        "usdt": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "usdc": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "dai": "0x6B175474E89094C44Da98b954EesdfeeB131e",
    }

    def __init__(self, api_key: str = None, rate_limit_delay: float = 0.25):
        """
        Initialize collector.

        Args:
            api_key: Etherscan API key (or set ETHERSCAN_API_KEY env var)
            rate_limit_delay: Delay between requests (free tier: 5 calls/sec)
        """
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def _request(self, params: dict) -> dict:
        """Make API request with rate limiting."""
        params["chainid"] = 1  # Ethereum mainnet
        if self.api_key:
            params["apikey"] = self.api_key

        response = self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        data = response.json()
        if data.get("status") == "0" and "rate limit" in data.get("message", "").lower():
            print("Rate limited, waiting 5 seconds...")
            time.sleep(5)
            return self._request(params)

        return data

    def get_token_transfers(
        self,
        contract_address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 1000,
    ) -> list:
        """
        Get ERC20 token transfer events.

        Note: Limited to 10,000 results per query on free tier.
        """
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": "desc",
        }

        data = self._request(params)
        return data.get("result", [])

    def get_gas_oracle(self) -> dict:
        """Get current gas prices."""
        params = {
            "module": "gastracker",
            "action": "gasoracle",
        }
        data = self._request(params)
        return data.get("result", {})

    def get_daily_gas_price(self, start_date: str, end_date: str) -> list:
        """
        Get historical daily average gas prices.

        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
        """
        params = {
            "module": "stats",
            "action": "dailyavggasprice",
            "startdate": start_date,
            "enddate": end_date,
            "sort": "asc",
        }
        data = self._request(params)
        return data.get("result", [])

    def get_eth_supply(self) -> dict:
        """Get current ETH supply statistics."""
        params = {
            "module": "stats",
            "action": "ethsupply2",
        }
        data = self._request(params)
        return data.get("result", {})

    def collect_transfer_metrics(
        self,
        coin_key: str,
        num_pages: int = 10,
    ) -> pd.DataFrame:
        """
        Collect recent large transfers for a stablecoin.

        Args:
            coin_key: Stablecoin key
            num_pages: Number of pages to fetch (1000 transfers each)

        Returns:
            DataFrame with transfer data
        """
        contract = self.CONTRACTS.get(coin_key)
        if not contract:
            raise ValueError(f"Unknown contract for: {coin_key}")

        print(f"Collecting {coin_key} transfers from Etherscan...")

        all_transfers = []
        for page in range(1, num_pages + 1):
            transfers = self.get_token_transfers(
                contract_address=contract,
                page=page,
                offset=1000,
            )
            if not transfers:
                break
            all_transfers.extend(transfers)
            print(f"  Page {page}: {len(transfers)} transfers")

        if not all_transfers:
            print("No transfers found (API key may be required)")
            return pd.DataFrame()

        df = pd.DataFrame(all_transfers)

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")
        df["value"] = df["value"].astype(float) / 1e6  # USDT has 6 decimals
        df["gas_price"] = df["gasPrice"].astype(float) / 1e9  # Convert to Gwei
        df["gas_used"] = df["gasUsed"].astype(int)

        df["coin"] = coin_key

        return df[["timestamp", "coin", "value", "gas_price", "gas_used",
                   "from", "to", "hash", "blockNumber"]]

    def aggregate_transfer_metrics(self, df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
        """
        Aggregate transfer data to regular intervals.

        Args:
            df: Raw transfer DataFrame
            freq: Aggregation frequency

        Returns:
            Aggregated metrics DataFrame
        """
        if df.empty:
            return pd.DataFrame()

        df = df.set_index("timestamp").sort_index()

        agg = df.resample(freq).agg(
            transfer_count=("value", "count"),
            total_volume=("value", "sum"),
            mean_transfer=("value", "mean"),
            max_transfer=("value", "max"),
            unique_senders=("from", "nunique"),
            unique_receivers=("to", "nunique"),
            avg_gas_price=("gas_price", "mean"),
        ).reset_index()

        # Whale transfer count (> $1M)
        whale_transfers = df[df["value"] > 1_000_000].resample(freq).size()
        agg["whale_transfers"] = agg["timestamp"].map(
            whale_transfers.to_dict()
        ).fillna(0).astype(int)

        return agg

    def save_data(self, df: pd.DataFrame, coin_key: str, suffix: str = "") -> Path:
        """Save DataFrame to CSV."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{coin_key}_onchain{suffix}.csv"
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
        return filepath


class DuneCollector:
    """
    Placeholder for Dune Analytics integration.

    Dune provides SQL access to blockchain data. Requires API key.
    See: https://dune.com/docs/api/
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DUNE_API_KEY")

    def execute_query(self, query_id: int) -> pd.DataFrame:
        """Execute a saved Dune query and return results."""
        if not self.api_key:
            raise ValueError("DUNE_API_KEY required")

        # Implementation would go here
        # See Dune API docs for details
        raise NotImplementedError("Implement based on Dune API docs")


def main():
    """Collect USDT on-chain data."""
    collector = EtherscanCollector()

    # Check if API key is set
    if not collector.api_key:
        print("Warning: ETHERSCAN_API_KEY not set")
        print("Set it with: export ETHERSCAN_API_KEY=your_key")
        print("Get a free key at: https://etherscan.io/apis")
        print("\nSkipping on-chain data collection for now.")
        return None

    # Collect recent transfers
    df = collector.collect_transfer_metrics("usdt", num_pages=10)

    if not df.empty:
        print(f"\nCollected {len(df)} transfers")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total volume: ${df['value'].sum():,.0f}")

        collector.save_data(df, "usdt", "_transfers")

        # Aggregate to hourly metrics
        agg_df = collector.aggregate_transfer_metrics(df)
        collector.save_data(agg_df, "usdt", "_hourly")

        return df, agg_df

    return None


if __name__ == "__main__":
    main()
