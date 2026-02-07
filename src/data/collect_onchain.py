"""
Collect on-chain metrics from Etherscan and other block explorers.

Provides: Transaction counts, gas prices, whale transactions, contract interactions.

Supports two collection modes:
- Block-range mode (default when start_date provided): converts dates to block numbers,
  recursively splits ranges that hit the 10k result cap.
- Page mode (legacy): fetches N pages of 1000 results each.
"""

import os
import time
from datetime import datetime, timedelta, timezone
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

    CONTRACTS = {
        "usdt": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "usdc": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "dai": "0x6B175474E89094C44Da98b954EesdfeeB131e",
    }

    TOKEN_DECIMALS = {
        "usdt": 6,
        "usdc": 6,
        "dai": 18,
    }

    MAX_RESULTS_PER_QUERY = 10_000
    MAX_SPLIT_DEPTH = 10

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

    def get_block_by_timestamp(self, timestamp: int, closest: str = "before") -> int:
        """
        Get the block number for a given Unix timestamp.

        Args:
            timestamp: Unix timestamp
            closest: "before" or "after"

        Returns:
            Block number
        """
        params = {
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": timestamp,
            "closest": closest,
        }
        data = self._request(params)
        result = data.get("result")
        if result and str(result).isdigit():
            return int(result)
        raise ValueError(f"Could not resolve block for timestamp {timestamp}: {data}")

    def get_token_transfers(
        self,
        contract_address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 1000,
        sort: str = "asc",
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
            "sort": sort,
        }

        data = self._request(params)
        return data.get("result", [])

    def _fetch_transfers_in_block_range(
        self,
        contract: str,
        start_block: int,
        end_block: int,
        depth: int = 0,
    ) -> list:
        """
        Fetch all transfers in a block range, recursively splitting if results are truncated.

        Args:
            contract: Token contract address
            start_block: Start block (inclusive)
            end_block: End block (inclusive)
            depth: Current recursion depth

        Returns:
            List of transfer dicts
        """
        transfers = self.get_token_transfers(
            contract_address=contract,
            start_block=start_block,
            end_block=end_block,
            page=1,
            offset=self.MAX_RESULTS_PER_QUERY,
            sort="asc",
        )

        if not isinstance(transfers, list):
            print(f"  Unexpected response for blocks {start_block}-{end_block}: {transfers}")
            return []

        if len(transfers) < self.MAX_RESULTS_PER_QUERY:
            return transfers

        # Hit the cap — need to split
        if depth >= self.MAX_SPLIT_DEPTH:
            print(f"  Warning: max split depth reached at blocks {start_block}-{end_block}, "
                  f"returning {len(transfers)} (possibly incomplete)")
            return transfers

        if start_block == end_block:
            print(f"  Warning: single block {start_block} has {len(transfers)} transfers (cap hit)")
            return transfers

        mid_block = (start_block + end_block) // 2
        print(f"  Splitting blocks {start_block}-{end_block} at {mid_block} (depth={depth + 1})")

        left = self._fetch_transfers_in_block_range(contract, start_block, mid_block, depth + 1)
        right = self._fetch_transfers_in_block_range(contract, mid_block + 1, end_block, depth + 1)

        return left + right

    def _get_token_decimals(self, coin_key: str) -> int:
        """Get the number of decimals for a token."""
        return self.TOKEN_DECIMALS.get(coin_key, 18)

    def _format_transfer_df(self, transfers: list, coin_key: str) -> pd.DataFrame:
        """
        Convert raw transfer dicts to a formatted DataFrame.

        Args:
            transfers: List of transfer dicts from Etherscan API
            coin_key: Stablecoin key (for decimal conversion)

        Returns:
            Formatted DataFrame
        """
        if not transfers:
            return pd.DataFrame()

        df = pd.DataFrame(transfers)

        decimals = self._get_token_decimals(coin_key)
        df["timestamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")
        df["value"] = df["value"].astype(float) / (10 ** decimals)
        df["gas_price"] = df["gasPrice"].astype(float) / 1e9  # Convert to Gwei
        df["gas_used"] = df["gasUsed"].astype(int)
        df["coin"] = coin_key

        return df[["timestamp", "coin", "value", "gas_price", "gas_used",
                    "from", "to", "hash", "blockNumber"]]

    def _collect_by_block_range(
        self,
        coin_key: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Collect transfers using block-range pagination with adaptive splitting.

        Args:
            coin_key: Stablecoin key
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with transfer data
        """
        contract = self.CONTRACTS.get(coin_key)
        if not contract:
            raise ValueError(f"Unknown contract for: {coin_key}")

        print(f"Collecting {coin_key} transfers via block-range mode...")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")

        # Convert dates to block numbers
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        print("  Resolving start block...")
        start_block = self.get_block_by_timestamp(start_ts, closest="after")
        print(f"    Start block: {start_block}")

        print("  Resolving end block...")
        end_block = self.get_block_by_timestamp(end_ts, closest="before")
        print(f"    End block: {end_block}")

        if start_block > end_block:
            print("  Warning: start_block > end_block, no data in range")
            return pd.DataFrame()

        # Fetch with adaptive splitting
        all_transfers = self._fetch_transfers_in_block_range(
            contract, start_block, end_block
        )

        if not all_transfers:
            print("  No transfers found in block range")
            return pd.DataFrame()

        # Deduplicate by hash + logIndex
        seen = set()
        unique_transfers = []
        for t in all_transfers:
            key = (t.get("hash"), t.get("logIndex"))
            if key not in seen:
                seen.add(key)
                unique_transfers.append(t)

        if len(unique_transfers) < len(all_transfers):
            print(f"  Deduplicated: {len(all_transfers)} -> {len(unique_transfers)} transfers")

        df = self._format_transfer_df(unique_transfers, coin_key)
        print(f"  Collected {len(df)} transfers")

        return df

    def _collect_by_pages(
        self,
        coin_key: str,
        num_pages: int = 10,
    ) -> pd.DataFrame:
        """
        Collect transfers using legacy page-based pagination.

        Args:
            coin_key: Stablecoin key
            num_pages: Number of pages to fetch (1000 transfers each)

        Returns:
            DataFrame with transfer data
        """
        contract = self.CONTRACTS.get(coin_key)
        if not contract:
            raise ValueError(f"Unknown contract for: {coin_key}")

        print(f"Collecting {coin_key} transfers from Etherscan (page mode)...")

        all_transfers = []
        for page in range(1, num_pages + 1):
            transfers = self.get_token_transfers(
                contract_address=contract,
                page=page,
                offset=1000,
                sort="desc",
            )
            if not transfers:
                break
            all_transfers.extend(transfers)
            print(f"  Page {page}: {len(transfers)} transfers")

        if not all_transfers:
            print("No transfers found (API key may be required)")
            return pd.DataFrame()

        return self._format_transfer_df(all_transfers, coin_key)

    def collect_transfer_metrics(
        self,
        coin_key: str,
        start_date: datetime = None,
        end_date: datetime = None,
        num_pages: int = None,
    ) -> pd.DataFrame:
        """
        Collect transfer data for a stablecoin.

        When start_date is provided, uses block-range mode with adaptive splitting
        for complete coverage. Otherwise falls back to legacy page mode.

        Args:
            coin_key: Stablecoin key
            start_date: Start date for block-range mode
            end_date: End date for block-range mode (defaults to now)
            num_pages: Number of pages for page mode (default 10)

        Returns:
            DataFrame with transfer data
        """
        if start_date is not None:
            if end_date is None:
                end_date = datetime.now(timezone.utc)
            return self._collect_by_block_range(coin_key, start_date, end_date)
        else:
            return self._collect_by_pages(coin_key, num_pages=num_pages or 10)

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
    """Collect USDT on-chain data (block-range mode demo)."""
    collector = EtherscanCollector()

    # Check if API key is set
    if not collector.api_key:
        print("Warning: ETHERSCAN_API_KEY not set")
        print("Set it with: export ETHERSCAN_API_KEY=your_key")
        print("Get a free key at: https://etherscan.io/apis")
        print("\nSkipping on-chain data collection for now.")
        return None

    # Collect transfers using block-range mode (1 week demo)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    df = collector.collect_transfer_metrics("usdt", start_date=start_date, end_date=end_date)

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
