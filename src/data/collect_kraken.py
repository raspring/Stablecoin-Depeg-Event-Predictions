"""
Collect trading data from Kraken API.

Provides: OHLCV data for stablecoin-to-USD fiat pairs (USDTUSD, USDCUSD, DAIUSD).
These fiat pairs give the most direct measure of a stablecoin's dollar value.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR


class KrakenCollector:
    """Collect data from Kraken public API."""

    BASE_URL = "https://api.kraken.com/0/public"

    # Stablecoin fiat pairs (direct USD pricing)
    PAIRS = {
        "usdt": ["USDTUSD"],
        "usdc": ["USDCUSD"],
        "dai": ["DAIUSD"],
    }

    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_ohlc(
        self,
        pair: str,
        interval: int = 60,
        since: int = None,
    ) -> dict:
        """
        Get OHLC data from Kraken.

        Args:
            pair: Trading pair (e.g., USDTUSD)
            interval: Interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Return data since given Unix timestamp

        Returns:
            Raw JSON response dict
        """
        url = f"{self.BASE_URL}/OHLC"
        params = {
            "pair": pair,
            "interval": interval,
        }
        if since is not None:
            params["since"] = since

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        data = response.json()
        if data.get("error") and len(data["error"]) > 0:
            raise ValueError(f"Kraken API error: {data['error']}")

        return data

    def get_historical_ohlc(
        self,
        pair: str,
        interval: int = 60,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """
        Get historical OHLC data with pagination.

        Args:
            pair: Trading pair
            interval: Interval in minutes
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        since = int(start_date.timestamp()) if start_date else None
        end_ts = int(end_date.timestamp()) if end_date else int(datetime.now().timestamp())

        all_records = []

        print(f"Collecting {pair} OHLC data...")

        while True:
            data = self.get_ohlc(pair=pair, interval=interval, since=since)

            # Kraken returns data under the pair key (which may differ from input)
            result_key = None
            for key in data.get("result", {}):
                if key != "last":
                    result_key = key
                    break

            if result_key is None:
                break

            records = data["result"][result_key]
            if not records:
                break

            # Filter records within our date range
            filtered = [r for r in records if r[0] <= end_ts]
            all_records.extend(filtered)

            # Get the 'last' timestamp for pagination
            last = data["result"].get("last")
            if last is None:
                break

            # If last record is beyond end_date, stop
            if records[-1][0] > end_ts:
                break

            # If we got fewer than 720 records, we've reached the end
            if len(records) < 720:
                break

            since = last
            print(f"  Collected {len(all_records)} records...")

        if not all_records:
            return pd.DataFrame()

        # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(all_records, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])

        # Remove duplicates (pagination overlap)
        df = df.drop_duplicates(subset=["time"], keep="last")

        # Convert types
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        for col in ["open", "high", "low", "close", "vwap", "volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["count"].astype(int)

        # Calculate derived metrics
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"].replace(0, 1)
        df["vwap_deviation"] = (df["vwap"] - df["close"]) / df["close"].replace(0, 1)

        df["symbol"] = pair

        return df[["timestamp", "symbol", "open", "high", "low", "close",
                    "volume", "vwap", "trades", "spread_proxy", "vwap_deviation"]]

    def collect_stablecoin_trading_data(
        self,
        coin_key: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> dict:
        """
        Collect trading data for stablecoin fiat pairs.

        Returns:
            Dict with DataFrames for each pair. Empty dict for coins without Kraken pairs.
        """
        pairs = self.PAIRS.get(coin_key, [])
        if not pairs:
            print(f"  No Kraken pairs available for {coin_key}")
            return {}

        results = {}

        for pair in pairs:
            df = self.get_historical_ohlc(
                pair=pair,
                interval=60,
                start_date=start_date,
                end_date=end_date,
            )
            results[pair] = df
            print(f"  {pair}: {len(df)} records")

        return results

    def save_data(self, data: dict, coin_key: str) -> list:
        """Save trading data to CSV files."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        paths = []

        for pair, df in data.items():
            filepath = RAW_DATA_DIR / f"{coin_key}_kraken_{pair.lower()}.csv"
            df.to_csv(filepath, index=False)
            paths.append(filepath)
            print(f"Saved {filepath}")

        return paths


def main():
    """Collect USDT fiat pair data from Kraken."""
    collector = KrakenCollector()

    data = collector.collect_stablecoin_trading_data(
        coin_key="usdt",
        start_date=datetime(2020, 1, 1),
    )

    collector.save_data(data, "usdt")

    # Summary
    for pair, df in data.items():
        print(f"\n{pair}:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Avg close price: ${df['close'].mean():.4f}")
        print(f"  Avg VWAP: ${df['vwap'].mean():.4f}")


if __name__ == "__main__":
    main()
