"""
Collect trading data from Binance API.

Provides: OHLCV data, order book snapshots, funding rates.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR


class BinanceCollector:
    """Collect data from Binance public API."""

    BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_URL = "https://fapi.binance.com/fapi/v1"

    # Stablecoin trading pairs
    PAIRS = {
        "usdt": ["BTCUSDT", "ETHUSDT"],  # Major pairs for volume/liquidity
        "usdc": ["BTCUSDC", "ETHUSDC"],
        "dai": ["DAIUSDT"],  # DAI traded against USDT
    }

    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000,
    ) -> list:
        """
        Get candlestick/kline data.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Kline interval (1m, 5m, 1h, 1d, etc.)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of records (max 1000)

        Returns:
            List of kline data
        """
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)
        return response.json()

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """
        Get historical klines with pagination.

        Args:
            symbol: Trading pair
            interval: Kline interval
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        start_ts = int(start_date.timestamp() * 1000) if start_date else None
        end_ts = int(end_date.timestamp() * 1000) if end_date else int(datetime.now().timestamp() * 1000)

        all_klines = []
        current_start = start_ts

        print(f"Collecting {symbol} klines...")

        while True:
            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ts,
                limit=1000,
            )

            if not klines:
                break

            all_klines.extend(klines)
            current_start = klines[-1][0] + 1  # Next millisecond after last

            if len(klines) < 1000:
                break

            print(f"  Collected {len(all_klines)} records...")

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)

        # Calculate additional metrics
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]  # Volatility proxy
        df["buy_ratio"] = df["taker_buy_volume"] / df["volume"].replace(0, 1)  # Buy pressure

        df["symbol"] = symbol

        return df[["timestamp", "symbol", "open", "high", "low", "close",
                   "volume", "quote_volume", "trades", "taker_buy_volume",
                   "buy_ratio", "spread_proxy"]]

    def get_funding_rate_history(
        self,
        symbol: str,
        start_time: int = None,
        end_time: int = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get funding rate history for perpetual futures.

        Funding rates can indicate market sentiment/stress.
        """
        url = f"{self.FUTURES_URL}/fundingRate"
        params = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        response = self.session.get(url, params=params)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        data = response.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)

        return df[["timestamp", "symbol", "funding_rate"]]

    def collect_stablecoin_trading_data(
        self,
        coin_key: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> dict:
        """
        Collect trading data for stablecoin pairs.

        Returns:
            Dict with DataFrames for each pair
        """
        pairs = self.PAIRS.get(coin_key, [])
        results = {}

        for pair in pairs:
            df = self.get_historical_klines(
                symbol=pair,
                interval="1h",
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
            filepath = RAW_DATA_DIR / f"{coin_key}_binance_{pair.lower()}.csv"
            df.to_csv(filepath, index=False)
            paths.append(filepath)
            print(f"Saved {filepath}")

        return paths


def main():
    """Collect USDT trading data from Binance."""
    collector = BinanceCollector()

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
        print(f"  Avg daily volume: ${df['quote_volume'].mean():,.0f}")


if __name__ == "__main__":
    main()
