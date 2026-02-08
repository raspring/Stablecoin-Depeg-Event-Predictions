"""
Master script to collect all data sources for a stablecoin.
"""

import argparse
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.collect_prices import CoinGeckoCollector
from src.data.collect_binance import BinanceCollector
from src.data.collect_kraken import KrakenCollector
from src.data.collect_defillama import DefiLlamaCollector
from src.data.collect_onchain import EtherscanCollector
from src.data.collect_market import MarketDataCollector


def collect_all(
    coin_key: str,
    start_date: datetime = None,
    include_onchain: bool = True,
):
    """
    Collect all available data for a stablecoin.

    Args:
        coin_key: Stablecoin key (usdt, usdc, dai, etc.)
        start_date: Start date for historical data
        include_onchain: Whether to collect on-chain data (requires API key)
    """
    if start_date is None:
        start_date = datetime(2020, 1, 1)

    print(f"=" * 60)
    print(f"Collecting data for {coin_key.upper()}")
    print(f"Start date: {start_date.date()}")
    print(f"=" * 60)

    results = {}

    # 1. CoinGecko price data
    print("\n[1/6] CoinGecko - Price Data")
    print("-" * 40)
    try:
        cg_collector = CoinGeckoCollector()
        cg_df = cg_collector.collect_stablecoin_data(coin_key, from_date=start_date)
        cg_collector.save_data(cg_df, coin_key)
        results["coingecko"] = len(cg_df)
        print(f"Success: {len(cg_df)} records")
    except Exception as e:
        print(f"Error: {e}")
        results["coingecko"] = 0

    # 2. Binance trading data
    print("\n[2/6] Binance - Trading Data")
    print("-" * 40)
    try:
        binance_collector = BinanceCollector()
        binance_data = binance_collector.collect_stablecoin_trading_data(
            coin_key, start_date=start_date
        )
        binance_collector.save_data(binance_data, coin_key)
        results["binance"] = sum(len(df) for df in binance_data.values())
        print(f"Success: {results['binance']} total records")
    except Exception as e:
        print(f"Error: {e}")
        results["binance"] = 0

    # 3. Kraken fiat pair data
    print("\n[3/6] Kraken - Fiat Pair Data")
    print("-" * 40)
    try:
        kraken_collector = KrakenCollector()
        kraken_data = kraken_collector.collect_stablecoin_trading_data(
            coin_key, start_date=start_date
        )
        if kraken_data:
            kraken_collector.save_data(kraken_data, coin_key)
            results["kraken"] = sum(len(df) for df in kraken_data.values())
            print(f"Success: {results['kraken']} total records")
        else:
            results["kraken"] = 0
            print("No Kraken pairs available for this coin")
    except Exception as e:
        print(f"Error: {e}")
        results["kraken"] = 0

    # 4. DefiLlama stablecoin metrics
    print("\n[4/6] DefiLlama - Stablecoin Metrics")
    print("-" * 40)
    try:
        defillama_collector = DefiLlamaCollector()
        dl_df = defillama_collector.collect_stablecoin_data(coin_key)
        defillama_collector.save_data(dl_df, coin_key, "_metrics")

        chain_df = defillama_collector.collect_chain_distribution(coin_key)
        defillama_collector.save_data(chain_df, coin_key, "_chains")

        results["defillama"] = len(dl_df) + len(chain_df)
        print(f"Success: {len(dl_df)} metrics + {len(chain_df)} chain records")
    except Exception as e:
        print(f"Error: {e}")
        results["defillama"] = 0

    # 5. On-chain data (optional)
    print("\n[5/6] Etherscan - On-Chain Data")
    print("-" * 40)
    if include_onchain:
        try:
            onchain_collector = EtherscanCollector()
            if onchain_collector.api_key:
                onchain_df = onchain_collector.collect_transfer_metrics(coin_key, start_date=start_date)
                if not onchain_df.empty:
                    onchain_collector.save_data(onchain_df, coin_key, "_transfers")
                    agg_df = onchain_collector.aggregate_transfer_metrics(onchain_df)
                    onchain_collector.save_data(agg_df, coin_key, "_hourly")
                    results["onchain"] = len(onchain_df)
                    print(f"Success: {len(onchain_df)} transfers")
                else:
                    results["onchain"] = 0
                    print("No data returned")
            else:
                print("Skipped: ETHERSCAN_API_KEY not set")
                results["onchain"] = 0
        except Exception as e:
            print(f"Error: {e}")
            results["onchain"] = 0
    else:
        print("Skipped (include_onchain=False)")
        results["onchain"] = 0

    # 6. Market data
    print("\n[6/6] Market Data - BTC/ETH & Sentiment")
    print("-" * 40)
    try:
        market_collector = MarketDataCollector()

        fng_df = market_collector.get_fear_greed_index()
        market_collector.save_data(fng_df, "market_fear_greed.csv")

        prices_df = market_collector.get_btc_eth_prices(from_date=start_date)
        market_collector.save_data(prices_df, "market_btc_eth.csv")

        results["market"] = len(fng_df) + len(prices_df)
        print(f"Success: {len(fng_df)} sentiment + {len(prices_df)} price records")
    except Exception as e:
        print(f"Error: {e}")
        results["market"] = 0

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    for source, count in results.items():
        status = "OK" if count > 0 else "FAILED/SKIPPED"
        print(f"  {source:15} {count:>10} records  [{status}]")
    print(f"\n  Total records: {sum(results.values()):,}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect stablecoin data")
    parser.add_argument(
        "coin",
        choices=["usdt", "usdc", "dai", "frax", "all"],
        help="Stablecoin to collect data for",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-onchain",
        action="store_true",
        help="Skip on-chain data collection",
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    if args.coin == "all":
        for coin in ["usdt", "usdc", "dai", "frax"]:
            collect_all(coin, start_date, include_onchain=not args.no_onchain)
    else:
        collect_all(args.coin, start_date, include_onchain=not args.no_onchain)


if __name__ == "__main__":
    main()
