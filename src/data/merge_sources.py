"""
Merge all data sources into processed files.

Combines:
- Binance trading data (OHLCV, spread, buy pressure)
- Kraken fiat pair data (direct USD price, VWAP, spread)
- DefiLlama supply metrics
- CoinGecko direct USD prices
- Fear & Greed index
- FRED macro indicators (DXY, VIX, Treasury rates)
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_binance_daily(coin: str) -> pd.DataFrame:
    """Load and aggregate Binance hourly data to daily."""
    pair = "btcusdt" if coin == "usdt" else "btcusdc"
    filepath = RAW_DATA_DIR / f"{coin}_binance_{pair}.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Aggregate to daily
    daily = df.groupby("date").agg(
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        trades=("trades", "sum"),
        spread_proxy=("spread_proxy", "mean"),
        buy_ratio=("buy_ratio", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["timestamp_btc"] = daily["date"]

    return daily


def load_kraken_daily(coin: str) -> pd.DataFrame:
    """Load and aggregate Kraken hourly fiat pair data to daily."""
    pair_map = {"usdt": "usdtusd", "usdc": "usdcusd", "dai": "daiusd"}
    pair = pair_map.get(coin)
    if pair is None:
        return pd.DataFrame()

    filepath = RAW_DATA_DIR / f"{coin}_kraken_{pair}.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Aggregate to daily
    daily = df.groupby("date").agg(
        kr_price=("close", "last"),
        kr_volume=("volume", "sum"),
        kr_trades=("trades", "sum"),
        kr_spread_proxy=("spread_proxy", "mean"),
        kr_vwap=("vwap", "last"),
        kr_vwap_deviation=("vwap_deviation", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])

    return daily


def load_defillama(coin: str) -> pd.DataFrame:
    """Load DefiLlama supply metrics."""
    filepath = RAW_DATA_DIR / f"{coin}_defillama_metrics.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Rename timestamp to avoid confusion
    df = df.rename(columns={"timestamp": "timestamp_dl"})

    return df


def load_coingecko(coin: str) -> pd.DataFrame:
    """Load CoinGecko direct USD prices."""
    filepath = RAW_DATA_DIR / f"{coin}_prices.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Keep only relevant columns, rename to avoid conflicts
    df = df[["date", "price", "market_cap", "volume"]].copy()
    df = df.rename(columns={
        "price": "cg_price",
        "market_cap": "cg_market_cap",
        "volume": "cg_volume",
    })

    # Aggregate if multiple entries per day
    df = df.groupby("date").agg({
        "cg_price": "mean",
        "cg_market_cap": "mean",
        "cg_volume": "sum",
    }).reset_index()

    return df


def load_fear_greed() -> pd.DataFrame:
    """Load Fear & Greed index."""
    filepath = RAW_DATA_DIR / "market_fear_greed.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Keep only what we need
    df = df[["date", "fear_greed_value", "fear_greed_class"]].copy()

    return df


def load_fred() -> pd.DataFrame:
    """Load FRED macro indicators."""
    filepath = RAW_DATA_DIR / "fred_macro.csv"

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])

    # Rename columns with fred_ prefix to avoid conflicts
    rename_cols = {col: f"fred_{col.lower()}" for col in df.columns if col != "date"}
    df = df.rename(columns=rename_cols)

    return df


def merge_coin_data(coin: str, fred_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge all sources for a single coin."""
    print(f"\n=== Merging {coin.upper()} data ===")

    # Load sources
    binance = load_binance_daily(coin)
    kraken = load_kraken_daily(coin)
    defillama = load_defillama(coin)
    coingecko = load_coingecko(coin)
    fear_greed = load_fear_greed()

    print(f"Binance: {len(binance)} days")
    print(f"Kraken: {len(kraken)} days")
    print(f"DefiLlama: {len(defillama)} days")
    print(f"CoinGecko: {len(coingecko)} days")
    print(f"Fear & Greed: {len(fear_greed)} days")
    if fred_df is not None:
        print(f"FRED: {len(fred_df)} days")

    # Start with Binance + DefiLlama (inner join - need both)
    if binance.empty or defillama.empty:
        print("Missing core data, skipping")
        return pd.DataFrame()

    merged = binance.merge(defillama, on="date", how="inner")
    print(f"After Binance + DefiLlama: {len(merged)} days")

    # Add CoinGecko (left join - optional)
    if not coingecko.empty:
        merged = merged.merge(coingecko, on="date", how="left")
        print(f"After CoinGecko: {len(merged)} days ({merged['cg_price'].notna().sum()} with CG data)")

    # Add Kraken fiat pair data (left join - optional)
    if not kraken.empty:
        merged = merged.merge(kraken, on="date", how="left")
        print(f"After Kraken: {len(merged)} days ({merged['kr_price'].notna().sum()} with Kraken data)")

    # Add Fear & Greed (left join - optional)
    if not fear_greed.empty:
        merged = merged.merge(fear_greed, on="date", how="left")
        print(f"After Fear & Greed: {len(merged)} days ({merged['fear_greed_value'].notna().sum()} with F&G data)")

    # Add FRED macro data (left join - optional)
    if fred_df is not None and not fred_df.empty:
        merged = merged.merge(fred_df, on="date", how="left")
        fred_col = [c for c in merged.columns if c.startswith("fred_")][0] if any(c.startswith("fred_") for c in merged.columns) else None
        if fred_col:
            print(f"After FRED: {len(merged)} days ({merged[fred_col].notna().sum()} with FRED data)")

    # Add coin identifier
    merged["coin"] = coin

    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def create_processed_files():
    """Create all processed data files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load FRED data once (shared across coins)
    fred_df = load_fred()

    all_data = []

    for coin in ["usdt", "usdc"]:
        df = merge_coin_data(coin, fred_df=fred_df)

        if not df.empty:
            # Save individual coin file
            filepath = PROCESSED_DATA_DIR / f"{coin}_merged_daily.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved: {filepath}")

            all_data.append(df)

    # Create combined file
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(["coin", "date"]).reset_index(drop=True)

        filepath = PROCESSED_DATA_DIR / "combined_stablecoins_daily.csv"
        combined.to_csv(filepath, index=False)
        print(f"\nSaved combined: {filepath}")
        print(f"Total records: {len(combined)}")

        # Summary
        print("\n=== Summary ===")
        print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
        print(f"USDT records: {len(combined[combined['coin'] == 'usdt'])}")
        print(f"USDC records: {len(combined[combined['coin'] == 'usdc'])}")

        # New columns
        new_cols = [c for c in combined.columns if c.startswith("cg_") or c.startswith("kr_") or c.startswith("fear_") or c.startswith("fred_")]
        if new_cols:
            print(f"New features added: {new_cols}")

    return combined if all_data else pd.DataFrame()


if __name__ == "__main__":
    create_processed_files()
