"""
Feature engineering for stablecoin depeg prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, DEPEG_THRESHOLD


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for depeg prediction model.

    Args:
        df: Merged daily data with price, volume, and supply metrics

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # === Price Features ===
    # BTC returns and momentum
    df['btc_return_1d'] = df['close'].pct_change()
    df['btc_return_7d'] = df['close'].pct_change(periods=7)
    df['btc_return_30d'] = df['close'].pct_change(periods=30)

    # BTC volatility (rolling std of returns)
    df['btc_volatility_7d'] = df['btc_return_1d'].rolling(7).std()
    df['btc_volatility_30d'] = df['btc_return_1d'].rolling(30).std()

    # BTC drawdown from recent high
    df['btc_rolling_max_30d'] = df['close'].rolling(30).max()
    df['btc_drawdown_30d'] = (df['close'] - df['btc_rolling_max_30d']) / df['btc_rolling_max_30d']

    # === Volume Features ===
    # Volume moving averages
    df['volume_ma_7d'] = df['quote_volume'].rolling(7).mean()
    df['volume_ma_30d'] = df['quote_volume'].rolling(30).mean()

    # Volume ratio (current vs average)
    df['volume_ratio_7d'] = df['quote_volume'] / df['volume_ma_7d']
    df['volume_ratio_30d'] = df['quote_volume'] / df['volume_ma_30d']

    # Volume trend
    df['volume_change_7d'] = df['quote_volume'].pct_change(periods=7)

    # Trade count features
    df['trades_ma_7d'] = df['trades'].rolling(7).mean()
    df['trades_ratio'] = df['trades'] / df['trades_ma_7d']

    # === Volatility Features ===
    # Spread (intra-day volatility) moving averages
    df['spread_ma_7d'] = df['spread_proxy'].rolling(7).mean()
    df['spread_ma_30d'] = df['spread_proxy'].rolling(30).mean()

    # Spread spikes
    df['spread_zscore'] = (df['spread_proxy'] - df['spread_ma_30d']) / df['spread_proxy'].rolling(30).std()

    # === Buy Pressure Features ===
    df['buy_ratio_ma_7d'] = df['buy_ratio'].rolling(7).mean()
    df['buy_ratio_deviation'] = df['buy_ratio'] - df['buy_ratio_ma_7d']

    # === Supply Features ===
    # Supply momentum
    df['supply_change_1d'] = df['total_circulating'].pct_change()
    df['supply_change_7d'] = df['total_circulating'].pct_change(periods=7)
    df['supply_change_30d'] = df['total_circulating'].pct_change(periods=30)

    # Supply acceleration
    df['supply_acceleration'] = df['supply_change_1d'] - df['supply_change_1d'].shift(1)

    # Supply volatility
    df['supply_volatility_7d'] = df['supply_change_1d'].rolling(7).std()

    # === USDT Price Features ===
    # Current deviation from peg
    df['price_deviation'] = df['implied_price'] - 1.0
    df['abs_deviation'] = df['price_deviation'].abs()

    # Deviation trend
    df['deviation_ma_7d'] = df['price_deviation'].rolling(7).mean()
    df['deviation_increasing'] = (df['price_deviation'] > df['deviation_ma_7d']).astype(int)

    # === Interaction Features ===
    # Stress indicator (high volatility + volume spike)
    df['stress_indicator'] = df['spread_zscore'] * df['volume_ratio_7d']

    # Flight to safety (negative BTC return + supply increase)
    df['flight_to_safety'] = (-df['btc_return_1d']) * df['supply_change_1d'].clip(lower=0)

    return df


def create_target(
    df: pd.DataFrame,
    threshold: float = 0.005,  # 0.5% default for more positive cases
    horizon_days: int = 7,
) -> pd.DataFrame:
    """
    Create prediction target: will deviation exceed threshold in next N days?

    Args:
        df: DataFrame with price_deviation column
        threshold: Deviation threshold
        horizon_days: Prediction horizon

    Returns:
        DataFrame with target column
    """
    df = df.copy()

    # Forward-looking max absolute deviation
    df['future_max_deviation'] = (
        df['abs_deviation']
        .rolling(horizon_days, min_periods=1)
        .max()
        .shift(-horizon_days)
    )

    # Binary target
    df['target'] = (df['future_max_deviation'] >= threshold).astype(int)

    return df


def prepare_modeling_data(
    df: pd.DataFrame,
    threshold: float = 0.005,
    horizon_days: int = 7,
) -> tuple:
    """
    Prepare data for modeling.

    Returns:
        Tuple of (X, y, feature_names, df_with_features)
    """
    # Create features
    df = create_features(df)

    # Create target
    df = create_target(df, threshold=threshold, horizon_days=horizon_days)

    # Define feature columns
    feature_cols = [
        # BTC price features
        'btc_return_1d', 'btc_return_7d', 'btc_return_30d',
        'btc_volatility_7d', 'btc_volatility_30d',
        'btc_drawdown_30d',

        # Volume features
        'volume_ratio_7d', 'volume_ratio_30d', 'volume_change_7d',
        'trades_ratio',

        # Volatility features
        'spread_proxy', 'spread_ma_7d', 'spread_zscore',

        # Buy pressure
        'buy_ratio', 'buy_ratio_deviation',

        # Supply features
        'supply_change_1d', 'supply_change_7d', 'supply_change_30d',
        'supply_acceleration', 'supply_volatility_7d',

        # Current state
        'price_deviation', 'abs_deviation', 'deviation_increasing',

        # Interaction features
        'stress_indicator', 'flight_to_safety',
    ]

    # Drop rows with NaN (from rolling calculations)
    df_clean = df.dropna(subset=feature_cols + ['target'])

    X = df_clean[feature_cols]
    y = df_clean['target']

    return X, y, feature_cols, df_clean


if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv(PROCESSED_DATA_DIR / "usdt_merged_daily.csv")
    X, y, features, df_full = prepare_modeling_data(df)

    print(f"Features: {len(features)}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Positive rate: {y.mean()*100:.2f}%")
