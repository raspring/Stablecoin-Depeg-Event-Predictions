"""
Multi-stablecoin depeg prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR


def create_features_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for multi-coin dataset."""
    df = df.copy()

    # Process each coin separately for time-series features
    result_dfs = []

    for coin in df['coin'].unique():
        coin_df = df[df['coin'] == coin].copy()
        coin_df = coin_df.sort_values('date').reset_index(drop=True)

        # === Price Features ===
        coin_df['btc_return_1d'] = coin_df['close'].pct_change()
        coin_df['btc_return_7d'] = coin_df['close'].pct_change(periods=7)
        coin_df['btc_volatility_7d'] = coin_df['btc_return_1d'].rolling(7).std()
        coin_df['btc_volatility_30d'] = coin_df['btc_return_1d'].rolling(30).std()

        # BTC drawdown
        coin_df['btc_rolling_max_30d'] = coin_df['close'].rolling(30).max()
        coin_df['btc_drawdown_30d'] = (coin_df['close'] - coin_df['btc_rolling_max_30d']) / coin_df['btc_rolling_max_30d']

        # === Volume Features ===
        coin_df['volume_ma_7d'] = coin_df['quote_volume'].rolling(7).mean()
        coin_df['volume_ratio_7d'] = coin_df['quote_volume'] / coin_df['volume_ma_7d']

        # === Volatility Features ===
        coin_df['spread_ma_7d'] = coin_df['spread_proxy'].rolling(7).mean()
        coin_df['spread_zscore'] = (
            (coin_df['spread_proxy'] - coin_df['spread_proxy'].rolling(30).mean()) /
            coin_df['spread_proxy'].rolling(30).std()
        )

        # === Supply Features ===
        coin_df['supply_change_1d'] = coin_df['total_circulating'].pct_change()
        coin_df['supply_change_7d'] = coin_df['total_circulating'].pct_change(periods=7)
        coin_df['supply_volatility_7d'] = coin_df['supply_change_1d'].rolling(7).std()

        # === USDT/USDC Price Features ===
        coin_df['price_deviation'] = coin_df['implied_price'] - 1.0
        coin_df['abs_deviation'] = coin_df['price_deviation'].abs()
        coin_df['deviation_ma_7d'] = coin_df['price_deviation'].rolling(7).mean()

        # === Interaction Features ===
        coin_df['stress_indicator'] = coin_df['spread_zscore'] * coin_df['volume_ratio_7d']

        result_dfs.append(coin_df)

    return pd.concat(result_dfs, ignore_index=True)


def create_target_multi(df: pd.DataFrame, threshold: float = 0.01, horizon_days: int = 7) -> pd.DataFrame:
    """Create target for multi-coin dataset."""
    df = df.copy()

    # Process each coin separately
    result_dfs = []

    for coin in df['coin'].unique():
        coin_df = df[df['coin'] == coin].copy()
        coin_df = coin_df.sort_values('date').reset_index(drop=True)

        # Forward-looking max deviation
        coin_df['future_max_deviation'] = (
            coin_df['abs_deviation']
            .rolling(horizon_days, min_periods=1)
            .max()
            .shift(-horizon_days)
        )
        coin_df['target'] = (coin_df['future_max_deviation'] >= threshold).astype(int)

        result_dfs.append(coin_df)

    return pd.concat(result_dfs, ignore_index=True)


def prepare_multi_coin_data(df: pd.DataFrame, threshold: float = 0.01, horizon_days: int = 7):
    """Prepare multi-coin data for modeling."""

    # Create features
    df = create_features_multi(df)

    # Create target
    df = create_target_multi(df, threshold=threshold, horizon_days=horizon_days)

    # Feature columns
    feature_cols = [
        'btc_return_1d', 'btc_return_7d',
        'btc_volatility_7d', 'btc_volatility_30d', 'btc_drawdown_30d',
        'volume_ratio_7d',
        'spread_proxy', 'spread_ma_7d', 'spread_zscore',
        'buy_ratio',
        'supply_change_1d', 'supply_change_7d', 'supply_volatility_7d',
        'price_deviation', 'abs_deviation',
        'stress_indicator',
    ]

    # Add coin dummy variable
    df['is_usdc'] = (df['coin'] == 'usdc').astype(int)
    feature_cols.append('is_usdc')

    # Drop NaN
    df_clean = df.dropna(subset=feature_cols + ['target'])

    X = df_clean[feature_cols]
    y = df_clean['target']

    return X, y, feature_cols, df_clean


def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print('='*60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    if len(np.unique(y_true)) > 1 and y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        print(f"\nROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision (PR-AUC): {avg_precision:.4f}")

        return {
            'model': model_name,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
            'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        }
    return None


def train_models(X, y, threshold):
    """Train and evaluate models."""

    print(f"\n{'#'*60}")
    print(f"MULTI-COIN DEPEG PREDICTION")
    print(f"Threshold: {threshold*100:.2f}%")
    print(f"{'#'*60}")

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: {sum(y==0)} negative, {sum(y==1)} positive ({y.mean()*100:.2f}%)")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_np = y.values

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=5)

    # Class weight
    class_weight = {0: 1, 1: sum(y==0) / sum(y==1)} if sum(y==1) > 0 else None
    print(f"Class weight: {class_weight[1]:.2f}")

    results = []

    # Logistic Regression
    print("\n" + "-"*40)
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    y_pred_lr = np.zeros(len(y))
    y_prob_lr = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X_scaled):
        lr.fit(X_scaled[train_idx], y_np[train_idx])
        y_pred_lr[test_idx] = lr.predict(X_scaled[test_idx])
        y_prob_lr[test_idx] = lr.predict_proba(X_scaled[test_idx])[:, 1]

    test_mask = y_prob_lr > 0
    result = evaluate_model(y_np[test_mask], y_pred_lr[test_mask], y_prob_lr[test_mask],
                           "Logistic Regression")
    if result:
        results.append(result)

    # Random Forest
    print("\n" + "-"*40)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced',
                                random_state=42, n_jobs=-1)

    y_pred_rf = np.zeros(len(y))
    y_prob_rf = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X_scaled):
        rf.fit(X_scaled[train_idx], y_np[train_idx])
        y_pred_rf[test_idx] = rf.predict(X_scaled[test_idx])
        y_prob_rf[test_idx] = rf.predict_proba(X_scaled[test_idx])[:, 1]

    result = evaluate_model(y_np[test_mask], y_pred_rf[test_mask], y_prob_rf[test_mask],
                           "Random Forest")
    if result:
        results.append(result)

    # Gradient Boosting
    print("\n" + "-"*40)
    print("Training Gradient Boosting...")
    sample_weight = np.where(y_np == 1, class_weight[1], 1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    random_state=42)

    y_pred_gb = np.zeros(len(y))
    y_prob_gb = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X_scaled):
        gb.fit(X_scaled[train_idx], y_np[train_idx], sample_weight=sample_weight[train_idx])
        y_pred_gb[test_idx] = gb.predict(X_scaled[test_idx])
        y_prob_gb[test_idx] = gb.predict_proba(X_scaled[test_idx])[:, 1]

    result = evaluate_model(y_np[test_mask], y_pred_gb[test_mask], y_prob_gb[test_mask],
                           "Gradient Boosting")
    if result:
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    rf.fit(X_scaled, y_np)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.to_string(index=False))

    return results_df, rf, scaler


def main():
    """Run multi-coin model."""
    print("Loading combined stablecoin data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "combined_stablecoins_daily.csv")

    print(f"Total records: {len(df)}")
    print(f"Coins: {df['coin'].value_counts().to_dict()}")

    # Try different thresholds
    for threshold in [0.005, 0.01]:
        X, y, features, df_full = prepare_multi_coin_data(df, threshold=threshold, horizon_days=7)

        if y.sum() >= 30:  # Need enough positive cases
            results, model, scaler = train_models(X, y, threshold)
        else:
            print(f"\nSkipping {threshold*100}% threshold - only {y.sum()} positive cases")


if __name__ == "__main__":
    main()
