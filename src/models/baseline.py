"""
Baseline models for stablecoin depeg prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.features.engineering import prepare_modeling_data


def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Print evaluation metrics for a model."""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print('='*60)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # ROC-AUC
    if len(np.unique(y_true)) > 1:
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


def train_and_evaluate(X, y, threshold=0.005):
    """Train multiple models and evaluate using time series cross-validation."""

    print(f"\n{'#'*60}")
    print(f"DEPEG PREDICTION MODEL EVALUATION")
    print(f"Threshold: {threshold*100:.2f}%")
    print(f"{'#'*60}")

    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution: 0={sum(y==0)}, 1={sum(y==1)} ({y.mean()*100:.2f}% positive)")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Calculate class weights
    class_weight = {0: 1, 1: sum(y==0) / sum(y==1)} if sum(y==1) > 0 else None
    print(f"Class weight for positive class: {class_weight[1]:.2f}" if class_weight else "")

    results = []

    # Convert y to numpy for indexing
    y_np = y.values if hasattr(y, 'values') else np.array(y)

    # Model 1: Logistic Regression (baseline)
    print("\n" + "-"*40)
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    y_pred_lr = np.zeros(len(y))
    y_prob_lr = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X_scaled):
        lr.fit(X_scaled[train_idx], y_np[train_idx])
        y_pred_lr[test_idx] = lr.predict(X_scaled[test_idx])
        y_prob_lr[test_idx] = lr.predict_proba(X_scaled[test_idx])[:, 1]

    # Only evaluate on test indices (exclude first fold which was never tested)
    test_mask = y_prob_lr > 0
    result = evaluate_model(y_np[test_mask], y_pred_lr[test_mask], y_prob_lr[test_mask], "Logistic Regression (Balanced)")
    if result:
        results.append(result)

    # Model 2: Random Forest
    print("\n" + "-"*40)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    y_pred_rf = np.zeros(len(y))
    y_prob_rf = np.zeros(len(y))
    for train_idx, test_idx in tscv.split(X_scaled):
        rf.fit(X_scaled[train_idx], y_np[train_idx])
        y_pred_rf[test_idx] = rf.predict(X_scaled[test_idx])
        y_prob_rf[test_idx] = rf.predict_proba(X_scaled[test_idx])[:, 1]

    result = evaluate_model(y_np[test_mask], y_pred_rf[test_mask], y_prob_rf[test_mask], "Random Forest (Balanced)")
    if result:
        results.append(result)

    # Model 3: Gradient Boosting
    print("\n" + "-"*40)
    print("Training Gradient Boosting...")
    sample_weight = np.where(y_np == 1, class_weight[1], 1) if class_weight else None

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    y_pred_gb = np.zeros(len(y))
    y_prob_gb = np.zeros(len(y))

    for train_idx, test_idx in tscv.split(X_scaled):
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        gb.fit(X_scaled[train_idx], y_np[train_idx], sample_weight=sw_train)
        y_pred_gb[test_idx] = gb.predict(X_scaled[test_idx])
        y_prob_gb[test_idx] = gb.predict_proba(X_scaled[test_idx])[:, 1]

    result = evaluate_model(y_np[test_mask], y_pred_gb[test_mask], y_prob_gb[test_mask], "Gradient Boosting (Weighted)")
    if result:
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Feature importance from Random Forest
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    rf.fit(X_scaled, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(15).to_string(index=False))

    return results_df, rf, scaler


def main():
    """Run baseline model training and evaluation."""
    # Load data
    print("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "usdt_merged_daily.csv")

    # Prepare data with 0.5% threshold (more positive cases)
    print("Preparing features...")
    X, y, feature_names, df_full = prepare_modeling_data(df, threshold=0.005, horizon_days=7)

    # Train and evaluate
    results, best_model, scaler = train_and_evaluate(X, y, threshold=0.005)

    # Also try with 1% threshold
    print("\n\n" + "#"*60)
    print("REPEATING WITH 1% THRESHOLD")
    print("#"*60)
    X_1pct, y_1pct, _, _ = prepare_modeling_data(df, threshold=0.01, horizon_days=7)
    print(f"Skipping 1% threshold - only {y_1pct.sum()} positive cases")
    print("With so few positive cases, time series CV folds may have no positive samples")
    print("Recommendation: Use 0.5% threshold or add more stablecoins to dataset")

    return results, best_model


if __name__ == "__main__":
    main()
