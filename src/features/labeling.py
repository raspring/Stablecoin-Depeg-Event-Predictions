"""
Label depeg events in stablecoin price data.
"""

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DEPEG_THRESHOLD


def label_depeg_events(
    df: pd.DataFrame,
    threshold: float = DEPEG_THRESHOLD,
    min_duration_hours: int = 1,
) -> pd.DataFrame:
    """
    Label depeg events based on price deviation threshold.

    Args:
        df: DataFrame with 'price', 'peg', 'timestamp' columns
        threshold: Deviation threshold (e.g., 0.01 for 1%)
        min_duration_hours: Minimum hours deviation must persist

    Returns:
        DataFrame with depeg labels added
    """
    df = df.copy()

    # Calculate deviation if not present
    if "abs_deviation" not in df.columns:
        df["deviation"] = (df["price"] - df["peg"]) / df["peg"]
        df["abs_deviation"] = df["deviation"].abs()

    # Label individual points exceeding threshold
    df["exceeds_threshold"] = df["abs_deviation"] >= threshold

    # Identify depeg events (contiguous periods above threshold)
    df["depeg_event_id"] = (
        df["exceeds_threshold"].ne(df["exceeds_threshold"].shift()).cumsum()
    )
    df.loc[~df["exceeds_threshold"], "depeg_event_id"] = 0

    # Filter by minimum duration
    if min_duration_hours > 0 and "timestamp" in df.columns:
        event_durations = df[df["depeg_event_id"] > 0].groupby("depeg_event_id").agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
        )
        event_durations["duration_hours"] = (
            (event_durations["end"] - event_durations["start"]).dt.total_seconds() / 3600
        )

        valid_events = event_durations[
            event_durations["duration_hours"] >= min_duration_hours
        ].index

        df.loc[~df["depeg_event_id"].isin(valid_events), "depeg_event_id"] = 0

    # Binary label
    df["is_depeg"] = (df["depeg_event_id"] > 0).astype(int)

    return df


def create_prediction_target(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    threshold: float = DEPEG_THRESHOLD,
) -> pd.DataFrame:
    """
    Create forward-looking prediction target.

    Args:
        df: DataFrame with timestamp and price data
        horizon_hours: Hours ahead to predict
        threshold: Depeg threshold

    Returns:
        DataFrame with 'target' column (1 if depeg within horizon)
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calculate if deviation exceeds threshold
    df["exceeds_threshold"] = df["abs_deviation"] >= threshold

    # For each point, check if any point within horizon exceeds threshold
    df["target"] = 0

    # Convert horizon to number of rows (assuming hourly data)
    time_diff = df["timestamp"].diff().median()
    rows_per_hour = pd.Timedelta(hours=1) / time_diff if time_diff else 1
    horizon_rows = int(horizon_hours * rows_per_hour)

    # Rolling forward look
    for i in range(len(df) - 1, -1, -1):
        end_idx = min(i + horizon_rows, len(df))
        if df.loc[i:end_idx, "exceeds_threshold"].any():
            df.loc[i, "target"] = 1

    return df


def summarize_depeg_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize depeg events.

    Args:
        df: DataFrame with depeg labels

    Returns:
        Summary DataFrame of depeg events
    """
    if "depeg_event_id" not in df.columns:
        raise ValueError("Run label_depeg_events first")

    events = df[df["depeg_event_id"] > 0].groupby("depeg_event_id").agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        max_deviation=("abs_deviation", "max"),
        mean_deviation=("abs_deviation", "mean"),
        num_observations=("price", "count"),
    )

    events["duration_hours"] = (
        (events["end_time"] - events["start_time"]).dt.total_seconds() / 3600
    )

    return events.sort_values("start_time")
