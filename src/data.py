
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw transaction data from CSV or Parquet."""
    data_path = Path(path)
    logger.info("Loading raw data from %s", data_path)

    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info("Raw data loaded with shape %s", df.shape)
    return df


def extract_target(df: pd.DataFrame, target_col: str = "isFraud") -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target from features without mutating the input."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    logger.info("Target extracted with %d samples", len(y))
    return X, y


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic type casting for known columns."""
    df = df.copy()
    if "type" in df.columns:
        df["type"] = df["type"].astype("category")
    return df


def create_missing_flags(df: pd.DataFrame, balance_cols: Iterable[str]) -> pd.DataFrame:
    """Create missing-value indicator features for balance columns."""
    df = df.copy()
    for col in balance_cols:
        if col in df.columns:
            df[f"{col}_miss"] = df[col].isna().astype(int)

    miss_flags = [f"{col}_miss" for col in balance_cols if f"{col}_miss" in df.columns]
    if miss_flags:
        df["any_balance_missing"] = df[miss_flags].any(axis=1).astype(int)

    logger.info("Missing flags created for %d balance columns", len(miss_flags))
    return df


def winsorize_series(series: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    """Clip values based on percentiles to limit extreme outliers."""
    lo, hi = series.quantile([p_low, p_high])
    return series.clip(lower=lo, upper=hi)


def handle_outliers(df: pd.DataFrame, balance_cols: Iterable[str]) -> pd.DataFrame:
    """Apply gentle outlier handling to balance columns and amount."""
    df = df.copy()
    for col in balance_cols:
        if col in df.columns and df[col].notna().sum() > 100:
            df[col] = winsorize_series(df[col], p_low=0.005, p_high=0.995)

    if "amount" in df.columns:
        df.loc[df["amount"] < 0, "amount"] = 0

    logger.info("Outliers handled for balance columns and amount")
    return df


def create_balance_features(df: pd.DataFrame, balance_cols: Iterable[str]) -> pd.DataFrame:
    """Create balance-based engineered features."""
    df = df.copy()
    for col in balance_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["diff_orig"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["diff_dest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["inconsistent_orig"] = (np.abs(df["diff_orig"]) > 0.01).astype(int)
    df["inconsistent_dest"] = (np.abs(df["diff_dest"]) > 0.01).astype(int)
    df["emptied_account"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    df["dest_account_new"] = (df["oldbalanceDest"] == 0).astype(int)

    logger.info("Balance features created")
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create amount-based engineered features."""
    df = df.copy()
    df["log1p_amount"] = np.log1p(df["amount"])
    df["amount_vs_balance_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_vs_balance_dest"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["amount_equals_balance"] = (np.abs(df["amount"] - df["oldbalanceOrg"]) < 0.01).astype(int)
    df["amount_is_round"] = (df["amount"] % 10000 == 0).astype(int)

    logger.info("Amount features created")
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from 'step'."""
    df = df.copy()
    if "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        df["day_of_month"] = (df["step"] // 24) % 30
        df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
        df["is_weekend"] = (df["day_of_month"] % 7).isin([0, 6]).astype(int)

    logger.info("Temporal features created")
    return df


def create_log_transformations(df: pd.DataFrame, log_cols: Iterable[str]) -> pd.DataFrame:
    """Add log1p transformations for selected columns."""
    df = df.copy()
    for col in log_cols:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col])

    logger.info("Log transformations created for %d columns", len(list(log_cols)))
    return df


def drop_unused_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop columns not intended for modeling."""
    df = df.copy()
    existing = [col for col in columns if col in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    logger.info("Dropped %d unused columns", len(existing))
    return df


def drop_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    sample_size: int | None = None,
    random_state: int = 42,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Drop highly colinear numeric features based on absolute correlation.

    Args:
        df: Input DataFrame
        threshold: Absolute correlation threshold for dropping
        sample_size: Optional row sample size for correlation computation
        random_state: Random seed for sampling
        exclude: Columns to exclude from dropping

    Returns:
        DataFrame with colinear features removed
    """
    df = df.copy()
    exclude_set = set(exclude or [])

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        logger.info("Not enough numeric features for correlation analysis")
        return df

    if sample_size is not None and len(numeric_df) > sample_size:
        numeric_df = numeric_df.sample(n=sample_size, random_state=random_state)

    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    to_drop = [col for col in to_drop if col not in exclude_set]

    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        logger.info(
            "Dropped %d highly correlated features (threshold=%.2f): %s",
            len(to_drop),
            threshold,
            to_drop,
        )
    else:
        logger.info("No features dropped for correlation threshold %.2f", threshold)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run model-agnostic cleaning steps."""
    df = cast_types(df)
    balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df = create_missing_flags(df, balance_cols)
    df = handle_outliers(df, balance_cols)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering steps without model-specific preprocessing."""
    balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df = create_balance_features(df, balance_cols)
    df = create_amount_features(df)
    df = create_temporal_features(df)
    df = create_log_transformations(df, balance_cols)

    cols_to_drop = [
        "nameOrig",
        "nameDest",
    ]
    df = drop_unused_columns(df, cols_to_drop)
    return df


def prepare_dataset(path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load, clean, and feature-engineer the dataset.

    Returns:
        X: Feature dataframe
        y: Target series
    """
    df_raw = load_raw_data(path)
    X, y = extract_target(df_raw, target_col="isFraud")
    X = clean_data(X)
    X = engineer_features(X)
    logger.info("Prepared dataset with features shape %s and target shape %s", X.shape, y.shape)
    return X, y
