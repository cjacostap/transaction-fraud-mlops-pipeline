"""
MLflow integration utilities for fraud detection pipeline.

This module provides helper functions to log parameters, metrics, artifacts,
and tags to MLflow tracking server.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setup_mlflow(config: Dict[str, Any]) -> None:
    """
    Setup MLflow tracking URI and experiment.
    
    Priority order for tracking URI:
    1. Environment variable MLFLOW_TRACKING_URI
    2. Config file mlflow.tracking_uri
    3. Default: http://localhost:5000

    Args:
        config: Configuration dictionary with mlflow settings
    """
    import os
    
    mlflow_cfg = config.get("mlflow", {})
    
    # Priority: env var > config > default
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "http://localhost:5001")
    )
    experiment_name = mlflow_cfg.get("experiment_name", "fraud-detection-experiments")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("MLflow experiment: %s", experiment_name)


def log_training_params(config: Dict[str, Any]) -> None:
    """
    Log training configuration parameters to MLflow.

    Args:
        config: Configuration dictionary
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping training params logging")
        return
    
    params_to_log = {
        "test_size": config.get("test_size"),
        "val_size": config.get("val_size"),
        "random_state": config.get("random_state"),
        "use_smote": config.get("use_smote"),
        "smote_sampling_strategy": config.get("smote_sampling_strategy"),
        "use_class_weights": config.get("use_class_weights"),
        "drop_collinear": config.get("drop_collinear"),
        "corr_threshold": config.get("corr_threshold"),
        "hidden_layers": str(config.get("hidden_layers")),
        "dropout_rate": config.get("dropout_rate"),
        "l2_reg": config.get("l2_reg"),
        "activation": config.get("activation"),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "learning_rate": config.get("learning_rate"),
        "patience_early_stop": config.get("patience_early_stop"),
        "patience_reduce_lr": config.get("patience_reduce_lr"),
        "scaler_type": config.get("scaler_type"),
        "use_optuna": config.get("use_optuna"),
    }
    
    # Remove None values
    params_to_log = {k: v for k, v in params_to_log.items() if v is not None}
    
    mlflow.log_params(params_to_log)
    logger.info("Logged %d training parameters to MLflow", len(params_to_log))


def log_data_statistics(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Log dataset statistics to MLflow as metrics.

    Args:
        X: Features DataFrame
        y: Target Series
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping data statistics logging")
        return
    
    stats = {
        "data_n_samples": len(X),
        "data_n_features": X.shape[1],
        "data_fraud_rate": float(y.mean()),
        "data_n_fraud": int(y.sum()),
        "data_n_legitimate": int((y == 0).sum()),
    }
    
    mlflow.log_metrics(stats)
    logger.info("Logged dataset statistics to MLflow")


def log_evaluation_metrics(metrics: Dict[str, Any]) -> None:
    """
    Log evaluation metrics to MLflow.

    Args:
        metrics: Dictionary with evaluation metrics
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping evaluation metrics logging")
        return
    
    metrics_to_log = {
        "test_accuracy": metrics.get("accuracy"),
        "test_precision": metrics.get("precision"),
        "test_recall": metrics.get("recall"),
        "test_f1_score": metrics.get("f1_score"),
        "test_auc_roc": metrics.get("auc_roc"),
        "test_auc_pr": metrics.get("auc_pr"),
        "optimal_threshold": metrics.get("optimal_threshold"),
        "optimal_f1_score": metrics.get("optimal_f1_score"),
    }
    
    # Log confusion matrix components
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        metrics_to_log.update({
            "test_true_negatives": cm.get("true_negatives"),
            "test_false_positives": cm.get("false_positives"),
            "test_false_negatives": cm.get("false_negatives"),
            "test_true_positives": cm.get("true_positives"),
        })
    
    # Remove None values
    metrics_to_log = {k: v for k, v in metrics_to_log.items() if v is not None}
    
    mlflow.log_metrics(metrics_to_log)
    logger.info("Logged %d evaluation metrics to MLflow", len(metrics_to_log))


def log_optuna_results(tuned_params: Dict[str, Any]) -> None:
    """
    Log Optuna hyperparameter tuning results to MLflow.

    Args:
        tuned_params: Dictionary with best hyperparameters from Optuna
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping Optuna results logging")
        return
    
    params_to_log = {f"optuna_{k}": v for k, v in tuned_params.items()}
    
    # Convert lists to strings for MLflow
    for k, v in params_to_log.items():
        if isinstance(v, (list, tuple)):
            params_to_log[k] = str(v)
    
    mlflow.log_params(params_to_log)
    logger.info("Logged Optuna tuning results to MLflow")


def log_model_artifacts(
    model_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """
    Log additional artifacts (plots, reports) to MLflow.

    Args:
        model_dir: Directory with model artifacts
        reports_dir: Directory with evaluation reports
        figures_dir: Directory with plots/figures
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping artifacts logging")
        return
    
    # Log reports
    if reports_dir.exists():
        for report_file in reports_dir.glob("*.txt"):
            mlflow.log_artifact(str(report_file), artifact_path="reports")
        for report_file in reports_dir.glob("*.json"):
            mlflow.log_artifact(str(report_file), artifact_path="reports")
    
    # Log figures
    if figures_dir.exists():
        for figure_file in figures_dir.glob("*.png"):
            mlflow.log_artifact(str(figure_file), artifact_path="figures")
    
    logger.info("Logged additional artifacts to MLflow")


def log_tags(tags: Dict[str, Any]) -> None:
    """
    Log custom tags to MLflow run.

    Args:
        tags: Dictionary of tags (key-value pairs)
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run, skipping tags logging")
        return
    
    # Convert all values to strings
    tags = {k: str(v) for k, v in tags.items()}
    
    mlflow.set_tags(tags)
    logger.info("Logged %d tags to MLflow", len(tags))


def get_dvc_data_hash(data_path: str) -> str:
    """
    Get hash of data file for tracking data versions.
    
    Strategy:
    1. If .dvc file exists, log it as artifact and extract hash
    2. Otherwise, compute MD5 hash of data file
    3. If all fails, return "unknown"

    Args:
        data_path: Path to data file

    Returns:
        Hash string (first 8 characters)
    """
    path = Path(data_path)
    dvc_file = path.with_suffix(path.suffix + ".dvc")
    
    # Strategy 1: Use DVC file if it exists
    if dvc_file.exists():
        try:
            # Log .dvc file as artifact for full reproducibility
            if mlflow.active_run():
                mlflow.log_artifact(str(dvc_file), artifact_path="data_version")
                logger.info("Logged DVC file: %s", dvc_file.name)
            
            # Extract hash from .dvc file
            import yaml
            with open(dvc_file) as f:
                dvc_data = yaml.safe_load(f)
                # DVC stores MD5 hash in 'outs' section
                if "outs" in dvc_data and len(dvc_data["outs"]) > 0:
                    md5_hash = dvc_data["outs"][0].get("md5", "")
                    if md5_hash:
                        return md5_hash[:8]
        except Exception as e:
            logger.warning("Could not parse DVC file: %s", e)
    
    # Strategy 2: Compute MD5 hash directly
    try:
        if path.exists():
            hasher = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:8]
    except Exception as e:
        logger.warning("Could not compute data hash: %s", e)
    
    return "unknown"
