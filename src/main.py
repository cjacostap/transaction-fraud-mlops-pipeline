#!/usr/bin/env python3
"""
Main training pipeline for fraud detection model.

This is the single entry point for training the fraud detection model.
It orchestrates data loading, preprocessing, training, and evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models.signature import infer_signature

from src.mlflow_integration import (
    get_dvc_data_hash,
    log_data_statistics,
    log_evaluation_metrics,
    log_model_artifacts,
    log_optuna_results,
    log_tags,
    log_training_params,
    setup_mlflow,
)
from src.mlflow_pyfunc_wrapper import FraudModelWrapper
from src.data import drop_highly_correlated_features, prepare_dataset
from src.model import run_optuna_tuning, save_artifacts, train_final_model
from src.validation import evaluate_model, plot_training_history


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix in [".yaml", ".yml"]:
            import yaml

            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    logger.info("Configuration loaded from %s", config_path)
    return config


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to raw data CSV (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for models and results (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (overrides config)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation on test set (for quick training tests)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training (overrides config)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="Test size ratio (overrides config)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        help="Validation size ratio (overrides config)",
    )
    parser.add_argument(
        "--use-smote",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable SMOTE (overrides config)",
    )
    parser.add_argument(
        "--smote-sampling-strategy",
        type=float,
        help="SMOTE sampling strategy (overrides config)",
    )
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable class weights (overrides config)",
    )
    parser.add_argument(
        "--drop-collinear",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable collinearity dropping (overrides config)",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        help="Correlation threshold for dropping features (overrides config)",
    )
    parser.add_argument(
        "--corr-sample-size",
        type=int,
        help="Sample size for correlation computation (overrides config)",
    )
    parser.add_argument(
        "--corr-exclude",
        nargs="*",
        type=str,
        help="Feature names to exclude from collinearity dropping",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        help="Hidden layer sizes (e.g., --hidden-layers 256 128 64)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        help="Dropout rate (overrides config)",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        help="L2 regularization factor (overrides config)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="Activation function name (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--patience-early-stop",
        type=int,
        help="Early stopping patience (overrides config)",
    )
    parser.add_argument(
        "--patience-reduce-lr",
        type=int,
        help="ReduceLROnPlateau patience (overrides config)",
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        help="Scaler type: standard or robust (overrides config)",
    )
    parser.add_argument(
        "--use-optuna",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable Optuna tuning (overrides config)",
    )
    parser.add_argument(
        "--optuna-n-trials-coarse",
        type=int,
        help="Optuna coarse trials (overrides config)",
    )
    parser.add_argument(
        "--optuna-n-trials-fine",
        type=int,
        help="Optuna fine trials (overrides config)",
    )
    parser.add_argument(
        "--optuna-epochs",
        type=int,
        help="Epochs per Optuna trial (overrides config)",
    )
    parser.add_argument(
        "--optuna-patience",
        type=int,
        help="Early stopping patience for Optuna (overrides config)",
    )
    parser.add_argument(
        "--optuna-cv-folds",
        type=int,
        help="Number of CV folds for Optuna (overrides config)",
    )
    parser.add_argument(
        "--optuna-val-size",
        type=float,
        help="Hold-out validation size for Optuna (overrides config)",
    )
    parser.add_argument(
        "--optuna-subsample",
        type=int,
        help="Subsample size for Optuna (overrides config)",
    )
    parser.add_argument(
        "--optuna-direction",
        type=str,
        choices=["maximize", "minimize"],
        help="Optuna optimization direction (overrides config)",
    )
    parser.add_argument(
        "--optuna-metric",
        type=str,
        help="Optuna metric name (overrides config)",
    )

    return parser.parse_args()


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration with command line arguments.
    CLI args take precedence over config file.

    Args:
        config: Configuration from file
        args: Parsed command line arguments

    Returns:
        Merged configuration
    """
    if args.data_path:
        config["data_path"] = args.data_path

    if args.output_dir:
        config["output_dir"] = args.output_dir

    if args.seed is not None:
        config["random_state"] = args.seed

    if args.epochs is not None:
        config["epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.test_size is not None:
        config["test_size"] = args.test_size

    if args.val_size is not None:
        config["val_size"] = args.val_size

    if args.use_smote is not None:
        config["use_smote"] = args.use_smote

    if args.smote_sampling_strategy is not None:
        config["smote_sampling_strategy"] = args.smote_sampling_strategy

    if args.use_class_weights is not None:
        config["use_class_weights"] = args.use_class_weights

    if args.drop_collinear is not None:
        config["drop_collinear"] = args.drop_collinear

    if args.corr_threshold is not None:
        config["corr_threshold"] = args.corr_threshold

    if args.corr_sample_size is not None:
        config["corr_sample_size"] = args.corr_sample_size

    if args.corr_exclude is not None:
        config["corr_exclude"] = args.corr_exclude

    if args.hidden_layers is not None:
        config["hidden_layers"] = args.hidden_layers

    if args.dropout_rate is not None:
        config["dropout_rate"] = args.dropout_rate

    if args.l2_reg is not None:
        config["l2_reg"] = args.l2_reg

    if args.activation is not None:
        config["activation"] = args.activation

    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    if args.patience_early_stop is not None:
        config["patience_early_stop"] = args.patience_early_stop

    if args.patience_reduce_lr is not None:
        config["patience_reduce_lr"] = args.patience_reduce_lr

    if args.scaler_type is not None:
        config["scaler_type"] = args.scaler_type

    if args.use_optuna is not None:
        config["use_optuna"] = args.use_optuna

    if args.optuna_n_trials_coarse is not None:
        config["optuna_n_trials_coarse"] = args.optuna_n_trials_coarse

    if args.optuna_n_trials_fine is not None:
        config["optuna_n_trials_fine"] = args.optuna_n_trials_fine

    if args.optuna_epochs is not None:
        config["optuna_epochs"] = args.optuna_epochs

    if args.optuna_patience is not None:
        config["optuna_patience"] = args.optuna_patience

    if args.optuna_cv_folds is not None:
        config["optuna_cv_folds"] = args.optuna_cv_folds

    if args.optuna_val_size is not None:
        config["optuna_val_size"] = args.optuna_val_size

    if args.optuna_subsample is not None:
        config["optuna_subsample"] = args.optuna_subsample

    if args.optuna_direction is not None:
        config["optuna_direction"] = args.optuna_direction

    if args.optuna_metric is not None:
        config["optuna_metric"] = args.optuna_metric

    return config


def split_data(
    X, y, config: Dict[str, Any]
) -> tuple:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        config: Configuration dictionary

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    test_size = config.get("test_size", 0.2)
    val_size = config.get("val_size", 0.15)
    random_state = config.get("random_state", 42)

    logger.info("Splitting data: test_size=%.2f, val_size=%.2f", test_size, val_size)

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info("Data split completed:")
    logger.info("  Train: %d samples (%.1f%%)", len(X_train), len(X_train) / len(X) * 100)
    logger.info("  Val:   %d samples (%.1f%%)", len(X_val), len(X_val) / len(X) * 100)
    logger.info("  Test:  %d samples (%.1f%%)", len(X_test), len(X_test) / len(X) * 100)
    logger.info("  Fraud rate - Train: %.4f%%, Val: %.4f%%, Test: %.4f%%",
                y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100)

    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_summary_report(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    training_time: float,
    model_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """
    Generate executive summary report.

    Args:
        config: Configuration used
        metrics: Evaluation metrics
        training_time: Time taken for training (seconds)
        output_dir: Output directory
    """
    summary = f"""
{'='*70}
FRAUD DETECTION MODEL - TRAINING SUMMARY
{'='*70}

Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)

CONFIGURATION
{'='*70}
Data Path: {config.get('data_path')}
Output Directory: {config.get('output_dir')}
Output Root: {config.get('output_root', 'outputs')}
Random Seed: {config.get('random_state')}

Data Split:
  - Test Size: {config.get('test_size', 0.2):.1%}
  - Validation Size: {config.get('val_size', 0.15):.1%}

Model Architecture:
  - Hidden Layers: {config.get('hidden_layers', [256, 128, 64, 32])}
  - Dropout Rate: {config.get('dropout_rate', 0.3)}
  - L2 Regularization: {config.get('l2_reg', 0.001)}
  - Learning Rate: {config.get('learning_rate', 0.001)}

Training:
  - Epochs: {config.get('epochs', 50)}
  - Batch Size: {config.get('batch_size', 512)}
  - Use SMOTE: {config.get('use_smote', False)}
  - SMOTE Strategy: {config.get('smote_sampling_strategy', 0.3)}
  - Use Class Weights: {config.get('use_class_weights', False)}
  - Scaler Type: {config.get('scaler_type', 'standard')}
  - Drop Collinear: {config.get('drop_collinear', False)}
  - Corr Threshold: {config.get('corr_threshold', 0.95)}
  - Corr Sample Size: {config.get('corr_sample_size', None)}

EVALUATION RESULTS
{'='*70}
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1_score']:.4f}
AUC-ROC:   {metrics['auc_roc']:.4f}
AUC-PR:    {metrics['auc_pr']:.4f}

Confusion Matrix:
  True Negatives:  {metrics['confusion_matrix']['true_negatives']:,}
  False Positives: {metrics['confusion_matrix']['false_positives']:,}
  False Negatives: {metrics['confusion_matrix']['false_negatives']:,}
  True Positives:  {metrics['confusion_matrix']['true_positives']:,}

Optimal Threshold: {metrics.get('optimal_threshold', 0.5):.3f}
Optimal F1-Score:  {metrics.get('optimal_f1_score', 0.0):.4f}

ARTIFACTS
{'='*70}
Model: {model_dir / 'fraud_detection_model.keras'}
Pipeline: {model_dir / 'preprocessing_pipeline.pkl'}
Metrics: {reports_dir / 'evaluation_metrics.json'}
Plots: {figures_dir / '*.png'}

{'='*70}
"""

    # Save summary
    summary_path = reports_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    # Print to console
    print(summary)

    logger.info("Summary report saved to %s", summary_path)


def main():
    """Main training pipeline."""
    start_time = datetime.now()

    try:
        # Parse arguments
        args = parse_args()

        # Load configuration
        try:
            config = load_config(args.config)
            setup_mlflow(config)

        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            config = {
                "data_path": "data/raw/onlinefraud.csv",
                "output_root": "outputs",
                "output_dir": "outputs/models",
                "test_size": 0.2,
                "val_size": 0.15,
                "random_state": 42,
                "use_smote": True,
                "smote_sampling_strategy": 0.3,
                "use_class_weights": False,
                "drop_collinear": False,
                "corr_threshold": 0.95,
                "corr_sample_size": 200000,
                "corr_exclude": [],
                "hidden_layers": [256, 128, 64, 32],
                "dropout_rate": 0.3,
                "l2_reg": 0.001,
                "activation": "relu",
                "batch_size": 512,
                "epochs": 50,
                "learning_rate": 0.001,
                "patience_early_stop": 15,
                "patience_reduce_lr": 7,
                "scaler_type": "standard",
                "use_optuna": False,
                "optuna_n_trials_coarse": 20,
                "optuna_n_trials_fine": 30,
                "optuna_epochs": 15,
                "optuna_patience": 5,
                "optuna_cv_folds": 1,
                "optuna_val_size": 0.15,
                "optuna_subsample": 500000,
                "optuna_direction": "maximize",
                "optuna_metric": "auc_pr",
            }

        # Merge with CLI arguments
        config = merge_config_with_args(config, args)

        logger.info("="*70)
        logger.info("FRAUD DETECTION MODEL - TRAINING PIPELINE")
        logger.info("="*70)
        
        # ─────────────────────────────────────────────
        # START MLFLOW RUN
        # ─────────────────────────────────────────────

        with mlflow.start_run(run_name=f"fraud-training-{datetime.now():%Y%m%d-%H%M}"):

            # ─────────────────────────────────────────────
            # LOGGING
            # ─────────────────────────────────────────────

            # Guarda en MLflow los parámetros de configuración: ephocs, batch size, learning rate, etc.
            log_training_params(config)

            # Guarda en MLflow el hash del dataset: (si usas DVC)

            # Esto responde a la pregunta: "¿Qué dataset usamos para este experimento?"
            data_hash = get_dvc_data_hash(config["data_path"])
            log_tags({
                "data_hash": data_hash,
                "project": "fraud-detection",
                "framework": "tensorflow",
            })

            # ─────────────────────────────────────────────
            # LOAD AND PROCESS DATA
            # ─────────────────────────────────────────────

            logger.info("\n[1/7] Loading and processing data...")
            data_path = config.get("data_path", "data/raw/onlinefraud.csv")
            X, y = prepare_dataset(data_path)
            logger.info("Dataset prepared: X shape=%s, y shape=%s", X.shape, y.shape)

            if config.get("drop_collinear", False):
                logger.info("Dropping highly correlated features...")
                X = drop_highly_correlated_features(
                    X,
                    threshold=config.get("corr_threshold", 0.95),
                    sample_size=config.get("corr_sample_size", None),
                    random_state=config.get("random_state", 42),
                    exclude=config.get("corr_exclude", []),
                )
                logger.info("Post-correlation feature shape: %s", X.shape)

            # Guarda en MLflow las estadísticas del dataset: (media, desviación estándar, etc.)
            log_data_statistics(X, y)


            # Step 2: Split data
            logger.info("\n[2/7] Splitting data...")
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

            # Step 3: Optional Optuna tuning
            if config.get("use_optuna", False):
                logger.info("\n[3/7] Running Optuna tuning...")
                tuned_params = run_optuna_tuning(X_train, y_train, config)
                log_optuna_results(tuned_params)
                config.update(tuned_params)
                logger.info("Optuna tuning completed. Best params applied to config.")
            else:
                logger.info("\n[3/7] Skipping Optuna tuning")

            # Step 4: Train model (with validation for early stopping)
            logger.info("\n[4/7] Training model with validation...")
            model, history, preprocessing_pipeline = train_final_model(
                X_train, y_train, X_val, y_val, config, use_validation=True
            )
            logger.info("Initial training completed")

            # Prepare output directories
            output_root = Path(config.get("output_root", "outputs"))
            model_dir = Path(config.get("output_dir", output_root / "models"))
            reports_dir = output_root / "reports"
            figures_dir = output_root / "figures"
            model_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_evaluation:
                logger.info("\n[5/7] Evaluating model on test set...")
                
                # Preprocess test data
                X_test_processed = preprocessing_pipeline.transform(X_test)
                
                # Evaluate
                metrics = evaluate_model(
                    model=model,
                    X_test=X_test_processed,
                    y_test=y_test.values if hasattr(y_test, "values") else y_test,
                    output_dir=reports_dir,
                    threshold=0.5,
                    figures_dir=figures_dir,
                )
                log_evaluation_metrics(metrics)
                logger.info("Evaluation completed")

                # Plot training history
                logger.info("Generating training history plots...")
                plot_training_history(history, figures_dir)
            else:
                logger.info("\n[5/7] Skipping evaluation (--skip-evaluation flag)")
                metrics = {}

            # Step 6: Retrain final model on train+val (no validation)
            logger.info("\n[6/7] Retraining final model on train+val...")
            X_train_full = pd.concat([X_train, X_val], axis=0)
            y_train_full = pd.concat([y_train, y_val], axis=0)
            final_model, final_history, final_pipeline = train_final_model(
                X_train_full, y_train_full, None, None, config, use_validation=False
            )


            # Step 7: Save artifacts:
            # Guarda en MLflow el modelo, el pipeline de preprocesamiento y los nombres de las features.
            logger.info("\n[7/7] Saving artifacts...")
            feature_names = X_train.columns.tolist()
            save_artifacts(
                model=final_model,
                preprocessing_pipeline=final_pipeline,
                feature_names=feature_names,
                config=config,
                output_dir=model_dir,
            )

            # El bloque comentado es el que usaba para guardar el modelo en MLflow usando tensorflow.
            # mlflow.tensorflow.log_model(
            #     model=final_model,
            #     artifact_path="model",
            #     signature=signature,
            #     registered_model_name=model_name,
            # )


            # ─────────────────────────────────────────────
            # REGISTER MODEL IN MLFLOW MODEL REGISTRY
            # ─────────────────────────────────────────────

            if config.get("model_registry", {}).get("enabled", False):

                model_name = config["model_registry"]["model_name"]

                # Define paths to model and pipeline
                model_path = model_dir / "fraud_detection_model.keras"
                pipeline_path = model_dir / "preprocessing_pipeline.pkl"

                # Sample for signature
                X_sample_raw = X_train_full.sample(50, random_state=42)
                X_sample = final_pipeline.transform(X_sample_raw)

                signature = infer_signature(
                    X_sample_raw,  # Raw features (before preprocessing)
                    final_model.predict(X_sample)
                )

            # En este bloque se guarda el modelo en MLflow usando pyfunc (no solo tensorflow sino processing pipeline).
            # Esto es útil porque permite usar el modelo en otros frameworks, como por ejemplo, en un servidor de FastAPI.

                model_info = mlflow.pyfunc.log_model(
                    artifact_path="fraud_model",
                    python_model=FraudModelWrapper(),
                    artifacts={
                        "model": str(model_path),
                        "pipeline": str(pipeline_path),
                    },
                    signature=signature,
                    registered_model_name=model_name,
                )
            
                # Add tags and description to the registered model version
                client = mlflow.tracking.MlflowClient()
                
                # Get the latest version number that was just registered
                latest_versions = client.get_latest_versions(model_name, stages=["None"])
                if latest_versions:
                    version = latest_versions[0].version
                    
                    # Set tags on the registered model (model-level)
                    client.set_registered_model_tag(model_name, "task", "fraud_detection")
                    client.set_registered_model_tag(model_name, "framework", "tensorflow")
                    client.set_registered_model_tag(model_name, "model_type", "deep_neural_network")
                    
                    # Set tags on the model version (version-level)
                    client.set_model_version_tag(model_name, version, "stage", "training")
                    client.set_model_version_tag(model_name, version, "used_smote", str(config.get("use_smote")))
                    client.set_model_version_tag(model_name, version, "used_optuna", str(config.get("use_optuna")))
                    if not args.skip_evaluation:
                        client.set_model_version_tag(model_name, version, "test_auc_pr", f"{metrics.get('auc_pr', 0.0):.4f}")
                        client.set_model_version_tag(model_name, version, "test_f1", f"{metrics.get('f1_score', 0.0):.4f}")
                    
                    # Update model version description
                    if not args.skip_evaluation:
                        description = f"""Fraud Detection Deep Neural Network

**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Performance on Test Set:**
- AUC-PR: {metrics.get('auc_pr', 0.0):.4f}
- AUC-ROC: {metrics.get('auc_roc', 0.0):.4f}
- F1-Score: {metrics.get('f1_score', 0.0):.4f}
- Precision: {metrics.get('precision', 0.0):.4f}
- Recall: {metrics.get('recall', 0.0):.4f}

**Configuration:**
- Hidden Layers: {config.get('hidden_layers')}
- Dropout: {config.get('dropout_rate')}
- Learning Rate: {config.get('learning_rate')}
- SMOTE: {config.get('use_smote')}
- Optuna Tuning: {config.get('use_optuna')}
"""
                    else:
                        description = f"""Fraud Detection Deep Neural Network

**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Configuration:**
- Hidden Layers: {config.get('hidden_layers')}
- Dropout: {config.get('dropout_rate')}
- Learning Rate: {config.get('learning_rate')}
- SMOTE: {config.get('use_smote')}
- Optuna Tuning: {config.get('use_optuna')}
"""
                    
                    client.update_model_version(
                        name=model_name,
                        version=version,
                        description=description
                    )
                    
                    logger.info("Model version %s registered with tags and description", version)
            
                # (Opcional) log extra de artefactos del experimento
                log_model_artifacts(
                    model_dir=model_dir,
                    reports_dir=reports_dir,
                    figures_dir=figures_dir,
                )

            logger.info("All artifacts saved to %s", model_dir)

            # Generate summary report
            if not args.skip_evaluation:
                training_time = (datetime.now() - start_time).total_seconds()
                generate_summary_report(
                    config, metrics, training_time, model_dir, reports_dir, figures_dir
                )

            logger.info("\n" + "="*70)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)

            log_tags({
                "stage": "training",
                "final_model": "true",
                "used_optuna": config.get("use_optuna"),
                "used_smote": config.get("use_smote"),
            })

            return 0

    except Exception as e:
        logger.error("="*70)
        logger.error("TRAINING PIPELINE FAILED")
        logger.error("="*70)
        logger.exception("Error: %s", str(e))
        return 1



if __name__ == "__main__":
    sys.exit(main())
