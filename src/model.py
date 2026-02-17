from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import optuna

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Random seeds set to %d", seed)


def _stratified_subsample(
    X: pd.DataFrame, y: pd.Series | np.ndarray, sample_size: int, random_state: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a stratified subsample for faster tuning."""
    if sample_size >= len(X):
        return X, y.values if isinstance(y, pd.Series) else y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=random_state,
        stratify=y,
    )
    return X_sample, y_sample.values if isinstance(y_sample, pd.Series) else y_sample


def _suggest_hyperparameters(
    trial: optuna.Trial,
    stage: str,
    best_params: dict | None = None,
) -> dict:
    """Suggest hyperparameters for Optuna tuning."""
    params: dict = {}

    if stage == "coarse":
        n_layers = trial.suggest_int("n_layers", 2, 5)
        params["hidden_layers"] = [
            trial.suggest_int(f"units_l{i+1}", 32, 512, log=True) for i in range(n_layers)
        ]
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)
        params["l2_reg"] = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
        params["scaler_type"] = trial.suggest_categorical("scaler_type", ["standard", "robust"])
        params["use_smote"] = trial.suggest_categorical("use_smote", [True, False])
        params["smote_sampling_strategy"] = trial.suggest_float(
            "smote_sampling_strategy", 0.1, 0.5
        )
        return params

    if stage == "fine" and best_params:
        base_layers = best_params.get("hidden_layers", [256, 128, 64, 32])
        params["hidden_layers"] = [
            trial.suggest_int(
                f"units_l{i+1}",
                max(16, int(units * 0.7)),
                int(units * 1.3),
            )
            for i, units in enumerate(base_layers)
        ]
        params["dropout_rate"] = trial.suggest_float(
            "dropout_rate",
            max(0.05, best_params.get("dropout_rate", 0.3) - 0.1),
            min(0.6, best_params.get("dropout_rate", 0.3) + 0.1),
        )
        params["l2_reg"] = trial.suggest_float(
            "l2_reg",
            max(1e-6, best_params.get("l2_reg", 0.001) / 3),
            min(1e-1, best_params.get("l2_reg", 0.001) * 3),
            log=True,
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            max(1e-6, best_params.get("learning_rate", 0.001) / 3),
            min(1e-1, best_params.get("learning_rate", 0.001) * 3),
            log=True,
        )
        params["batch_size"] = trial.suggest_categorical(
            "batch_size", [best_params.get("batch_size", 512), 256, 1024]
        )
        params["scaler_type"] = trial.suggest_categorical(
            "scaler_type", [best_params.get("scaler_type", "standard")]
        )
        params["use_smote"] = best_params.get("use_smote", False)
        params["smote_sampling_strategy"] = best_params.get("smote_sampling_strategy", 0.3)
        return params

    raise ValueError(f"Unknown tuning stage: {stage}")


def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    config: Dict[str, Any],
    stage: str,
    best_params: dict | None = None,
) -> float:
    """Optuna objective function for hyperparameter tuning."""
    params = _suggest_hyperparameters(trial, stage, best_params)

    # Fix seed per trial for reproducibility
    set_seed(config.get("random_state", 42) + trial.number)

    # Optional subsample for tuning speed
    subsample_size = config.get("optuna_subsample", None)
    if subsample_size:
        X_used, y_used = _stratified_subsample(
            X, y, subsample_size, config.get("random_state", 42)
        )
    else:
        X_used, y_used = X, y

    cv_folds = config.get("optuna_cv_folds", 1)
    if cv_folds <= 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X_used,
            y_used,
            test_size=config.get("optuna_val_size", 0.15),
            random_state=config.get("random_state", 42),
            stratify=y_used,
        )
        score = _train_and_score(
            X_train, y_train, X_val, y_val, config, params
        )
        return score

    # Cross-validation (expensive; use only if configured)
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=config.get("random_state", 42)
    )
    scores = []
    for train_idx, val_idx in cv.split(X_used, y_used):
        X_train = X_used.iloc[train_idx]
        y_train = y_used[train_idx]
        X_val = X_used.iloc[val_idx]
        y_val = y_used[val_idx]
        scores.append(_train_and_score(X_train, y_train, X_val, y_val, config, params))

    return float(np.mean(scores))


def _train_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config: Dict[str, Any],
    params: dict,
) -> float:
    """Train a model for one fold and return AUC-PR score."""
    categorical_features = ["type"] if "type" in X_train.columns else []
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    preprocessing_pipeline = build_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        scaler_type=params.get("scaler_type", "standard"),
    )

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    if params.get("use_smote", False):
        X_train_processed, y_train = apply_smote(
            X_train_processed,
            y_train,
            sampling_strategy=params.get("smote_sampling_strategy", 0.3),
            random_state=config.get("random_state", 42),
        )

    model = build_model(
        input_dim=X_train_processed.shape[1],
        hidden_layers=params.get("hidden_layers"),
        dropout_rate=params.get("dropout_rate", 0.3),
        l2_reg=params.get("l2_reg", 0.001),
        activation=config.get("activation", "relu"),
        learning_rate=params.get("learning_rate", 0.001),
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc_pr",
        patience=config.get("optuna_patience", 5),
        mode="max",
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=config.get("optuna_epochs", 15),
        batch_size=params.get("batch_size", 512),
        callbacks=[early_stop],
        verbose=0,
    )

    y_proba = model.predict(X_val_processed, verbose=0).flatten()
    score = average_precision_score(y_val, y_proba)

    # Clear session to avoid graph buildup across trials
    keras.backend.clear_session()
    return float(score)


def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    config: Dict[str, Any],
) -> dict:
    """Run Optuna tuning with optional coarse → fine strategy."""
    y_array = y_train.values if isinstance(y_train, pd.Series) else y_train

    study_direction = config.get("optuna_direction", "maximize")
    metric_name = config.get("optuna_metric", "auc_pr")

    # Coarse search
    logger.info("Starting Optuna coarse search...")
    study_coarse = optuna.create_study(direction=study_direction)
    study_coarse.optimize(
        lambda trial: _objective(trial, X_train, y_array, config, stage="coarse"),
        n_trials=config.get("optuna_n_trials_coarse", 20),
        timeout=config.get("optuna_timeout", None),
    )
    best_coarse = study_coarse.best_params
    logger.info("Optuna coarse search complete. Best %s=%.4f", metric_name, study_coarse.best_value)

    # Fine search
    logger.info("Starting Optuna fine search...")
    study_fine = optuna.create_study(direction=study_direction)
    study_fine.optimize(
        lambda trial: _objective(
            trial, X_train, y_array, config, stage="fine", best_params=best_coarse
        ),
        n_trials=config.get("optuna_n_trials_fine", 30),
        timeout=config.get("optuna_timeout", None),
    )
    best_fine = study_fine.best_params
    logger.info("Optuna fine search complete. Best %s=%.4f", metric_name, study_fine.best_value)

    # Merge best params into config
    tuned_params = best_coarse.copy()
    tuned_params.update(best_fine)
    return tuned_params


def build_preprocessing_pipeline(
    categorical_features: list[str],
    numerical_features: list[str],
    scaler_type: str = "standard",
) -> Pipeline:
    """
    Build sklearn preprocessing pipeline with encoding and scaling.

    Args:
        categorical_features: Columns to one-hot encode
        numerical_features: Columns to scale
        scaler_type: 'standard' or 'robust'

    Returns:
        sklearn Pipeline ready to fit/transform
    """
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", scaler, numerical_features),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])
    logger.info(
        "Preprocessing pipeline created with %d categorical and %d numerical features",
        len(categorical_features),
        len(numerical_features),
    )
    return pipeline


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to training data only.

    Args:
        X_train: Training features (already scaled)
        y_train: Training target
        sampling_strategy: Ratio of minority class after resampling
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    logger.info("Applying SMOTE with sampling_strategy=%.2f", sampling_strategy)
    original_count = len(y_train)
    fraud_count_before = int(y_train.sum())

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    fraud_count_after = int(y_resampled.sum())
    logger.info(
        "SMOTE completed: %d → %d samples, fraud: %d → %d",
        original_count,
        len(y_resampled),
        fraud_count_before,
        fraud_count_after,
    )
    return X_resampled, y_resampled


def build_model(
    input_dim: int,
    hidden_layers: list[int] = None,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    activation: str = "relu",
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build deep neural network for fraud detection.

    Args:
        input_dim: Number of input features
        hidden_layers: List of units per hidden layer
        dropout_rate: Dropout probability
        l2_reg: L2 regularization coefficient
        activation: Activation function for hidden layers
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    if hidden_layers is None:
        hidden_layers = [256, 128, 64, 32]

    inputs = layers.Input(shape=(input_dim,), name="input_layer")
    x = inputs

    for i, units in enumerate(hidden_layers):
        x = layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer="he_normal",
            name=f"dense_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"batchnorm_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(units=1, activation="sigmoid", name="output_layer")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FraudDetectionNN")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc_roc"),
            keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )

    logger.info(
        "Model built with %d layers, %d total parameters",
        len(hidden_layers),
        model.count_params(),
    )
    return model


def create_callbacks(
    output_dir: Path,
    patience_early_stop: int = 15,
    patience_reduce_lr: int = 7,
) -> list[callbacks.Callback]:
    """
    Create Keras callbacks for training.

    Args:
        output_dir: Directory to save checkpoints and logs
        patience_early_stop: Epochs to wait before early stopping
        patience_reduce_lr: Epochs to wait before reducing LR

    Returns:
        List of Keras callbacks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model_checkpoint.keras"
    tensorboard_dir = output_dir / "tensorboard_logs"

    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_auc_pr",
            patience=patience_early_stop,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            factor=0.5,
            patience=patience_reduce_lr,
            mode="max",
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=str(tensorboard_dir), histogram_freq=1),
    ]

    logger.info("Created %d callbacks for training", len(callback_list))
    return callback_list


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced data.

    Args:
        y_train: Training target array (should be ORIGINAL, not SMOTE-resampled)

    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    # Validate that data is not already balanced
    fraud_ratio = y_train.mean()
    if fraud_ratio > 0.2:
        logger.warning(
            "Class distribution seems balanced (%.2f%% fraud). "
            "Using class_weights may not be needed or may indicate SMOTE was already applied.",
            fraud_ratio * 100,
        )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))

    logger.info("Class weights computed: %s", class_weights)
    return class_weights


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    batch_size: int = 512,
    epochs: int = 50,
    class_weights: Dict[int, float] | None = None,
    callbacks_list: list[callbacks.Callback] | None = None,
) -> keras.callbacks.History:
    """
    Train the model with given data and configuration.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        class_weights: Class weight dictionary
        callbacks_list: List of Keras callbacks

    Returns:
        Training history object
    """
    logger.info("Starting model training with %d samples", len(X_train))

    fit_kwargs: Dict[str, Any] = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "class_weight": class_weights,
        "verbose": 1,
    }

    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
    if callbacks_list:
        fit_kwargs["callbacks"] = callbacks_list

    history = model.fit(**fit_kwargs)

    logger.info("Training completed after %d epochs", len(history.history["loss"]))
    return history


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | np.ndarray | None,
    config: Dict[str, Any],
    use_validation: bool = True,
) -> Tuple[keras.Model, keras.callbacks.History, Pipeline]:
    """
    Train final production model with full pipeline.

    This is the main training entry point that:
    1. Builds preprocessing pipeline
    2. Applies preprocessing + SMOTE
    3. Builds and trains model
    4. Returns trained artifacts

    Args:
        X_train: Raw training features (DataFrame)
        y_train: Training target
        X_val: Raw validation features (DataFrame)
        y_val: Validation target
        config: Configuration dictionary

    Returns:
        Trained model, training history, fitted preprocessing pipeline
    """
    set_seed(config.get("random_state", 42))

    # Identify feature types
    categorical_features = ["type"] if "type" in X_train.columns else []
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    # Build and fit preprocessing pipeline
    preprocessing_pipeline = build_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        scaler_type=config.get("scaler_type", "standard"),
    )

    logger.info("Fitting preprocessing pipeline on training data")
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_val_processed = None
    if use_validation and X_val is not None:
        X_val_processed = preprocessing_pipeline.transform(X_val)

    # Store original y_train before any modifications (needed for class_weights)
    y_train_original = y_train.values if isinstance(y_train, pd.Series) else y_train.copy()

    if config.get("use_smote", False) and config.get("use_class_weights", False):
        raise ValueError("Use either SMOTE or class weights, not both.")

    # Apply SMOTE if configured
    if config.get("use_smote", False):
        X_train_processed, y_train = apply_smote(
            X_train_processed,
            y_train.values if isinstance(y_train, pd.Series) else y_train,
            sampling_strategy=config.get("smote_sampling_strategy", 0.3),
            random_state=config.get("random_state", 42),
        )

    # Compute class weights from ORIGINAL data (before SMOTE)
    class_weights = None
    if config.get("use_class_weights", False):
        class_weights = compute_class_weights(y_train_original)

    # Build model
    model = build_model(
        input_dim=X_train_processed.shape[1],
        hidden_layers=config.get("hidden_layers", [256, 128, 64, 32]),
        dropout_rate=config.get("dropout_rate", 0.3),
        l2_reg=config.get("l2_reg", 0.001),
        activation=config.get("activation", "relu"),
        learning_rate=config.get("learning_rate", 0.001),
    )

    callbacks_list: list[callbacks.Callback] | None = None
    if use_validation and X_val_processed is not None:
        output_dir = Path(config.get("output_dir", "outputs/models"))
        callbacks_list = create_callbacks(
            output_dir=output_dir,
            patience_early_stop=config.get("patience_early_stop", 15),
            patience_reduce_lr=config.get("patience_reduce_lr", 7),
        )

    # Train model
    history = train_model(
        model=model,
        X_train=X_train_processed,
        y_train=y_train,
        X_val=X_val_processed,
        y_val=y_val,
        batch_size=config.get("batch_size", 512),
        epochs=config.get("epochs", 50),
        class_weights=class_weights,
        callbacks_list=callbacks_list,
    )

    logger.info("Final model training completed successfully")
    return model, history, preprocessing_pipeline


def save_artifacts(
    model: keras.Model,
    preprocessing_pipeline: Pipeline,
    feature_names: list[str],
    config: Dict[str, Any],
    output_dir: str | Path,
) -> None:
    """
    Save all model artifacts for production deployment.

    Args:
        model: Trained Keras model
        preprocessing_pipeline: Fitted preprocessing pipeline
        feature_names: List of feature column names
        config: Configuration used for training
        output_dir: Directory to save artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "fraud_detection_model.keras"
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Save preprocessing pipeline
    pipeline_path = output_dir / "preprocessing_pipeline.pkl"
    joblib.dump(preprocessing_pipeline, pipeline_path)
    logger.info("Preprocessing pipeline saved to %s", pipeline_path)

    # Save feature names
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Feature names saved to %s", features_path)

    # Save configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Training config saved to %s", config_path)


def load_artifacts(model_dir: str | Path) -> Tuple[keras.Model, Pipeline, list[str], Dict]:
    """
    Load all saved artifacts for inference.

    Args:
        model_dir: Directory containing saved artifacts

    Returns:
        model, preprocessing_pipeline, feature_names, config

    Raises:
        FileNotFoundError: If any required artifact is missing
    """
    model_dir = Path(model_dir)

    # Define required artifact paths
    model_path = model_dir / "fraud_detection_model.keras"
    pipeline_path = model_dir / "preprocessing_pipeline.pkl"
    features_path = model_dir / "feature_names.json"
    config_path = model_dir / "training_config.json"

    required_files = [model_path, pipeline_path, features_path, config_path]

    # Check for missing files
    missing = [f.name for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts in {model_dir}: {', '.join(missing)}"
        )

    # Load all artifacts
    model = keras.models.load_model(model_path)
    preprocessing_pipeline = joblib.load(pipeline_path)

    with open(features_path) as f:
        feature_names = json.load(f)

    with open(config_path) as f:
        config = json.load(f)

    logger.info("Artifacts loaded from %s", model_dir)
    return model, preprocessing_pipeline, feature_names, config
