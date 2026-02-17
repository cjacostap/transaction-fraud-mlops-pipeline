from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow import keras

logger = logging.getLogger(__name__)


def predict(
    raw_features: pd.DataFrame,
    model: keras.Model,
    preprocessing_pipeline: Pipeline,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Make predictions on new transaction data.

    This function is designed to be used in production (API/batch inference).

    Args:
        raw_features: Raw feature DataFrame (after data.py processing)
        model: Trained Keras model
        preprocessing_pipeline: Fitted preprocessing pipeline
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary with predictions and probabilities
    """
    # Preprocess features
    X_processed = preprocessing_pipeline.transform(raw_features)

    # Get predictions
    probabilities = model.predict(X_processed, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)

    # Compute risk levels
    risk_levels = np.where(
        probabilities > 0.7, "HIGH", np.where(probabilities > 0.3, "MEDIUM", "LOW")
    )

    results = {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "risk_levels": risk_levels.tolist(),
        "threshold_used": threshold,
        "n_samples": len(raw_features),
    }

    logger.info("Predictions generated for %d samples", len(raw_features))
    return results


def predict_single(
    transaction: Dict[str, Any],
    model: keras.Model,
    preprocessing_pipeline: Pipeline,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Make prediction on a single transaction (useful for API endpoints).

    Args:
        transaction: Dictionary with transaction features
        model: Trained Keras model
        preprocessing_pipeline: Fitted preprocessing pipeline
        threshold: Decision threshold

    Returns:
        Dictionary with prediction result
    """
    # Convert single transaction to DataFrame
    df = pd.DataFrame([transaction])
    
    # Use batch predict function
    result = predict(df, model, preprocessing_pipeline, threshold)
    
    # Extract single result
    return {
        "is_fraud": bool(result["predictions"][0]),
        "probability": float(result["probabilities"][0]),
        "risk_level": result["risk_levels"][0],
        "threshold_used": threshold,
    }


def load_model_for_inference(model_dir: str | Path) -> tuple[keras.Model, Pipeline]:
    """
    Load model and preprocessing pipeline for inference.

    This is a lightweight version that only loads what's needed for prediction.

    Args:
        model_dir: Directory containing saved artifacts

    Returns:
        (model, preprocessing_pipeline)
    """
    from .model import load_artifacts

    model, preprocessing_pipeline, _, _ = load_artifacts(model_dir)
    logger.info("Model loaded for inference from %s", model_dir)
    return model, preprocessing_pipeline
