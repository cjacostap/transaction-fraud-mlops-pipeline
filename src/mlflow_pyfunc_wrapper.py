"""
Custom MLflow PyFunc wrapper for fraud detection model.

This wrapper encapsulates both the Keras model and the preprocessing pipeline,
allowing them to be served together as a single deployable unit.
"""

from __future__ import annotations

import joblib
import mlflow
from tensorflow import keras


class FraudModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc wrapper that combines preprocessing and prediction.
    
    This ensures that raw features are always preprocessed correctly before
    being passed to the model, even in production serving environments.
    """

    def load_context(self, context):
        """
        Load model and preprocessing pipeline from artifacts.
        
        Args:
            context: MLflow context with artifacts dictionary
        """
        # Load Keras model
        self.model = keras.models.load_model(context.artifacts["model"])
        
        # Load preprocessing pipeline
        self.pipeline = joblib.load(context.artifacts["pipeline"])

    def predict(self, context, model_input):
        """
        Preprocess input and make predictions.
        
        Args:
            context: MLflow context (unused but required by interface)
            model_input: Raw features (pandas DataFrame or numpy array)
            
        Returns:
            Predicted probabilities (numpy array)
        """
        # Apply preprocessing pipeline
        X_processed = self.pipeline.transform(model_input)
        
        # Get predictions
        predictions = self.model.predict(X_processed, verbose=0)
        
        return predictions
