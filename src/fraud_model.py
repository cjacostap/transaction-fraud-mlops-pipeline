"""
Neural Network model for Online Payment Fraud Detection.
Implements a Deep Neural Network (DNN) using Keras/TensorFlow.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from pathlib import Path


class FraudDNNModel:
    """
    Deep Neural Network model for fraud detection.
    """
    
    def __init__(self, input_dim: int, 
                 hidden_layers: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize the DNN model.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons in each hidden layer (default [128, 64, 32])
            dropout_rate: Dropout rate for regularization (default 0.2)
            learning_rate: Learning rate for Adam optimizer (default 0.001)
            random_state: Random state for reproducibility
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        try:
            import tensorflow as tf
            tf.random.set_seed(random_state)
        except ImportError:
            pass
    
    def build_model(self):
        """
        Build the DNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer (first hidden layer)
        model.add(Dense(self.hidden_layers[0], 
                       input_dim=self.input_dim, 
                       activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Additional hidden layers
        for neurons in self.hidden_layers[1:]:
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer (binary classification)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 10,
              batch_size: int = 32,
              verbose: int = 1,
              early_stopping_patience: int = 3,
              save_best_model: bool = False,
              model_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the DNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs (default 10)
            batch_size: Batch size (default 32)
            verbose: Verbosity level (default 1)
            early_stopping_patience: Patience for early stopping (default 3)
            save_best_model: Whether to save the best model (default False)
            model_save_path: Path to save the best model (optional)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        print("\n" + "=" * 70)
        print("FRAUD DETECTION - MODEL TRAINING")
        print("=" * 70)
        print(f"Model Architecture:")
        self.model.summary()
        print("\n" + "=" * 70)
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Model checkpoint
        if save_best_model and model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                model_save_path,
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train the model
        print(f"\nTraining model...")
        print(f"  Training samples: {len(X_train):,}")
        if validation_data:
            print(f"  Validation samples: {len(X_val):,}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)
        
        return self.history.history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded. Call build_model() or load_model() first.")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded. Call build_model() or load_model() first.")
        
        probabilities = self.model.predict(X, verbose=0)
        return probabilities.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input features
            y: True labels
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        
        # Get predictions
        y_pred = self.predict(X, threshold=threshold)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        from keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from: {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            String summary of the model
        """
        if self.model is None:
            return "Model has not been built yet."
        
        from io import StringIO
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()
        self.model.summary()
        summary = summary_buffer.getvalue()
        sys.stdout = old_stdout
        
        return summary