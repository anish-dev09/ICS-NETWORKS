"""
CNN Models for ICS Intrusion Detection

This module implements Convolutional Neural Network architectures
for detecting attacks in ICS sensor data.

Author: Anish
Date: November 6, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


class CNN1DDetector:
    """
    1D Convolutional Neural Network for ICS intrusion detection.
    
    Architecture:
    - Conv1D layers with increasing filters
    - MaxPooling for dimensionality reduction
    - Dense layers for classification
    - Dropout for regularization
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (60, 83),
        filters: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dense_units: List[int] = [128, 64],
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001
    ):
        """
        Initialize 1D-CNN detector.
        
        Args:
            input_shape: (timesteps, sensors)
            filters: List of filters for Conv1D layers
            kernel_size: Kernel size for convolutions
            dense_units: List of units for dense layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the 1D-CNN architecture."""
        model = models.Sequential(name='CNN1D_Detector')
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape, name='input'))
        
        # Convolutional blocks
        for i, n_filters in enumerate(self.filters):
            model.add(layers.Conv1D(
                filters=n_filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.MaxPooling1D(
                pool_size=2,
                name=f'maxpool_{i+1}'
            ))
        
        # Global pooling
        model.add(layers.GlobalMaxPooling1D(name='global_maxpool'))
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            model.add(layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            model.add(layers.Dropout(
                self.dropout_rate if i == 0 else self.dropout_rate * 0.6,
                name=f'dropout_{i+1}'
            ))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            self.build_model()
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        class_weight: Optional[Dict[int, float]] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the CNN model.
        
        Args:
            X_train: Training sequences (n_samples, timesteps, sensors)
            y_train: Training labels (n_samples,)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            class_weight: Dictionary of class weights for imbalance
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Calculate class weights if not provided
        if class_weight is None:
            n_normal = np.sum(y_train == 0)
            n_attack = np.sum(y_train == 1)
            class_weight = {
                0: 1.0,
                1: n_normal / n_attack if n_attack > 0 else 1.0
            }
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\n{'='*60}")
        print("Training 1D-CNN Model")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)} ({np.sum(y_train)} attacks)")
        print(f"Validation samples: {len(X_val)} ({np.sum(y_val)} attacks)")
        print(f"Class weights: {class_weight}")
        print(f"{'='*60}\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Input sequences
            threshold: Decision threshold
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities >= threshold).astype(int).flatten()
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict attack probabilities.
        
        Args:
            X: Input sequences
            
        Returns:
            Probabilities
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        probabilities = self.model.predict(X, verbose=0).flatten()
        return probabilities
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            threshold: Decision threshold
            
        Returns:
            Dictionary with metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        # Get predictions
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        results = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'threshold': threshold,
            'n_samples': len(y_test),
            'n_attacks': int(np.sum(y_test))
        }
        
        return results
    
    def print_metrics(self, results: Dict):
        """Print formatted evaluation metrics."""
        print(f"\n{'='*60}")
        print("CNN Model Evaluation Results")
        print(f"{'='*60}")
        print(f"Threshold: {results['threshold']:.2f}")
        print(f"Samples: {results['n_samples']} ({results['n_attacks']} attacks)")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {results['true_negatives']:5d}  |  FP: {results['false_positives']:5d}")
        print(f"  FN: {results['false_negatives']:5d}  |  TP: {results['true_positives']:5d}")
        print(f"{'='*60}\n")
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model (e.g., 'model.h5' or 'model.keras')
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save(filepath)
        print(f"✓ Model saved to: {filepath}")
        
        # Save history if available
        if self.history is not None:
            history_path = filepath.replace('.h5', '_history.json').replace('.keras', '_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
            print(f"✓ Training history saved to: {history_path}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from: {filepath}")
        
        # Try to load history
        history_path = filepath.replace('.h5', '_history.json').replace('.keras', '_history.json')
        try:
            with open(history_path, 'r') as f:
                history_dict = json.load(f)
                self.history = type('History', (), {'history': history_dict})()
            print(f"✓ Training history loaded from: {history_path}")
        except:
            print(f"⚠ Training history not found")


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights.
    
    Args:
        y: Labels array
        
    Returns:
        Dictionary with class weights
    """
    n_samples = len(y)
    n_normal = np.sum(y == 0)
    n_attack = np.sum(y == 1)
    
    weight_normal = n_samples / (2 * n_normal) if n_normal > 0 else 1.0
    weight_attack = n_samples / (2 * n_attack) if n_attack > 0 else 1.0
    
    return {0: weight_normal, 1: weight_attack}


if __name__ == "__main__":
    print("Testing CNN1DDetector...")
    
    # Create dummy data
    n_samples = 1000
    timesteps = 60
    n_sensors = 83
    
    X_train = np.random.randn(n_samples, timesteps, n_sensors)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    X_val = np.random.randn(200, timesteps, n_sensors)
    y_val = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
    
    print(f"✓ Train shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"✓ Val shape: {X_val.shape}, labels: {y_val.shape}")
    
    # Build model
    cnn = CNN1DDetector(input_shape=(timesteps, n_sensors))
    model = cnn.build_model()
    
    print(f"\n✓ Model built successfully!")
    print(f"  Total params: {model.count_params():,}")
    
    # Quick training test (1 epoch)
    print(f"\n✓ Running quick training test...")
    history = cnn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=2,
        batch_size=32,
        verbose=0
    )
    
    print(f"✓ Training completed!")
    
    # Test evaluation
    results = cnn.evaluate(X_val, y_val)
    cnn.print_metrics(results)
    
    print("✅ CNN1DDetector test passed!")
