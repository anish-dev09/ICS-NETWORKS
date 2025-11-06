"""
Machine Learning Models for ICS Intrusion Detection

This module implements ML-based anomaly detection models including:
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
- Model ensemble with voting

Author: Anish
Date: November 6, 2025
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")


class MLDetector:
    """
    Machine Learning based intrusion detector.
    
    Supports multiple ML algorithms:
    - Random Forest
    - XGBoost
    - Ensemble methods
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize ML detector.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'svm')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model: Optional[Union[RandomForestClassifier, Any]] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.training_time: Optional[float] = None
        self.params = kwargs
        
        # Initialize the model
        self._init_model()
        
    def _init_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.params.get('n_estimators', 200),
                max_depth=self.params.get('max_depth', 20),
                min_samples_split=self.params.get('min_samples_split', 5),
                min_samples_leaf=self.params.get('min_samples_leaf', 2),
                random_state=self.params.get('random_state', 42),
                n_jobs=self.params.get('n_jobs', -1),
                class_weight=self.params.get('class_weight', 'balanced')
            )
            print(f"✅ Random Forest initialized with {self.params.get('n_estimators', 200)} trees")
            
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE or xgb is None:
                raise ImportError("XGBoost is not installed!")
            
            self.model = xgb.XGBClassifier(  # type: ignore
                n_estimators=self.params.get('n_estimators', 200),
                learning_rate=self.params.get('learning_rate', 0.1),
                max_depth=self.params.get('max_depth', 10),
                subsample=self.params.get('subsample', 0.8),
                colsample_bytree=self.params.get('colsample_bytree', 0.8),
                random_state=self.params.get('random_state', 42),
                n_jobs=self.params.get('n_jobs', -1),
                scale_pos_weight=self.params.get('scale_pos_weight', 10)
            )
            print(f"✅ XGBoost initialized with {self.params.get('n_estimators', 200)} estimators")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        print(f"\n🔧 Training {self.model_type.upper()} model...")
        print(f"   Training samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Attack samples: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        
        start_time = time.time()
        
        # Train the model
        if self.model is not None:
            self.model.fit(X, y)  # type: ignore
        
        self.training_time = time.time() - start_time
        
        # Extract feature importance
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_  # type: ignore
            }).sort_values('importance', ascending=False)
        
        print(f"✅ Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (0=normal, 1=attack)
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities [prob_normal, prob_attack]
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Evaluating {self.model_type.upper()} model...")
        
        start_time = time.time()
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        predict_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            'train_time': self.training_time,
            'predict_time': predict_time
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print evaluation metrics in a nice format."""
        print(f"\n{'='*60}")
        print(f"{self.model_type.upper()} PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f} ({metrics['fpr']*100:.2f}%)")
        print(f"  TPR:       {metrics['tpr']:.4f} ({metrics['tpr']*100:.2f}%)")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {metrics['true_negative']:6d}  FP: {metrics['false_positive']:6d}")
        print(f"    FN: {metrics['false_negative']:6d}  TP: {metrics['true_positive']:6d}")
        print(f"\n  Training Time:   {metrics['train_time']:.2f}s")
        print(f"  Prediction Time: {metrics['predict_time']:.2f}s")
        print(f"{'='*60}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available!")
        
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_time': self.training_time,
            'params': self.params
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            MLDetector instance
        """
        data = joblib.load(filepath)
        detector = cls(model_type=data['model_type'], **data['params'])
        detector.model = data['model']
        detector.feature_importance = data['feature_importance']
        detector.training_time = data['training_time']
        print(f"✅ Model loaded from {filepath}")
        return detector


class EnsembleDetector:
    """
    Ensemble detector combining multiple ML models.
    
    Uses weighted voting to combine predictions from multiple models.
    """
    
    def __init__(self, models: List[MLDetector], weights: Optional[List[float]] = None):
        """
        Initialize ensemble detector.
        
        Args:
            models: List of trained ML detectors
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models!")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"✅ Ensemble detector initialized with {len(models)} models")
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            print(f"   Model {i+1}: {model.model_type} (weight: {weight:.2f})")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted average probabilities
        """
        # Get predictions from all models
        all_proba = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_proba.append(proba)
        
        # Weighted average
        ensemble_proba = np.zeros_like(all_proba[0])
        for proba, weight in zip(all_proba, self.weights):
            ensemble_proba += proba * weight
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict labels using weighted ensemble.
        
        Args:
            X: Feature matrix
            threshold: Decision threshold
            
        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Evaluating ENSEMBLE model...")
        
        start_time = time.time()
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        predict_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            'predict_time': predict_time
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics


def train_and_compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             models_config: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """
    Train and compare multiple ML models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models_config: Optional configuration for each model
        
    Returns:
        Dictionary containing trained models and results
    """
    if models_config is None:
        models_config = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 10,
                'random_state': 42
            }
        }
    
    results = {
        'models': {},
        'metrics': {},
        'comparison': None
    }
    
    print("\n" + "="*80)
    print("TRAINING AND COMPARING ML MODELS")
    print("="*80)
    
    # Train each model
    for model_name, config in models_config.items():
        if model_name == 'xgboost' and not XGBOOST_AVAILABLE:
            print(f"\n⚠️  Skipping {model_name} (not installed)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*80}")
        
        # Initialize and train
        detector = MLDetector(model_type=model_name, **config)
        detector.fit(X_train, y_train)
        
        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        detector.print_metrics(metrics)
        
        # Store results
        results['models'][model_name] = detector
        results['metrics'][model_name] = metrics
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results['metrics'].items():
        comparison_data.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'fpr': metrics['fpr'],
            'train_time': metrics['train_time'],
            'predict_time': metrics['predict_time']
        })
    
    results['comparison'] = pd.DataFrame(comparison_data)
    
    return results


if __name__ == "__main__":
    print("Machine Learning Models for ICS Intrusion Detection")
    print("=" * 60)
    print("\nThis module provides:")
    print("  1. Random Forest Classifier")
    print("  2. XGBoost Classifier")
    print("  3. Model ensemble with voting")
    print("  4. Comprehensive evaluation metrics")
    print("\nUsage:")
    print("  from src.models.ml_models import MLDetector, train_and_compare_models")
    print("  detector = MLDetector(model_type='random_forest')")
    print("  detector.fit(X_train, y_train)")
    print("  metrics = detector.evaluate(X_test, y_test)")
