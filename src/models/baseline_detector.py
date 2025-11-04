"""
Baseline Anomaly Detectors for ICS Networks
Includes simple statistical and ML-based methods for sanity checks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class BaselineDetector:
    """
    Base class for baseline anomaly detectors.
    """
    
    def __init__(self, method='zscore', threshold=3.0):
        """
        Initialize baseline detector.
        
        Args:
            method (str): Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold (float): Threshold for anomaly detection
        """
        self.method = method
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.fitted = False
        
        # For statistical methods
        self.mean_ = None
        self.std_ = None
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        
        # For Isolation Forest
        self.model = None
    
    def fit(self, X, y=None):
        """
        Fit the detector on training data.
        
        Args:
            X (np.ndarray): Training data (n_samples, n_features)
            y (np.ndarray, optional): Labels (not used for unsupervised)
        
        Returns:
            self
        """
        if self.method == 'zscore':
            self._fit_zscore(X)
        elif self.method == 'iqr':
            self._fit_iqr(X)
        elif self.method == 'isolation_forest':
            self._fit_isolation_forest(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.fitted = True
        return self
    
    def _fit_zscore(self, X):
        """Fit Z-score based detector."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Handle zero std
        self.std_[self.std_ == 0] = 1.0
    
    def _fit_iqr(self, X):
        """Fit IQR (Interquartile Range) based detector."""
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        # Handle zero IQR
        self.iqr_[self.iqr_ == 0] = 1.0
    
    def _fit_isolation_forest(self, X):
        """Fit Isolation Forest detector."""
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit Isolation Forest
        self.model = IsolationForest(
            contamination=0.1,  # Expected proportion of anomalies
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
    
    def predict(self, X):
        """
        Predict anomalies in data.
        
        Args:
            X (np.ndarray): Data to predict (n_samples, n_features)
        
        Returns:
            np.ndarray: Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        if self.method == 'zscore':
            return self._predict_zscore(X)
        elif self.method == 'iqr':
            return self._predict_iqr(X)
        elif self.method == 'isolation_forest':
            return self._predict_isolation_forest(X)
    
    def _predict_zscore(self, X):
        """Predict using Z-score method."""
        # Calculate Z-scores
        z_scores = np.abs((X - self.mean_) / self.std_)
        
        # Any feature with z-score > threshold is anomaly
        anomalies = np.any(z_scores > self.threshold, axis=1).astype(int)
        
        return anomalies
    
    def _predict_iqr(self, X):
        """Predict using IQR method."""
        # Calculate bounds
        lower_bound = self.q1_ - (self.threshold * self.iqr_)
        upper_bound = self.q3_ + (self.threshold * self.iqr_)
        
        # Check if any feature is outside bounds
        anomalies = np.any((X < lower_bound) | (X > upper_bound), axis=1).astype(int)
        
        return anomalies
    
    def _predict_isolation_forest(self, X):
        """Predict using Isolation Forest."""
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_scaled)
        
        # Convert to 0/1 (0 = normal, 1 = anomaly)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies
    
    def predict_with_scores(self, X):
        """
        Predict anomalies with anomaly scores.
        
        Args:
            X (np.ndarray): Data to predict
        
        Returns:
            tuple: (predictions, scores)
        """
        predictions = self.predict(X)
        
        if self.method == 'zscore':
            # Use max z-score as anomaly score
            z_scores = np.abs((X - self.mean_) / self.std_)
            scores = np.max(z_scores, axis=1)
        
        elif self.method == 'iqr':
            # Calculate distance from bounds
            lower_bound = self.q1_ - (self.threshold * self.iqr_)
            upper_bound = self.q3_ + (self.threshold * self.iqr_)
            
            lower_dist = np.maximum(0, lower_bound - X)
            upper_dist = np.maximum(0, X - upper_bound)
            
            scores = np.max(lower_dist + upper_dist, axis=1)
        
        elif self.method == 'isolation_forest':
            # Use decision function (negative = more anomalous)
            X_scaled = self.scaler.transform(X)
            scores = -self.model.decision_function(X_scaled)
        
        return predictions, scores
    
    def evaluate(self, X, y_true):
        """
        Evaluate detector performance.
        
        Args:
            X (np.ndarray): Test data
            y_true (np.ndarray): True labels
        
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Calculate false positive rate
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics in a formatted way."""
        print(f"\n{'='*60}")
        print(f"Baseline Detector Evaluation ({self.method.upper()})")
        print(f"{'='*60}")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision:          {metrics['precision']:.4f}")
        print(f"Recall:             {metrics['recall']:.4f}")
        print(f"F1-Score:           {metrics['f1_score']:.4f}")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"True Positive Rate:  {metrics['true_positive_rate']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Normal  Attack")
        print(f"Actual Normal   {metrics['confusion_matrix'][0,0]:6d}  {metrics['confusion_matrix'][0,1]:6d}")
        print(f"       Attack   {metrics['confusion_matrix'][1,0]:6d}  {metrics['confusion_matrix'][1,1]:6d}")
        print(f"{'='*60}\n")


class MultiMethodDetector:
    """
    Ensemble of multiple baseline methods.
    """
    
    def __init__(self, methods=['zscore', 'iqr', 'isolation_forest'], voting='majority'):
        """
        Initialize multi-method detector.
        
        Args:
            methods (list): List of methods to use
            voting (str): Voting strategy ('majority', 'unanimous', 'any')
        """
        self.methods = methods
        self.voting = voting
        self.detectors = {method: BaselineDetector(method) for method in methods}
    
    def fit(self, X, y=None):
        """Fit all detectors."""
        for method, detector in self.detectors.items():
            print(f"Training {method} detector...")
            detector.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict using voting."""
        predictions = []
        
        for method, detector in self.detectors.items():
            pred = detector.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.voting == 'majority':
            # Majority vote
            final_pred = (np.sum(predictions, axis=0) > len(self.methods) / 2).astype(int)
        
        elif self.voting == 'unanimous':
            # All must agree
            final_pred = np.all(predictions == 1, axis=0).astype(int)
        
        elif self.voting == 'any':
            # Any detector flags as anomaly
            final_pred = np.any(predictions == 1, axis=0).astype(int)
        
        return final_pred
    
    def evaluate(self, X, y_true):
        """Evaluate ensemble performance."""
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return metrics


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Baseline Detectors...")
    
    # Generate synthetic normal data
    np.random.seed(42)
    X_normal = np.random.randn(1000, 10)
    
    # Generate synthetic anomalies
    X_anomaly = np.random.randn(100, 10) * 3 + 5
    
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(1000), np.ones(100)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test each method
    for method in ['zscore', 'iqr', 'isolation_forest']:
        detector = BaselineDetector(method=method)
        detector.fit(X_train)
        metrics = detector.evaluate(X_test, y_test)
        detector.print_metrics(metrics)
