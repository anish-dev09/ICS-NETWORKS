"""
Online Learning System for ICS Anomaly Detection
Handles evolving threats through incremental learning and concept drift detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ConceptDriftDetector:
    """
    Detects concept drift in data streams using multiple methods.
    """
    
    def __init__(self, method: str = 'adwin', warning_level: float = 2.0, drift_level: float = 3.0):
        """
        Initialize drift detector.
        
        Args:
            method (str): Detection method ('adwin', 'ddm', 'eddm', 'page_hinkley')
            warning_level (float): Warning threshold
            drift_level (float): Drift threshold
        """
        self.method = method
        self.warning_level = warning_level
        self.drift_level = drift_level
        
        # ADWIN parameters
        self.window = deque(maxlen=1000)
        self.adwin_delta = 0.002
        
        # DDM parameters
        self.ddm_min = np.inf
        self.ddm_std_min = np.inf
        self.error_count = 0
        self.sample_count = 0
        
        # Page-Hinkley parameters
        self.ph_sum = 0
        self.ph_min = 0
        self.ph_lambda = 50
        self.ph_alpha = 0.9999
        
        # Drift status
        self.in_warning = False
        self.in_drift = False
        self.drift_detected_at = None
        
    def add_element(self, error: float) -> Dict[str, bool]:
        """
        Add new element and check for drift.
        
        Args:
            error (float): Prediction error (0 or 1 for classification)
            
        Returns:
            dict: {'warning': bool, 'drift': bool}
        """
        if self.method == 'adwin':
            return self._adwin_detect(error)
        elif self.method == 'ddm':
            return self._ddm_detect(error)
        elif self.method == 'eddm':
            return self._eddm_detect(error)
        elif self.method == 'page_hinkley':
            return self._page_hinkley_detect(error)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _adwin_detect(self, error: float) -> Dict[str, bool]:
        """ADWIN drift detection."""
        self.window.append(error)
        
        if len(self.window) < 2:
            return {'warning': False, 'drift': False}
        
        # Compute change detection
        window_array = np.array(self.window)
        n = len(window_array)
        
        # Split window and compare distributions
        drift_detected = False
        for i in range(1, n):
            w1 = window_array[:i]
            w2 = window_array[i:]
            
            if len(w1) < 5 or len(w2) < 5:
                continue
            
            # Compare means
            mu1, mu2 = np.mean(w1), np.mean(w2)
            var1, var2 = np.var(w1), np.var(w2)
            
            # Hoeffding bound
            m = 1.0 / (1.0/len(w1) + 1.0/len(w2))
            epsilon = np.sqrt(2.0 * np.log(2.0/self.adwin_delta) / m)
            
            if np.abs(mu1 - mu2) > epsilon:
                drift_detected = True
                self.window.clear()
                break
        
        self.in_drift = drift_detected
        if drift_detected:
            self.drift_detected_at = time.time()
        
        return {'warning': False, 'drift': drift_detected}
    
    def _ddm_detect(self, error: float) -> Dict[str, bool]:
        """Drift Detection Method (DDM)."""
        self.sample_count += 1
        self.error_count += error
        
        if self.sample_count < 30:
            return {'warning': False, 'drift': False}
        
        # Error rate and standard deviation
        p = self.error_count / self.sample_count
        s = np.sqrt(p * (1 - p) / self.sample_count)
        
        # Update minimum
        if p + s < self.ddm_min + self.ddm_std_min:
            self.ddm_min = p
            self.ddm_std_min = s
        
        # Check for drift
        warning = p + s >= self.ddm_min + self.warning_level * self.ddm_std_min
        drift = p + s >= self.ddm_min + self.drift_level * self.ddm_std_min
        
        if drift:
            # Reset
            self.ddm_min = np.inf
            self.ddm_std_min = np.inf
            self.error_count = 0
            self.sample_count = 0
            self.drift_detected_at = time.time()
        
        self.in_warning = warning
        self.in_drift = drift
        
        return {'warning': warning, 'drift': drift}
    
    def _eddm_detect(self, error: float) -> Dict[str, bool]:
        """Early Drift Detection Method (EDDM)."""
        # Simplified EDDM based on distance between errors
        self.sample_count += 1
        
        if error == 1:
            self.error_count += 1
        
        if self.sample_count < 30:
            return {'warning': False, 'drift': False}
        
        # Average distance between errors
        if self.error_count > 0:
            avg_distance = self.sample_count / self.error_count
            std_distance = np.sqrt(avg_distance)
        else:
            return {'warning': False, 'drift': False}
        
        # Update minimum
        if avg_distance + 2*std_distance < self.ddm_min + 2*self.ddm_std_min:
            self.ddm_min = avg_distance
            self.ddm_std_min = std_distance
        
        # Check for drift
        warning = avg_distance + 2*std_distance <= (self.ddm_min + 2*self.ddm_std_min) * 0.95
        drift = avg_distance + 2*std_distance <= (self.ddm_min + 2*self.ddm_std_min) * 0.90
        
        if drift:
            self.ddm_min = np.inf
            self.ddm_std_min = np.inf
            self.error_count = 0
            self.sample_count = 0
            self.drift_detected_at = time.time()
        
        self.in_warning = warning
        self.in_drift = drift
        
        return {'warning': warning, 'drift': drift}
    
    def _page_hinkley_detect(self, error: float) -> Dict[str, bool]:
        """Page-Hinkley drift detection."""
        # Update cumulative sum
        self.ph_sum = self.ph_alpha * self.ph_sum + (error - 0.5)
        
        # Update minimum
        if self.ph_sum < self.ph_min:
            self.ph_min = self.ph_sum
        
        # Check for drift
        drift = (self.ph_sum - self.ph_min) > self.ph_lambda
        
        if drift:
            self.ph_sum = 0
            self.ph_min = 0
            self.drift_detected_at = time.time()
        
        self.in_drift = drift
        
        return {'warning': False, 'drift': drift}
    
    def reset(self):
        """Reset detector state."""
        self.window.clear()
        self.ddm_min = np.inf
        self.ddm_std_min = np.inf
        self.error_count = 0
        self.sample_count = 0
        self.ph_sum = 0
        self.ph_min = 0
        self.in_warning = False
        self.in_drift = False


class IncrementalLearner:
    """
    Incremental learning for online model updates.
    """
    
    def __init__(self, model_type: str = 'sgd'):
        """
        Initialize incremental learner.
        
        Args:
            model_type (str): Model type ('sgd', 'passive_aggressive', 'perceptron')
        """
        self.model_type = model_type
        self.model = None
        self.n_updates = 0
        self.performance_history = []
        
    def initialize_model(self, n_features: int):
        """
        Initialize incremental model.
        
        Args:
            n_features (int): Number of features
        """
        from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
        
        if self.model_type == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',
                learning_rate='optimal',
                random_state=42
            )
        elif self.model_type == 'passive_aggressive':
            self.model = PassiveAggressiveClassifier(
                C=1.0,
                random_state=42
            )
        elif self.model_type == 'perceptron':
            self.model = Perceptron(
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Initialize with dummy data
        X_dummy = np.zeros((2, n_features))
        y_dummy = np.array([0, 1])
        self.model.fit(X_dummy, y_dummy)
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """
        Incremental update.
        
        Args:
            X (np.ndarray): New data
            y (np.ndarray): New labels
            classes (np.ndarray): Class labels
        """
        if self.model is None:
            self.initialize_model(X.shape[1])
        
        if classes is None:
            classes = np.array([0, 1])
        
        self.model.partial_fit(X, y, classes=classes)
        self.n_updates += 1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with current model.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate current model.
        
        Args:
            X (np.ndarray): Test data
            y (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        score = self.model.score(X, y)
        self.performance_history.append(score)
        return score


class ModelEnsemble:
    """
    Ensemble of models with different ages for robustness.
    """
    
    def __init__(self, max_models: int = 5):
        """
        Initialize model ensemble.
        
        Args:
            max_models (int): Maximum number of models to maintain
        """
        self.max_models = max_models
        self.models = deque(maxlen=max_models)
        self.model_ages = deque(maxlen=max_models)
        self.model_weights = deque(maxlen=max_models)
        
    def add_model(self, model, weight: float = 1.0):
        """
        Add new model to ensemble.
        
        Args:
            model: Trained model
            weight (float): Model weight
        """
        self.models.append(model)
        self.model_ages.append(time.time())
        self.model_weights.append(weight)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction (weighted voting).
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted voting
        weights = np.array(self.model_weights)
        weights = weights / weights.sum()
        
        ensemble_pred = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            ensemble_pred += predictions[i] * weights[i]
        
        return (ensemble_pred > 0.5).astype(int)
    
    def update_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Update model weights based on performance.
        
        Args:
            X (np.ndarray): Validation data
            y (np.ndarray): True labels
        """
        from sklearn.metrics import accuracy_score
        
        new_weights = []
        for model in self.models:
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            new_weights.append(acc)
        
        self.model_weights = deque(new_weights, maxlen=self.max_models)


class OnlineLearningSystem:
    """
    Integrated online learning system with drift detection and incremental updates.
    """
    
    def __init__(self, 
                 drift_method: str = 'adwin',
                 model_type: str = 'sgd',
                 use_ensemble: bool = True,
                 retrain_window: int = 1000):
        """
        Initialize online learning system.
        
        Args:
            drift_method (str): Drift detection method
            model_type (str): Incremental model type
            use_ensemble (bool): Use model ensemble
            retrain_window (int): Window size for retraining
        """
        self.drift_detector = ConceptDriftDetector(method=drift_method)
        self.learner = IncrementalLearner(model_type=model_type)
        self.ensemble = ModelEnsemble() if use_ensemble else None
        self.retrain_window = retrain_window
        
        # Data buffer for retraining
        self.buffer = deque(maxlen=retrain_window)
        self.buffer_labels = deque(maxlen=retrain_window)
        
        # Statistics
        self.drift_count = 0
        self.update_count = 0
        self.performance_over_time = []
        
    def process_batch(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Process batch of data with online learning.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): True labels
            
        Returns:
            dict: Processing results
        """
        results = {
            'drift_detected': False,
            'model_updated': False,
            'predictions': None,
            'errors': None
        }
        
        # Get predictions
        try:
            predictions = self.learner.predict(X)
            results['predictions'] = predictions
        except ValueError:
            # Model not initialized yet
            self.learner.initialize_model(X.shape[1])
            predictions = self.learner.predict(X)
            results['predictions'] = predictions
        
        # Compute errors
        errors = (predictions != y).astype(float)
        results['errors'] = errors
        
        # Check for drift
        for error in errors:
            drift_status = self.drift_detector.add_element(error)
            
            if drift_status['drift']:
                self.drift_count += 1
                results['drift_detected'] = True
                
                # Retrain on buffer
                if len(self.buffer) > 0:
                    X_buffer = np.array(self.buffer)
                    y_buffer = np.array(self.buffer_labels)
                    
                    # Add current model to ensemble before retraining
                    if self.ensemble:
                        self.ensemble.add_model(self.learner.model)
                    
                    # Retrain
                    self.learner.initialize_model(X.shape[1])
                    self.learner.partial_fit(X_buffer, y_buffer)
                    
                    results['model_updated'] = True
                    self.update_count += 1
        
        # Incremental update (if no drift or after drift)
        self.learner.partial_fit(X, y)
        
        # Update buffer
        for i in range(len(X)):
            self.buffer.append(X[i])
            self.buffer_labels.append(y[i])
        
        # Track performance
        score = self.learner.score(X, y)
        self.performance_over_time.append(score)
        results['current_score'] = score
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            dict: System statistics
        """
        return {
            'drift_count': self.drift_count,
            'update_count': self.update_count,
            'performance_history': self.performance_over_time,
            'buffer_size': len(self.buffer),
            'avg_recent_performance': np.mean(self.performance_over_time[-100:]) if len(self.performance_over_time) > 0 else 0
        }


if __name__ == "__main__":
    print("Testing Online Learning System...")
    
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("Test 1: Concept Drift Detection (ADWIN)")
    print("="*60)
    
    detector = ConceptDriftDetector(method='adwin')
    
    # Simulate data stream with concept drift
    n_samples = 500
    
    # First concept (low error rate)
    errors1 = np.random.binomial(1, 0.1, n_samples//2)
    
    # Second concept (high error rate - drift!)
    errors2 = np.random.binomial(1, 0.4, n_samples//2)
    
    errors = np.concatenate([errors1, errors2])
    
    drift_points = []
    for i, error in enumerate(errors):
        status = detector.add_element(error)
        if status['drift']:
            drift_points.append(i)
            print(f"Drift detected at sample {i}")
    
    print(f"\nTotal drifts detected: {len(drift_points)}")
    print(f"Expected drift around sample {n_samples//2}")
    
    print("\n" + "="*60)
    print("Test 2: DDM Drift Detection")
    print("="*60)
    
    detector_ddm = ConceptDriftDetector(method='ddm', warning_level=2.0, drift_level=3.0)
    
    drift_points_ddm = []
    warning_points = []
    
    for i, error in enumerate(errors):
        status = detector_ddm.add_element(error)
        if status['warning']:
            warning_points.append(i)
        if status['drift']:
            drift_points_ddm.append(i)
            print(f"DDM drift detected at sample {i}")
    
    print(f"\nWarnings: {len(warning_points)}")
    print(f"Drifts: {len(drift_points_ddm)}")
    
    print("\n" + "="*60)
    print("Test 3: Incremental Learning")
    print("="*60)
    
    learner = IncrementalLearner(model_type='sgd')
    
    # Generate training data
    n_features = 10
    X1 = np.random.randn(100, n_features) * 0.5 + 1
    y1 = np.zeros(100)
    
    X2 = np.random.randn(100, n_features) * 0.5 + 3
    y2 = np.ones(100)
    
    X_train = np.vstack([X1, X2])
    y_train = np.hstack([y1, y2])
    
    # Incremental training
    batch_size = 20
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        learner.partial_fit(X_batch, y_batch)
    
    print(f"\nIncremental updates: {learner.n_updates}")
    
    # Test
    X_test = np.vstack([
        np.random.randn(50, n_features) * 0.5 + 1,
        np.random.randn(50, n_features) * 0.5 + 3
    ])
    y_test = np.hstack([np.zeros(50), np.ones(50)])
    
    score = learner.score(X_test, y_test)
    print(f"Test accuracy: {score:.4f}")
    
    print("\n" + "="*60)
    print("Test 4: Model Ensemble")
    print("="*60)
    
    ensemble = ModelEnsemble(max_models=3)
    
    # Create multiple models
    for i in range(3):
        model = IncrementalLearner(model_type='sgd')
        model.initialize_model(n_features)
        
        # Train on slightly different data
        noise = np.random.randn(len(X_train), n_features) * 0.1
        model.partial_fit(X_train + noise, y_train)
        
        ensemble.add_model(model.model, weight=1.0)
    
    print(f"\nEnsemble size: {len(ensemble.models)}")
    
    # Ensemble prediction
    pred = ensemble.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, pred)
    print(f"Ensemble accuracy: {acc:.4f}")
    
    # Update weights
    ensemble.update_weights(X_test, y_test)
    print(f"Updated weights: {list(ensemble.model_weights)}")
    
    print("\n" + "="*60)
    print("Test 5: Integrated Online Learning System")
    print("="*60)
    
    system = OnlineLearningSystem(
        drift_method='adwin',
        model_type='sgd',
        use_ensemble=True,
        retrain_window=200
    )
    
    # Simulate data stream with drift
    n_batches = 20
    batch_size = 50
    
    print("\nProcessing data stream...")
    
    for batch_idx in range(n_batches):
        # Introduce concept drift halfway
        if batch_idx < n_batches // 2:
            X_batch = np.random.randn(batch_size, n_features) * 0.5 + 1
            y_batch = np.zeros(batch_size)
        else:
            X_batch = np.random.randn(batch_size, n_features) * 1.0 + 2
            y_batch = np.ones(batch_size)
        
        # Add some noise
        noise_mask = np.random.rand(batch_size) < 0.1
        y_batch[noise_mask] = 1 - y_batch[noise_mask]
        
        # Process batch
        result = system.process_batch(X_batch, y_batch)
        
        if result['drift_detected']:
            print(f"Batch {batch_idx}: DRIFT DETECTED, model updated")
        
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: Score = {result['current_score']:.4f}")
    
    # Get final statistics
    stats = system.get_statistics()
    
    print(f"\n✓ Online learning completed:")
    print(f"  Total drifts detected: {stats['drift_count']}")
    print(f"  Model updates: {stats['update_count']}")
    print(f"  Average recent performance: {stats['avg_recent_performance']:.4f}")
    print(f"  Buffer size: {stats['buffer_size']}")
    
    print("\n✓ Online learning system working!")
