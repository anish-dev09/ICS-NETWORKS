"""
Multi-Layer Ensemble Detector for Zero-Day Attack Detection
Combines statistical, ML, DL, and protocol validation for robust detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from src.models.baseline_detector import BaselineDetector

try:
    from src.models.deep_anomaly_detector import AutoencoderDetector, LSTMDetector
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("Warning: Deep learning models not available. Install TensorFlow for full functionality.")

try:
    from src.models.protocol_validator import ICSProtocolValidator, SemanticAnalyzer
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False
    print("Warning: Protocol validators not available.")


class ZeroDayEnsembleDetector:
    """
    Multi-layer ensemble for zero-day detection.
    Combines multiple detection methods with weighted voting.
    """
    
    def __init__(self, input_dim, weights=None, enable_deep_learning=True, 
                 enable_protocol_validation=True, sequence_length=10):
        """
        Initialize ensemble detector.
        
        Args:
            input_dim (int): Number of input features
            weights (dict): Weights for each detection layer
            enable_deep_learning (bool): Enable deep learning models
            enable_protocol_validation (bool): Enable protocol validation
            sequence_length (int): Sequence length for LSTM
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.enable_dl = enable_deep_learning and DL_AVAILABLE
        self.enable_protocol = enable_protocol_validation and PROTOCOL_AVAILABLE
        
        # Default weights for each detection layer
        if weights is None:
            if self.enable_dl and self.enable_protocol:
                # All layers enabled
                self.weights = {
                    'statistical': 0.15,
                    'isolation_forest': 0.15,
                    'autoencoder': 0.25,
                    'lstm': 0.25,
                    'protocol': 0.10,
                    'semantic': 0.10
                }
            elif self.enable_dl:
                # Only DL enabled
                self.weights = {
                    'statistical': 0.20,
                    'isolation_forest': 0.20,
                    'autoencoder': 0.30,
                    'lstm': 0.30,
                }
            elif self.enable_protocol:
                # Only protocol validation enabled
                self.weights = {
                    'statistical': 0.30,
                    'isolation_forest': 0.30,
                    'protocol': 0.20,
                    'semantic': 0.20
                }
            else:
                # Only baseline methods
                self.weights = {
                    'statistical': 0.50,
                    'isolation_forest': 0.50,
                }
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Initialize detectors
        print(f"[Ensemble] Initializing detectors...")
        self.statistical_detector = BaselineDetector(method='zscore', threshold=3.0)
        self.isolation_detector = BaselineDetector(method='isolation_forest')
        
        if self.enable_dl:
            self.autoencoder = AutoencoderDetector(input_dim=input_dim, encoding_dim=min(32, input_dim//2))
            self.lstm = LSTMDetector(input_dim=input_dim, sequence_length=sequence_length)
            print(f"[Ensemble] Deep learning models enabled")
        else:
            self.autoencoder = None
            self.lstm = None
            print(f"[Ensemble] Deep learning models disabled")
        
        if self.enable_protocol:
            self.protocol_validator = ICSProtocolValidator()
            self.semantic_analyzer = SemanticAnalyzer()
            print(f"[Ensemble] Protocol validation enabled")
        else:
            self.protocol_validator = None
            self.semantic_analyzer = None
            print(f"[Ensemble] Protocol validation disabled")
        
        self.fitted = False
        
        print(f"[Ensemble] Active layers: {list(self.weights.keys())}")
        print(f"[Ensemble] Layer weights: {self.weights}")
    
    def fit(self, X, X_temporal=None, epochs=50, batch_size=32, verbose=1):
        """
        Train all detectors.
        
        Args:
            X (np.ndarray): Training data (normal traffic only)
            X_temporal (np.ndarray): Temporal data for LSTM (optional)
            epochs (int): Training epochs for deep learning models
            batch_size (int): Batch size
            verbose (int): Verbosity level
        
        Returns:
            self
        """
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE DETECTOR")
        print("="*60)
        
        # Train statistical detector
        print("\n[1/6] Training Statistical (Z-score) Detector...")
        self.statistical_detector.fit(X)
        print("[1/6] ✓ Statistical detector trained")
        
        # Train Isolation Forest
        print("\n[2/6] Training Isolation Forest Detector...")
        self.isolation_detector.fit(X)
        print("[2/6] ✓ Isolation Forest trained")
        
        # Train Autoencoder
        if self.enable_dl and self.autoencoder:
            print("\n[3/6] Training Autoencoder...")
            self.autoencoder.fit(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
            print("[3/6] ✓ Autoencoder trained")
        else:
            print("\n[3/6] Autoencoder skipped (disabled)")
        
        # Train LSTM
        if self.enable_dl and self.lstm:
            print("\n[4/6] Training LSTM...")
            if X_temporal is not None:
                self.lstm.fit(X_temporal, epochs=epochs, batch_size=batch_size, verbose=verbose)
            else:
                self.lstm.fit(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
            print("[4/6] ✓ LSTM trained")
        else:
            print("\n[4/6] LSTM skipped (disabled)")
        
        # Protocol validator (no training needed)
        if self.enable_protocol:
            print("\n[5/6] Protocol Validator: Ready")
            print("[6/6] Semantic Analyzer: Ready")
        else:
            print("\n[5/6] Protocol Validator: Disabled")
            print("[6/6] Semantic Analyzer: Disabled")
        
        self.fitted = True
        
        print("\n" + "="*60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*60)
        
        return self
    
    def predict(self, X, command_data=None, sensor_data=None, return_details=False):
        """
        Predict using ensemble of all detectors.
        
        Args:
            X (np.ndarray): Test data (n_samples, n_features)
            command_data (list): Protocol command data (optional)
            sensor_data (dict): Physical sensor readings (optional)
            return_details (bool): Return detailed results
        
        Returns:
            np.ndarray or dict: Predictions (and details if requested)
        """
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        n_samples = X.shape[0]
        results = {}
        scores = {}
        
        # Statistical detection
        stat_pred = self.statistical_detector.predict(X)
        results['statistical'] = stat_pred
        scores['statistical'] = np.mean(stat_pred)
        
        # Isolation Forest
        iso_pred = self.isolation_detector.predict(X)
        results['isolation_forest'] = iso_pred
        scores['isolation_forest'] = np.mean(iso_pred)
        
        # Autoencoder
        if self.enable_dl and self.autoencoder:
            ae_pred, ae_scores = self.autoencoder.predict_with_scores(X)
            results['autoencoder'] = ae_pred
            scores['autoencoder'] = np.mean(ae_pred)
        
        # LSTM
        if self.enable_dl and self.lstm:
            lstm_pred = self.lstm.predict(X)
            results['lstm'] = lstm_pred
            scores['lstm'] = np.mean(lstm_pred)
        
        # Protocol validation
        if self.enable_protocol and command_data:
            protocol_violations = 0
            for cmd in command_data:
                validation = self.protocol_validator.validate_command(cmd)
                if not validation['is_valid']:
                    protocol_violations += 1
            protocol_score = min(protocol_violations / len(command_data), 1.0) if command_data else 0
            scores['protocol'] = protocol_score
        elif 'protocol' in self.weights:
            scores['protocol'] = 0.0
        
        # Semantic analysis
        if self.enable_protocol and command_data:
            semantic_risks = []
            for cmd in command_data:
                analysis = self.semantic_analyzer.analyze_intent(cmd)
                semantic_risks.append(analysis['risk_score'] / 100.0)
            semantic_score = min(np.mean(semantic_risks), 1.0) if semantic_risks else 0
            scores['semantic'] = semantic_score
        elif 'semantic' in self.weights:
            scores['semantic'] = 0.0
        
        # Weighted ensemble voting
        ensemble_score = sum(scores.get(key, 0) * weight 
                           for key, weight in self.weights.items())
        
        # Per-sample predictions (combine all detector predictions)
        per_sample_scores = np.zeros(n_samples)
        weight_sum = 0
        
        for key, weight in self.weights.items():
            if key in results:
                per_sample_scores += results[key] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            per_sample_scores /= weight_sum
        
        final_predictions = (per_sample_scores > 0.5).astype(int)
        
        if return_details:
            return {
                'predictions': final_predictions,
                'ensemble_score': ensemble_score,
                'per_sample_scores': per_sample_scores,
                'individual_scores': scores,
                'individual_predictions': results,
                'confidence': self._calculate_confidence(scores),
                'active_detectors': list(self.weights.keys())
            }
        
        return final_predictions
    
    def _calculate_confidence(self, scores):
        """Calculate detection confidence based on agreement."""
        if len(scores) == 0:
            return 0.0
        
        # High confidence if most detectors agree
        predictions = [1 if s > 0.5 else 0 for s in scores.values()]
        if len(predictions) == 0:
            return 0.0
        
        agreement = max(sum(predictions), len(predictions) - sum(predictions)) / len(predictions)
        return agreement
    
    def evaluate(self, X, y_true, command_data=None, sensor_data=None):
        """
        Evaluate ensemble performance.
        
        Args:
            X (np.ndarray): Test data
            y_true (np.ndarray): True labels
            command_data (list): Protocol command data (optional)
            sensor_data (dict): Physical sensor readings (optional)
        
        Returns:
            dict: Comprehensive evaluation metrics
        """
        result = self.predict(X, command_data, sensor_data, return_details=True)
        y_pred = result['predictions']
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'ensemble_confidence': result['confidence'],
            'ensemble_score': result['ensemble_score'],
            'individual_scores': result['individual_scores'],
            'detection_rate': np.mean(y_pred),
            'false_positive_rate': np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1),
            'false_negative_rate': np.sum((y_pred == 0) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Individual detector performance
        individual_metrics = {}
        for detector_name, predictions in result['individual_predictions'].items():
            individual_metrics[detector_name] = {
                'accuracy': accuracy_score(y_true, predictions),
                'precision': precision_score(y_true, predictions, zero_division=0),
                'recall': recall_score(y_true, predictions, zero_division=0),
                'f1_score': f1_score(y_true, predictions, zero_division=0)
            }
        
        metrics['individual_detector_metrics'] = individual_metrics
        
        return metrics
    
    def explain_detection(self, result):
        """
        Provide human-readable explanation of detection.
        
        Args:
            result (dict): Detection result from predict with return_details=True
        
        Returns:
            str: Explanation
        """
        explanation = []
        
        predictions = result['predictions']
        anomaly_count = np.sum(predictions)
        total_count = len(predictions)
        
        explanation.append("="*60)
        explanation.append("ENSEMBLE DETECTION REPORT")
        explanation.append("="*60)
        
        if anomaly_count > 0:
            explanation.append(f"\n⚠️  ANOMALIES DETECTED: {anomaly_count}/{total_count} samples ({anomaly_count/total_count*100:.1f}%)")
            explanation.append(f"Overall Confidence: {result['confidence']:.2%}")
            explanation.append(f"Ensemble Score: {result['ensemble_score']:.4f}")
            explanation.append(f"\nActive Detectors: {', '.join(result['active_detectors'])}")
            explanation.append("\nDetection Breakdown:")
            
            for detector, score in result['individual_scores'].items():
                weight = self.weights.get(detector, 0)
                status = "ANOMALY" if score > 0.5 else "NORMAL"
                explanation.append(f"  • {detector.upper():20s}: {score:.4f} (weight: {weight:.2f}) [{status}]")
            
            explanation.append("\n" + "-"*60)
            explanation.append("RECOMMENDATION: Investigate flagged traffic immediately")
            
        else:
            explanation.append(f"\n✓ All traffic appears NORMAL ({total_count} samples)")
            explanation.append(f"Confidence: {result['confidence']:.2%}")
            explanation.append(f"Ensemble Score: {result['ensemble_score']:.4f}")
        
        explanation.append("="*60)
        
        return "\n".join(explanation)
    
    def print_metrics(self, metrics):
        """Print evaluation metrics in readable format."""
        print("\n" + "="*60)
        print("ENSEMBLE EVALUATION METRICS")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"\nEnsemble Metrics:")
        print(f"  Confidence:      {metrics['ensemble_confidence']:.4f}")
        print(f"  Ensemble Score:  {metrics['ensemble_score']:.4f}")
        print(f"  Detection Rate:  {metrics['detection_rate']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        
        print(f"\nIndividual Detector Performance:")
        for detector, det_metrics in metrics.get('individual_detector_metrics', {}).items():
            print(f"\n  {detector.upper()}:")
            print(f"    Accuracy:  {det_metrics['accuracy']:.4f}")
            print(f"    Precision: {det_metrics['precision']:.4f}")
            print(f"    Recall:    {det_metrics['recall']:.4f}")
            print(f"    F1-Score:  {det_metrics['f1_score']:.4f}")
        
        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix:")
            print(f"  [[TN={metrics['confusion_matrix'][0,0]:4d}, FP={metrics['confusion_matrix'][0,1]:4d}]")
            print(f"   [FN={metrics['confusion_matrix'][1,0]:4d}, TP={metrics['confusion_matrix'][1,1]:4d}]]")
        
        print("="*60)


if __name__ == "__main__":
    print("Testing Zero-Day Ensemble Detector...")
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Normal traffic
    X_normal = np.random.randn(1000, 20)
    
    # Anomalous traffic (zero-day attacks)
    X_anomaly = np.random.randn(200, 20) * 3 + 5
    
    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(1000), np.ones(200)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Use only normal data for training (unsupervised learning)
    X_train_normal = X_train[y_train == 0]
    
    print(f"\nDataset:")
    print(f"  Training samples (normal): {len(X_train_normal)}")
    print(f"  Test samples: {len(X_test)} (normal: {np.sum(y_test==0)}, anomaly: {np.sum(y_test==1)})")
    
    # Create and train ensemble
    print("\n" + "="*60)
    print("INITIALIZING ENSEMBLE")
    print("="*60)
    
    ensemble = ZeroDayEnsembleDetector(
        input_dim=20,
        enable_deep_learning=DL_AVAILABLE,
        enable_protocol_validation=False,  # No protocol data in this test
        sequence_length=5
    )
    
    # Train
    ensemble.fit(X_train_normal, epochs=20, batch_size=32, verbose=0)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING ENSEMBLE")
    print("="*60)
    
    metrics = ensemble.evaluate(X_test, y_test)
    ensemble.print_metrics(metrics)
    
    # Get detailed prediction for a few samples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    sample_X = X_test[:10]
    sample_y = y_test[:10]
    
    result = ensemble.predict(sample_X, return_details=True)
    print(ensemble.explain_detection(result))
    
    print(f"\nActual labels: {sample_y}")
    print(f"Predicted:     {result['predictions']}")
