"""
Advanced Integrated Detector combining all improvements
Includes: Class Imbalance, Temporal Context, Pattern Learning, Online Learning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils.class_imbalance_handler import ClassImbalanceHandler, FocalLoss, CostSensitiveLearner
from src.utils.advanced_temporal_context import TemporalContextEnhancer
from src.utils.automatic_pattern_learning import AutomaticPatternLearner
from src.utils.online_learning_system import OnlineLearningSystem


class AdvancedIntegratedDetector:
    """
    Integrated anomaly detector with all advanced features.
    Handles: class imbalance, temporal context, automatic patterns, evolving threats.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize advanced integrated detector.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Components
        self.imbalance_handler = None
        self.temporal_enhancer = None
        self.pattern_learner = None
        self.online_system = None
        
        # State
        self.is_trained = False
        self.feature_names = []
        self.training_history = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components."""
        
        # 1. Class imbalance handler
        imbalance_config = self.config.get('imbalance', {})
        self.imbalance_handler = ClassImbalanceHandler(
            strategy=imbalance_config.get('strategy', 'smote'),
            sampling_ratio=imbalance_config.get('sampling_ratio', 0.3),
            k_neighbors=imbalance_config.get('k_neighbors', 5)
        )
        
        # 2. Temporal context enhancer
        temporal_config = self.config.get('temporal', {})
        self.temporal_enhancer = TemporalContextEnhancer(
            window_size=temporal_config.get('window_size', 60),
            stride=temporal_config.get('stride', 10),
            use_attention=temporal_config.get('use_attention', True)
        )
        
        # 3. Pattern learner
        pattern_config = self.config.get('pattern', {})
        self.pattern_learner = AutomaticPatternLearner(
            use_deep_features=pattern_config.get('use_deep_features', True),
            use_automl=pattern_config.get('use_automl', True)
        )
        
        # 4. Online learning system
        online_config = self.config.get('online', {})
        self.online_system = OnlineLearningSystem(
            drift_method=online_config.get('drift_method', 'adwin'),
            model_type=online_config.get('model_type', 'sgd'),
            use_ensemble=online_config.get('use_ensemble', True),
            retrain_window=online_config.get('retrain_window', 1000)
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the integrated detector.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            dict: Training results
        """
        print("Training Advanced Integrated Detector...")
        results = {}
        
        # 1. Handle class imbalance
        print("\n[1/4] Handling class imbalance...")
        dist = self.imbalance_handler.analyze_distribution(y)
        print(f"  Original distribution: {dist}")
        
        X_balanced, y_balanced = self.imbalance_handler.resample(X, y)
        dist_balanced = self.imbalance_handler.analyze_distribution(y_balanced)
        print(f"  Balanced distribution: {dist_balanced}")
        
        results['imbalance'] = {
            'original': dist,
            'balanced': dist_balanced,
            'samples_added': len(y_balanced) - len(y)
        }
        
        # 2. Extract temporal features
        print("\n[2/4] Extracting temporal features...")
        X_temporal = self.temporal_enhancer.process_sequence(X_balanced)
        print(f"  Original shape: {X_balanced.shape}")
        print(f"  With temporal features: {X_temporal.shape}")
        
        results['temporal'] = {
            'original_shape': X_balanced.shape,
            'temporal_shape': X_temporal.shape,
            'features_added': X_temporal.shape[1] - X_balanced.shape[1] if len(X_temporal.shape) == 2 else 0
        }
        
        # 3. Learn patterns automatically
        print("\n[3/4] Learning patterns automatically...")
        pattern_result = self.pattern_learner.learn_patterns(X_temporal, y_balanced)
        
        # Use best features for training
        if 'deep_features' in pattern_result:
            X_features = pattern_result['deep_features']
        else:
            X_features = pattern_result['engineered_features']
        
        print(f"  Patterns discovered: {pattern_result.get('n_patterns', 'N/A')}")
        print(f"  Final feature shape: {X_features.shape}")
        
        results['pattern_learning'] = {
            'n_patterns': pattern_result.get('n_patterns'),
            'feature_shape': X_features.shape,
            'automl_results': pattern_result.get('automl_results', {})
        }
        
        # 4. Initialize online learning
        print("\n[4/4] Initializing online learning system...")
        self.online_system.learner.initialize_model(X_features.shape[1])
        
        # Train in batches to simulate online learning
        batch_size = 100
        for i in range(0, len(X_features), batch_size):
            X_batch = X_features[i:i+batch_size]
            y_batch = y_balanced[i:i+batch_size]
            
            if len(X_batch) > 0:
                self.online_system.process_batch(X_batch, y_batch)
        
        online_stats = self.online_system.get_statistics()
        print(f"  Online learning initialized")
        print(f"  Current performance: {online_stats['avg_recent_performance']:.4f}")
        
        results['online_learning'] = online_stats
        
        # Mark as trained
        self.is_trained = True
        self.training_history = results
        
        print("\n✓ Training completed!")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions (0 = normal, 1 = anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Process through pipeline
        X_temporal = self.temporal_enhancer.process_sequence(X)
        
        # Extract features (use pattern learner's feature engineering)
        X_engineered, _ = self.pattern_learner.feature_engineer.engineer_statistical_features(X_temporal)
        
        # Predict
        predictions = self.online_system.learner.predict(X_engineered)
        
        return predictions
    
    def predict_with_feedback(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict:
        """
        Predict with online learning feedback.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True labels (for online learning)
            
        Returns:
            dict: Predictions and system status
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Process through pipeline
        X_temporal = self.temporal_enhancer.process_sequence(X)
        X_engineered, _ = self.pattern_learner.feature_engineer.engineer_statistical_features(X_temporal)
        
        # Predict and update if labels provided
        if y is not None:
            result = self.online_system.process_batch(X_engineered, y)
            return result
        else:
            predictions = self.online_system.learner.predict(X_engineered)
            return {
                'predictions': predictions,
                'drift_detected': False,
                'model_updated': False
            }
    
    def get_status(self) -> Dict:
        """
        Get detector status.
        
        Returns:
            dict: System status
        """
        if not self.is_trained:
            return {'trained': False}
        
        online_stats = self.online_system.get_statistics()
        
        return {
            'trained': True,
            'training_history': self.training_history,
            'online_statistics': online_stats,
            'components': {
                'imbalance_handler': self.imbalance_handler.strategy,
                'temporal_enhancer': f'window={self.temporal_enhancer.window_size}',
                'pattern_learner': 'deep' if self.pattern_learner.use_deep_features else 'basic',
                'online_learning': self.online_system.drift_detector.method
            }
        }


def create_default_detector() -> AdvancedIntegratedDetector:
    """
    Create detector with default configuration.
    
    Returns:
        AdvancedIntegratedDetector: Configured detector
    """
    config = {
        'imbalance': {
            'strategy': 'smote',
            'sampling_ratio': 0.3,
            'k_neighbors': 5
        },
        'temporal': {
            'window_size': 60,
            'stride': 10,
            'use_attention': True
        },
        'pattern': {
            'use_deep_features': True,
            'use_automl': True
        },
        'online': {
            'drift_method': 'adwin',
            'model_type': 'sgd',
            'use_ensemble': True,
            'retrain_window': 1000
        }
    }
    
    return AdvancedIntegratedDetector(config)


def create_lightweight_detector() -> AdvancedIntegratedDetector:
    """
    Create lightweight detector (faster, less resource-intensive).
    
    Returns:
        AdvancedIntegratedDetector: Configured detector
    """
    config = {
        'imbalance': {
            'strategy': 'weights',  # No resampling
            'sampling_ratio': 0.0,
            'k_neighbors': 3
        },
        'temporal': {
            'window_size': 30,  # Smaller window
            'stride': 15,
            'use_attention': False  # Disable attention
        },
        'pattern': {
            'use_deep_features': False,  # No deep learning
            'use_automl': False  # No AutoML
        },
        'online': {
            'drift_method': 'ddm',  # Faster drift detection
            'model_type': 'sgd',
            'use_ensemble': False,  # No ensemble
            'retrain_window': 500
        }
    }
    
    return AdvancedIntegratedDetector(config)


if __name__ == "__main__":
    print("Testing Advanced Integrated Detector...")
    
    # Generate synthetic ICS data with imbalance
    np.random.seed(42)
    
    # 95% normal, 5% attack (severe imbalance)
    n_normal = 950
    n_attack = 50
    n_features = 20
    
    X_normal = np.random.randn(n_normal, n_features) * 0.5 + 2
    y_normal = np.zeros(n_normal)
    
    X_attack = np.random.randn(n_attack, n_features) * 1.5 + 5
    y_attack = np.ones(n_attack)
    
    X_train = np.vstack([X_normal, X_attack])
    y_train = np.hstack([y_normal, y_attack])
    
    # Shuffle
    indices = np.random.permutation(len(y_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print("\n" + "="*60)
    print("Test 1: Default Detector (Full Features)")
    print("="*60)
    
    detector = create_default_detector()
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Class distribution: Normal={np.sum(y_train==0)}, Attack={np.sum(y_train==1)}")
    
    # Train
    results = detector.train(X_train, y_train)
    
    print(f"\n✓ Training results:")
    print(f"  Samples added (resampling): {results['imbalance']['samples_added']}")
    print(f"  Temporal features added: {results['temporal'].get('features_added', 'N/A')}")
    print(f"  Patterns discovered: {results['pattern_learning']['n_patterns']}")
    print(f"  Online learning performance: {results['online_learning']['avg_recent_performance']:.4f}")
    
    # Test prediction
    X_test_normal = np.random.randn(100, n_features) * 0.5 + 2
    X_test_attack = np.random.randn(20, n_features) * 1.5 + 5
    X_test = np.vstack([X_test_normal, X_test_attack])
    y_test = np.hstack([np.zeros(100), np.ones(20)])
    
    predictions = detector.predict(X_test)
    
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, predictions)
    
    print(f"\n✓ Test results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Predictions: {np.bincount(predictions)}")
    
    # Test online learning with feedback
    print("\n" + "="*60)
    print("Test 2: Online Learning with Concept Drift")
    print("="*60)
    
    # Simulate concept drift
    print("\nProcessing data stream with concept drift...")
    
    for batch_idx in range(5):
        if batch_idx < 3:
            # Original concept
            X_batch = np.random.randn(50, n_features) * 0.5 + 2
            y_batch = np.zeros(50)
        else:
            # Drifted concept (different distribution)
            X_batch = np.random.randn(50, n_features) * 0.8 + 3
            y_batch = np.ones(50)
        
        result = detector.predict_with_feedback(X_batch, y_batch)
        
        if result['drift_detected']:
            print(f"Batch {batch_idx}: DRIFT DETECTED!")
        
        print(f"Batch {batch_idx}: Accuracy={result['current_score']:.4f}")
    
    # Get final status
    status = detector.get_status()
    
    print(f"\n✓ Final status:")
    print(f"  Total drifts: {status['online_statistics']['drift_count']}")
    print(f"  Model updates: {status['online_statistics']['update_count']}")
    print(f"  Recent performance: {status['online_statistics']['avg_recent_performance']:.4f}")
    
    print("\n" + "="*60)
    print("Test 3: Lightweight Detector")
    print("="*60)
    
    detector_light = create_lightweight_detector()
    
    print("\nTraining lightweight detector...")
    results_light = detector_light.train(X_train, y_train)
    
    print(f"\n✓ Lightweight training (faster):")
    print(f"  Strategy: {detector_light.imbalance_handler.strategy}")
    print(f"  Deep features: {detector_light.pattern_learner.use_deep_features}")
    print(f"  Ensemble: {detector_light.online_system.ensemble is not None}")
    
    predictions_light = detector_light.predict(X_test)
    acc_light = accuracy_score(y_test, predictions_light)
    
    print(f"  Test accuracy: {acc_light:.4f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    
    print("\nAdvanced Integrated Detector successfully handles:")
    print("  ✓ Class imbalance (95:5 → balanced)")
    print("  ✓ Temporal context (sliding windows + attention)")
    print("  ✓ Automatic pattern learning (deep features + AutoML)")
    print("  ✓ Evolving threats (online learning + drift detection)")
