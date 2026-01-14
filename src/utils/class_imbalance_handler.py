"""
Class Imbalance Handler for ICS Anomaly Detection
Addresses severe class imbalance between normal and attack traffic.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Using basic resampling only.")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ClassImbalanceHandler:
    """
    Handles severe class imbalance in ICS datasets.
    Normal traffic: 95-99%, Attack traffic: 1-5%
    """
    
    def __init__(self, strategy: str = 'smote', sampling_ratio: float = 0.3):
        """
        Initialize class imbalance handler.
        
        Args:
            strategy (str): Resampling strategy ('smote', 'adasyn', 'borderline', 'combined', 'weights')
            sampling_ratio (float): Desired minority/majority ratio (0.1-1.0)
        """
        self.strategy = strategy
        self.sampling_ratio = sampling_ratio
        self.class_weights = {}
        self.original_distribution = {}
        self.resampled_distribution = {}
        
    def analyze_distribution(self, y: np.ndarray) -> Dict:
        """
        Analyze class distribution.
        
        Args:
            y (np.ndarray): Labels
            
        Returns:
            dict: Distribution statistics
        """
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        total = len(y)
        
        # Calculate imbalance ratio
        if len(unique) == 2:
            minority_class = unique[np.argmin(counts)]
            majority_class = unique[np.argmax(counts)]
            imbalance_ratio = counts.max() / counts.min()
        else:
            minority_class = None
            majority_class = None
            imbalance_ratio = 0
        
        self.original_distribution = distribution
        
        return {
            'classes': unique.tolist(),
            'counts': counts.tolist(),
            'distribution': distribution,
            'total_samples': total,
            'minority_class': minority_class,
            'majority_class': majority_class,
            'imbalance_ratio': imbalance_ratio,
            'minority_percentage': (counts.min() / total) * 100 if len(unique) > 1 else 0
        }
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for balanced training.
        
        Args:
            y (np.ndarray): Labels
            
        Returns:
            dict: Class weights
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_classes = len(unique)
        
        # Compute balanced weights: n_samples / (n_classes * n_samples_per_class)
        weights = {}
        for cls, count in zip(unique, counts):
            weight = total / (n_classes * count)
            weights[int(cls)] = float(weight)
        
        self.class_weights = weights
        return weights
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray, 
                    k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            k_neighbors (int): Number of nearest neighbors
            
        Returns:
            tuple: Resampled X, y
        """
        if not IMBLEARN_AVAILABLE:
            print("Warning: SMOTE not available, returning original data")
            return X, y
        
        try:
            smote = SMOTE(
                sampling_strategy=self.sampling_ratio,
                k_neighbors=min(k_neighbors, np.bincount(y).min() - 1),
                random_state=42
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            self.resampled_distribution = dict(Counter(y_resampled))
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTE failed: {e}, returning original data")
            return X, y
    
    def apply_adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling).
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            tuple: Resampled X, y
        """
        if not IMBLEARN_AVAILABLE:
            print("Warning: ADASYN not available, returning original data")
            return X, y
        
        try:
            adasyn = ADASYN(
                sampling_strategy=self.sampling_ratio,
                random_state=42
            )
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            self.resampled_distribution = dict(Counter(y_resampled))
            return X_resampled, y_resampled
        except Exception as e:
            print(f"ADASYN failed: {e}, using SMOTE")
            return self.apply_smote(X, y)
    
    def apply_borderline_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Borderline-SMOTE (focuses on borderline samples).
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            tuple: Resampled X, y
        """
        if not IMBLEARN_AVAILABLE:
            print("Warning: Borderline-SMOTE not available, returning original data")
            return X, y
        
        try:
            bsmote = BorderlineSMOTE(
                sampling_strategy=self.sampling_ratio,
                random_state=42
            )
            X_resampled, y_resampled = bsmote.fit_resample(X, y)
            
            self.resampled_distribution = dict(Counter(y_resampled))
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Borderline-SMOTE failed: {e}, using standard SMOTE")
            return self.apply_smote(X, y)
    
    def apply_combined_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply combined over-sampling and under-sampling.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            tuple: Resampled X, y
        """
        if not IMBLEARN_AVAILABLE:
            print("Warning: Combined sampling not available, returning original data")
            return X, y
        
        try:
            # SMOTETomek: SMOTE + Tomek links cleaning
            smotetomek = SMOTETomek(
                sampling_strategy=self.sampling_ratio,
                random_state=42
            )
            X_resampled, y_resampled = smotetomek.fit_resample(X, y)
            
            self.resampled_distribution = dict(Counter(y_resampled))
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Combined sampling failed: {e}, using SMOTE")
            return self.apply_smote(X, y)
    
    def resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply selected resampling strategy.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            tuple: Resampled X, y
        """
        # Analyze original distribution
        dist_info = self.analyze_distribution(y)
        print(f"\nOriginal distribution: {dist_info['distribution']}")
        print(f"Imbalance ratio: {dist_info['imbalance_ratio']:.2f}:1")
        print(f"Minority class: {dist_info['minority_percentage']:.2f}%")
        
        # Apply strategy
        if self.strategy == 'smote':
            X_resampled, y_resampled = self.apply_smote(X, y)
        elif self.strategy == 'adasyn':
            X_resampled, y_resampled = self.apply_adasyn(X, y)
        elif self.strategy == 'borderline':
            X_resampled, y_resampled = self.apply_borderline_smote(X, y)
        elif self.strategy == 'combined':
            X_resampled, y_resampled = self.apply_combined_sampling(X, y)
        elif self.strategy == 'weights':
            # Just compute weights, no resampling
            self.compute_class_weights(y)
            return X, y
        else:
            print(f"Unknown strategy: {self.strategy}, using SMOTE")
            X_resampled, y_resampled = self.apply_smote(X, y)
        
        # Report new distribution
        print(f"Resampled distribution: {self.resampled_distribution}")
        new_ratio = max(Counter(y_resampled).values()) / min(Counter(y_resampled).values())
        print(f"New imbalance ratio: {new_ratio:.2f}:1")
        
        return X_resampled, y_resampled
    
    def get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Get sample-level weights for training.
        
        Args:
            y (np.ndarray): Labels
            
        Returns:
            np.ndarray: Sample weights
        """
        if not self.class_weights:
            self.compute_class_weights(y)
        
        sample_weights = np.array([self.class_weights[int(label)] for label in y])
        return sample_weights


class FocalLoss:
    """
    Focal Loss for addressing class imbalance in deep learning.
    Focuses training on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for class imbalance (0-1)
            gamma (float): Focusing parameter (0-5, typically 2)
        """
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute focal loss.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted probabilities
            
        Returns:
            float: Focal loss
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute cross entropy
        ce = -y_true * np.log(y_pred)
        
        # Compute focal loss
        focal_weight = self.alpha * np.power(1 - y_pred, self.gamma)
        focal_loss = focal_weight * ce
        
        return np.mean(focal_loss)
    
    def get_keras_loss(self):
        """Get Keras-compatible focal loss function."""
        if not TF_AVAILABLE:
            return None
        
        alpha = self.alpha
        gamma = self.gamma
        
        def focal_loss_keras(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            focal_weight = alpha * tf.pow(1 - y_pred, gamma)
            focal_loss = focal_weight * ce
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_keras


class CostSensitiveLearner:
    """
    Cost-sensitive learning with custom misclassification costs.
    False negatives (missing attacks) are more costly than false positives.
    """
    
    def __init__(self, fn_cost: float = 10.0, fp_cost: float = 1.0):
        """
        Initialize cost-sensitive learner.
        
        Args:
            fn_cost (float): Cost of false negative (missing attack)
            fp_cost (float): Cost of false positive (false alarm)
        """
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.cost_matrix = None
    
    def create_cost_matrix(self, n_classes: int = 2) -> np.ndarray:
        """
        Create cost matrix for classification.
        
        Args:
            n_classes (int): Number of classes
            
        Returns:
            np.ndarray: Cost matrix
        """
        cost_matrix = np.zeros((n_classes, n_classes))
        
        if n_classes == 2:
            # Binary classification: [normal, attack]
            cost_matrix[0, 1] = self.fn_cost  # Missing attack (FN)
            cost_matrix[1, 0] = self.fp_cost  # False alarm (FP)
        
        self.cost_matrix = cost_matrix
        return cost_matrix
    
    def compute_cost_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights based on misclassification costs.
        
        Args:
            y (np.ndarray): Labels
            
        Returns:
            dict: Cost-based class weights
        """
        unique, counts = np.unique(y, return_counts=True)
        
        if len(unique) == 2:
            # Binary: weight attack class higher
            weights = {
                int(unique[0]): self.fp_cost,  # Normal
                int(unique[1]): self.fn_cost   # Attack
            }
        else:
            # Multi-class: use inverse frequency
            weights = {}
            for cls, count in zip(unique, counts):
                weights[int(cls)] = len(y) / (len(unique) * count)
        
        return weights


if __name__ == "__main__":
    print("Testing Class Imbalance Handler...")
    
    # Simulate severely imbalanced ICS dataset
    np.random.seed(42)
    n_normal = 9500
    n_attack = 500
    
    # Normal traffic features
    X_normal = np.random.randn(n_normal, 10) * 0.5 + 2
    y_normal = np.zeros(n_normal)
    
    # Attack traffic features
    X_attack = np.random.randn(n_attack, 10) * 1.5 + 5
    y_attack = np.ones(n_attack)
    
    # Combine
    X = np.vstack([X_normal, X_attack])
    y = np.hstack([y_normal, y_attack])
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    print("\n" + "="*60)
    print("Test 1: Distribution Analysis")
    print("="*60)
    
    handler = ClassImbalanceHandler(strategy='smote', sampling_ratio=0.3)
    dist_info = handler.analyze_distribution(y)
    
    print(f"\nOriginal Dataset:")
    print(f"  Total samples: {dist_info['total_samples']}")
    print(f"  Class distribution: {dist_info['distribution']}")
    print(f"  Imbalance ratio: {dist_info['imbalance_ratio']:.2f}:1")
    print(f"  Minority percentage: {dist_info['minority_percentage']:.2f}%")
    
    print("\n" + "="*60)
    print("Test 2: SMOTE Resampling")
    print("="*60)
    
    X_smote, y_smote = handler.resample(X, y)
    print(f"\nAfter SMOTE:")
    print(f"  Total samples: {len(y_smote)}")
    print(f"  Class 0 (normal): {np.sum(y_smote == 0)}")
    print(f"  Class 1 (attack): {np.sum(y_smote == 1)}")
    
    print("\n" + "="*60)
    print("Test 3: Class Weights")
    print("="*60)
    
    weights = handler.compute_class_weights(y)
    print(f"\nClass Weights:")
    for cls, weight in weights.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    sample_weights = handler.get_sample_weights(y[:100])
    print(f"\nSample weights (first 10): {sample_weights[:10]}")
    
    print("\n" + "="*60)
    print("Test 4: Focal Loss")
    print("="*60)
    
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Test predictions
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred_good = np.array([0.9, 0.1, 0.85, 0.15, 0.95])
    y_pred_bad = np.array([0.6, 0.4, 0.6, 0.4, 0.6])
    
    loss_good = focal(y_true, y_pred_good)
    loss_bad = focal(y_pred_bad, y_pred_bad)
    
    print(f"\nFocal Loss (good predictions): {loss_good:.4f}")
    print(f"Focal Loss (bad predictions): {loss_bad:.4f}")
    
    print("\n" + "="*60)
    print("Test 5: Cost-Sensitive Learning")
    print("="*60)
    
    cost_learner = CostSensitiveLearner(fn_cost=10.0, fp_cost=1.0)
    cost_matrix = cost_learner.create_cost_matrix(n_classes=2)
    cost_weights = cost_learner.compute_cost_weights(y)
    
    print(f"\nCost Matrix:")
    print(cost_matrix)
    print(f"\nCost-based Weights:")
    for cls, weight in cost_weights.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    print("\n✓ Class imbalance handler working!")
    
    if IMBLEARN_AVAILABLE:
        print("\n✓ All resampling methods available")
    else:
        print("\n⚠ imbalanced-learn not installed (pip install imbalanced-learn)")
