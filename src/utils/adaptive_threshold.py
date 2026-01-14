"""
Adaptive Threshold Manager for ICS Anomaly Detection
Dynamically adjusts detection thresholds based on operational context,
time-of-day patterns, and traffic characteristics to reduce false positives.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime, time
import json


class AdaptiveThresholdManager:
    """
    Manages dynamic thresholds that adapt to operational context.
    Reduces false positives by understanding normal operational variations.
    """
    
    def __init__(self, learning_window: int = 1000, confidence_level: float = 0.95):
        """
        Initialize adaptive threshold manager.
        
        Args:
            learning_window (int): Number of samples for baseline learning
            confidence_level (float): Statistical confidence level (0.0-1.0)
        """
        self.learning_window = learning_window
        self.confidence_level = confidence_level
        
        # Historical data for threshold adaptation
        self.normal_scores = deque(maxlen=learning_window)
        self.attack_scores = deque(maxlen=learning_window)
        
        # Context-specific thresholds
        self.time_of_day_thresholds = {}  # Hour -> threshold
        self.day_of_week_thresholds = {}  # Weekday -> threshold
        self.operational_mode_thresholds = {}  # Mode -> threshold
        
        # Baseline statistics
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_percentiles = {}
        
        # Operational context
        self.current_mode = 'normal'  # normal, maintenance, startup, shutdown
        self.is_learning = True
        self.learned_samples = 0
    
    def update_baseline(self, scores: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Update baseline statistics from normal traffic.
        
        Args:
            scores (np.ndarray): Anomaly scores
            labels (np.ndarray, optional): True labels (0=normal, 1=attack)
        """
        if labels is not None:
            # Separate normal and attack scores
            normal_mask = labels == 0
            attack_mask = labels == 1
            
            normal_scores_batch = scores[normal_mask]
            attack_scores_batch = scores[attack_mask]
            
            self.normal_scores.extend(normal_scores_batch)
            self.attack_scores.extend(attack_scores_batch)
        else:
            # Assume all are normal during learning phase
            self.normal_scores.extend(scores)
        
        self.learned_samples += len(scores)
        
        # Update baseline statistics
        if len(self.normal_scores) >= 100:
            self.baseline_mean = np.mean(self.normal_scores)
            self.baseline_std = np.std(self.normal_scores)
            
            # Calculate percentiles for dynamic thresholding
            self.baseline_percentiles = {
                'p90': np.percentile(self.normal_scores, 90),
                'p95': np.percentile(self.normal_scores, 95),
                'p99': np.percentile(self.normal_scores, 99),
                'p99.9': np.percentile(self.normal_scores, 99.9)
            }
            
            if self.learned_samples >= self.learning_window:
                self.is_learning = False
    
    def get_adaptive_threshold(self, context: Optional[Dict] = None) -> float:
        """
        Get threshold adapted to current operational context.
        
        Args:
            context (dict): Operational context with keys:
                - timestamp: Current time
                - mode: operational mode (normal, maintenance, startup, etc.)
                - traffic_load: Current traffic load (0.0-1.0)
                
        Returns:
            float: Adaptive threshold
        """
        if self.is_learning or self.baseline_mean is None:
            # Use conservative threshold during learning
            return 50.0
        
        # Start with statistical threshold (mean + k*std)
        k = self._get_confidence_multiplier()
        base_threshold = self.baseline_mean + k * self.baseline_std
        
        # Apply context adjustments
        if context:
            # Time-of-day adjustment
            if 'timestamp' in context:
                dt = datetime.fromtimestamp(context['timestamp'])
                hour = dt.hour
                
                # Higher threshold during business hours (more benign activity)
                if 8 <= hour <= 18:
                    base_threshold *= 1.3
                # Lower threshold during off-hours (less normal activity)
                elif hour < 6 or hour > 22:
                    base_threshold *= 0.8
            
            # Operational mode adjustment
            if 'mode' in context:
                mode = context['mode']
                
                if mode == 'maintenance':
                    base_threshold *= 2.0  # Much higher during maintenance
                elif mode == 'startup':
                    base_threshold *= 1.5  # Higher during startup
                elif mode == 'shutdown':
                    base_threshold *= 1.5  # Higher during shutdown
                elif mode == 'emergency':
                    base_threshold *= 0.6  # Lower during emergency (be cautious)
            
            # Traffic load adjustment
            if 'traffic_load' in context:
                load = context['traffic_load']
                
                # Higher threshold during high traffic (more benign anomalies)
                if load > 0.7:
                    base_threshold *= 1.2
                elif load < 0.3:
                    base_threshold *= 0.9
        
        # Ensure reasonable bounds
        min_threshold = self.baseline_percentiles.get('p95', 30.0)
        max_threshold = self.baseline_percentiles.get('p99.9', 100.0)
        
        return np.clip(base_threshold, min_threshold, max_threshold)
    
    def _get_confidence_multiplier(self) -> float:
        """Get multiplier for standard deviation based on confidence level."""
        # Map confidence level to z-score
        confidence_map = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
            0.999: 3.291
        }
        
        # Find closest confidence level
        closest = min(confidence_map.keys(), key=lambda x: abs(x - self.confidence_level))
        return confidence_map[closest]
    
    def should_alert(self, score: float, context: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Determine if score should trigger alert based on adaptive threshold.
        
        Args:
            score (float): Anomaly score
            context (dict): Operational context
            
        Returns:
            tuple: (should_alert, reason)
        """
        threshold = self.get_adaptive_threshold(context)
        
        if score > threshold:
            # Determine severity
            if score > threshold * 2:
                reason = f"CRITICAL: Score {score:.1f} >> threshold {threshold:.1f}"
            elif score > threshold * 1.5:
                reason = f"HIGH: Score {score:.1f} > threshold {threshold:.1f}"
            else:
                reason = f"MEDIUM: Score {score:.1f} slightly above threshold {threshold:.1f}"
            
            return True, reason
        
        return False, f"Normal: Score {score:.1f} <= threshold {threshold:.1f}"
    
    def optimize_threshold(self, val_scores: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Find optimal threshold that maximizes F1 score on validation set.
        
        Args:
            val_scores (np.ndarray): Validation anomaly scores
            val_labels (np.ndarray): Validation labels
            
        Returns:
            float: Optimal threshold
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Try different thresholds
        thresholds = np.linspace(0, 100, 200)
        best_f1 = 0
        best_threshold = 50.0
        best_metrics = {}
        
        for threshold in thresholds:
            predictions = (val_scores > threshold).astype(int)
            f1 = f1_score(val_labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'f1': f1,
                    'precision': precision_score(val_labels, predictions, zero_division=0),
                    'recall': recall_score(val_labels, predictions, zero_division=0),
                    'threshold': threshold
                }
        
        # Update baseline with this optimal threshold
        self.baseline_mean = best_threshold - 2 * self.baseline_std if self.baseline_std else best_threshold
        
        print(f"\n[Threshold Optimization]")
        print(f"  Best Threshold: {best_threshold:.2f}")
        print(f"  F1 Score: {best_metrics['f1']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        
        return best_threshold
    
    def get_statistics(self) -> Dict:
        """Get current threshold statistics."""
        return {
            'is_learning': self.is_learning,
            'learned_samples': self.learned_samples,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'baseline_percentiles': self.baseline_percentiles,
            'normal_score_count': len(self.normal_scores),
            'attack_score_count': len(self.attack_scores),
            'current_mode': self.current_mode
        }
    
    def save(self, filepath: str):
        """Save threshold configuration."""
        config = {
            'learning_window': self.learning_window,
            'confidence_level': self.confidence_level,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'baseline_percentiles': self.baseline_percentiles,
            'normal_scores': list(self.normal_scores),
            'is_learning': self.is_learning,
            'learned_samples': self.learned_samples
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Threshold configuration saved to {filepath}")
    
    def load(self, filepath: str):
        """Load threshold configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.learning_window = config['learning_window']
        self.confidence_level = config['confidence_level']
        self.baseline_mean = config['baseline_mean']
        self.baseline_std = config['baseline_std']
        self.baseline_percentiles = config['baseline_percentiles']
        self.normal_scores = deque(config['normal_scores'], maxlen=self.learning_window)
        self.is_learning = config['is_learning']
        self.learned_samples = config['learned_samples']
        
        print(f"Threshold configuration loaded from {filepath}")


class ContextAwareFilter:
    """
    Filters out known benign anomalies based on operational context.
    Maintains whitelist of acceptable unusual behaviors.
    """
    
    def __init__(self):
        """Initialize context-aware filter."""
        self.maintenance_windows = []  # List of (start_time, end_time) tuples
        self.whitelisted_ips = set()
        self.whitelisted_operations = set()
        self.benign_patterns = []
        
        # Known benign anomaly patterns
        self.known_benign = {
            'maintenance_window': [],
            'software_update': [],
            'configuration_change': [],
            'backup_operation': [],
            'system_restart': []
        }
    
    def add_maintenance_window(self, start_time: datetime, end_time: datetime):
        """Add scheduled maintenance window."""
        self.maintenance_windows.append((start_time, end_time))
    
    def is_maintenance_time(self, timestamp: float) -> bool:
        """Check if timestamp falls in maintenance window."""
        dt = datetime.fromtimestamp(timestamp)
        
        for start, end in self.maintenance_windows:
            if start <= dt <= end:
                return True
        
        return False
    
    def whitelist_ip(self, ip_address: str):
        """Add IP to whitelist (e.g., admin workstation)."""
        self.whitelisted_ips.add(ip_address)
    
    def whitelist_operation(self, operation: str):
        """Add operation to whitelist (e.g., 'firmware_update')."""
        self.whitelisted_operations.add(operation)
    
    def should_suppress_alert(self, detection_result: Dict, context: Dict) -> Tuple[bool, str]:
        """
        Determine if alert should be suppressed due to benign context.
        
        Args:
            detection_result (dict): Anomaly detection result
            context (dict): Operational context
            
        Returns:
            tuple: (suppress, reason)
        """
        # Check maintenance window
        if 'timestamp' in context and self.is_maintenance_time(context['timestamp']):
            return True, "Maintenance window - anomaly expected"
        
        # Check whitelisted source
        if 'src_ip' in context and context['src_ip'] in self.whitelisted_ips:
            return True, f"Whitelisted source: {context['src_ip']}"
        
        # Check operational mode
        if context.get('mode') in ['maintenance', 'startup', 'shutdown']:
            if detection_result.get('severity') in ['low', 'medium']:
                return True, f"Benign anomaly during {context['mode']} mode"
        
        # Check whitelisted operation
        if 'operation' in context and context['operation'] in self.whitelisted_operations:
            return True, f"Whitelisted operation: {context['operation']}"
        
        # Check known benign patterns
        if 'pattern_type' in detection_result:
            pattern = detection_result['pattern_type']
            if pattern in self.known_benign and len(self.known_benign[pattern]) > 0:
                return True, f"Known benign pattern: {pattern}"
        
        return False, "Alert should be raised"
    
    def learn_benign_pattern(self, pattern_type: str, features: np.ndarray):
        """Learn new benign anomaly pattern."""
        if pattern_type not in self.known_benign:
            self.known_benign[pattern_type] = []
        
        self.known_benign[pattern_type].append(features)


if __name__ == "__main__":
    print("Testing Adaptive Threshold Manager...")
    
    # Create manager
    manager = AdaptiveThresholdManager(learning_window=500, confidence_level=0.95)
    
    print("\n" + "="*60)
    print("Test 1: Baseline Learning")
    print("="*60)
    
    # Simulate normal traffic scores
    normal_scores = np.random.gamma(2, 10, 600)  # Mean ~20
    manager.update_baseline(normal_scores)
    
    stats = manager.get_statistics()
    print(f"Learned Samples: {stats['learned_samples']}")
    print(f"Baseline Mean: {stats['baseline_mean']:.2f}")
    print(f"Baseline Std: {stats['baseline_std']:.2f}")
    print(f"P95: {stats['baseline_percentiles']['p95']:.2f}")
    print(f"P99: {stats['baseline_percentiles']['p99']:.2f}")
    print(f"Learning Complete: {not stats['is_learning']}")
    
    print("\n" + "="*60)
    print("Test 2: Adaptive Thresholds")
    print("="*60)
    
    # Test different contexts
    contexts = [
        {'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(), 'mode': 'normal'},
        {'timestamp': datetime(2024, 1, 15, 2, 0).timestamp(), 'mode': 'normal'},
        {'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(), 'mode': 'maintenance'},
        {'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(), 'mode': 'emergency'}
    ]
    
    for ctx in contexts:
        threshold = manager.get_adaptive_threshold(ctx)
        dt = datetime.fromtimestamp(ctx['timestamp'])
        print(f"\nContext: {dt.strftime('%H:%M')} - {ctx['mode']}")
        print(f"  Threshold: {threshold:.2f}")
    
    print("\n" + "="*60)
    print("Test 3: Alert Decision")
    print("="*60)
    
    test_scores = [25, 45, 75, 120]
    context = {'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(), 'mode': 'normal'}
    
    for score in test_scores:
        should_alert, reason = manager.should_alert(score, context)
        print(f"\nScore {score}: {'ALERT' if should_alert else 'OK'}")
        print(f"  {reason}")
    
    print("\n" + "="*60)
    print("Test 4: Context-Aware Filtering")
    print("="*60)
    
    filter_sys = ContextAwareFilter()
    filter_sys.add_maintenance_window(
        datetime(2024, 1, 15, 2, 0),
        datetime(2024, 1, 15, 4, 0)
    )
    filter_sys.whitelist_ip('192.168.1.10')
    
    test_contexts = [
        {'timestamp': datetime(2024, 1, 15, 3, 0).timestamp(), 'mode': 'maintenance'},
        {'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(), 'src_ip': '192.168.1.10'},
        {'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(), 'src_ip': '10.0.0.50'}
    ]
    
    for ctx in test_contexts:
        suppress, reason = filter_sys.should_suppress_alert({}, ctx)
        print(f"\nContext: {ctx}")
        print(f"  Suppress: {suppress} - {reason}")
    
    print("\n✓ Adaptive threshold system working!")
