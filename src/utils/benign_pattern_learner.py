"""
Benign Pattern Learner for ICS Anomaly Detection
Automatically learns and recognizes benign operational patterns to reduce false positives.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


class BenignPatternLearner:
    """
    Learns benign operational patterns in ICS environments.
    Uses clustering and pattern templates to distinguish normal anomalies from attacks.
    """
    
    def __init__(self, learning_threshold: int = 100, confidence_threshold: float = 0.8):
        """
        Initialize benign pattern learner.
        
        Args:
            learning_threshold (int): Minimum occurrences to establish pattern
            confidence_threshold (float): Confidence required to classify as benign
        """
        self.learning_threshold = learning_threshold
        self.confidence_threshold = confidence_threshold
        
        # Pattern storage
        self.operation_patterns = defaultdict(list)  # Operation type -> feature vectors
        self.pattern_templates = {}  # Pattern name -> template
        self.pattern_counts = Counter()  # Pattern -> occurrence count
        
        # Known benign patterns
        self.known_patterns = self._initialize_known_patterns()
        
        # Learned clusters
        self.clusters = []
        self.cluster_labels = {}
        
        # Feature statistics
        self.feature_stats = {}
        
    def _initialize_known_patterns(self) -> Dict:
        """Initialize templates for common benign patterns."""
        return {
            'maintenance_window': {
                'description': 'Scheduled maintenance operations',
                'indicators': ['config_change', 'service_restart', 'firmware_update'],
                'typical_duration': (900, 3600),  # 15 min to 1 hour
                'frequency_pattern': 'weekly',
                'confidence': 0.95
            },
            'backup_operation': {
                'description': 'Data backup procedures',
                'indicators': ['high_read_rate', 'sequential_access', 'off_hours'],
                'typical_duration': (600, 1800),  # 10-30 minutes
                'frequency_pattern': 'daily',
                'confidence': 0.90
            },
            'software_update': {
                'description': 'Software or firmware updates',
                'indicators': ['connection_change', 'protocol_anomaly', 'timing_anomaly'],
                'typical_duration': (300, 1200),  # 5-20 minutes
                'frequency_pattern': 'monthly',
                'confidence': 0.85
            },
            'operator_training': {
                'description': 'Training or testing activities',
                'indicators': ['unusual_sequence', 'repeated_commands', 'test_mode'],
                'typical_duration': (1800, 7200),  # 30 min to 2 hours
                'frequency_pattern': 'irregular',
                'confidence': 0.75
            },
            'system_startup': {
                'description': 'System initialization after restart',
                'indicators': ['protocol_reconnect', 'initialization', 'calibration'],
                'typical_duration': (60, 300),  # 1-5 minutes
                'frequency_pattern': 'rare',
                'confidence': 0.90
            },
            'configuration_sync': {
                'description': 'Configuration synchronization',
                'indicators': ['bulk_transfer', 'config_read', 'sequential_poll'],
                'typical_duration': (30, 180),  # 30 sec to 3 minutes
                'frequency_pattern': 'hourly',
                'confidence': 0.85
            }
        }
    
    def learn_pattern(self, features: Dict, label: Optional[str] = None):
        """
        Learn from observed pattern.
        
        Args:
            features (dict): Feature vector from detection
            label (str): Optional label for supervised learning
        """
        # Extract feature vector
        feature_vector = self._extract_feature_vector(features)
        
        # Add to operation-specific patterns
        operation_type = features.get('operation_type', 'unknown')
        self.operation_patterns[operation_type].append(feature_vector)
        
        # Update feature statistics
        self._update_feature_stats(features)
        
        # If labeled, update pattern counts
        if label:
            self.pattern_counts[label] += 1
    
    def _extract_feature_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to normalized vector."""
        # Extract key features for clustering
        vector = []
        
        # Statistical features
        vector.append(features.get('packet_rate', 0))
        vector.append(features.get('byte_rate', 0))
        vector.append(features.get('connection_count', 0))
        
        # Protocol features
        vector.append(features.get('unique_commands', 0))
        vector.append(features.get('error_rate', 0))
        vector.append(features.get('retry_rate', 0))
        
        # Temporal features
        vector.append(features.get('burst_score', 0))
        vector.append(features.get('periodicity', 0))
        
        # Behavioral features
        vector.append(features.get('entropy', 0))
        vector.append(features.get('symmetry', 0))
        
        return np.array(vector, dtype=np.float32)
    
    def _update_feature_stats(self, features: Dict):
        """Update running statistics for features."""
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if key not in self.feature_stats:
                    self.feature_stats[key] = {
                        'values': [],
                        'mean': 0,
                        'std': 0,
                        'min': value,
                        'max': value
                    }
                
                stats = self.feature_stats[key]
                stats['values'].append(value)
                
                # Keep only recent values (max 10000)
                if len(stats['values']) > 10000:
                    stats['values'] = stats['values'][-10000:]
                
                # Update statistics
                stats['mean'] = np.mean(stats['values'])
                stats['std'] = np.std(stats['values'])
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
    
    def cluster_patterns(self, operation_type: Optional[str] = None):
        """
        Cluster learned patterns using DBSCAN.
        
        Args:
            operation_type (str): Cluster specific operation type or all
        """
        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available, skipping clustering")
            return
        
        # Get patterns to cluster
        if operation_type:
            patterns = self.operation_patterns.get(operation_type, [])
        else:
            patterns = []
            for op_patterns in self.operation_patterns.values():
                patterns.extend(op_patterns)
        
        if len(patterns) < 10:
            return
        
        # Convert to array
        X = np.array(patterns)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster with DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        # Store clusters
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]
            
            # Compute cluster statistics
            cluster_info = {
                'centroid': np.mean(cluster_points, axis=0),
                'std': np.std(cluster_points, axis=0),
                'size': len(cluster_points),
                'label': f"cluster_{len(self.clusters)}"
            }
            
            self.clusters.append(cluster_info)
    
    def is_benign_pattern(self, features: Dict) -> Dict:
        """
        Determine if features match a known benign pattern.
        
        Args:
            features (dict): Feature dict from detection
            
        Returns:
            dict: Pattern classification result
        """
        result = {
            'is_benign': False,
            'confidence': 0.0,
            'pattern_name': None,
            'reasons': []
        }
        
        # Check against known patterns
        for pattern_name, pattern_info in self.known_patterns.items():
            match_score = self._match_pattern(features, pattern_info)
            
            if match_score > result['confidence']:
                result['confidence'] = match_score
                result['pattern_name'] = pattern_name
        
        # Check if confidence exceeds threshold
        if result['confidence'] >= self.confidence_threshold:
            result['is_benign'] = True
            result['reasons'].append(f"Matches {result['pattern_name']} pattern")
        
        # Check against learned clusters
        if SKLEARN_AVAILABLE and len(self.clusters) > 0:
            cluster_match = self._match_cluster(features)
            if cluster_match['confidence'] > result['confidence']:
                result['is_benign'] = True
                result['confidence'] = cluster_match['confidence']
                result['pattern_name'] = cluster_match['cluster_label']
                result['reasons'].append("Matches learned cluster pattern")
        
        # Add contextual information
        result['operation_type'] = features.get('operation_type', 'unknown')
        result['timestamp'] = features.get('timestamp', 0)
        
        return result
    
    def _match_pattern(self, features: Dict, pattern_info: Dict) -> float:
        """
        Calculate match score between features and pattern template.
        
        Args:
            features (dict): Observed features
            pattern_info (dict): Pattern template
            
        Returns:
            float: Match confidence (0-1)
        """
        match_score = 0.0
        indicator_count = len(pattern_info['indicators'])
        
        if indicator_count == 0:
            return 0.0
        
        # Check indicators
        matched_indicators = 0
        for indicator in pattern_info['indicators']:
            if self._check_indicator(features, indicator):
                matched_indicators += 1
        
        # Base score from indicator matching
        match_score = matched_indicators / indicator_count
        
        # Adjust for duration if available
        if 'duration' in features:
            duration = features['duration']
            expected_range = pattern_info['typical_duration']
            
            if expected_range[0] <= duration <= expected_range[1]:
                match_score *= 1.2  # Boost if duration matches
            elif duration > expected_range[1] * 2:
                match_score *= 0.7  # Penalize if much longer
        
        # Adjust for timing patterns
        if 'timestamp' in features:
            timestamp = features['timestamp']
            dt = datetime.fromtimestamp(timestamp)
            
            frequency = pattern_info['frequency_pattern']
            if frequency == 'off_hours' and (dt.hour < 6 or dt.hour > 22):
                match_score *= 1.1
            elif frequency == 'weekly' and dt.weekday() == 6:  # Sunday
                match_score *= 1.1
        
        # Cap at pattern's confidence level
        return min(match_score, pattern_info['confidence'])
    
    def _check_indicator(self, features: Dict, indicator: str) -> bool:
        """Check if specific indicator is present in features."""
        # Map indicators to feature checks
        checks = {
            'config_change': lambda f: f.get('config_modified', False),
            'service_restart': lambda f: f.get('service_restarted', False),
            'firmware_update': lambda f: f.get('firmware_changed', False),
            'high_read_rate': lambda f: f.get('read_rate', 0) > f.get('write_rate', 0) * 5,
            'sequential_access': lambda f: f.get('sequential_pattern', 0) > 0.7,
            'off_hours': lambda f: datetime.fromtimestamp(f.get('timestamp', 0)).hour not in range(6, 22),
            'connection_change': lambda f: f.get('new_connections', 0) > 0,
            'protocol_anomaly': lambda f: f.get('protocol_anomaly_score', 0) > 50,
            'timing_anomaly': lambda f: f.get('timing_anomaly_score', 0) > 50,
            'unusual_sequence': lambda f: f.get('sequence_anomaly', 0) > 0.5,
            'repeated_commands': lambda f: f.get('command_repetition', 0) > 0.6,
            'test_mode': lambda f: f.get('test_mode', False),
            'protocol_reconnect': lambda f: f.get('reconnects', 0) > 0,
            'initialization': lambda f: f.get('init_sequence', False),
            'calibration': lambda f: f.get('calibration_mode', False),
            'bulk_transfer': lambda f: f.get('bulk_data', False),
            'config_read': lambda f: f.get('config_read', False),
            'sequential_poll': lambda f: f.get('polling_pattern', 0) > 0.8,
        }
        
        check_func = checks.get(indicator)
        if check_func:
            try:
                return check_func(features)
            except:
                return False
        
        # Fallback: check if indicator string appears in feature keys/values
        for key, value in features.items():
            if indicator in str(key).lower() or indicator in str(value).lower():
                return True
        
        return False
    
    def _match_cluster(self, features: Dict) -> Dict:
        """Match features against learned clusters."""
        if not self.clusters:
            return {'confidence': 0.0, 'cluster_label': None}
        
        feature_vector = self._extract_feature_vector(features)
        
        best_match = None
        best_distance = float('inf')
        
        for cluster in self.clusters:
            centroid = cluster['centroid']
            std = cluster['std']
            
            # Compute normalized distance
            if np.any(std == 0):
                distance = np.linalg.norm(feature_vector - centroid)
            else:
                distance = np.linalg.norm((feature_vector - centroid) / std)
            
            if distance < best_distance:
                best_distance = distance
                best_match = cluster
        
        if best_match is None:
            return {'confidence': 0.0, 'cluster_label': None}
        
        # Convert distance to confidence (inverse exponential)
        confidence = np.exp(-best_distance / 3.0)
        
        return {
            'confidence': confidence,
            'cluster_label': best_match['label']
        }
    
    def add_benign_pattern(self, name: str, description: str, 
                           indicators: List[str], duration_range: Tuple[int, int],
                           frequency: str, confidence: float = 0.85):
        """
        Add custom benign pattern template.
        
        Args:
            name (str): Pattern name
            description (str): Human-readable description
            indicators (list): List of indicator strings
            duration_range (tuple): (min_seconds, max_seconds)
            frequency (str): Frequency pattern (e.g., 'daily', 'weekly')
            confidence (float): Base confidence level
        """
        self.known_patterns[name] = {
            'description': description,
            'indicators': indicators,
            'typical_duration': duration_range,
            'frequency_pattern': frequency,
            'confidence': confidence
        }
    
    def get_statistics(self) -> Dict:
        """Get learner statistics."""
        return {
            'known_patterns': len(self.known_patterns),
            'learned_patterns': sum(len(patterns) for patterns in self.operation_patterns.values()),
            'clusters': len(self.clusters),
            'operation_types': len(self.operation_patterns),
            'most_common_patterns': self.pattern_counts.most_common(5)
        }


if __name__ == "__main__":
    print("Testing Benign Pattern Learner...")
    
    learner = BenignPatternLearner(learning_threshold=10, confidence_threshold=0.75)
    
    print("\n" + "="*60)
    print("Test 1: Known Pattern Recognition")
    print("="*60)
    
    # Test maintenance window pattern
    maintenance_features = {
        'timestamp': datetime(2024, 1, 14, 22, 0).timestamp(),  # 10 PM
        'config_modified': True,
        'service_restarted': True,
        'duration': 1200,  # 20 minutes
        'operation_type': 'maintenance'
    }
    
    result = learner.is_benign_pattern(maintenance_features)
    print(f"\nMaintenance Pattern:")
    print(f"  Is Benign: {result['is_benign']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Pattern: {result['pattern_name']}")
    if result['reasons']:
        print(f"  Reasons: {', '.join(result['reasons'])}")
    
    # Test backup operation
    backup_features = {
        'timestamp': datetime(2024, 1, 15, 2, 0).timestamp(),  # 2 AM
        'read_rate': 5000,
        'write_rate': 100,
        'sequential_pattern': 0.95,
        'duration': 900,  # 15 minutes
        'operation_type': 'backup'
    }
    
    result = learner.is_benign_pattern(backup_features)
    print(f"\nBackup Pattern:")
    print(f"  Is Benign: {result['is_benign']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Pattern: {result['pattern_name']}")
    
    # Test attack (should not match)
    attack_features = {
        'timestamp': datetime(2024, 1, 15, 14, 0).timestamp(),  # 2 PM
        'unauthorized_access': True,
        'privilege_escalation': True,
        'unusual_destination': True,
        'duration': 30,
        'operation_type': 'attack'
    }
    
    result = learner.is_benign_pattern(attack_features)
    print(f"\nAttack Pattern:")
    print(f"  Is Benign: {result['is_benign']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Pattern: {result['pattern_name']}")
    
    print("\n" + "="*60)
    print("Test 2: Pattern Learning")
    print("="*60)
    
    # Learn from normal operations
    for i in range(50):
        features = {
            'packet_rate': 100 + np.random.randn() * 10,
            'byte_rate': 5000 + np.random.randn() * 500,
            'connection_count': 5,
            'unique_commands': 10 + np.random.randint(0, 5),
            'error_rate': 0.01 + np.random.randn() * 0.005,
            'retry_rate': 0.02,
            'burst_score': 0.3,
            'periodicity': 0.8,
            'entropy': 3.5 + np.random.randn() * 0.5,
            'symmetry': 0.9,
            'operation_type': 'normal_ops'
        }
        learner.learn_pattern(features, label='normal_operation')
    
    stats = learner.get_statistics()
    print(f"Known patterns: {stats['known_patterns']}")
    print(f"Learned patterns: {stats['learned_patterns']}")
    print(f"Operation types: {stats['operation_types']}")
    
    print("\n" + "="*60)
    print("Test 3: Custom Pattern Addition")
    print("="*60)
    
    # Add custom pattern
    learner.add_benign_pattern(
        name='database_optimization',
        description='Monthly database optimization',
        indicators=['high_write_rate', 'sequential_access', 'off_hours'],
        duration_range=(1800, 5400),  # 30-90 minutes
        frequency='monthly',
        confidence=0.80
    )
    
    print(f"Added custom pattern: database_optimization")
    print(f"Total known patterns: {len(learner.known_patterns)}")
    
    # Test custom pattern
    db_opt_features = {
        'timestamp': datetime(2024, 1, 1, 3, 0).timestamp(),  # 3 AM
        'read_rate': 1000,
        'write_rate': 8000,  # High write
        'sequential_pattern': 0.9,
        'duration': 3600,  # 1 hour
        'operation_type': 'database_maintenance'
    }
    
    result = learner.is_benign_pattern(db_opt_features)
    print(f"\nCustom Pattern Match:")
    print(f"  Is Benign: {result['is_benign']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Pattern: {result['pattern_name']}")
    
    print("\n" + "="*60)
    print("Test 4: Clustering (if sklearn available)")
    print("="*60)
    
    if SKLEARN_AVAILABLE:
        learner.cluster_patterns()
        print(f"Clusters created: {len(learner.clusters)}")
        
        for i, cluster in enumerate(learner.clusters[:3]):  # Show first 3
            print(f"\nCluster {i+1}:")
            print(f"  Size: {cluster['size']} patterns")
            print(f"  Label: {cluster['label']}")
    else:
        print("sklearn not available, skipping clustering test")
    
    print("\n✓ Benign pattern learner working!")
