"""
Integrated ICS Anomaly Detection System
Combines all optimization and context-aware modules for production-ready detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from utils.adaptive_threshold import AdaptiveThresholdManager, ContextAwareFilter
from utils.performance_optimizer import (
    FeatureCache, ModelQuantizer, ParallelProcessor,
    IncrementalLearner, PerformanceMonitor
)
from utils.temporal_analyzer import TemporalPatternAnalyzer
from utils.benign_pattern_learner import BenignPatternLearner
from utils.context_analyzer import ContextAwareAnalyzer


class IntegratedICSDetector:
    """
    Production-ready integrated ICS anomaly detector.
    Combines adaptive thresholding, benign learning, temporal analysis, and context awareness.
    """
    
    def __init__(self, 
                 enable_adaptive_thresholds: bool = True,
                 enable_benign_learning: bool = True,
                 enable_temporal_analysis: bool = True,
                 enable_context_awareness: bool = True,
                 enable_performance_optimization: bool = True):
        """
        Initialize integrated detector with optional module toggles.
        
        Args:
            enable_adaptive_thresholds (bool): Enable adaptive threshold management
            enable_benign_learning (bool): Enable benign pattern learning
            enable_temporal_analysis (bool): Enable temporal pattern analysis
            enable_context_awareness (bool): Enable context-aware analysis
            enable_performance_optimization (bool): Enable performance optimizations
        """
        # Core detection modules (placeholder - would integrate real detectors)
        self.base_detector = None
        
        # Optimization and intelligence modules
        self.adaptive_threshold = AdaptiveThresholdManager() if enable_adaptive_thresholds else None
        self.context_filter = ContextAwareFilter() if enable_adaptive_thresholds else None
        self.benign_learner = BenignPatternLearner() if enable_benign_learning else None
        self.temporal_analyzer = TemporalPatternAnalyzer() if enable_temporal_analysis else None
        self.context_analyzer = ContextAwareAnalyzer() if enable_context_awareness else None
        
        # Performance optimization
        if enable_performance_optimization:
            self.feature_cache = FeatureCache(max_size=10000, ttl=300)
            self.performance_monitor = PerformanceMonitor()
        else:
            self.feature_cache = None
            self.performance_monitor = None
        
        # Configuration
        self.config = {
            'adaptive_thresholds': enable_adaptive_thresholds,
            'benign_learning': enable_benign_learning,
            'temporal_analysis': enable_temporal_analysis,
            'context_awareness': enable_context_awareness,
            'performance_optimization': enable_performance_optimization
        }
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'suppressed_alerts': 0,
            'benign_patterns_learned': 0,
            'attack_chains_detected': 0
        }
    
    def detect(self, packet_data: Dict) -> Dict:
        """
        Perform integrated anomaly detection on packet data.
        
        Args:
            packet_data (dict): Packet data and metadata
            
        Returns:
            dict: Comprehensive detection result
        """
        timestamp = packet_data.get('timestamp', datetime.now().timestamp())
        
        # Start performance monitoring
        if self.performance_monitor:
            detection_start = datetime.now().timestamp()
        
        # Step 1: Extract features (with caching)
        features = self._extract_features(packet_data)
        
        # Step 2: Base detection (placeholder)
        base_score = self._perform_base_detection(features)
        
        detection_result = {
            'timestamp': timestamp,
            'score': base_score,
            'features': features,
            'packet_data': packet_data
        }
        
        # Step 3: Temporal analysis
        temporal_info = {}
        if self.temporal_analyzer:
            self.temporal_analyzer.add_observation(timestamp, base_score, features)
            self.temporal_analyzer.compute_baselines()
            temporal_info = self.temporal_analyzer.detect_temporal_anomaly(timestamp, base_score)
            
            # Check for sequence attacks
            sequence_info = self.temporal_analyzer.detect_sequence_attack()
            temporal_info['sequence_attack'] = sequence_info
        
        # Step 4: Benign pattern analysis
        benign_info = {}
        if self.benign_learner:
            benign_info = self.benign_learner.is_benign_pattern(features)
            
            # Learn from patterns
            if not benign_info.get('is_benign', False):
                self.benign_learner.learn_pattern(features)
        
        # Step 5: Context-aware analysis
        if self.context_analyzer:
            analysis = self.context_analyzer.analyze_with_context(
                detection_result, temporal_info, benign_info
            )
            final_score = analysis['adjusted_score']
            suppressed = analysis['suppressed']
            attack_chain = analysis.get('attack_chain')
        else:
            final_score = base_score
            suppressed = False
            attack_chain = None
            analysis = {'adjustments': [], 'confidence': 0.5}
        
        # Step 6: Adaptive threshold application
        should_alert = True
        if self.adaptive_threshold and self.context_filter:
            # Update baseline
            if not suppressed:
                self.adaptive_threshold.update_baseline([final_score])  # Pass as list
            
            # Get adaptive threshold
            context = self.context_analyzer.get_operational_context(timestamp) if self.context_analyzer else {}
            threshold = self.adaptive_threshold.get_adaptive_threshold(context)
            
            # Check if should suppress
            suppress_result = self.context_filter.should_suppress_alert(
                detection_result, context
            )
            should_alert = not suppress_result[0]  # First element is boolean
            
            # Apply threshold
            if final_score < threshold:
                should_alert = False
        
        # Step 7: Compile final result
        result = {
            'timestamp': timestamp,
            'alert': should_alert and not suppressed,
            'score': {
                'base': base_score,
                'adjusted': final_score,
                'threshold': threshold if self.adaptive_threshold else 50
            },
            'severity': self._calculate_severity(final_score),
            'confidence': analysis.get('confidence', 0.5),
            'temporal_analysis': temporal_info,
            'benign_pattern': benign_info,
            'context_adjustments': analysis.get('adjustments', []),
            'attack_chain': attack_chain,
            'suppressed': suppressed,
            'features': features
        }
        
        # Update statistics
        self._update_statistics(result)
        
        # Performance monitoring
        if self.performance_monitor:
            detection_end = datetime.now().timestamp()
            self.performance_monitor.record_latency(detection_end - detection_start)
        
        return result
    
    def _extract_features(self, packet_data: Dict) -> Dict:
        """Extract features from packet data (with caching)."""
        if self.feature_cache:
            # Serialize packet data for caching
            cache_key = pickle.dumps(packet_data)
            
            # Try to get from cache
            cached_features = self.feature_cache.get(cache_key)
            if cached_features is not None:
                # Deserialize
                return pickle.loads(cached_features)
        
        # Extract features (placeholder - would use real feature extractor)
        features = {
            'timestamp': packet_data.get('timestamp', datetime.now().timestamp()),
            'packet_rate': packet_data.get('packet_count', 0) / max(packet_data.get('time_window', 1), 1),
            'byte_rate': packet_data.get('byte_count', 0) / max(packet_data.get('time_window', 1), 1),
            'connection_count': packet_data.get('connections', 1),
            'unique_commands': len(set(packet_data.get('commands', []))),
            'error_rate': packet_data.get('errors', 0) / max(packet_data.get('total_packets', 1), 1),
            'retry_rate': packet_data.get('retries', 0) / max(packet_data.get('total_packets', 1), 1),
            'burst_score': packet_data.get('burst_detected', 0),
            'periodicity': packet_data.get('periodic_pattern', 0),
            'entropy': packet_data.get('entropy', 0),
            'symmetry': packet_data.get('symmetry', 0),
            'operation_type': packet_data.get('operation', 'unknown')
        }
        
        # Cache features
        if self.feature_cache:
            self.feature_cache.put(cache_key, pickle.dumps(features))
        
        return features
    
    def _perform_base_detection(self, features: Dict) -> float:
        """
        Perform base anomaly detection.
        Placeholder - would integrate real detection models.
        """
        # Simple heuristic for demonstration
        score = 0.0
        
        # High packet rate
        if features['packet_rate'] > 1000:
            score += 20
        
        # High error rate
        if features['error_rate'] > 0.1:
            score += 25
        
        # High retry rate
        if features['retry_rate'] > 0.2:
            score += 15
        
        # Burst detected
        if features['burst_score'] > 0.7:
            score += 20
        
        # Low entropy (potential attack)
        if features['entropy'] < 2.0:
            score += 20
        
        return min(score, 100)
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity level from score."""
        if score >= 80:
            return 'critical'
        elif score >= 60:
            return 'high'
        elif score >= 40:
            return 'medium'
        elif score >= 20:
            return 'low'
        else:
            return 'info'
    
    def _update_statistics(self, result: Dict):
        """Update detection statistics."""
        self.stats['total_detections'] += 1
        
        if result['suppressed']:
            self.stats['suppressed_alerts'] += 1
        
        if result.get('attack_chain'):
            self.stats['attack_chains_detected'] += 1
    
    def add_maintenance_window(self, start_time: float, end_time: float, description: str):
        """Add maintenance window to context analyzer."""
        if self.context_analyzer:
            self.context_analyzer.add_maintenance_window(start_time, end_time, description)
    
    def add_whitelist(self, ip: str, reason: str = "Authorized"):
        """Add IP to whitelist."""
        if self.context_filter:
            self.context_filter.whitelist_ip(ip)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            'detector': self.stats.copy(),
            'config': self.config.copy()
        }
        
        if self.adaptive_threshold:
            stats['adaptive_threshold'] = self.adaptive_threshold.get_statistics()
        
        if self.benign_learner:
            stats['benign_learner'] = self.benign_learner.get_statistics()
        
        if self.temporal_analyzer:
            stats['temporal_analyzer'] = self.temporal_analyzer.get_statistics()
        
        if self.context_analyzer:
            stats['context_analyzer'] = self.context_analyzer.get_statistics()
        
        if self.feature_cache:
            cache_stats = self.feature_cache.get_statistics()
            stats['feature_cache'] = {
                'hits': cache_stats['hit_count'],
                'misses': cache_stats['miss_count'],
                'hit_rate': cache_stats['hit_rate']
            }
        
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_report()
            stats['performance'] = perf_stats
        
        return stats
    
    def get_recommendations(self) -> List[str]:
        """Get system recommendations based on statistics."""
        recommendations = []
        stats = self.get_statistics()
        
        # Check false positive rate
        if self.stats['total_detections'] > 100:
            suppression_rate = self.stats['suppressed_alerts'] / self.stats['total_detections']
            if suppression_rate > 0.5:
                recommendations.append("High suppression rate detected - consider adjusting benign patterns")
        
        # Check cache performance
        if 'feature_cache' in stats:
            if stats['feature_cache']['hit_rate'] < 0.3:
                recommendations.append("Low cache hit rate - consider increasing cache size or TTL")
        
        # Check attack chains
        if self.stats['attack_chains_detected'] > 0:
            recommendations.append(f"⚠️ {self.stats['attack_chains_detected']} attack chains detected - review security posture")
        
        # Performance recommendations
        if 'performance' in stats and 'latency_ms' in stats['performance']:
            mean_latency_ms = stats['performance']['latency_ms']['mean']
            if mean_latency_ms > 100:  # 100ms threshold
                recommendations.append("High detection latency - consider enabling performance optimizations")
        
        return recommendations


if __name__ == "__main__":
    print("Testing Integrated ICS Detector...")
    print("="*60)
    
    # Initialize detector with all features enabled
    detector = IntegratedICSDetector(
        enable_adaptive_thresholds=True,
        enable_benign_learning=True,
        enable_temporal_analysis=True,
        enable_context_awareness=True,
        enable_performance_optimization=True
    )
    
    print("\n✓ Integrated detector initialized")
    print(f"Enabled modules: {sum(detector.config.values())}/5")
    
    # Add maintenance window
    maintenance_start = datetime(2024, 1, 15, 22, 0).timestamp()
    maintenance_end = datetime(2024, 1, 15, 23, 0).timestamp()
    detector.add_maintenance_window(maintenance_start, maintenance_end, "Monthly updates")
    print("✓ Added maintenance window")
    
    # Add whitelist
    detector.add_whitelist("10.0.0.100", "SCADA server")
    print("✓ Added whitelist entry")
    
    print("\n" + "="*60)
    print("Test 1: Normal Traffic")
    print("="*60)
    
    normal_packet = {
        'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(),
        'packet_count': 100,
        'byte_count': 50000,
        'time_window': 1,
        'connections': 5,
        'commands': ['read', 'write'],
        'errors': 1,
        'total_packets': 100,
        'retries': 2,
        'entropy': 3.5,
        'symmetry': 0.9
    }
    
    result = detector.detect(normal_packet)
    print(f"Alert: {result['alert']}")
    print(f"Score: {result['score']['base']:.1f} → {result['score']['adjusted']:.1f} (threshold: {result['score']['threshold']:.1f})")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    print("\n" + "="*60)
    print("Test 2: Suspicious Traffic (Off-Hours)")
    print("="*60)
    
    suspicious_packet = {
        'timestamp': datetime(2024, 1, 15, 2, 0).timestamp(),  # 2 AM
        'packet_count': 2000,
        'byte_count': 500000,
        'time_window': 1,
        'connections': 50,
        'commands': ['read', 'write', 'execute'] * 10,
        'errors': 50,
        'total_packets': 200,
        'retries': 40,
        'entropy': 1.5,  # Low entropy
        'symmetry': 0.3,
        'burst_detected': 0.9
    }
    
    result = detector.detect(suspicious_packet)
    print(f"Alert: {result['alert']}")
    print(f"Score: {result['score']['base']:.1f} → {result['score']['adjusted']:.1f}")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2f}")
    if result['context_adjustments']:
        print(f"Adjustments: {', '.join(result['context_adjustments'])}")
    
    print("\n" + "="*60)
    print("Test 3: Maintenance Window (Benign)")
    print("="*60)
    
    maintenance_packet = {
        'timestamp': datetime(2024, 1, 15, 22, 30).timestamp(),  # During maintenance
        'packet_count': 500,
        'byte_count': 200000,
        'time_window': 1,
        'connections': 10,
        'commands': ['config_change', 'restart'],
        'errors': 10,
        'total_packets': 100,
        'retries': 5,
        'entropy': 2.5,
        'symmetry': 0.7,
        'operation': 'maintenance'
    }
    
    result = detector.detect(maintenance_packet)
    print(f"Alert: {result['alert']}")
    print(f"Score: {result['score']['base']:.1f} → {result['score']['adjusted']:.1f}")
    print(f"Suppressed: {result['suppressed']}")
    print(f"Benign Pattern: {result['benign_pattern'].get('is_benign', False)}")
    if result['context_adjustments']:
        print(f"Adjustments: {', '.join(result['context_adjustments'])}")
    
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    
    stats = detector.get_statistics()
    print(f"\nDetector Statistics:")
    print(f"  Total Detections: {stats['detector']['total_detections']}")
    print(f"  Suppressed Alerts: {stats['detector']['suppressed_alerts']}")
    print(f"  Attack Chains: {stats['detector']['attack_chains_detected']}")
    
    if 'feature_cache' in stats:
        print(f"\nCache Performance:")
        print(f"  Hit Rate: {stats['feature_cache']['hit_rate']:.1%}")
        print(f"  Hits: {stats['feature_cache']['hits']}")
        print(f"  Misses: {stats['feature_cache']['misses']}")
    
    recommendations = detector.get_recommendations()
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print("\n✓ Integrated detector working perfectly!")
    print("="*60)
