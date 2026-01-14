"""
Enhanced Temporal Analysis for ICS Anomaly Detection
Performs long-term trend analysis, seasonal patterns, and multi-stage attack correlation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in ICS traffic for advanced anomaly detection.
    Detects slow attacks, seasonal patterns, and multi-stage attack sequences.
    """
    
    def __init__(self, window_hours: int = 24, history_days: int = 30):
        """
        Initialize temporal pattern analyzer.
        
        Args:
            window_hours (int): Hours for short-term pattern analysis
            history_days (int): Days for long-term trend analysis
        """
        self.window_hours = window_hours
        self.history_days = history_days
        
        # Time-series data storage
        self.hourly_stats = defaultdict(list)  # Hour -> [scores]
        self.daily_stats = defaultdict(list)   # Day -> [scores]
        self.weekly_stats = defaultdict(list)  # Week -> [scores]
        
        # Event sequences
        self.event_sequence = deque(maxlen=1000)
        self.attack_stages = defaultdict(list)
        
        # Baseline patterns
        self.hourly_baseline = {}  # Hour -> {mean, std}
        self.daily_baseline = {}   # Day -> {mean, std}
        
        # Trend detection
        self.trend_window = deque(maxlen=168)  # 1 week of hourly data
    
    def add_observation(self, timestamp: float, score: float, 
                       features: Optional[Dict] = None):
        """
        Add temporal observation for pattern analysis.
        
        Args:
            timestamp (float): Unix timestamp
            score (float): Anomaly score
            features (dict): Optional feature dict for sequence analysis
        """
        dt = datetime.fromtimestamp(timestamp)
        
        # Store by temporal granularity
        hour = dt.hour
        day = dt.weekday()
        week = dt.isocalendar()[1]
        
        self.hourly_stats[hour].append(score)
        self.daily_stats[day].append(score)
        self.weekly_stats[week].append(score)
        
        # Add to trend window
        self.trend_window.append((timestamp, score))
        
        # Add to event sequence if features provided
        if features:
            self.event_sequence.append({
                'timestamp': timestamp,
                'score': score,
                'features': features
            })
    
    def compute_baselines(self):
        """Compute baseline statistics for each time period."""
        # Hourly baseline (24 hours)
        for hour in range(24):
            if hour in self.hourly_stats and len(self.hourly_stats[hour]) >= 10:
                scores = self.hourly_stats[hour]
                self.hourly_baseline[hour] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'p95': np.percentile(scores, 95)
                }
        
        # Daily baseline (7 days)
        for day in range(7):
            if day in self.daily_stats and len(self.daily_stats[day]) >= 10:
                scores = self.daily_stats[day]
                self.daily_baseline[day] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'p95': np.percentile(scores, 95)
                }
    
    def detect_temporal_anomaly(self, timestamp: float, score: float) -> Dict:
        """
        Detect if score is anomalous considering temporal context.
        
        Args:
            timestamp (float): Unix timestamp
            score (float): Anomaly score
            
        Returns:
            dict: Temporal anomaly analysis
        """
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day = dt.weekday()
        
        result = {
            'is_temporal_anomaly': False,
            'temporal_score': 0.0,
            'reasons': [],
            'context': {
                'hour': hour,
                'day': day,
                'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                            'Friday', 'Saturday', 'Sunday'][day]
            }
        }
        
        # Check against hourly baseline
        if hour in self.hourly_baseline:
            baseline = self.hourly_baseline[hour]
            z_score = (score - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
            
            if abs(z_score) > 3:  # 3-sigma rule
                result['is_temporal_anomaly'] = True
                result['temporal_score'] += abs(z_score) * 10
                result['reasons'].append(f"Unusual for hour {hour:02d}:00 (z={z_score:.2f})")
        
        # Check against daily baseline
        if day in self.daily_baseline:
            baseline = self.daily_baseline[day]
            z_score = (score - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
            
            if abs(z_score) > 3:
                result['is_temporal_anomaly'] = True
                result['temporal_score'] += abs(z_score) * 10
                result['reasons'].append(f"Unusual for {result['context']['day_name']} (z={z_score:.2f})")
        
        # Detect trend changes
        trend_info = self.detect_trend()
        if trend_info['trend'] == 'increasing' and score > trend_info['predicted'] * 1.5:
            result['is_temporal_anomaly'] = True
            result['temporal_score'] += 30
            result['reasons'].append("Score exceeds increasing trend prediction")
        
        return result
    
    def detect_trend(self) -> Dict:
        """
        Detect trend in recent time-series data.
        
        Returns:
            dict: Trend information
        """
        if len(self.trend_window) < 10:
            return {'trend': 'insufficient_data', 'slope': 0, 'predicted': 0}
        
        # Extract timestamps and scores
        times = np.array([t for t, s in self.trend_window])
        scores = np.array([s for t, s in self.trend_window])
        
        # Normalize time to hours from start
        times_norm = (times - times[0]) / 3600.0
        
        # Linear regression for trend
        A = np.vstack([times_norm, np.ones(len(times_norm))]).T
        slope, intercept = np.linalg.lstsq(A, scores, rcond=None)[0]
        
        # Classify trend
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0.1:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        # Predict next value
        next_time = times_norm[-1] + 1  # 1 hour ahead
        predicted = slope * next_time + intercept
        
        return {
            'trend': trend,
            'slope': slope,
            'intercept': intercept,
            'predicted': predicted,
            'r_squared': self._compute_r_squared(times_norm, scores, slope, intercept)
        }
    
    def _compute_r_squared(self, x: np.ndarray, y: np.ndarray, 
                          slope: float, intercept: float) -> float:
        """Compute R-squared for linear fit."""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def detect_sequence_attack(self, window_minutes: int = 60) -> Dict:
        """
        Detect multi-stage attack sequences.
        Looks for suspicious patterns like reconnaissance -> exploitation -> data exfiltration.
        
        Args:
            window_minutes (int): Time window for sequence detection
            
        Returns:
            dict: Sequence attack analysis
        """
        if len(self.event_sequence) < 3:
            return {'sequence_detected': False, 'reason': 'Insufficient events'}
        
        # Get recent events within window
        current_time = self.event_sequence[-1]['timestamp']
        window_start = current_time - (window_minutes * 60)
        
        recent_events = [e for e in self.event_sequence 
                        if e['timestamp'] >= window_start]
        
        if len(recent_events) < 3:
            return {'sequence_detected': False, 'reason': 'Insufficient recent events'}
        
        # Extract event characteristics
        scores = [e['score'] for e in recent_events]
        
        # Detect patterns
        patterns = []
        
        # Pattern 1: Escalating scores (reconnaissance -> attack)
        if self._is_escalating(scores):
            patterns.append('escalating_threat')
        
        # Pattern 2: Burst pattern (rapid attacks)
        if self._is_burst_pattern(recent_events):
            patterns.append('burst_attack')
        
        # Pattern 3: Periodic pattern (automated attack)
        if self._is_periodic(recent_events):
            patterns.append('automated_attack')
        
        # Pattern 4: Multi-target (lateral movement)
        if self._is_multi_target(recent_events):
            patterns.append('lateral_movement')
        
        return {
            'sequence_detected': len(patterns) > 0,
            'patterns': patterns,
            'event_count': len(recent_events),
            'time_span_minutes': (recent_events[-1]['timestamp'] - recent_events[0]['timestamp']) / 60,
            'severity': 'critical' if len(patterns) >= 2 else 'high' if len(patterns) == 1 else 'medium'
        }
    
    def _is_escalating(self, scores: List[float], threshold: float = 0.3) -> bool:
        """Check if scores are escalating (increasing trend)."""
        if len(scores) < 3:
            return False
        
        # Check if each score is higher than average of previous
        escalating_count = 0
        for i in range(2, len(scores)):
            prev_avg = np.mean(scores[:i])
            if scores[i] > prev_avg * (1 + threshold):
                escalating_count += 1
        
        return escalating_count >= len(scores) // 2
    
    def _is_burst_pattern(self, events: List[Dict], 
                         max_interval: float = 5.0) -> bool:
        """Check for burst pattern (rapid succession of events)."""
        if len(events) < 5:
            return False
        
        # Check inter-arrival times
        intervals = []
        for i in range(1, len(events)):
            interval = events[i]['timestamp'] - events[i-1]['timestamp']
            intervals.append(interval)
        
        # Burst if most intervals are small
        small_intervals = sum(1 for i in intervals if i < max_interval)
        return small_intervals >= len(intervals) * 0.7
    
    def _is_periodic(self, events: List[Dict], tolerance: float = 0.3) -> bool:
        """Check for periodic pattern (regular intervals)."""
        if len(events) < 4:
            return False
        
        # Calculate inter-arrival times
        intervals = []
        for i in range(1, len(events)):
            interval = events[i]['timestamp'] - events[i-1]['timestamp']
            intervals.append(interval)
        
        # Check if intervals are similar (low coefficient of variation)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return False
        
        cv = std_interval / mean_interval
        return cv < tolerance
    
    def _is_multi_target(self, events: List[Dict]) -> bool:
        """Check if events target multiple destinations (lateral movement)."""
        # Check if events have destination information
        destinations = set()
        
        for event in events:
            if 'features' in event and 'destination' in event['features']:
                destinations.add(event['features']['destination'])
        
        # Multi-target if >= 3 different destinations
        return len(destinations) >= 3
    
    def get_seasonal_pattern(self, hour: int) -> Optional[Dict]:
        """
        Get seasonal pattern for specific hour.
        
        Args:
            hour (int): Hour of day (0-23)
            
        Returns:
            dict: Seasonal pattern or None
        """
        return self.hourly_baseline.get(hour)
    
    def get_statistics(self) -> Dict:
        """Get temporal analysis statistics."""
        return {
            'hours_with_baseline': len(self.hourly_baseline),
            'days_with_baseline': len(self.daily_baseline),
            'event_sequence_length': len(self.event_sequence),
            'trend_window_length': len(self.trend_window),
            'total_observations': sum(len(scores) for scores in self.hourly_stats.values())
        }


if __name__ == "__main__":
    print("Testing Enhanced Temporal Analyzer...")
    
    analyzer = TemporalPatternAnalyzer(window_hours=24, history_days=7)
    
    print("\n" + "="*60)
    print("Test 1: Baseline Learning")
    print("="*60)
    
    # Simulate one week of normal traffic
    base_time = datetime(2024, 1, 15, 0, 0).timestamp()
    
    for day in range(7):
        for hour in range(24):
            timestamp = base_time + day * 86400 + hour * 3600
            
            # Simulate hourly patterns (lower at night, higher during day)
            if 8 <= hour <= 18:
                base_score = 20 + np.random.randn() * 5
            else:
                base_score = 10 + np.random.randn() * 3
            
            analyzer.add_observation(timestamp, base_score)
    
    analyzer.compute_baselines()
    stats = analyzer.get_statistics()
    
    print(f"Hours with baseline: {stats['hours_with_baseline']}/24")
    print(f"Days with baseline: {stats['days_with_baseline']}/7")
    print(f"Total observations: {stats['total_observations']}")
    
    # Show sample baselines
    print(f"\nSample hourly baselines:")
    for hour in [2, 10, 18]:
        if hour in analyzer.hourly_baseline:
            baseline = analyzer.hourly_baseline[hour]
            print(f"  Hour {hour:02d}:00 - Mean: {baseline['mean']:.2f}, P95: {baseline['p95']:.2f}")
    
    print("\n" + "="*60)
    print("Test 2: Temporal Anomaly Detection")
    print("="*60)
    
    # Test anomalous scores at different times
    test_cases = [
        (datetime(2024, 1, 22, 10, 0).timestamp(), 25, "Normal daytime"),
        (datetime(2024, 1, 22, 10, 0).timestamp(), 85, "High daytime"),
        (datetime(2024, 1, 22, 2, 0).timestamp(), 45, "High nighttime"),
    ]
    
    for timestamp, score, description in test_cases:
        result = analyzer.detect_temporal_anomaly(timestamp, score)
        print(f"\n{description} (score={score}):")
        print(f"  Temporal Anomaly: {result['is_temporal_anomaly']}")
        print(f"  Temporal Score: {result['temporal_score']:.1f}")
        if result['reasons']:
            print(f"  Reasons: {', '.join(result['reasons'])}")
    
    print("\n" + "="*60)
    print("Test 3: Trend Detection")
    print("="*60)
    
    # Add increasing trend
    current_time = datetime(2024, 1, 22, 12, 0).timestamp()
    for i in range(24):
        timestamp = current_time + i * 3600
        score = 20 + i * 2 + np.random.randn() * 2  # Increasing trend
        analyzer.add_observation(timestamp, score)
    
    trend = analyzer.detect_trend()
    print(f"Trend: {trend['trend']}")
    print(f"Slope: {trend['slope']:.4f} points/hour")
    print(f"Predicted next: {trend['predicted']:.2f}")
    print(f"R-squared: {trend['r_squared']:.4f}")
    
    print("\n" + "="*60)
    print("Test 4: Sequence Attack Detection")
    print("="*60)
    
    # Simulate attack sequence
    attack_start = datetime(2024, 1, 23, 14, 0).timestamp()
    
    # Reconnaissance phase (low scores, rapid)
    for i in range(5):
        analyzer.add_observation(
            attack_start + i * 30,  # 30 sec intervals
            30 + np.random.randn() * 3,
            {'destination': f'10.0.0.{10+i}'}
        )
    
    # Exploitation phase (medium scores)
    for i in range(3):
        analyzer.add_observation(
            attack_start + 300 + i * 60,
            60 + np.random.randn() * 5,
            {'destination': '10.0.0.50'}
        )
    
    # Data exfiltration (high scores)
    for i in range(2):
        analyzer.add_observation(
            attack_start + 600 + i * 120,
            90 + np.random.randn() * 5,
            {'destination': '10.0.0.50'}
        )
    
    sequence_result = analyzer.detect_sequence_attack(window_minutes=30)
    
    print(f"Sequence Detected: {sequence_result['sequence_detected']}")
    if sequence_result['sequence_detected']:
        print(f"Patterns: {', '.join(sequence_result['patterns'])}")
        print(f"Event Count: {sequence_result['event_count']}")
        print(f"Time Span: {sequence_result['time_span_minutes']:.1f} minutes")
        print(f"Severity: {sequence_result['severity']}")
    
    print("\n✓ Enhanced temporal analyzer working!")
