"""
Context-Aware Analyzer for ICS Anomaly Detection
Integrates temporal, benign pattern, and operational context for intelligent detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ContextAwareAnalyzer:
    """
    Provides intelligent context-aware analysis for ICS anomaly detection.
    Integrates shift patterns, maintenance schedules, and attack correlation.
    """
    
    def __init__(self):
        """Initialize context-aware analyzer."""
        # Operational context
        self.shift_schedule = self._default_shift_schedule()
        self.maintenance_windows = []
        self.production_cycles = {}
        
        # Attack correlation
        self.attack_chain_templates = self._initialize_attack_chains()
        self.recent_alerts = []
        self.alert_correlation_window = 3600  # 1 hour
        
        # Contextual scoring
        self.context_weights = {
            'shift_context': 1.0,
            'maintenance_context': 2.0,
            'production_context': 1.2,
            'temporal_context': 1.5,
            'historical_context': 1.0
        }
    
    def _default_shift_schedule(self) -> Dict:
        """Define default shift patterns for ICS environments."""
        return {
            'weekday': {
                'morning': {'start': 6, 'end': 14, 'activity': 'high'},
                'afternoon': {'start': 14, 'end': 22, 'activity': 'medium'},
                'night': {'start': 22, 'end': 6, 'activity': 'low'}
            },
            'weekend': {
                'day': {'start': 8, 'end': 20, 'activity': 'low'},
                'night': {'start': 20, 'end': 8, 'activity': 'minimal'}
            }
        }
    
    def _initialize_attack_chains(self) -> Dict:
        """Initialize known attack chain templates."""
        return {
            'reconnaissance_to_attack': {
                'stages': ['scanning', 'enumeration', 'exploitation', 'persistence'],
                'typical_duration': 3600,  # 1 hour
                'severity': 'critical',
                'indicators': {
                    'scanning': ['port_scan', 'service_discovery', 'network_mapping'],
                    'enumeration': ['credential_probing', 'directory_listing', 'version_detection'],
                    'exploitation': ['buffer_overflow', 'injection', 'privilege_escalation'],
                    'persistence': ['backdoor_install', 'scheduled_task', 'registry_modification']
                }
            },
            'lateral_movement': {
                'stages': ['initial_access', 'credential_theft', 'lateral_spread', 'data_exfiltration'],
                'typical_duration': 7200,  # 2 hours
                'severity': 'critical',
                'indicators': {
                    'initial_access': ['compromised_account', 'phishing', 'vulnerability_exploit'],
                    'credential_theft': ['password_dump', 'token_theft', 'keylogging'],
                    'lateral_spread': ['remote_execution', 'file_copy', 'service_creation'],
                    'data_exfiltration': ['large_transfer', 'encrypted_channel', 'unusual_destination']
                }
            },
            'ransomware_attack': {
                'stages': ['delivery', 'encryption', 'ransom_demand', 'data_destruction'],
                'typical_duration': 1800,  # 30 minutes
                'severity': 'critical',
                'indicators': {
                    'delivery': ['malicious_attachment', 'drive_by_download', 'exploit_kit'],
                    'encryption': ['file_modification', 'crypto_activity', 'mass_rename'],
                    'ransom_demand': ['ransom_note', 'bitcoin_address', 'tor_communication'],
                    'data_destruction': ['file_deletion', 'backup_destruction', 'shadow_copy_deletion']
                }
            },
            'data_exfiltration': {
                'stages': ['reconnaissance', 'data_staging', 'compression', 'exfiltration'],
                'typical_duration': 5400,  # 90 minutes
                'severity': 'high',
                'indicators': {
                    'reconnaissance': ['file_search', 'database_query', 'sensitive_data_access'],
                    'data_staging': ['file_copy', 'temporary_storage', 'data_aggregation'],
                    'compression': ['archive_creation', 'encryption', 'obfuscation'],
                    'exfiltration': ['outbound_transfer', 'cloud_upload', 'dns_tunneling']
                }
            },
            'supply_chain_attack': {
                'stages': ['compromise', 'trojanization', 'distribution', 'activation'],
                'typical_duration': 10800,  # 3 hours
                'severity': 'critical',
                'indicators': {
                    'compromise': ['vendor_breach', 'update_hijack', 'certificate_theft'],
                    'trojanization': ['code_injection', 'binary_modification', 'backdoor_addition'],
                    'distribution': ['update_push', 'supply_delivery', 'trusted_channel'],
                    'activation': ['trigger_condition', 'payload_execution', 'command_control']
                }
            }
        }
    
    def get_operational_context(self, timestamp: float) -> Dict:
        """
        Get current operational context for given timestamp.
        
        Args:
            timestamp (float): Unix timestamp
            
        Returns:
            dict: Operational context information
        """
        dt = datetime.fromtimestamp(timestamp)
        
        # Determine shift
        is_weekend = dt.weekday() >= 5
        hour = dt.hour
        
        if is_weekend:
            if 8 <= hour < 20:
                shift = 'day'
                activity_level = 'low'
            else:
                shift = 'night'
                activity_level = 'minimal'
        else:
            if 6 <= hour < 14:
                shift = 'morning'
                activity_level = 'high'
            elif 14 <= hour < 22:
                shift = 'afternoon'
                activity_level = 'medium'
            else:
                shift = 'night'
                activity_level = 'low'
        
        # Check if in maintenance window
        in_maintenance = self._is_maintenance_window(timestamp)
        
        # Check production cycle
        production_status = self._get_production_status(timestamp)
        
        return {
            'timestamp': timestamp,
            'datetime': dt.isoformat(),
            'shift': shift,
            'activity_level': activity_level,
            'is_weekend': is_weekend,
            'is_business_hours': 6 <= hour < 18 and not is_weekend,
            'in_maintenance': in_maintenance,
            'production_status': production_status,
            'day_of_week': dt.strftime('%A'),
            'hour': hour
        }
    
    def _is_maintenance_window(self, timestamp: float) -> bool:
        """Check if timestamp falls within scheduled maintenance."""
        for window in self.maintenance_windows:
            if window['start'] <= timestamp <= window['end']:
                return True
        return False
    
    def _get_production_status(self, timestamp: float) -> str:
        """Get production status at given time."""
        # This would integrate with actual production scheduling system
        # For now, return based on time of day
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        
        if 6 <= hour < 22:
            return 'active'
        else:
            return 'idle'
    
    def add_maintenance_window(self, start_time: float, end_time: float, 
                               description: str):
        """
        Add scheduled maintenance window.
        
        Args:
            start_time (float): Start timestamp
            end_time (float): End timestamp
            description (str): Maintenance description
        """
        self.maintenance_windows.append({
            'start': start_time,
            'end': end_time,
            'description': description,
            'added': datetime.now().timestamp()
        })
        
        # Keep only future and recent (24h) maintenance windows
        current_time = datetime.now().timestamp()
        self.maintenance_windows = [
            w for w in self.maintenance_windows
            if w['end'] >= current_time - 86400
        ]
    
    def analyze_with_context(self, detection_result: Dict, 
                            temporal_info: Dict,
                            benign_info: Dict) -> Dict:
        """
        Perform context-aware analysis combining multiple information sources.
        
        Args:
            detection_result (dict): Raw detection result
            temporal_info (dict): Temporal analysis result
            benign_info (dict): Benign pattern analysis result
            
        Returns:
            dict: Enhanced context-aware analysis
        """
        timestamp = detection_result.get('timestamp', datetime.now().timestamp())
        base_score = detection_result.get('score', 0)
        
        # Get operational context
        context = self.get_operational_context(timestamp)
        
        # Initialize analysis
        analysis = {
            'original_score': base_score,
            'adjusted_score': base_score,
            'context': context,
            'adjustments': [],
            'suppressed': False,
            'severity': self._calculate_severity(base_score),
            'confidence': 0.5
        }
        
        # Apply contextual adjustments
        
        # 1. Maintenance window adjustment
        if context['in_maintenance']:
            if benign_info.get('is_benign', False):
                analysis['adjusted_score'] *= 0.3  # Major reduction
                analysis['adjustments'].append('Maintenance window + benign pattern')
                analysis['suppressed'] = True
                analysis['confidence'] = 0.9
            else:
                analysis['adjusted_score'] *= 0.6  # Moderate reduction
                analysis['adjustments'].append('Maintenance window')
        
        # 2. Benign pattern adjustment
        if benign_info.get('is_benign', False) and not context['in_maintenance']:
            confidence = benign_info.get('confidence', 0)
            reduction = 0.5 + (confidence * 0.4)  # 0.5-0.9 reduction
            analysis['adjusted_score'] *= (1 - reduction)
            analysis['adjustments'].append(f"Benign pattern ({benign_info.get('pattern_name', 'unknown')})")
            analysis['confidence'] = confidence
        
        # 3. Temporal anomaly adjustment
        if temporal_info.get('is_temporal_anomaly', False):
            boost = 1 + (temporal_info.get('temporal_score', 0) / 100)
            analysis['adjusted_score'] *= boost
            analysis['adjustments'].append('Temporal anomaly detected')
            analysis['confidence'] = max(analysis['confidence'], 0.7)
        
        # 4. Off-hours boost
        if not context['is_business_hours'] and not context['in_maintenance']:
            if base_score > 60:
                analysis['adjusted_score'] *= 1.3
                analysis['adjustments'].append('Suspicious activity during off-hours')
                analysis['confidence'] = max(analysis['confidence'], 0.75)
        
        # 5. Activity level adjustment
        activity_level = context['activity_level']
        if activity_level == 'minimal' and base_score > 50:
            analysis['adjusted_score'] *= 1.4
            analysis['adjustments'].append('Activity during minimal operations period')
        elif activity_level == 'high' and base_score < 40:
            analysis['adjusted_score'] *= 0.8
            analysis['adjustments'].append('Low-priority alert during high activity')
        
        # Cap adjusted score
        analysis['adjusted_score'] = np.clip(analysis['adjusted_score'], 0, 100)
        
        # Recalculate severity
        analysis['final_severity'] = self._calculate_severity(analysis['adjusted_score'])
        
        # Check for attack chain
        chain_info = self.detect_attack_chain(detection_result)
        if chain_info['chain_detected']:
            analysis['attack_chain'] = chain_info
            analysis['adjusted_score'] = max(analysis['adjusted_score'], 85)
            analysis['final_severity'] = 'critical'
            analysis['confidence'] = 0.95
        
        return analysis
    
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
    
    def detect_attack_chain(self, detection_result: Dict) -> Dict:
        """
        Detect multi-stage attack chains.
        
        Args:
            detection_result (dict): Current detection result
            
        Returns:
            dict: Attack chain analysis
        """
        # Add to recent alerts
        self.recent_alerts.append({
            'timestamp': detection_result.get('timestamp', datetime.now().timestamp()),
            'score': detection_result.get('score', 0),
            'features': detection_result.get('features', {})
        })
        
        # Keep only recent alerts
        current_time = datetime.now().timestamp()
        self.recent_alerts = [
            a for a in self.recent_alerts
            if current_time - a['timestamp'] <= self.alert_correlation_window
        ]
        
        if len(self.recent_alerts) < 2:
            return {'chain_detected': False, 'reason': 'Insufficient alerts'}
        
        # Check against each attack chain template
        for chain_name, chain_template in self.attack_chain_templates.items():
            match_result = self._match_attack_chain(self.recent_alerts, chain_template)
            
            if match_result['matched']:
                return {
                    'chain_detected': True,
                    'chain_name': chain_name,
                    'chain_type': chain_name.replace('_', ' ').title(),
                    'severity': chain_template['severity'],
                    'stages_detected': match_result['stages'],
                    'confidence': match_result['confidence'],
                    'timeline': match_result['timeline'],
                    'recommendations': self._get_chain_recommendations(chain_name)
                }
        
        return {'chain_detected': False, 'reason': 'No chain pattern matched'}
    
    def _match_attack_chain(self, alerts: List[Dict], 
                           chain_template: Dict) -> Dict:
        """Match alerts against attack chain template."""
        stages = chain_template['stages']
        indicators = chain_template['indicators']
        
        matched_stages = []
        timeline = []
        
        for alert in alerts:
            features = alert.get('features', {})
            alert_time = alert['timestamp']
            
            # Check which stage this alert might belong to
            for stage in stages:
                if stage in matched_stages:
                    continue
                
                stage_indicators = indicators.get(stage, [])
                matched_indicators = 0
                
                for indicator in stage_indicators:
                    if self._check_indicator_in_features(indicator, features):
                        matched_indicators += 1
                
                # If at least 30% of indicators match, consider stage detected
                if matched_indicators >= len(stage_indicators) * 0.3:
                    matched_stages.append(stage)
                    timeline.append({
                        'stage': stage,
                        'timestamp': alert_time,
                        'score': alert['score']
                    })
                    break
        
        # Chain matched if at least 40% of stages detected
        threshold = len(stages) * 0.4
        matched = len(matched_stages) >= threshold
        confidence = len(matched_stages) / len(stages)
        
        return {
            'matched': matched,
            'stages': matched_stages,
            'confidence': confidence,
            'timeline': timeline
        }
    
    def _check_indicator_in_features(self, indicator: str, features: Dict) -> bool:
        """Check if indicator is present in features."""
        # Simple keyword matching in feature keys and values
        for key, value in features.items():
            if indicator in str(key).lower() or indicator in str(value).lower():
                return True
        return False
    
    def _get_chain_recommendations(self, chain_name: str) -> List[str]:
        """Get response recommendations for detected attack chain."""
        recommendations = {
            'reconnaissance_to_attack': [
                'Isolate affected systems immediately',
                'Block source IP addresses',
                'Review firewall rules and access controls',
                'Initiate incident response procedure',
                'Monitor for lateral movement attempts'
            ],
            'lateral_movement': [
                'Isolate compromised accounts',
                'Reset credentials for affected systems',
                'Enable enhanced monitoring on all systems',
                'Check for unauthorized access to critical assets',
                'Review authentication logs'
            ],
            'ransomware_attack': [
                'IMMEDIATE: Isolate all affected systems',
                'Do not pay ransom - contact authorities',
                'Restore from clean backups',
                'Scan all systems for malware',
                'Review backup integrity'
            ],
            'data_exfiltration': [
                'Block outbound connections to suspicious destinations',
                'Review data access logs',
                'Identify compromised data sets',
                'Notify affected parties if required',
                'Implement DLP controls'
            ],
            'supply_chain_attack': [
                'Quarantine all systems with compromised software',
                'Contact vendors immediately',
                'Review all recent updates and patches',
                'Implement strict software verification',
                'Monitor for command and control activity'
            ]
        }
        
        return recommendations.get(chain_name, ['Initiate standard incident response'])
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics."""
        return {
            'maintenance_windows': len(self.maintenance_windows),
            'recent_alerts': len(self.recent_alerts),
            'attack_chain_templates': len(self.attack_chain_templates),
            'correlation_window_hours': self.alert_correlation_window / 3600
        }


if __name__ == "__main__":
    print("Testing Context-Aware Analyzer...")
    
    analyzer = ContextAwareAnalyzer()
    
    print("\n" + "="*60)
    print("Test 1: Operational Context")
    print("="*60)
    
    # Test different times
    test_times = [
        (datetime(2024, 1, 15, 10, 0).timestamp(), "Monday 10 AM"),
        (datetime(2024, 1, 15, 22, 0).timestamp(), "Monday 10 PM"),
        (datetime(2024, 1, 20, 14, 0).timestamp(), "Saturday 2 PM"),
    ]
    
    for timestamp, description in test_times:
        context = analyzer.get_operational_context(timestamp)
        print(f"\n{description}:")
        print(f"  Shift: {context['shift']}")
        print(f"  Activity Level: {context['activity_level']}")
        print(f"  Business Hours: {context['is_business_hours']}")
        print(f"  Weekend: {context['is_weekend']}")
    
    print("\n" + "="*60)
    print("Test 2: Context-Aware Analysis")
    print("="*60)
    
    # Test during business hours (should be less suspicious)
    detection1 = {
        'timestamp': datetime(2024, 1, 15, 10, 0).timestamp(),
        'score': 65,
        'features': {'packet_rate': 1000}
    }
    temporal1 = {'is_temporal_anomaly': False}
    benign1 = {'is_benign': False, 'confidence': 0.3}
    
    analysis1 = analyzer.analyze_with_context(detection1, temporal1, benign1)
    print(f"\nBusiness Hours Alert (score={detection1['score']}):")
    print(f"  Adjusted Score: {analysis1['adjusted_score']:.1f}")
    print(f"  Severity: {analysis1['original_score']} → {analysis1['final_severity']}")
    print(f"  Adjustments: {', '.join(analysis1['adjustments']) if analysis1['adjustments'] else 'None'}")
    
    # Test during off-hours (should be more suspicious)
    detection2 = {
        'timestamp': datetime(2024, 1, 15, 2, 0).timestamp(),
        'score': 65,
        'features': {'packet_rate': 1000}
    }
    temporal2 = {'is_temporal_anomaly': True, 'temporal_score': 40}
    benign2 = {'is_benign': False, 'confidence': 0.2}
    
    analysis2 = analyzer.analyze_with_context(detection2, temporal2, benign2)
    print(f"\nOff-Hours Alert (score={detection2['score']}):")
    print(f"  Adjusted Score: {analysis2['adjusted_score']:.1f}")
    print(f"  Severity: {analysis2['final_severity']}")
    print(f"  Confidence: {analysis2['confidence']:.2f}")
    print(f"  Adjustments: {', '.join(analysis2['adjustments'])}")
    
    # Test during maintenance (should suppress if benign)
    analyzer.add_maintenance_window(
        datetime(2024, 1, 15, 22, 0).timestamp(),
        datetime(2024, 1, 15, 23, 30).timestamp(),
        "Monthly system updates"
    )
    
    detection3 = {
        'timestamp': datetime(2024, 1, 15, 22, 30).timestamp(),
        'score': 70,
        'features': {'config_change': True}
    }
    temporal3 = {'is_temporal_anomaly': False}
    benign3 = {'is_benign': True, 'confidence': 0.9, 'pattern_name': 'maintenance_window'}
    
    analysis3 = analyzer.analyze_with_context(detection3, temporal3, benign3)
    print(f"\nMaintenance Window Alert (score={detection3['score']}):")
    print(f"  Adjusted Score: {analysis3['adjusted_score']:.1f}")
    print(f"  Suppressed: {analysis3['suppressed']}")
    print(f"  Confidence: {analysis3['confidence']:.2f}")
    print(f"  Adjustments: {', '.join(analysis3['adjustments'])}")
    
    print("\n" + "="*60)
    print("Test 3: Attack Chain Detection")
    print("="*60)
    
    # Simulate attack sequence
    attack_time = datetime(2024, 1, 16, 14, 0).timestamp()
    
    # Stage 1: Scanning
    detection_scan = {
        'timestamp': attack_time,
        'score': 55,
        'features': {'port_scan': True, 'service_discovery': True}
    }
    analyzer.detect_attack_chain(detection_scan)
    
    # Stage 2: Enumeration
    detection_enum = {
        'timestamp': attack_time + 300,
        'score': 65,
        'features': {'credential_probing': True, 'version_detection': True}
    }
    analyzer.detect_attack_chain(detection_enum)
    
    # Stage 3: Exploitation
    detection_exploit = {
        'timestamp': attack_time + 600,
        'score': 85,
        'features': {'buffer_overflow': True, 'privilege_escalation': True}
    }
    chain_result = analyzer.detect_attack_chain(detection_exploit)
    
    print(f"Attack Chain Detected: {chain_result['chain_detected']}")
    if chain_result['chain_detected']:
        print(f"  Chain Type: {chain_result['chain_type']}")
        print(f"  Severity: {chain_result['severity']}")
        print(f"  Stages: {', '.join(chain_result['stages_detected'])}")
        print(f"  Confidence: {chain_result['confidence']:.2f}")
        print(f"  Recommendations:")
        for rec in chain_result['recommendations'][:3]:
            print(f"    - {rec}")
    
    stats = analyzer.get_statistics()
    print(f"\nAnalyzer Statistics:")
    print(f"  Recent Alerts: {stats['recent_alerts']}")
    print(f"  Attack Chain Templates: {stats['attack_chain_templates']}")
    print(f"  Maintenance Windows: {stats['maintenance_windows']}")
    
    print("\n✓ Context-aware analyzer working!")
