"""
Encrypted Traffic Anomaly Detection for ICS Networks
Handles TLS/SSL encrypted ICS traffic using metadata and flow analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import struct


class TLSFingerprinter:
    """
    Extract features from TLS handshake without decryption.
    Uses JA3/JA3S fingerprinting for client/server identification.
    """
    
    # TLS Record Types
    RECORD_TYPES = {
        0x14: 'ChangeCipherSpec',
        0x15: 'Alert',
        0x16: 'Handshake',
        0x17: 'ApplicationData'
    }
    
    # TLS Handshake Types
    HANDSHAKE_TYPES = {
        0x01: 'ClientHello',
        0x02: 'ServerHello',
        0x0B: 'Certificate',
        0x10: 'ClientKeyExchange',
        0x14: 'Finished'
    }
    
    def __init__(self):
        """Initialize TLS fingerprinter."""
        self.handshake_history = deque(maxlen=50)
    
    def is_tls_traffic(self, packet_bytes: bytes) -> bool:
        """
        Check if packet contains TLS traffic.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            
        Returns:
            bool: True if TLS traffic detected
        """
        if len(packet_bytes) < 5:
            return False
        
        # Check for TLS record header (0x16 = Handshake, 0x17 = Application Data)
        content_type = packet_bytes[0]
        
        # TLS versions: 0x0301 (TLS 1.0), 0x0302 (TLS 1.1), 0x0303 (TLS 1.2), 0x0304 (TLS 1.3)
        if len(packet_bytes) >= 3:
            version = struct.unpack('>H', packet_bytes[1:3])[0]
            if content_type in [0x14, 0x15, 0x16, 0x17] and version in [0x0301, 0x0302, 0x0303, 0x0304]:
                return True
        
        return False
    
    def extract_tls_features(self, packet_bytes: bytes) -> Dict:
        """
        Extract TLS metadata features without decryption.
        
        Args:
            packet_bytes (bytes): Raw TLS packet
            
        Returns:
            dict: TLS features
        """
        if len(packet_bytes) < 5:
            return {'valid': False, 'error': 'Packet too short'}
        
        features = {
            'valid': True,
            'is_encrypted': True,
            'content_type': packet_bytes[0],
            'content_type_name': self.RECORD_TYPES.get(packet_bytes[0], 'Unknown'),
            'tls_version': struct.unpack('>H', packet_bytes[1:3])[0],
            'record_length': struct.unpack('>H', packet_bytes[3:5])[0],
            'total_length': len(packet_bytes)
        }
        
        # Extract handshake-specific features
        if features['content_type'] == 0x16 and len(packet_bytes) > 5:  # Handshake
            handshake_type = packet_bytes[5]
            features['handshake_type'] = handshake_type
            features['handshake_type_name'] = self.HANDSHAKE_TYPES.get(handshake_type, 'Unknown')
            
            # Extract ClientHello/ServerHello details
            if handshake_type in [0x01, 0x02]:  # ClientHello or ServerHello
                features.update(self._parse_hello_message(packet_bytes[5:]))
        
        # Check for suspicious patterns
        features['anomalies'] = self._detect_tls_anomalies(features)
        
        self.handshake_history.append(features)
        
        return features
    
    def _parse_hello_message(self, hello_bytes: bytes) -> Dict:
        """Parse ClientHello or ServerHello for fingerprinting."""
        result = {}
        
        try:
            if len(hello_bytes) < 40:
                return result
            
            # Handshake length (3 bytes)
            handshake_length = struct.unpack('>I', b'\x00' + hello_bytes[1:4])[0]
            result['handshake_length'] = handshake_length
            
            # TLS version in handshake
            result['handshake_version'] = struct.unpack('>H', hello_bytes[4:6])[0]
            
            # Random (32 bytes)
            result['random'] = hello_bytes[6:38].hex()
            
            # Session ID length
            if len(hello_bytes) > 38:
                session_id_length = hello_bytes[38]
                result['session_id_length'] = session_id_length
                
                offset = 39 + session_id_length
                
                # Cipher suites
                if len(hello_bytes) > offset + 2:
                    cipher_suites_length = struct.unpack('>H', hello_bytes[offset:offset+2])[0]
                    result['cipher_suites_length'] = cipher_suites_length
                    result['cipher_suite_count'] = cipher_suites_length // 2
                    
                    # Extract cipher suite IDs (for JA3 fingerprinting)
                    cipher_offset = offset + 2
                    cipher_suites = []
                    for i in range(0, min(cipher_suites_length, 100), 2):
                        if cipher_offset + i + 2 <= len(hello_bytes):
                            cipher_id = struct.unpack('>H', hello_bytes[cipher_offset+i:cipher_offset+i+2])[0]
                            cipher_suites.append(cipher_id)
                    
                    result['cipher_suites'] = cipher_suites
                    result['has_weak_ciphers'] = any(cs in self._get_weak_ciphers() for cs in cipher_suites)
        
        except Exception as e:
            result['parse_error'] = str(e)
        
        return result
    
    def _get_weak_ciphers(self) -> List[int]:
        """List of known weak cipher suites."""
        return [
            0x0004,  # TLS_RSA_WITH_RC4_128_MD5
            0x0005,  # TLS_RSA_WITH_RC4_128_SHA
            0x000A,  # TLS_RSA_WITH_3DES_EDE_CBC_SHA
            0x002F,  # TLS_RSA_WITH_AES_128_CBC_SHA (vulnerable to BEAST)
        ]
    
    def _detect_tls_anomalies(self, features: Dict) -> List[str]:
        """Detect anomalies in TLS handshake."""
        anomalies = []
        
        # Check for weak ciphers
        if features.get('has_weak_ciphers'):
            anomalies.append('weak_cipher_suite')
        
        # Check for suspicious record length
        if features.get('record_length', 0) > 16384:  # Max TLS record is 16KB
            anomalies.append('oversized_tls_record')
        
        # Check for version downgrade
        tls_version = features.get('tls_version', 0)
        handshake_version = features.get('handshake_version', 0)
        if handshake_version and tls_version and handshake_version < tls_version:
            anomalies.append('version_downgrade_attempt')
        
        # Check for unusual cipher suite count
        cipher_count = features.get('cipher_suite_count', 0)
        if cipher_count > 50 or cipher_count == 0:
            anomalies.append('unusual_cipher_suite_count')
        
        # Check handshake sequence
        if len(self.handshake_history) >= 3:
            recent_types = [h.get('handshake_type') for h in list(self.handshake_history)[-3:]]
            # Normal sequence: ClientHello -> ServerHello -> Certificate
            if recent_types.count(0x01) > 2:  # Too many ClientHellos
                anomalies.append('repeated_client_hello')
        
        return anomalies
    
    def compute_ja3_fingerprint(self, features: Dict) -> Optional[str]:
        """
        Compute JA3 fingerprint (simplified version).
        JA3 = MD5(TLSVersion, CipherSuites, Extensions, EllipticCurves, ECPointFormats)
        
        Returns:
            str: JA3 hash or None
        """
        if not features.get('cipher_suites'):
            return None
        
        # Create fingerprint string
        tls_version = features.get('handshake_version', 0)
        cipher_suites = ','.join(str(c) for c in features['cipher_suites'])
        
        fingerprint_str = f"{tls_version},{cipher_suites}"
        
        # Simple hash (in production, use MD5)
        import hashlib
        return hashlib.md5(fingerprint_str.encode()).hexdigest()


class FlowAnalyzer:
    """
    Analyze traffic flows for encrypted ICS communications.
    Uses statistical features without packet content inspection.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize flow analyzer.
        
        Args:
            window_size (int): Number of packets to analyze in sliding window
        """
        self.window_size = window_size
        self.packet_history = deque(maxlen=window_size)
        self.flow_stats = {}
    
    def add_packet(self, packet_info: Dict):
        """
        Add packet to flow analysis.
        
        Args:
            packet_info (dict): Must contain 'timestamp', 'size', 'direction', 'src', 'dst'
        """
        self.packet_history.append(packet_info)
        
        # Update flow statistics
        flow_key = self._get_flow_key(packet_info)
        if flow_key not in self.flow_stats:
            self.flow_stats[flow_key] = {
                'packet_count': 0,
                'byte_count': 0,
                'start_time': packet_info['timestamp'],
                'last_seen': packet_info['timestamp']
            }
        
        self.flow_stats[flow_key]['packet_count'] += 1
        self.flow_stats[flow_key]['byte_count'] += packet_info['size']
        self.flow_stats[flow_key]['last_seen'] = packet_info['timestamp']
    
    def _get_flow_key(self, packet_info: Dict) -> Tuple:
        """Create flow identifier from 5-tuple."""
        return (
            packet_info.get('src', ''),
            packet_info.get('dst', ''),
            packet_info.get('src_port', 0),
            packet_info.get('dst_port', 0),
            packet_info.get('protocol', 'TCP')
        )
    
    def extract_flow_features(self) -> np.ndarray:
        """
        Extract statistical features from packet flow.
        
        Returns:
            np.ndarray: 20-dimensional feature vector
        """
        if len(self.packet_history) == 0:
            return np.zeros(20, dtype=np.float32)
        
        packets = list(self.packet_history)
        sizes = [p['size'] for p in packets]
        
        # Extract inter-arrival times
        if len(packets) > 1:
            timestamps = [p['timestamp'] for p in packets]
            inter_arrival_times = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        else:
            inter_arrival_times = [0]
        
        features = np.array([
            # Packet size statistics
            np.mean(sizes),
            np.std(sizes),
            np.min(sizes),
            np.max(sizes),
            np.median(sizes),
            
            # Timing statistics
            np.mean(inter_arrival_times),
            np.std(inter_arrival_times),
            np.min(inter_arrival_times),
            np.max(inter_arrival_times),
            
            # Flow characteristics
            len(packets),
            sum(sizes),
            len(self.flow_stats),  # Number of active flows
            
            # Direction statistics
            sum(1 for p in packets if p.get('direction') == 'outbound'),
            sum(1 for p in packets if p.get('direction') == 'inbound'),
            
            # Burst detection
            self._compute_burst_score(sizes),
            self._compute_periodicity_score(inter_arrival_times),
            
            # Size distribution
            np.percentile(sizes, 25),
            np.percentile(sizes, 75),
            
            # Timing distribution
            np.percentile(inter_arrival_times, 75) if inter_arrival_times else 0,
            len(set(sizes)) / len(sizes) if sizes else 0  # Size diversity
        ], dtype=np.float32)
        
        return features
    
    def _compute_burst_score(self, sizes: List[int]) -> float:
        """Detect bursty traffic patterns."""
        if len(sizes) < 10:
            return 0.0
        
        # Compare variance in sliding windows
        window = 5
        variances = []
        for i in range(len(sizes) - window):
            window_variance = np.var(sizes[i:i+window])
            variances.append(window_variance)
        
        if variances:
            return np.max(variances) / (np.mean(sizes) ** 2) if np.mean(sizes) > 0 else 0
        return 0.0
    
    def _compute_periodicity_score(self, inter_arrival_times: List[float]) -> float:
        """Detect periodic communication patterns."""
        if len(inter_arrival_times) < 10:
            return 0.0
        
        # Use autocorrelation to detect periodicity
        times = np.array(inter_arrival_times)
        if len(times) < 2:
            return 0.0
        
        mean = np.mean(times)
        std = np.std(times)
        
        if std == 0:
            return 1.0  # Perfectly periodic
        
        # Coefficient of variation (lower = more periodic)
        cv = std / mean if mean > 0 else 0
        return np.exp(-cv)  # Convert to score (higher = more periodic)
    
    def detect_anomalies(self) -> Dict:
        """
        Detect anomalies in traffic flow.
        
        Returns:
            dict: Anomaly detection results
        """
        anomalies = []
        anomaly_score = 0
        
        if len(self.packet_history) == 0:
            return {
                'is_anomalous': False,
                'anomaly_score': 0,
                'detected_anomalies': [],
                'severity': 'low'
            }
        
        packets = list(self.packet_history)
        sizes = [p['size'] for p in packets]
        
        # Check for unusual packet sizes
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        if std_size > mean_size * 2:  # High variance
            anomalies.append('high_size_variance')
            anomaly_score += 20
        
        # Check for suspiciously small packets (possible covert channel)
        small_packets = sum(1 for s in sizes if s < 100)
        if small_packets > len(sizes) * 0.5:
            anomalies.append('many_small_packets')
            anomaly_score += 30
        
        # Check for flood patterns
        if len(packets) > self.window_size * 0.9:
            anomalies.append('potential_flood')
            anomaly_score += 40
        
        # Check flow statistics
        for flow_key, stats in self.flow_stats.items():
            duration = stats['last_seen'] - stats['start_time']
            if duration > 0:
                pps = stats['packet_count'] / duration
                if pps > 1000:  # More than 1000 packets/sec
                    anomalies.append('high_packet_rate')
                    anomaly_score += 35
        
        return {
            'is_anomalous': anomaly_score > 40,
            'anomaly_score': min(anomaly_score, 100),
            'detected_anomalies': anomalies,
            'severity': 'critical' if anomaly_score > 70 else 'high' if anomaly_score > 40 else 'medium' if anomaly_score > 20 else 'low',
            'recommendation': self._get_recommendation(anomaly_score)
        }
    
    def _get_recommendation(self, score: int) -> str:
        """Get recommendation based on anomaly score."""
        if score > 70:
            return "ALERT: Critical flow anomaly detected. Possible encrypted attack."
        elif score > 40:
            return "MONITOR: Suspicious traffic patterns in encrypted flow."
        elif score > 20:
            return "WATCH: Unusual encrypted traffic characteristics."
        else:
            return "NORMAL: Encrypted traffic within expected parameters."


class EncryptedTrafficDetector:
    """
    Combined detector for encrypted ICS traffic.
    Uses TLS fingerprinting + flow analysis for anomaly detection.
    """
    
    def __init__(self):
        """Initialize encrypted traffic detector."""
        self.tls_fingerprinter = TLSFingerprinter()
        self.flow_analyzer = FlowAnalyzer()
        self.detection_mode = 'encrypted'
    
    def process_packet(self, packet_bytes: bytes, packet_info: Dict) -> Dict:
        """
        Process encrypted packet and extract features.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            packet_info (dict): Metadata (timestamp, size, direction, src, dst, ports)
            
        Returns:
            dict: Detection results
        """
        result = {
            'is_encrypted': False,
            'tls_features': None,
            'flow_features': None,
            'combined_anomaly_score': 0
        }
        
        # Check if TLS traffic
        if self.tls_fingerprinter.is_tls_traffic(packet_bytes):
            result['is_encrypted'] = True
            
            # Extract TLS features
            tls_features = self.tls_fingerprinter.extract_tls_features(packet_bytes)
            result['tls_features'] = tls_features
            
            # TLS anomaly score
            tls_anomaly_count = len(tls_features.get('anomalies', []))
            tls_score = min(tls_anomaly_count * 25, 100)
        else:
            tls_score = 0
        
        # Add to flow analysis
        self.flow_analyzer.add_packet(packet_info)
        
        # Flow-based detection
        flow_anomalies = self.flow_analyzer.detect_anomalies()
        result['flow_anomalies'] = flow_anomalies
        flow_score = flow_anomalies['anomaly_score']
        
        # Combined scoring (weighted)
        result['combined_anomaly_score'] = 0.4 * tls_score + 0.6 * flow_score
        result['is_anomalous'] = result['combined_anomaly_score'] > 50
        
        return result
    
    def extract_ml_features(self, packet_bytes: bytes, packet_info: Dict) -> np.ndarray:
        """
        Extract features for machine learning models.
        
        Args:
            packet_bytes (bytes): Raw packet
            packet_info (dict): Packet metadata
            
        Returns:
            np.ndarray: 35-dimensional feature vector
        """
        # Add packet to flow
        self.flow_analyzer.add_packet(packet_info)
        
        # Get flow features (20 dimensions)
        flow_features = self.flow_analyzer.extract_flow_features()
        
        # Get TLS features if available (15 dimensions)
        tls_features = np.zeros(15, dtype=np.float32)
        if self.tls_fingerprinter.is_tls_traffic(packet_bytes):
            tls_info = self.tls_fingerprinter.extract_tls_features(packet_bytes)
            
            tls_features = np.array([
                tls_info.get('content_type', 0),
                tls_info.get('tls_version', 0),
                tls_info.get('record_length', 0),
                tls_info.get('total_length', 0),
                tls_info.get('handshake_type', 0),
                tls_info.get('handshake_length', 0),
                tls_info.get('handshake_version', 0),
                tls_info.get('session_id_length', 0),
                tls_info.get('cipher_suite_count', 0),
                tls_info.get('cipher_suites_length', 0),
                1 if tls_info.get('has_weak_ciphers') else 0,
                len(tls_info.get('anomalies', [])),
                1 if 'weak_cipher_suite' in tls_info.get('anomalies', []) else 0,
                1 if 'version_downgrade_attempt' in tls_info.get('anomalies', []) else 0,
                1 if 'oversized_tls_record' in tls_info.get('anomalies', []) else 0
            ], dtype=np.float32)
        
        # Combine features
        combined = np.concatenate([flow_features, tls_features])
        
        return combined
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for ML models."""
        flow_names = [
            'mean_packet_size', 'std_packet_size', 'min_packet_size', 'max_packet_size',
            'median_packet_size', 'mean_iat', 'std_iat', 'min_iat', 'max_iat',
            'packet_count', 'byte_count', 'flow_count', 'outbound_count', 'inbound_count',
            'burst_score', 'periodicity_score', 'size_p25', 'size_p75', 'iat_p75', 'size_diversity'
        ]
        
        tls_names = [
            'tls_content_type', 'tls_version', 'tls_record_length', 'tls_total_length',
            'tls_handshake_type', 'tls_handshake_length', 'tls_handshake_version',
            'tls_session_id_length', 'tls_cipher_count', 'tls_cipher_length',
            'tls_has_weak_ciphers', 'tls_anomaly_count', 'tls_weak_cipher_flag',
            'tls_version_downgrade_flag', 'tls_oversized_record_flag'
        ]
        
        return flow_names + tls_names


if __name__ == "__main__":
    print("Testing Encrypted Traffic Detection...")
    
    detector = EncryptedTrafficDetector()
    
    # Simulate TLS ClientHello packet (simplified)
    tls_packet = bytes([
        0x16,  # Content Type: Handshake
        0x03, 0x03,  # TLS 1.2
        0x00, 0x50,  # Length: 80 bytes
        0x01,  # Handshake Type: ClientHello
        0x00, 0x00, 0x4C,  # Handshake Length
        0x03, 0x03,  # Client Version: TLS 1.2
        # Random (32 bytes)
        *[0xFF] * 32,
        0x00,  # Session ID length
        0x00, 0x04,  # Cipher suites length: 4 bytes (2 suites)
        0xC0, 0x2F,  # Cipher suite 1
        0xC0, 0x30,  # Cipher suite 2
        0x01, 0x00  # Compression methods
    ])
    
    packet_info = {
        'timestamp': 1000.0,
        'size': len(tls_packet),
        'direction': 'outbound',
        'src': '192.168.1.100',
        'dst': '10.0.0.50',
        'src_port': 50000,
        'dst_port': 502,
        'protocol': 'TCP'
    }
    
    print("\n" + "="*60)
    print("Test 1: TLS Encrypted Packet")
    print("="*60)
    
    result = detector.process_packet(tls_packet, packet_info)
    print(f"Is Encrypted: {result['is_encrypted']}")
    print(f"Is Anomalous: {result['is_anomalous']}")
    print(f"Combined Score: {result['combined_anomaly_score']:.2f}")
    
    if result['tls_features']:
        print(f"\nTLS Features:")
        print(f"  Content Type: {result['tls_features']['content_type_name']}")
        print(f"  Version: {hex(result['tls_features']['tls_version'])}")
        print(f"  Handshake: {result['tls_features'].get('handshake_type_name', 'N/A')}")
        print(f"  Anomalies: {result['tls_features'].get('anomalies', [])}")
    
    # Extract ML features
    features = detector.extract_ml_features(tls_packet, packet_info)
    print(f"\nML Features Shape: {features.shape}")
    print(f"Feature Vector: {features[:10]}...")
    
    print("\n✓ Encrypted traffic detection working!")
