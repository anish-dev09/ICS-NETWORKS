"""
S7comm Protocol Parser
Siemens S7 Communication Protocol for S7-300/400/1200/1500 PLCs
"""

import struct
import numpy as np
from typing import Dict, List


class S7CommParser:
    """
    S7comm protocol parser for Siemens PLCs.
    Parses TPKT, COTP, and S7 protocol layers.
    """
    
    # Message types
    MESSAGE_TYPES = {
        0x01: "Job Request",
        0x02: "Ack",
        0x03: "Ack_Data",
        0x07: "Userdata"
    }
    
    # Function codes
    FUNCTION_CODES = {
        0x00: "CPU Services",
        0xF0: "Setup Communication",
        0x04: "Read Var",
        0x05: "Write Var",
        0x1A: "Request Download",
        0x1B: "Download Block",
        0x1C: "Download Ended",
        0x1D: "Start Upload",
        0x1E: "Upload",
        0x1F: "End Upload",
        0x28: "PLC Control",
        0x29: "PLC Stop",
        0x2A: "PLC Hot Restart",
        0x2B: "PLC Cold Restart"
    }
    
    # Error classes
    ERROR_CLASSES = {
        0x00: "No Error",
        0x81: "Application relationship error",
        0x82: "Object definition error",
        0x83: "No resources available",
        0x84: "Error on service processing",
        0x85: "Error on supplies",
        0x87: "Access error"
    }
    
    # Dangerous operations
    DANGEROUS_OPERATIONS = [0x28, 0x29, 0x2A, 0x2B, 0x1A, 0x1B]
    
    def __init__(self):
        """Initialize S7comm parser."""
        self.pdu_ref_history = []
        
    def parse_packet(self, packet_bytes: bytes) -> Dict:
        """
        Parse S7comm packet.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            
        Returns:
            dict: Parsed packet information
        """
        try:
            if len(packet_bytes) < 17:
                return {'valid': False, 'error': 'Packet too short', 'protocol': 's7comm'}
            
            # Parse TPKT header (4 bytes)
            tpkt_version = packet_bytes[0]
            tpkt_reserved = packet_bytes[1]
            tpkt_length = struct.unpack('>H', packet_bytes[2:4])[0]
            
            # Parse COTP header (minimum 3 bytes)
            cotp_length = packet_bytes[4]
            cotp_pdu_type = packet_bytes[5]
            cotp_tpdu_number = packet_bytes[6] if len(packet_bytes) > 6 else 0
            
            # Parse S7 header (starts at byte 7)
            s7_protocol_id = packet_bytes[7]
            s7_message_type = packet_bytes[8]
            s7_reserved = struct.unpack('>H', packet_bytes[9:11])[0]
            s7_pdu_ref = struct.unpack('>H', packet_bytes[11:13])[0]
            s7_param_length = struct.unpack('>H', packet_bytes[13:15])[0]
            s7_data_length = struct.unpack('>H', packet_bytes[15:17])[0]
            
            parsed = {
                'valid': True,
                'protocol': 's7comm',
                'tpkt_version': tpkt_version,
                'tpkt_length': tpkt_length,
                'cotp_length': cotp_length,
                'cotp_pdu_type': cotp_pdu_type,
                's7_protocol_id': s7_protocol_id,
                'message_type': s7_message_type,
                'message_type_name': self.MESSAGE_TYPES.get(s7_message_type, 'Unknown'),
                'pdu_ref': s7_pdu_ref,
                'param_length': s7_param_length,
                'data_length': s7_data_length,
                'raw_length': len(packet_bytes)
            }
            
            # Parse function code if present (Job Request/Ack_Data)
            if s7_param_length > 0 and len(packet_bytes) > 17:
                if s7_message_type in [0x01, 0x03]:  # Job Request or Ack_Data
                    function_code = packet_bytes[17]
                    parsed['function_code'] = function_code
                    parsed['function_name'] = self.FUNCTION_CODES.get(function_code, 'Unknown')
                    parsed['is_dangerous'] = function_code in self.DANGEROUS_OPERATIONS
                    
                    # Parse error codes if present
                    if s7_message_type == 0x03 and len(packet_bytes) > 18:
                        error_class = packet_bytes[18]
                        error_code = packet_bytes[19] if len(packet_bytes) > 19 else 0
                        parsed['error_class'] = error_class
                        parsed['error_class_name'] = self.ERROR_CLASSES.get(error_class, 'Unknown')
                        parsed['error_code'] = error_code
            
            # Track PDU references
            self.pdu_ref_history.append(s7_pdu_ref)
            if len(self.pdu_ref_history) > 20:
                self.pdu_ref_history.pop(0)
            
            # Validate
            parsed['anomalies'] = self._validate_packet(parsed, packet_bytes)
            
            return parsed
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'protocol': 's7comm',
                'raw_bytes': packet_bytes.hex() if packet_bytes else ''
            }
    
    def _validate_packet(self, parsed: Dict, packet_bytes: bytes) -> List[str]:
        """Validate S7comm packet."""
        anomalies = []
        
        # Check TPKT version (should be 3)
        if parsed.get('tpkt_version') != 3:
            anomalies.append('invalid_tpkt_version')
        
        # Check length consistency
        if parsed.get('tpkt_length') != len(packet_bytes):
            anomalies.append('tpkt_length_mismatch')
        
        # Check S7 protocol ID (should be 0x32)
        if parsed.get('s7_protocol_id') != 0x32:
            anomalies.append('invalid_s7_protocol_id')
        
        # Check message type validity
        if parsed.get('message_type') not in self.MESSAGE_TYPES:
            anomalies.append('invalid_message_type')
        
        # Check for dangerous functions
        if parsed.get('is_dangerous', False):
            anomalies.append('dangerous_function_detected')
        
        # Check for error responses
        error_class = parsed.get('error_class', 0)
        if error_class != 0:
            anomalies.append('error_response_detected')
        
        # Check PDU reference patterns
        if self._detect_pdu_ref_anomaly():
            anomalies.append('suspicious_pdu_reference_pattern')
        
        # Check param/data length bounds
        param_len = parsed.get('param_length', 0)
        data_len = parsed.get('data_length', 0)
        
        if param_len > 960 or data_len > 960:  # Max PDU is typically 960
            anomalies.append('excessive_parameter_or_data_length')
        
        return anomalies
    
    def _detect_pdu_ref_anomaly(self) -> bool:
        """Detect unusual PDU reference patterns."""
        if len(self.pdu_ref_history) < 5:
            return False
        
        # Check for repeated PDU references (possible replay attack)
        recent = self.pdu_ref_history[-5:]
        if len(set(recent)) < 3:  # Less than 3 unique values in last 5
            return True
        
        return False
    
    def extract_features(self, parsed: Dict) -> np.ndarray:
        """Extract features for ML models."""
        features = []
        
        # TPKT features
        features.append(parsed.get('tpkt_version', 0))
        features.append(parsed.get('tpkt_length', 0))
        
        # COTP features
        features.append(parsed.get('cotp_length', 0))
        features.append(parsed.get('cotp_pdu_type', 0))
        
        # S7 header features
        features.append(parsed.get('message_type', 0))
        features.append(parsed.get('pdu_ref', 0))
        features.append(parsed.get('param_length', 0))
        features.append(parsed.get('data_length', 0))
        
        # Function code features
        features.append(parsed.get('function_code', 0))
        features.append(int(parsed.get('is_dangerous', False)))
        
        # Error features
        features.append(parsed.get('error_class', 0))
        features.append(parsed.get('error_code', 0))
        
        # Anomaly features
        features.append(len(parsed.get('anomalies', [])))
        features.append(parsed.get('raw_length', 0))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            'tpkt_version',
            'tpkt_length',
            'cotp_length',
            'cotp_pdu_type',
            'message_type',
            'pdu_ref',
            'param_length',
            'data_length',
            'function_code',
            'is_dangerous',
            'error_class',
            'error_code',
            'anomaly_count',
            'raw_length'
        ]
    
    def detect_anomalies(self, parsed: Dict) -> Dict:
        """High-level anomaly detection."""
        anomaly_score = 0
        detected_anomalies = parsed.get('anomalies', [])
        
        # Scoring weights
        anomaly_weights = {
            'invalid_tpkt_version': 30,
            'tpkt_length_mismatch': 25,
            'invalid_s7_protocol_id': 50,
            'invalid_message_type': 40,
            'dangerous_function_detected': 70,
            'error_response_detected': 20,
            'suspicious_pdu_reference_pattern': 35,
            'excessive_parameter_or_data_length': 30
        }
        
        for anomaly in detected_anomalies:
            anomaly_score += anomaly_weights.get(anomaly, 10)
        
        # Additional checks
        # Unauthorized write operations
        if parsed.get('function_code') == 0x05:
            anomaly_score += 15
        
        # Block download/upload operations
        if parsed.get('function_code') in [0x1A, 0x1B, 0x1D, 0x1E]:
            anomaly_score += 25
        
        return {
            'is_anomalous': anomaly_score > 50,
            'anomaly_score': min(anomaly_score, 100),
            'detected_anomalies': detected_anomalies,
            'severity': 'critical' if anomaly_score > 80 else 'high' if anomaly_score > 50 else 'medium' if anomaly_score > 20 else 'low',
            'recommendation': self._get_recommendation(anomaly_score, parsed.get('function_code'))
        }
    
    def _get_recommendation(self, score: int, function_code: int = None) -> str:
        """Get recommendation based on score."""
        if score > 80:
            return "BLOCK: Critical S7comm violation. Immediate action required."
        elif score > 50:
            if function_code in [0x28, 0x29]:
                return "ALERT: PLC control operation detected. Verify authorization."
            return "ALERT: High-risk S7comm operation. Manual review recommended."
        elif score > 20:
            return "MONITOR: Moderate risk. Continue monitoring."
        else:
            return "ALLOW: Normal S7comm operation."


if __name__ == "__main__":
    print("Testing S7comm Parser...")
    
    parser = S7CommParser()
    
    # Test packet: S7comm Setup Communication
    test_packet = bytes([
        # TPKT Header
        0x03,        # Version
        0x00,        # Reserved
        0x00, 0x16,  # Length (22 bytes)
        # COTP Header
        0x02,        # Length
        0xF0,        # PDU type (Data)
        0x80,        # TPDU number
        # S7 Header
        0x32,        # Protocol ID
        0x01,        # Message type (Job Request)
        0x00, 0x00,  # Reserved
        0x00, 0x01,  # PDU reference
        0x00, 0x08,  # Parameter length
        0x00, 0x00,  # Data length
        # S7 Parameters
        0xF0,        # Function code (Setup Communication)
        0x00,        # Reserved
        0x00, 0x01,  # Max AMQ (calling)
        0x00, 0x01,  # Max AMQ (called)
        0x03, 0xC0   # PDU length (960)
    ])
    
    parsed = parser.parse_packet(test_packet)
    print("\n=== Parsed Packet ===")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
    
    features = parser.extract_features(parsed)
    print(f"\n=== Extracted Features ===")
    print(f"Values: {features}")
    
    anomaly_result = parser.detect_anomalies(parsed)
    print(f"\n=== Anomaly Detection ===")
    for key, value in anomaly_result.items():
        print(f"  {key}: {value}")
