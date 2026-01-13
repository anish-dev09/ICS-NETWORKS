"""
DNP3 (Distributed Network Protocol 3) Parser
Used in SCADA systems for electric utilities and water/wastewater systems.
"""

import struct
import numpy as np
from typing import Dict, List


class DNP3Parser:
    """
    DNP3 protocol parser with validation.
    Supports data link layer and application layer parsing.
    """
    
    # DNP3 function codes
    FUNCTION_CODES = {
        0x00: "CONFIRM",
        0x01: "READ",
        0x02: "WRITE",
        0x03: "SELECT",
        0x04: "OPERATE",
        0x05: "DIRECT_OPERATE",
        0x06: "DIRECT_OPERATE_NO_ACK",
        0x07: "IMMEDIATE_FREEZE",
        0x08: "IMMEDIATE_FREEZE_NO_ACK",
        0x09: "FREEZE_CLEAR",
        0x0A: "FREEZE_CLEAR_NO_ACK",
        0x0B: "FREEZE_AT_TIME",
        0x0C: "FREEZE_AT_TIME_NO_ACK",
        0x0D: "COLD_RESTART",
        0x0E: "WARM_RESTART",
        0x0F: "INITIALIZE_DATA",
        0x10: "INITIALIZE_APPLICATION",
        0x11: "START_APPLICATION",
        0x12: "STOP_APPLICATION",
        0x13: "SAVE_CONFIGURATION",
        0x14: "ENABLE_UNSOLICITED",
        0x15: "DISABLE_UNSOLICITED",
        0x16: "ASSIGN_CLASS",
        0x17: "DELAY_MEASUREMENT",
        0x18: "RECORD_CURRENT_TIME",
        0x19: "OPEN_FILE",
        0x1A: "CLOSE_FILE",
        0x1B: "DELETE_FILE",
        0x81: "RESPONSE",
        0x82: "UNSOLICITED_RESPONSE"
    }
    
    # Object groups
    OBJECT_GROUPS = {
        1: "Binary Input",
        2: "Binary Input Event",
        10: "Binary Output",
        20: "Binary Counter",
        21: "Frozen Counter",
        30: "Analog Input",
        32: "Analog Input Event",
        40: "Analog Output Status",
        41: "Analog Output Block",
        50: "Time and Date",
        60: "Class Objects",
        80: "Internal Indications",
        110: "Octet String",
        111: "Virtual Terminal Output"
    }
    
    # Dangerous function codes
    DANGEROUS_FUNCTIONS = [0x0D, 0x0E, 0x12, 0x1B]  # Restart, Stop App, Delete File
    
    def __init__(self):
        """Initialize DNP3 parser."""
        self.sequence_history = []
        
    def parse_packet(self, packet_bytes: bytes) -> Dict:
        """
        Parse DNP3 packet.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            
        Returns:
            dict: Parsed packet information
        """
        try:
            if len(packet_bytes) < 10:
                return {'valid': False, 'error': 'Packet too short', 'protocol': 'dnp3'}
            
            # Parse Data Link Layer
            start_bytes = packet_bytes[0:2]
            if start_bytes != b'\x05\x64':
                return {'valid': False, 'error': 'Invalid start bytes', 'protocol': 'dnp3'}
            
            length = packet_bytes[2]
            control = packet_bytes[3]
            dest = struct.unpack('<H', packet_bytes[4:6])[0]
            src = struct.unpack('<H', packet_bytes[6:8])[0]
            crc = struct.unpack('<H', packet_bytes[8:10])[0]
            
            # Extract control field components
            dir_bit = (control >> 7) & 0x01
            prm_bit = (control >> 6) & 0x01
            fcb_bit = (control >> 5) & 0x01
            fcv_bit = (control >> 4) & 0x01
            dl_function = control & 0x0F
            
            parsed = {
                'valid': True,
                'protocol': 'dnp3',
                'length': length,
                'direction': 'master_to_outstation' if dir_bit == 1 else 'outstation_to_master',
                'is_primary': prm_bit == 1,
                'frame_count_valid': fcv_bit == 1,
                'dest_address': dest,
                'src_address': src,
                'dl_function': dl_function,
                'crc': crc,
                'raw_length': len(packet_bytes)
            }
            
            # Verify CRC
            calculated_crc = self._calculate_crc(packet_bytes[0:8])
            parsed['crc_valid'] = (crc == calculated_crc)
            
            # Parse application layer if present
            if len(packet_bytes) > 10:
                app_data = packet_bytes[10:]
                parsed['application'] = self._parse_application_layer(app_data)
            
            # Detect anomalies
            parsed['anomalies'] = self._validate_packet(parsed)
            
            return parsed
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'protocol': 'dnp3',
                'raw_bytes': packet_bytes.hex() if packet_bytes else ''
            }
    
    def _parse_application_layer(self, data: bytes) -> Dict:
        """Parse DNP3 application layer."""
        if len(data) < 2:
            return {}
        
        app_control = data[0]
        function_code = data[1]
        
        # Extract application control fields
        fir = (app_control >> 7) & 0x01
        fin = (app_control >> 6) & 0x01
        con = (app_control >> 5) & 0x01
        uns = (app_control >> 4) & 0x01
        seq = app_control & 0x0F
        
        app_layer = {
            'function_code': function_code,
            'function_name': self.FUNCTION_CODES.get(function_code, 'Unknown'),
            'first_fragment': fir == 1,
            'final_fragment': fin == 1,
            'confirmation_required': con == 1,
            'unsolicited': uns == 1,
            'sequence': seq,
            'is_dangerous': function_code in self.DANGEROUS_FUNCTIONS
        }
        
        # Track sequence numbers
        self.sequence_history.append(seq)
        if len(self.sequence_history) > 20:
            self.sequence_history.pop(0)
        
        return app_layer
    
    def _calculate_crc(self, data: bytes) -> int:
        """
        Calculate DNP3 CRC-16.
        Uses polynomial 0x3D65 (reverse of 0xA6BC).
        """
        crc = 0
        polynomial = 0x3D65
        
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ polynomial
                else:
                    crc >>= 1
        
        # Invert bits
        crc = (~crc) & 0xFFFF
        
        return crc
    
    def _validate_packet(self, parsed: Dict) -> List[str]:
        """Validate DNP3 packet for anomalies."""
        anomalies = []
        
        # Check CRC
        if not parsed.get('crc_valid', False):
            anomalies.append('crc_mismatch')
        
        # Check address ranges (typical: 0-65519, 0xFFFF reserved for broadcast)
        dest = parsed.get('dest_address', 0)
        src = parsed.get('src_address', 0)
        
        if dest > 65535:
            anomalies.append('invalid_dest_address')
        if src > 65535:
            anomalies.append('invalid_src_address')
        
        # Check for broadcast abuse
        if dest == 0xFFFF or dest == 0xFFFD:
            anomalies.append('broadcast_address_used')
        
        # Check function code validity
        if 'application' in parsed:
            app = parsed['application']
            fc = app.get('function_code')
            
            if fc and fc not in self.FUNCTION_CODES:
                anomalies.append('invalid_function_code')
            
            # Check for dangerous functions
            if app.get('is_dangerous', False):
                anomalies.append('dangerous_function_detected')
            
            # Check sequence number consistency
            if self._detect_sequence_anomaly():
                anomalies.append('sequence_number_anomaly')
        
        # Check packet length
        if parsed.get('length', 0) > 292:  # Max DNP3 frame length
            anomalies.append('excessive_frame_length')
        
        return anomalies
    
    def _detect_sequence_anomaly(self) -> bool:
        """Detect unusual sequence number patterns."""
        if len(self.sequence_history) < 5:
            return False
        
        # Check for sequence number jumps
        recent = self.sequence_history[-5:]
        jumps = sum(1 for i in range(len(recent)-1) 
                   if abs(recent[i+1] - recent[i]) > 2)
        
        # More than 2 large jumps in recent history is suspicious
        return jumps > 2
    
    def extract_features(self, parsed: Dict) -> np.ndarray:
        """Extract features for ML models."""
        features = []
        
        # Data link features
        features.append(parsed.get('dest_address', 0))
        features.append(parsed.get('src_address', 0))
        features.append(int(parsed.get('is_primary', False)))
        features.append(parsed.get('length', 0))
        features.append(int(parsed.get('crc_valid', False)))
        features.append(parsed.get('dl_function', 0))
        
        # Application layer features
        if 'application' in parsed:
            app = parsed['application']
            features.append(app.get('function_code', 0))
            features.append(int(app.get('confirmation_required', False)))
            features.append(int(app.get('unsolicited', False)))
            features.append(app.get('sequence', 0))
            features.append(int(app.get('is_dangerous', False)))
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Anomaly features
        features.append(len(parsed.get('anomalies', [])))
        features.append(parsed.get('raw_length', 0))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            'dest_address',
            'src_address',
            'is_primary',
            'length',
            'crc_valid',
            'dl_function',
            'app_function_code',
            'confirmation_required',
            'unsolicited',
            'sequence',
            'is_dangerous',
            'anomaly_count',
            'raw_length'
        ]
    
    def detect_anomalies(self, parsed: Dict) -> Dict:
        """High-level anomaly detection."""
        anomaly_score = 0
        detected_anomalies = parsed.get('anomalies', [])
        
        # Scoring weights
        anomaly_weights = {
            'crc_mismatch': 40,
            'invalid_dest_address': 30,
            'invalid_src_address': 30,
            'broadcast_address_used': 25,
            'invalid_function_code': 50,
            'dangerous_function_detected': 60,
            'sequence_number_anomaly': 35,
            'excessive_frame_length': 25
        }
        
        for anomaly in detected_anomalies:
            anomaly_score += anomaly_weights.get(anomaly, 10)
        
        # Additional checks
        if 'application' in parsed:
            app = parsed['application']
            
            # Unsolicited responses can be suspicious
            if app.get('unsolicited', False):
                anomaly_score += 15
        
        return {
            'is_anomalous': anomaly_score > 50,
            'anomaly_score': min(anomaly_score, 100),
            'detected_anomalies': detected_anomalies,
            'severity': 'critical' if anomaly_score > 80 else 'high' if anomaly_score > 50 else 'medium' if anomaly_score > 20 else 'low',
            'recommendation': self._get_recommendation(anomaly_score)
        }
    
    def _get_recommendation(self, score: int) -> str:
        """Get recommendation based on score."""
        if score > 80:
            return "BLOCK: Critical DNP3 violation. Immediate action required."
        elif score > 50:
            return "ALERT: High-risk DNP3 operation. Manual review recommended."
        elif score > 20:
            return "MONITOR: Moderate risk. Continue monitoring."
        else:
            return "ALLOW: Normal DNP3 operation."


if __name__ == "__main__":
    print("Testing DNP3 Parser...")
    
    parser = DNP3Parser()
    
    # Test packet: DNP3 request
    test_packet = bytes([
        0x05, 0x64,  # Start bytes
        0x05,        # Length
        0xC4,        # Control (DIR=1, PRM=1, FCB=0, FCV=0, FUNC=4)
        0x01, 0x00,  # Dest address (1)
        0x00, 0x00,  # Src address (0)
        0x00, 0x00,  # CRC placeholder
        0xC0,        # App control (FIR=1, FIN=1, CON=0, UNS=0, SEQ=0)
        0x01         # Function code (READ)
    ])
    
    # Calculate and insert correct CRC
    crc = parser._calculate_crc(test_packet[0:8])
    test_packet = test_packet[:8] + struct.pack('<H', crc) + test_packet[10:]
    
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
