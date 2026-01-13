"""
Modbus TCP/RTU Protocol Parser
Parses and validates Modbus protocol packets for anomaly detection.
"""

import struct
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class ModbusParser:
    """
    Comprehensive Modbus protocol parser with validation.
    Supports both Modbus TCP and RTU variants.
    """
    
    # Modbus function codes
    FUNCTION_CODES = {
        0x01: "Read Coils",
        0x02: "Read Discrete Inputs",
        0x03: "Read Holding Registers",
        0x04: "Read Input Registers",
        0x05: "Write Single Coil",
        0x06: "Write Single Register",
        0x07: "Read Exception Status",
        0x08: "Diagnostics",
        0x0B: "Get Comm Event Counter",
        0x0C: "Get Comm Event Log",
        0x0F: "Write Multiple Coils",
        0x10: "Write Multiple Registers",
        0x11: "Report Server ID",
        0x14: "Read File Record",
        0x15: "Write File Record",
        0x16: "Mask Write Register",
        0x17: "Read/Write Multiple Registers",
        0x18: "Read FIFO Queue",
        0x2B: "Read Device Identification"
    }
    
    # Exception codes
    EXCEPTION_CODES = {
        0x01: "Illegal Function",
        0x02: "Illegal Data Address",
        0x03: "Illegal Data Value",
        0x04: "Slave Device Failure",
        0x05: "Acknowledge",
        0x06: "Slave Device Busy",
        0x08: "Memory Parity Error",
        0x0A: "Gateway Path Unavailable",
        0x0B: "Gateway Target Device Failed to Respond"
    }
    
    def __init__(self, protocol_type='tcp'):
        """
        Initialize Modbus parser.
        
        Args:
            protocol_type (str): 'tcp' or 'rtu'
        """
        self.protocol_type = protocol_type
        self.transaction_history = deque(maxlen=100)
        self.address_history = deque(maxlen=50)
        
    def parse_packet(self, packet_bytes: bytes) -> Dict:
        """
        Parse Modbus packet and extract features.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            
        Returns:
            dict: Parsed packet information
        """
        try:
            if self.protocol_type == 'tcp':
                return self._parse_tcp_packet(packet_bytes)
            else:
                return self._parse_rtu_packet(packet_bytes)
        except Exception as e:
            return {
                'valid': False,
                'protocol': f'modbus_{self.protocol_type}',
                'error': str(e),
                'raw_bytes': packet_bytes.hex() if packet_bytes else ''
            }
    
    def _parse_tcp_packet(self, packet_bytes: bytes) -> Dict:
        """Parse Modbus TCP packet (MBAP header + PDU)."""
        if len(packet_bytes) < 8:
            return {'valid': False, 'error': 'Packet too short', 'protocol': 'modbus_tcp'}
        
        # Parse MBAP Header (7 bytes)
        transaction_id = struct.unpack('>H', packet_bytes[0:2])[0]
        protocol_id = struct.unpack('>H', packet_bytes[2:4])[0]
        length = struct.unpack('>H', packet_bytes[4:6])[0]
        unit_id = packet_bytes[6]
        
        # Validate protocol ID (should be 0 for Modbus)
        if protocol_id != 0:
            return {
                'valid': False,
                'error': f'Invalid protocol ID: {protocol_id}',
                'anomaly_type': 'protocol_violation',
                'protocol': 'modbus_tcp'
            }
        
        # Parse PDU
        function_code = packet_bytes[7]
        
        # Check for exception response
        is_exception = (function_code & 0x80) != 0
        
        parsed = {
            'valid': True,
            'protocol': 'modbus_tcp',
            'transaction_id': transaction_id,
            'protocol_id': protocol_id,
            'length': length,
            'unit_id': unit_id,
            'function_code': function_code & 0x7F,
            'function_name': self.FUNCTION_CODES.get(function_code & 0x7F, 'Unknown'),
            'is_exception': is_exception,
            'raw_length': len(packet_bytes)
        }
        
        # Parse function-specific data
        if is_exception:
            if len(packet_bytes) > 8:
                exception_code = packet_bytes[8]
                parsed['exception_code'] = exception_code
                parsed['exception_name'] = self.EXCEPTION_CODES.get(exception_code, 'Unknown')
        else:
            parsed.update(self._parse_function_data(function_code, packet_bytes[8:]))
        
        # Validate packet structure
        parsed['anomalies'] = self._validate_packet(parsed)
        
        # Add to history
        self.transaction_history.append(parsed)
        if 'starting_address' in parsed:
            self.address_history.append(parsed['starting_address'])
        
        return parsed
    
    def _parse_rtu_packet(self, packet_bytes: bytes) -> Dict:
        """Parse Modbus RTU packet (Unit ID + PDU + CRC)."""
        if len(packet_bytes) < 4:
            return {'valid': False, 'error': 'Packet too short', 'protocol': 'modbus_rtu'}
        
        # Extract components
        unit_id = packet_bytes[0]
        function_code = packet_bytes[1]
        data = packet_bytes[2:-2]
        crc = struct.unpack('<H', packet_bytes[-2:])[0]
        
        # Verify CRC
        calculated_crc = self._calculate_crc(packet_bytes[:-2])
        crc_valid = (crc == calculated_crc)
        
        parsed = {
            'valid': crc_valid,
            'protocol': 'modbus_rtu',
            'unit_id': unit_id,
            'function_code': function_code & 0x7F,
            'function_name': self.FUNCTION_CODES.get(function_code & 0x7F, 'Unknown'),
            'is_exception': (function_code & 0x80) != 0,
            'crc': crc,
            'crc_valid': crc_valid,
            'crc_calculated': calculated_crc,
            'raw_length': len(packet_bytes)
        }
        
        if not crc_valid:
            parsed['anomalies'] = ['crc_mismatch']
        else:
            parsed['anomalies'] = []
            
        # Parse function data
        if not parsed['is_exception']:
            parsed.update(self._parse_function_data(function_code, data))
            parsed['anomalies'].extend(self._validate_packet(parsed))
        
        return parsed
    
    def _parse_function_data(self, function_code: int, data: bytes) -> Dict:
        """Parse function-specific data."""
        result = {}
        
        try:
            if function_code in [0x01, 0x02]:  # Read Coils/Discrete Inputs
                if len(data) >= 4:
                    starting_address = struct.unpack('>H', data[0:2])[0]
                    quantity = struct.unpack('>H', data[2:4])[0]
                    result.update({
                        'starting_address': starting_address,
                        'quantity': quantity,
                        'is_read': True,
                        'is_write': False
                    })
            
            elif function_code in [0x03, 0x04]:  # Read Holding/Input Registers
                if len(data) >= 4:
                    starting_address = struct.unpack('>H', data[0:2])[0]
                    quantity = struct.unpack('>H', data[2:4])[0]
                    result.update({
                        'starting_address': starting_address,
                        'quantity': quantity,
                        'is_read': True,
                        'is_write': False
                    })
            
            elif function_code == 0x05:  # Write Single Coil
                if len(data) >= 4:
                    address = struct.unpack('>H', data[0:2])[0]
                    value = struct.unpack('>H', data[2:4])[0]
                    result.update({
                        'address': address,
                        'value': value,
                        'is_read': False,
                        'is_write': True
                    })
            
            elif function_code == 0x06:  # Write Single Register
                if len(data) >= 4:
                    address = struct.unpack('>H', data[0:2])[0]
                    value = struct.unpack('>H', data[2:4])[0]
                    result.update({
                        'address': address,
                        'value': value,
                        'is_read': False,
                        'is_write': True
                    })
            
            elif function_code == 0x0F:  # Write Multiple Coils
                if len(data) >= 5:
                    starting_address = struct.unpack('>H', data[0:2])[0]
                    quantity = struct.unpack('>H', data[2:4])[0]
                    byte_count = data[4]
                    result.update({
                        'starting_address': starting_address,
                        'quantity': quantity,
                        'byte_count': byte_count,
                        'is_read': False,
                        'is_write': True
                    })
            
            elif function_code == 0x10:  # Write Multiple Registers
                if len(data) >= 5:
                    starting_address = struct.unpack('>H', data[0:2])[0]
                    quantity = struct.unpack('>H', data[2:4])[0]
                    byte_count = data[4]
                    result.update({
                        'starting_address': starting_address,
                        'quantity': quantity,
                        'byte_count': byte_count,
                        'is_read': False,
                        'is_write': True
                    })
            
            elif function_code == 0x17:  # Read/Write Multiple Registers
                if len(data) >= 9:
                    read_address = struct.unpack('>H', data[0:2])[0]
                    read_quantity = struct.unpack('>H', data[2:4])[0]
                    write_address = struct.unpack('>H', data[4:6])[0]
                    write_quantity = struct.unpack('>H', data[6:8])[0]
                    byte_count = data[8]
                    result.update({
                        'read_address': read_address,
                        'read_quantity': read_quantity,
                        'write_address': write_address,
                        'write_quantity': write_quantity,
                        'byte_count': byte_count,
                        'is_read': True,
                        'is_write': True
                    })
        
        except Exception as e:
            result['parse_error'] = str(e)
        
        return result
    
    def _validate_packet(self, parsed: Dict) -> List[str]:
        """
        Validate parsed packet for anomalies.
        
        Returns:
            list: List of detected anomalies
        """
        anomalies = []
        
        # Check function code validity
        if parsed.get('function_code') is not None and parsed['function_code'] not in self.FUNCTION_CODES:
            anomalies.append('invalid_function_code')
        
        # Check quantity bounds for read operations
        if 'quantity' in parsed and parsed['quantity'] is not None:
            quantity = parsed['quantity']
            fc = parsed.get('function_code')
            
            if fc in [0x01, 0x02] and (quantity < 1 or quantity > 2000):
                anomalies.append('invalid_coil_quantity')
            elif fc in [0x03, 0x04] and (quantity < 1 or quantity > 125):
                anomalies.append('invalid_register_quantity')
            elif fc == 0x0F and (quantity < 1 or quantity > 1968):
                anomalies.append('invalid_write_coil_quantity')
            elif fc == 0x10 and (quantity < 1 or quantity > 123):
                anomalies.append('invalid_write_register_quantity')
        
        # Check address bounds (typical range 0-65535)
        if 'starting_address' in parsed or 'address' in parsed:
            addr = parsed.get('starting_address') or parsed.get('address')
            if addr is not None and addr > 65535:
                anomalies.append('invalid_address')
            
            # Check for suspicious low addresses (system/config area)
            if addr is not None and addr < 100 and parsed.get('is_write', False):
                anomalies.append('suspicious_write_to_low_address')
        
        # Check for suspicious byte count
        if 'byte_count' in parsed and 'quantity' in parsed:
            fc = parsed['function_code']
            quantity = parsed['quantity']
            byte_count = parsed['byte_count']
            
            if fc == 0x10:  # Write Multiple Registers
                expected = quantity * 2
            elif fc == 0x0F:  # Write Multiple Coils
                expected = (quantity + 7) // 8
            else:
                expected = byte_count
            
            if byte_count != expected:
                anomalies.append('byte_count_mismatch')
        
        # Check packet length consistency
        if 'length' in parsed:
            # MBAP length should match actual PDU length + 1 (unit ID)
            expected_length = parsed['raw_length'] - 6  # Subtract MBAP header
            if parsed['length'] != expected_length:
                anomalies.append('length_field_mismatch')
        
        # Check for scanning patterns
        if self._detect_scanning_pattern():
            anomalies.append('scanning_pattern_detected')
        
        return anomalies
    
    def _calculate_crc(self, data: bytes) -> int:
        """Calculate Modbus RTU CRC-16."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc
    
    def _detect_scanning_pattern(self) -> bool:
        """Detect address scanning patterns."""
        if len(self.address_history) < 5:
            return False
        
        # Check for sequential address access
        recent = list(self.address_history)[-5:]
        sequential_count = sum(1 for i in range(len(recent)-1) 
                              if recent[i] + 1 == recent[i+1] or recent[i] + 10 == recent[i+1])
        
        # If more than 60% are sequential, flag as scan
        if sequential_count >= 3:
            return True
        
        return False
    
    def extract_features(self, parsed: Dict) -> np.ndarray:
        """
        Extract numerical features from parsed packet for ML models.
        
        Args:
            parsed (dict): Parsed packet dictionary
            
        Returns:
            np.ndarray: Feature vector
        """
        features = []
        
        # Basic features
        features.append(parsed.get('function_code', 0))
        features.append(int(parsed.get('is_exception', False)))
        features.append(int(parsed.get('is_read', False)))
        features.append(int(parsed.get('is_write', False)))
        features.append(parsed.get('unit_id', 0))
        
        # Address features
        features.append(parsed.get('starting_address', parsed.get('address', 0)))
        features.append(parsed.get('quantity', 0))
        
        # Packet size features
        features.append(parsed.get('raw_length', 0))
        features.append(parsed.get('length', 0))
        
        # Anomaly indicators
        features.append(len(parsed.get('anomalies', [])))
        features.append(int(not parsed.get('valid', True)))
        
        # Transaction features
        features.append(parsed.get('transaction_id', 0))
        
        # CRC validity (RTU only)
        features.append(int(parsed.get('crc_valid', True)))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return [
            'function_code',
            'is_exception',
            'is_read',
            'is_write',
            'unit_id',
            'address',
            'quantity',
            'raw_length',
            'length',
            'anomaly_count',
            'is_invalid',
            'transaction_id',
            'crc_valid'
        ]
    
    def detect_anomalies(self, parsed: Dict) -> Dict:
        """
        High-level anomaly detection based on protocol semantics.
        
        Args:
            parsed (dict): Parsed packet
            
        Returns:
            dict: Anomaly detection result
        """
        # Handle invalid/unparseable packets
        if not parsed.get('valid', True):
            return {
                'is_anomalous': True,
                'anomaly_score': 80,
                'detected_anomalies': ['parse_failure', parsed.get('error', 'unknown_error')],
                'severity': 'high',
                'recommendation': 'ALERT: Malformed Modbus packet. Possible attack or corruption.'
            }
        
        anomaly_score = 0
        detected_anomalies = parsed.get('anomalies', [])
        
        # Score based on anomaly types
        anomaly_weights = {
            'invalid_function_code': 50,
            'protocol_violation': 50,
            'invalid_address': 30,
            'invalid_coil_quantity': 30,
            'invalid_register_quantity': 30,
            'invalid_write_coil_quantity': 30,
            'invalid_write_register_quantity': 30,
            'crc_mismatch': 40,
            'length_field_mismatch': 25,
            'byte_count_mismatch': 20,
            'suspicious_write_to_low_address': 35,
            'scanning_pattern_detected': 30
        }
        
        for anomaly in detected_anomalies:
            anomaly_score += anomaly_weights.get(anomaly, 10)
        
        # Additional heuristics
        if parsed.get('is_exception'):
            anomaly_score += 15
        
        # Check for rapid repeated transactions
        if len(self.transaction_history) >= 10:
            recent_functions = [t.get('function_code') for t in list(self.transaction_history)[-10:]]
            if len(set(recent_functions)) == 1:
                anomaly_score += 20
                detected_anomalies.append('repeated_function_burst')
        
        return {
            'is_anomalous': anomaly_score > 50,
            'anomaly_score': min(anomaly_score, 100),
            'detected_anomalies': detected_anomalies,
            'severity': 'critical' if anomaly_score > 80 else 'high' if anomaly_score > 50 else 'medium' if anomaly_score > 20 else 'low',
            'recommendation': self._get_recommendation(anomaly_score)
        }
    
    def _get_recommendation(self, score: int) -> str:
        """Get recommendation based on anomaly score."""
        if score > 80:
            return "BLOCK: Critical protocol violation. Immediate action required."
        elif score > 50:
            return "ALERT: High-risk Modbus operation. Manual review recommended."
        elif score > 20:
            return "MONITOR: Moderate risk. Continue monitoring."
        else:
            return "ALLOW: Normal Modbus operation."
    
    def reset_history(self):
        """Reset transaction history."""
        self.transaction_history.clear()
        self.address_history.clear()


if __name__ == "__main__":
    print("Testing Modbus Parser...")
    
    parser = ModbusParser(protocol_type='tcp')
    
    # Test packet: Read Holding Registers (Function 0x03)
    test_packet = bytes([
        0x00, 0x01,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function code (Read Holding Registers)
        0x00, 0x00,  # Starting address
        0x00, 0x0A   # Quantity
    ])
    
    parsed = parser.parse_packet(test_packet)
    print("\n=== Parsed Packet ===")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
    
    features = parser.extract_features(parsed)
    print(f"\n=== Extracted Features ===")
    print(f"Values: {features}")
    print(f"Names: {parser.get_feature_names()}")
    
    anomaly_result = parser.detect_anomalies(parsed)
    print(f"\n=== Anomaly Detection ===")
    for key, value in anomaly_result.items():
        print(f"  {key}: {value}")
