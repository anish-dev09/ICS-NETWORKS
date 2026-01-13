"""
Multi-Protocol Detector and Router
Automatically detects ICS protocol type and routes to appropriate parser.
"""

from typing import Dict, Optional, List
import numpy as np

from .modbus_parser import ModbusParser
from .dnp3_parser import DNP3Parser
from .s7comm_parser import S7CommParser


class ProtocolDetector:
    """
    Automatically detect and parse multiple ICS protocols.
    Supports Modbus TCP/RTU, DNP3, and S7comm with automatic detection.
    """
    
    def __init__(self):
        """Initialize protocol detector with all parsers."""
        self.modbus_tcp_parser = ModbusParser(protocol_type='tcp')
        self.modbus_rtu_parser = ModbusParser(protocol_type='rtu')
        self.dnp3_parser = DNP3Parser()
        self.s7comm_parser = S7CommParser()
        
        # Port mappings (standard ICS protocol ports)
        self.port_to_protocol = {
            502: 'modbus_tcp',
            20000: 'dnp3',
            102: 's7comm',
            2404: 'modbus_tcp',  # Alternative Modbus port
            44818: 's7comm'      # Alternative S7comm port
        }
        
        self.detection_stats = {
            'modbus_tcp': 0,
            'modbus_rtu': 0,
            'dnp3': 0,
            's7comm': 0,
            'unknown': 0
        }
    
    def detect_protocol(self, packet_bytes: bytes, dest_port: Optional[int] = None, 
                       src_port: Optional[int] = None) -> str:
        """
        Detect protocol type from packet bytes or port.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            dest_port (int, optional): Destination port number
            src_port (int, optional): Source port number
            
        Returns:
            str: Detected protocol ('modbus_tcp', 'modbus_rtu', 'dnp3', 's7comm', 'unknown')
        """
        if not packet_bytes or len(packet_bytes) < 2:
            return 'unknown'
        
        # Try port-based detection first (most reliable)
        if dest_port and dest_port in self.port_to_protocol:
            protocol = self.port_to_protocol[dest_port]
            self.detection_stats[protocol] += 1
            return protocol
        
        if src_port and src_port in self.port_to_protocol:
            protocol = self.port_to_protocol[src_port]
            self.detection_stats[protocol] += 1
            return protocol
        
        # Signature-based detection
        protocol = self._signature_based_detection(packet_bytes)
        self.detection_stats[protocol] += 1
        return protocol
    
    def _signature_based_detection(self, packet_bytes: bytes) -> str:
        """
        Detect protocol using packet signatures.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            
        Returns:
            str: Detected protocol
        """
        # Check for DNP3 signature (0x05 0x64)
        if len(packet_bytes) >= 2 and packet_bytes[0:2] == b'\x05\x64':
            return 'dnp3'
        
        # Check for S7comm signature (TPKT version 3 + S7 protocol ID 0x32)
        if len(packet_bytes) >= 8:
            tpkt_version = packet_bytes[0]
            if tpkt_version == 0x03:
                # Check for S7 protocol ID at byte 7
                if len(packet_bytes) > 7 and packet_bytes[7] == 0x32:
                    return 's7comm'
        
        # Check for Modbus TCP (protocol ID = 0x0000 at bytes 2-3)
        if len(packet_bytes) >= 6:
            protocol_id = struct.unpack('>H', packet_bytes[2:4])[0]
            if protocol_id == 0x0000:
                return 'modbus_tcp'
        
        # Check for Modbus RTU (heuristic: unit ID + function code + CRC)
        if len(packet_bytes) >= 4:
            function_code = packet_bytes[1] & 0x7F
            # Check if function code is valid Modbus
            if function_code in [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0F, 0x10, 0x14, 0x15, 0x16, 0x17]:
                # Verify CRC
                crc = struct.unpack('<H', packet_bytes[-2:])[0]
                calculated_crc = self.modbus_rtu_parser._calculate_crc(packet_bytes[:-2])
                if crc == calculated_crc:
                    return 'modbus_rtu'
        
        return 'unknown'
    
    def parse_packet(self, packet_bytes: bytes, dest_port: Optional[int] = None,
                    src_port: Optional[int] = None, protocol_hint: Optional[str] = None) -> Dict:
        """
        Auto-detect protocol and parse packet.
        
        Args:
            packet_bytes (bytes): Raw packet bytes
            dest_port (int, optional): Destination port
            src_port (int, optional): Source port
            protocol_hint (str, optional): Force specific protocol parsing
            
        Returns:
            dict: Parsed packet with protocol-specific information
        """
        # Use hint if provided, otherwise auto-detect
        if protocol_hint:
            protocol = protocol_hint
        else:
            protocol = self.detect_protocol(packet_bytes, dest_port, src_port)
        
        # Route to appropriate parser
        if protocol == 'modbus_tcp':
            return self.modbus_tcp_parser.parse_packet(packet_bytes)
        elif protocol == 'modbus_rtu':
            return self.modbus_rtu_parser.parse_packet(packet_bytes)
        elif protocol == 'dnp3':
            return self.dnp3_parser.parse_packet(packet_bytes)
        elif protocol == 's7comm':
            return self.s7comm_parser.parse_packet(packet_bytes)
        else:
            return {
                'valid': False,
                'protocol': 'unknown',
                'error': 'Unable to detect protocol type',
                'raw_bytes': packet_bytes.hex() if packet_bytes else '',
                'raw_length': len(packet_bytes) if packet_bytes else 0
            }
    
    def parse_batch(self, packets: List[Dict]) -> List[Dict]:
        """
        Parse multiple packets efficiently.
        
        Args:
            packets (list): List of dicts with 'bytes', 'dest_port', 'src_port' keys
            
        Returns:
            list: List of parsed packet dictionaries
        """
        results = []
        for packet in packets:
            parsed = self.parse_packet(
                packet.get('bytes', b''),
                packet.get('dest_port'),
                packet.get('src_port'),
                packet.get('protocol_hint')
            )
            results.append(parsed)
        return results
    
    def extract_features(self, parsed: Dict) -> np.ndarray:
        """
        Extract features based on detected protocol.
        
        Args:
            parsed (dict): Parsed packet
            
        Returns:
            np.ndarray: Feature vector (padded to consistent length)
        """
        protocol = parsed.get('protocol', 'unknown')
        
        if protocol == 'modbus_tcp' or protocol == 'modbus_rtu':
            features = self.modbus_tcp_parser.extract_features(parsed)
        elif protocol == 'dnp3':
            features = self.dnp3_parser.extract_features(parsed)
        elif protocol == 's7comm':
            features = self.s7comm_parser.extract_features(parsed)
        else:
            # Return zero vector for unknown protocols
            features = np.zeros(15, dtype=np.float32)
        
        # Pad to consistent length (15 features)
        if len(features) < 15:
            features = np.pad(features, (0, 15 - len(features)), 'constant')
        elif len(features) > 15:
            features = features[:15]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get standardized feature names across all protocols."""
        return [
            'primary_identifier',      # function_code, dl_function, message_type
            'secondary_identifier',    # unit_id, src_address, pdu_ref
            'tertiary_identifier',     # dest_address, error_class
            'address_or_length',
            'quantity_or_param_length',
            'is_read',
            'is_write',
            'is_error',
            'packet_length',
            'header_length',
            'anomaly_count',
            'is_invalid',
            'is_dangerous',
            'validation_score',
            'protocol_specific'
        ]
    
    def detect_anomalies(self, parsed: Dict) -> Dict:
        """
        Unified anomaly detection across protocols.
        
        Args:
            parsed (dict): Parsed packet
            
        Returns:
            dict: Anomaly detection result
        """
        protocol = parsed.get('protocol', 'unknown')
        
        if protocol in ['modbus_tcp', 'modbus_rtu']:
            return self.modbus_tcp_parser.detect_anomalies(parsed)
        elif protocol == 'dnp3':
            return self.dnp3_parser.detect_anomalies(parsed)
        elif protocol == 's7comm':
            return self.s7comm_parser.detect_anomalies(parsed)
        else:
            return {
                'is_anomalous': True,
                'anomaly_score': 30,
                'detected_anomalies': ['unknown_protocol'],
                'severity': 'medium',
                'recommendation': 'INVESTIGATE: Unknown protocol detected.'
            }
    
    def get_supported_protocols(self) -> List[str]:
        """Get list of supported protocols."""
        return ['modbus_tcp', 'modbus_rtu', 'dnp3', 's7comm']
    
    def get_detection_stats(self) -> Dict:
        """Get protocol detection statistics."""
        total = sum(self.detection_stats.values())
        if total == 0:
            return self.detection_stats
        
        stats_with_percentages = {}
        for protocol, count in self.detection_stats.items():
            stats_with_percentages[protocol] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return stats_with_percentages
    
    def reset_stats(self):
        """Reset detection statistics."""
        for protocol in self.detection_stats:
            self.detection_stats[protocol] = 0


# Import struct at module level
import struct


if __name__ == "__main__":
    print("Testing Protocol Detector...")
    
    detector = ProtocolDetector()
    
    print("\n" + "="*60)
    print("Test 1: Modbus TCP Packet")
    print("="*60)
    
    modbus_packet = bytes([
        0x00, 0x01,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function code
        0x00, 0x00,  # Address
        0x00, 0x0A   # Quantity
    ])
    
    protocol = detector.detect_protocol(modbus_packet, 502)
    print(f"Detected Protocol: {protocol}")
    
    parsed = detector.parse_packet(modbus_packet, 502)
    print(f"\nParsed Keys: {list(parsed.keys())}")
    print(f"Function: {parsed.get('function_name')}")
    print(f"Valid: {parsed.get('valid')}")
    print(f"Anomalies: {parsed.get('anomalies')}")
    
    features = detector.extract_features(parsed)
    print(f"\nFeatures Shape: {features.shape}")
    print(f"Features: {features}")
    
    anomalies = detector.detect_anomalies(parsed)
    print(f"\nAnomaly Detection:")
    print(f"  Is Anomalous: {anomalies['is_anomalous']}")
    print(f"  Score: {anomalies['anomaly_score']}")
    print(f"  Severity: {anomalies['severity']}")
    
    print("\n" + "="*60)
    print("Detection Statistics")
    print("="*60)
    stats = detector.get_detection_stats()
    for protocol, data in stats.items():
        if isinstance(data, dict):
            print(f"{protocol}: {data['count']} ({data['percentage']:.1f}%)")
        else:
            print(f"{protocol}: {data}")
    
    print(f"\nSupported Protocols: {detector.get_supported_protocols()}")
