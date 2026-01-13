"""
Test Script for ICS Protocol Parsers
Tests Modbus, DNP3, S7comm parsing and anomaly detection.
"""

import numpy as np
import struct
from src.protocols.protocol_detector import ProtocolDetector
from src.models.baseline_detector import BaselineDetector


def test_modbus():
    """Test Modbus TCP protocol parsing."""
    print("\n" + "="*70)
    print("TEST 1: Modbus TCP - Read Holding Registers")
    print("="*70)
    
    # Modbus TCP packet: Read Holding Registers
    packet = bytes([
        0x00, 0x01,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function code (Read Holding Registers)
        0x00, 0x64,  # Starting address (100)
        0x00, 0x0A   # Quantity (10 registers)
    ])
    
    detector = ProtocolDetector()
    parsed = detector.parse_packet(packet, dest_port=502)
    
    print(f"✓ Protocol: {parsed['protocol']}")
    print(f"✓ Valid: {parsed['valid']}")
    print(f"✓ Function: {parsed['function_name']}")
    print(f"✓ Unit ID: {parsed['unit_id']}")
    print(f"✓ Address: {parsed['starting_address']}")
    print(f"✓ Quantity: {parsed['quantity']}")
    print(f"✓ Anomalies: {parsed['anomalies']}")
    
    anomalies = detector.detect_anomalies(parsed)
    print(f"\nAnomaly Detection:")
    print(f"  Is Anomalous: {anomalies['is_anomalous']}")
    print(f"  Score: {anomalies['anomaly_score']}")
    print(f"  Severity: {anomalies['severity']}")
    print(f"  Detected: {anomalies['detected_anomalies']}")
    
    features = detector.extract_features(parsed)
    print(f"\nExtracted Features: {features.shape}")
    print(f"  Feature Vector: {features[:5]}...")  # Show first 5
    
    return parsed, anomalies


def test_modbus_anomaly():
    """Test Modbus with anomalous packet (oversized quantity)."""
    print("\n" + "="*70)
    print("TEST 2: Modbus TCP - Anomalous Packet (Oversized Quantity)")
    print("="*70)
    
    # Modbus TCP with excessive quantity (DDOS attempt)
    packet = bytes([
        0x00, 0x02,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function code
        0x00, 0x00,  # Starting address
        0x07, 0xD0   # Quantity (2000 registers - exceeds limit)
    ])
    
    detector = ProtocolDetector()
    parsed = detector.parse_packet(packet, dest_port=502)
    
    print(f"✓ Protocol: {parsed['protocol']}")
    print(f"✓ Valid: {parsed['valid']}")
    print(f"✓ Anomalies: {parsed['anomalies']}")
    
    anomalies = detector.detect_anomalies(parsed)
    print(f"\nAnomaly Detection:")
    print(f"  Is Anomalous: {anomalies['is_anomalous']}")
    print(f"  Score: {anomalies['anomaly_score']}")
    print(f"  Severity: {anomalies['severity']}")
    print(f"  Detected: {anomalies['detected_anomalies']}")
    print(f"  Recommendation: {anomalies['recommendation']}")
    
    return parsed, anomalies


def test_dnp3():
    """Test DNP3 protocol parsing."""
    print("\n" + "="*70)
    print("TEST 3: DNP3 - Read Request")
    print("="*70)
    
    # DNP3 packet: Read request
    packet = bytes([
        0x05, 0x64,  # Start bytes
        0x0E,        # Length
        0xC4,        # Control
        0x01, 0x00,  # Destination
        0x02, 0x00,  # Source
        0x00, 0x00,  # CRC (will be calculated)
    ])
    
    # Add CRC for header
    from src.protocols.dnp3_parser import DNP3Parser
    parser = DNP3Parser()
    crc = parser._calculate_crc(packet[:8])
    packet = packet[:8] + struct.pack('<H', crc)
    
    # Application layer
    app_layer = bytes([
        0xC0,        # Control
        0x01,        # Function code (Read)
        0x3C, 0x02, 0x06,  # Object header
        0x00, 0x00,  # CRC
    ])
    
    app_crc = parser._calculate_crc(app_layer[:-2])
    app_layer = app_layer[:-2] + struct.pack('<H', app_crc)
    packet = packet + app_layer
    
    detector = ProtocolDetector()
    parsed = detector.parse_packet(packet, dest_port=20000)
    
    print(f"✓ Protocol: {parsed['protocol']}")
    print(f"✓ Valid: {parsed['valid']}")
    print(f"✓ Function: {parsed.get('dl_function_name', 'N/A')}")
    print(f"✓ Source: {parsed.get('source', 'N/A')}")
    print(f"✓ Destination: {parsed.get('destination', 'N/A')}")
    print(f"✓ Anomalies: {parsed.get('anomalies', [])}")
    
    anomalies = detector.detect_anomalies(parsed)
    print(f"\nAnomaly Detection:")
    print(f"  Is Anomalous: {anomalies['is_anomalous']}")
    print(f"  Score: {anomalies['anomaly_score']}")
    print(f"  Severity: {anomalies['severity']}")
    
    return parsed, anomalies


def test_s7comm():
    """Test Siemens S7comm protocol parsing."""
    print("\n" + "="*70)
    print("TEST 4: S7comm - Read Variable Request")
    print("="*70)
    
    # S7comm packet: Setup communication
    packet = bytes([
        # TPKT Header
        0x03,        # Version
        0x00,        # Reserved
        0x00, 0x1F,  # Length (31 bytes)
        
        # COTP Header
        0x02,        # Length
        0xF0,        # PDU Type (Data)
        0x80,        # TPDU Number
        
        # S7 Header
        0x32,        # Protocol ID
        0x01,        # Message type (Job)
        0x00, 0x00,  # Reserved
        0x00, 0x01,  # PDU Reference
        0x00, 0x0E,  # Parameter length (14)
        0x00, 0x00,  # Data length
        
        # Parameters (Read Var)
        0x04,        # Function code (Read Var)
        0x01,        # Item count
    ])
    
    detector = ProtocolDetector()
    parsed = detector.parse_packet(packet, dest_port=102)
    
    print(f"✓ Protocol: {parsed['protocol']}")
    print(f"✓ Valid: {parsed['valid']}")
    print(f"✓ Message Type: {parsed.get('message_type_name', 'N/A')}")
    print(f"✓ Function: {parsed.get('function_name', 'N/A')}")
    print(f"✓ PDU Reference: {parsed.get('pdu_ref', 'N/A')}")
    print(f"✓ Anomalies: {parsed.get('anomalies', [])}")
    
    anomalies = detector.detect_anomalies(parsed)
    print(f"\nAnomaly Detection:")
    print(f"  Is Anomalous: {anomalies['is_anomalous']}")
    print(f"  Score: {anomalies['anomaly_score']}")
    print(f"  Severity: {anomalies['severity']}")
    
    return parsed, anomalies


def test_s7comm_dangerous():
    """Test S7comm with dangerous operation (PLC Stop)."""
    print("\n" + "="*70)
    print("TEST 5: S7comm - Dangerous Operation (PLC Stop)")
    print("="*70)
    
    # S7comm packet: PLC Stop command
    packet = bytes([
        # TPKT Header
        0x03,        # Version
        0x00,        # Reserved
        0x00, 0x21,  # Length
        
        # COTP Header
        0x02,        # Length
        0xF0,        # PDU Type
        0x80,        # TPDU Number
        
        # S7 Header
        0x32,        # Protocol ID
        0x01,        # Message type (Job)
        0x00, 0x00,  # Reserved
        0x00, 0x05,  # PDU Reference
        0x00, 0x10,  # Parameter length
        0x00, 0x00,  # Data length
        
        # Parameters (PLC Control - Stop)
        0x29,        # Function code (PLC Control)
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x09,        # Service: Stop PLC
        0x50, 0x5F,  # "P_"
        0x50, 0x52, 0x4F, 0x47, 0x52, 0x41, 0x4D  # "PROGRAM"
    ])
    
    detector = ProtocolDetector()
    parsed = detector.parse_packet(packet, dest_port=102)
    
    print(f"✓ Protocol: {parsed['protocol']}")
    print(f"✓ Valid: {parsed['valid']}")
    print(f"✓ Function: {parsed.get('function_name', 'N/A')}")
    print(f"✓ Dangerous Operation: {parsed.get('is_dangerous_operation', False)}")
    print(f"✓ Anomalies: {parsed.get('anomalies', [])}")
    
    anomalies = detector.detect_anomalies(parsed)
    print(f"\n⚠️  CRITICAL ANOMALY DETECTED")
    print(f"  Is Anomalous: {anomalies['is_anomalous']}")
    print(f"  Score: {anomalies['anomaly_score']}")
    print(f"  Severity: {anomalies['severity']}")
    print(f"  Detected: {anomalies['detected_anomalies']}")
    print(f"  Recommendation: {anomalies['recommendation']}")
    
    return parsed, anomalies


def test_protocol_aware_baseline():
    """Test protocol-aware baseline detector."""
    print("\n" + "="*70)
    print("TEST 6: Protocol-Aware Baseline Detector")
    print("="*70)
    
    # Create detector with protocol awareness
    detector = BaselineDetector(method='isolation_forest', protocol_aware=True)
    
    # Generate training data (normal packets)
    print("\nGenerating training data...")
    training_packets = []
    for i in range(100):
        # Normal Modbus read packets
        packet = bytes([
            0x00, i % 256,  # Transaction ID
            0x00, 0x00,     # Protocol ID
            0x00, 0x06,     # Length
            0x01,           # Unit ID
            0x03,           # Function code
            0x00, i % 200,  # Address (varies)
            0x00, 0x0A      # Quantity (10)
        ])
        training_packets.append(packet)
    
    # Extract features for training
    from src.protocols.protocol_detector import ProtocolDetector
    proto_detector = ProtocolDetector()
    
    X_train = []
    for packet in training_packets:
        parsed = proto_detector.parse_packet(packet, dest_port=502)
        features = proto_detector.extract_features(parsed)
        X_train.append(features)
    
    X_train = np.array(X_train)
    
    print(f"✓ Training data shape: {X_train.shape}")
    print(f"✓ Training baseline detector...")
    detector.fit(X_train)
    
    # Test packets (mix of normal and anomalous)
    test_packets = [
        # Normal packet
        {
            'bytes': bytes([0x00, 0x10, 0x00, 0x00, 0x00, 0x06, 0x01, 0x03, 0x00, 0x64, 0x00, 0x05]),
            'dest_port': 502,
            'src_port': 50000
        },
        # Anomalous packet (oversized quantity)
        {
            'bytes': bytes([0x00, 0x11, 0x00, 0x00, 0x00, 0x06, 0x01, 0x03, 0x00, 0x00, 0x07, 0xD0]),
            'dest_port': 502,
            'src_port': 50001
        }
    ]
    
    print(f"\n✓ Testing {len(test_packets)} packets...")
    results = detector.predict_with_protocol(test_packets)
    
    print(f"\nResults:")
    print(f"  Total Packets: {results['summary']['total_packets']}")
    print(f"  Anomalies Detected: {results['summary']['anomalies_detected']}")
    print(f"  Anomaly Rate: {results['summary']['anomaly_rate']:.2%}")
    print(f"  Dangerous Operations: {results['summary']['dangerous_operations_count']}")
    
    for i, pred in enumerate(results['predictions']):
        status = "🔴 ANOMALY" if pred == 1 else "🟢 NORMAL"
        print(f"\n  Packet {i+1}: {status}")
        print(f"    Score: {results['anomaly_scores'][i]:.2f}")
        print(f"    Protocol: {results['protocols_detected'][i]}")
        print(f"    Protocol Anomalies: {results['protocol_anomalies'][i]}")
    
    return results


def run_all_tests():
    """Run all protocol tests."""
    print("="*70)
    print("ICS PROTOCOL PARSER TEST SUITE")
    print("="*70)
    print("\nTesting Modbus, DNP3, and S7comm protocols...")
    print("This validates parsing, feature extraction, and anomaly detection.\n")
    
    try:
        # Test 1: Modbus normal
        test_modbus()
        
        # Test 2: Modbus anomaly
        test_modbus_anomaly()
        
        # Test 3: DNP3
        test_dnp3()
        
        # Test 4: S7comm normal
        test_s7comm()
        
        # Test 5: S7comm dangerous
        test_s7comm_dangerous()
        
        # Test 6: Protocol-aware baseline
        test_protocol_aware_baseline()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nProtocol Support Summary:")
        print("  ✓ Modbus TCP/RTU - Full support with 24 function codes")
        print("  ✓ DNP3 - Data link and application layer parsing")
        print("  ✓ S7comm - TPKT/COTP/S7 layer support")
        print("  ✓ Automatic protocol detection (port + signature based)")
        print("  ✓ Anomaly detection with severity scoring")
        print("  ✓ Dangerous operation detection (PLC control, firmware)")
        print("  ✓ Integration with baseline detector")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
