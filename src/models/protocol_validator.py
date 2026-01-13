"""
ICS Protocol-Specific Validators for Zero-Day Detection
Validates protocol semantics, command sequences, and physical constraints.
"""

import numpy as np
from collections import deque, Counter
from datetime import datetime


class ICSProtocolValidator:
    """
    Validates ICS protocol commands and sequences.
    Detects protocol violations that may indicate zero-day attacks.
    """
    
    def __init__(self, protocol='modbus'):
        """
        Initialize protocol validator.
        
        Args:
            protocol (str): Protocol type ('modbus', 'dnp3', 's7comm')
        """
        self.protocol = protocol
        self.command_history = deque(maxlen=100)
        self.valid_commands = self._load_valid_commands()
        self.valid_transitions = self._load_valid_transitions()
        self.physical_constraints = self._load_physical_constraints()
        self.anomaly_count = 0
        self.total_validations = 0
    
    def _load_valid_commands(self):
        """Load valid command set for protocol."""
        if self.protocol == 'modbus':
            return {
                'read_coils': {'function_code': 1, 'max_quantity': 2000},
                'read_discrete_inputs': {'function_code': 2, 'max_quantity': 2000},
                'read_holding_registers': {'function_code': 3, 'max_quantity': 125},
                'read_input_registers': {'function_code': 4, 'max_quantity': 125},
                'write_single_coil': {'function_code': 5},
                'write_single_register': {'function_code': 6},
                'write_multiple_coils': {'function_code': 15, 'max_quantity': 1968},
                'write_multiple_registers': {'function_code': 16, 'max_quantity': 123},
                'read_write_multiple_registers': {'function_code': 23},
            }
        elif self.protocol == 'dnp3':
            return {
                'read': {'function_code': 1},
                'write': {'function_code': 2},
                'select': {'function_code': 3},
                'operate': {'function_code': 4},
                'direct_operate': {'function_code': 5},
                'freeze': {'function_code': 7},
            }
        elif self.protocol == 's7comm':
            return {
                'read_var': {'function': 0x04},
                'write_var': {'function': 0x05},
                'start_upload': {'function': 0x1d},
                'upload': {'function': 0x1e},
                'end_upload': {'function': 0x1f},
            }
        
        return {}
    
    def _load_valid_transitions(self):
        """Load valid command sequences."""
        # Define valid state transitions for ICS operations
        return {
            'startup': ['read_status', 'initialize', 'read_holding_registers', 'read_coils'],
            'read_status': ['read_sensors', 'write_setpoint', 'shutdown', 'read_status'],
            'write_setpoint': ['read_status', 'verify_setpoint', 'write_setpoint'],
            'verify_setpoint': ['read_status', 'emergency_stop', 'read_holding_registers'],
            'emergency_stop': ['read_status', 'startup'],
            'initialize': ['read_status', 'startup'],
            'read_sensors': ['read_status', 'write_setpoint', 'read_sensors'],
            'shutdown': ['read_status', 'startup']
        }
    
    def _load_physical_constraints(self):
        """Load physical process constraints."""
        return {
            'temperature': {'min': -50, 'max': 200, 'rate_of_change': 5.0, 'unit': 'C'},
            'pressure': {'min': 0, 'max': 1000, 'rate_of_change': 50.0, 'unit': 'PSI'},
            'flow_rate': {'min': 0, 'max': 500, 'rate_of_change': 20.0, 'unit': 'L/min'},
            'valve_position': {'min': 0, 'max': 100, 'rate_of_change': 10.0, 'unit': '%'},
            'level': {'min': 0, 'max': 100, 'rate_of_change': 5.0, 'unit': '%'},
            'rpm': {'min': 0, 'max': 5000, 'rate_of_change': 100.0, 'unit': 'RPM'},
        }
    
    def validate_command(self, command_data):
        """
        Validate individual command.
        
        Args:
            command_data (dict): Command parameters
                Expected keys: 'function_code', 'quantity', 'address', 'value', 'timestamp'
        
        Returns:
            dict: Validation result with anomaly indicators
        """
        self.total_validations += 1
        anomalies = []
        severity = 'none'
        
        # Check function code validity
        if 'function_code' in command_data:
            fc = command_data['function_code']
            valid_fcs = [cmd.get('function_code', -1) for cmd in self.valid_commands.values()]
            if fc not in valid_fcs and fc != -1:
                anomalies.append(f"Invalid function code: {fc}")
                severity = 'high'
        
        # Check quantity bounds
        if 'quantity' in command_data:
            qty = command_data['quantity']
            # Find corresponding command
            for cmd_name, cmd_spec in self.valid_commands.items():
                if cmd_spec.get('function_code') == command_data.get('function_code'):
                    if 'max_quantity' in cmd_spec and qty > cmd_spec['max_quantity']:
                        anomalies.append(f"Quantity {qty} exceeds max {cmd_spec['max_quantity']}")
                        severity = 'medium' if severity == 'none' else severity
                    if qty <= 0:
                        anomalies.append(f"Invalid quantity: {qty}")
                        severity = 'medium' if severity == 'none' else severity
        
        # Check address ranges
        if 'address' in command_data:
            addr = command_data['address']
            if addr < 0 or addr > 65535:
                anomalies.append(f"Address {addr} out of valid range [0-65535]")
                severity = 'high'
        
        # Check for malformed values
        if 'value' in command_data:
            val = command_data['value']
            if isinstance(val, (int, float)):
                if not np.isfinite(val):
                    anomalies.append(f"Non-finite value detected: {val}")
                    severity = 'critical'
        
        # Add to history
        self.command_history.append(command_data)
        
        if len(anomalies) > 0:
            self.anomaly_count += 1
        
        return {
            'is_valid': len(anomalies) == 0,
            'anomalies': anomalies,
            'severity': severity,
            'command_type': command_data.get('type', 'unknown')
        }
    
    def validate_sequence(self, command_sequence=None):
        """
        Validate command sequence for unusual patterns.
        
        Args:
            command_sequence (list): List of commands (optional, uses history if not provided)
        
        Returns:
            dict: Sequence validation result
        """
        if command_sequence is None:
            command_sequence = list(self.command_history)
        
        anomalies = []
        severity = 'none'
        
        if len(command_sequence) < 2:
            return {'is_valid': True, 'anomalies': [], 'severity': 'none'}
        
        # Check for rapid repeated commands (potential DoS)
        if len(command_sequence) >= 5:
            last_5 = [cmd.get('type', cmd.get('function_code', 'unknown')) for cmd in command_sequence[-5:]]
            if len(set(last_5)) == 1:
                anomalies.append("Suspicious repeated commands detected (possible DoS)")
                severity = 'high'
        
        # Check command transition validity
        for i in range(len(command_sequence) - 1):
            current_cmd = command_sequence[i].get('type', 'unknown')
            next_cmd = command_sequence[i + 1].get('type', 'unknown')
            
            if current_cmd in self.valid_transitions:
                if next_cmd not in self.valid_transitions[current_cmd] and next_cmd != 'unknown':
                    anomalies.append(f"Invalid transition: {current_cmd} -> {next_cmd}")
                    severity = 'medium' if severity == 'none' else severity
        
        # Check for suspicious patterns (e.g., scan-like behavior)
        if self._detect_scan_pattern(command_sequence):
            anomalies.append("Potential scanning behavior detected")
            severity = 'critical'
        
        # Check for burst behavior (many commands in short time)
        if self._detect_burst_pattern(command_sequence):
            anomalies.append("Command burst detected (unusual traffic pattern)")
            severity = 'medium' if severity == 'none' else severity
        
        return {
            'is_valid': len(anomalies) == 0,
            'anomalies': anomalies,
            'severity': severity,
            'sequence_length': len(command_sequence)
        }
    
    def _detect_scan_pattern(self, commands):
        """Detect scanning patterns (sequential address access)."""
        if len(commands) < 10:
            return False
        
        # Check for sequential address increments
        addresses = [cmd.get('address', -1) for cmd in commands[-10:] if 'address' in cmd]
        if len(addresses) < 5:
            return False
        
        # Check if addresses are sequential
        sequential_count = sum(1 for i in range(len(addresses) - 1) 
                              if addresses[i] != -1 and addresses[i+1] == addresses[i] + 1)
        
        # If more than 80% are sequential, flag as scan
        if sequential_count / (len(addresses) - 1) > 0.8:
            return True
        
        return False
    
    def _detect_burst_pattern(self, commands):
        """Detect command burst patterns."""
        if len(commands) < 10:
            return False
        
        # Check if timestamps are available
        recent = commands[-10:]
        timestamps = [cmd.get('timestamp', None) for cmd in recent]
        
        # If no timestamps, check by count
        if all(t is None for t in timestamps):
            return False
        
        # Calculate time span
        valid_timestamps = [t for t in timestamps if t is not None]
        if len(valid_timestamps) < 5:
            return False
        
        time_span = max(valid_timestamps) - min(valid_timestamps)
        
        # If 10+ commands in < 1 second, flag as burst
        if time_span < 1.0 and len(valid_timestamps) >= 10:
            return True
        
        return False
    
    def validate_physical_constraints(self, sensor_data, previous_data=None):
        """
        Validate physical process constraints.
        
        Args:
            sensor_data (dict): Current sensor readings
            previous_data (dict): Previous sensor readings
        
        Returns:
            dict: Physical constraint violations
        """
        violations = []
        severity = 'none'
        
        for sensor, value in sensor_data.items():
            if sensor in self.physical_constraints:
                constraints = self.physical_constraints[sensor]
                
                # Check bounds
                if value < constraints['min'] or value > constraints['max']:
                    violations.append(
                        f"{sensor} value {value:.2f}{constraints['unit']} outside bounds "
                        f"[{constraints['min']}, {constraints['max']}]{constraints['unit']}"
                    )
                    severity = 'critical'
                
                # Check rate of change
                if previous_data and sensor in previous_data:
                    prev_value = previous_data[sensor]
                    rate = abs(value - prev_value)
                    if rate > constraints['rate_of_change']:
                        violations.append(
                            f"{sensor} rate of change {rate:.2f}{constraints['unit']}/step "
                            f"exceeds limit {constraints['rate_of_change']}{constraints['unit']}/step"
                        )
                        severity = 'high' if severity == 'none' else severity
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'severity': severity,
            'sensor_count': len(sensor_data)
        }
    
    def get_statistics(self):
        """Get validation statistics."""
        return {
            'total_validations': self.total_validations,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.total_validations, 1),
            'command_history_size': len(self.command_history)
        }
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.anomaly_count = 0
        self.total_validations = 0
        self.command_history.clear()


class SemanticAnalyzer:
    """
    Analyzes semantic meaning of ICS commands.
    Detects malicious intent even in syntactically valid commands.
    """
    
    def __init__(self):
        self.dangerous_operations = [
            'emergency_shutdown',
            'disable_safety',
            'override_limits',
            'format_memory',
            'firmware_update',
            'disable_alarm',
            'bypass_interlock',
            'force_output',
            'clear_logs'
        ]
        
        # Critical memory/address ranges
        self.sensitive_addresses = {
            'firmware': range(0x0000, 0x1000),
            'safety_config': range(0x1000, 0x2000),
            'calibration': range(0x2000, 0x3000),
            'admin_settings': range(0xF000, 0x10000),
        }
        
        self.risk_history = []
    
    def analyze_intent(self, command):
        """
        Analyze command intent for malicious patterns.
        
        Args:
            command (dict): Command to analyze
                Expected keys: 'operation', 'address', 'timestamp', 'requires_admin', 
                              'has_admin', 'source_ip', 'function_code'
        
        Returns:
            dict: Intent analysis result
        """
        risk_score = 0
        flags = []
        
        # Check for dangerous operations
        operation = command.get('operation', command.get('type', ''))
        if operation in self.dangerous_operations:
            risk_score += 50
            flags.append(f"Dangerous operation: {operation}")
        
        # Check sensitive address access
        address = command.get('address', -1)
        for region, addr_range in self.sensitive_addresses.items():
            if address in addr_range:
                risk_score += 30
                flags.append(f"Access to sensitive region: {region} (addr: {address})")
                break
        
        # Check for unusual timing (off-hours access)
        if 'timestamp' in command:
            timestamp = command['timestamp']
            if isinstance(timestamp, (int, float)):
                # Convert to datetime if it's a Unix timestamp
                dt = datetime.fromtimestamp(timestamp)
                hour = dt.hour
            elif hasattr(timestamp, 'hour'):
                hour = timestamp.hour
            else:
                hour = 12  # Default to normal hours
            
            if hour < 6 or hour > 22:
                risk_score += 20
                flags.append(f"Off-hours operation (hour: {hour})")
        
        # Check for privilege escalation attempts
        if command.get('requires_admin', False) and not command.get('has_admin', True):
            risk_score += 40
            flags.append("Potential privilege escalation attempt")
        
        # Check for write operations to read-only regions
        func_code = command.get('function_code', 0)
        if func_code in [5, 6, 15, 16]:  # Write operations in Modbus
            if address in self.sensitive_addresses.get('firmware', []):
                risk_score += 35
                flags.append("Write attempt to firmware region")
        
        # Check for unusual source patterns
        source_ip = command.get('source_ip', '')
        if source_ip and not self._is_trusted_source(source_ip):
            risk_score += 15
            flags.append(f"Command from untrusted source: {source_ip}")
        
        # Determine severity based on risk score
        if risk_score > 80:
            severity = 'critical'
        elif risk_score > 50:
            severity = 'high'
        elif risk_score > 20:
            severity = 'medium'
        else:
            severity = 'low'
        
        result = {
            'risk_score': min(risk_score, 100),  # Cap at 100
            'is_suspicious': risk_score > 50,
            'flags': flags,
            'severity': severity,
            'recommendation': self._get_recommendation(risk_score)
        }
        
        self.risk_history.append(risk_score)
        
        return result
    
    def _is_trusted_source(self, source_ip):
        """Check if source IP is from trusted range."""
        # Define trusted IP ranges (example: local network)
        trusted_ranges = [
            '192.168.',
            '10.',
            '172.16.',
            '127.0.0.1'
        ]
        
        return any(source_ip.startswith(prefix) for prefix in trusted_ranges)
    
    def _get_recommendation(self, risk_score):
        """Get recommendation based on risk score."""
        if risk_score > 80:
            return "BLOCK: Critical threat detected. Immediate action required."
        elif risk_score > 50:
            return "ALERT: High-risk operation. Manual review recommended."
        elif risk_score > 20:
            return "MONITOR: Moderate risk. Continue monitoring."
        else:
            return "ALLOW: Low risk operation."
    
    def get_risk_statistics(self):
        """Get risk analysis statistics."""
        if not self.risk_history:
            return {
                'avg_risk': 0,
                'max_risk': 0,
                'high_risk_count': 0,
                'total_analyzed': 0
            }
        
        return {
            'avg_risk': np.mean(self.risk_history),
            'max_risk': max(self.risk_history),
            'high_risk_count': sum(1 for r in self.risk_history if r > 50),
            'total_analyzed': len(self.risk_history)
        }


if __name__ == "__main__":
    print("Testing ICS Protocol Validators...")
    
    # Test Protocol Validator
    print("\n" + "="*60)
    print("Testing Protocol Validator")
    print("="*60)
    
    validator = ICSProtocolValidator(protocol='modbus')
    
    # Test valid command
    valid_cmd = {
        'function_code': 3,
        'address': 100,
        'quantity': 10,
        'type': 'read_holding_registers'
    }
    result = validator.validate_command(valid_cmd)
    print(f"\nValid command result: {result}")
    
    # Test invalid command
    invalid_cmd = {
        'function_code': 99,  # Invalid
        'address': 70000,  # Out of range
        'quantity': 5000,  # Too large
        'type': 'read_holding_registers'
    }
    result = validator.validate_command(invalid_cmd)
    print(f"\nInvalid command result: {result}")
    
    # Test sequence validation
    sequence = [
        {'type': 'startup', 'address': i*10}
        for i in range(10)
    ]
    result = validator.validate_sequence(sequence)
    print(f"\nSequence validation result: {result}")
    
    # Test Semantic Analyzer
    print("\n" + "="*60)
    print("Testing Semantic Analyzer")
    print("="*60)
    
    analyzer = SemanticAnalyzer()
    
    # Test normal command
    normal_cmd = {
        'operation': 'read_status',
        'address': 5000,
        'timestamp': datetime.now().timestamp(),
        'has_admin': True,
        'source_ip': '192.168.1.10'
    }
    result = analyzer.analyze_intent(normal_cmd)
    print(f"\nNormal command analysis: {result}")
    
    # Test suspicious command
    suspicious_cmd = {
        'operation': 'firmware_update',
        'address': 0x0500,  # Firmware region
        'timestamp': datetime.now().replace(hour=2).timestamp(),  # 2 AM
        'requires_admin': True,
        'has_admin': False,
        'source_ip': '203.45.67.89'
    }
    result = analyzer.analyze_intent(suspicious_cmd)
    print(f"\nSuspicious command analysis: {result}")
    
    print(f"\nValidator statistics: {validator.get_statistics()}")
    print(f"Analyzer statistics: {analyzer.get_risk_statistics()}")
