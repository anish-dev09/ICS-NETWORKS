"""
ICS-Specific Feature Engineering for Zero-Day Detection
Extracts contextual, behavioral, and physical features.
"""

import numpy as np
import pandas as pd
from collections import Counter, deque
from datetime import datetime


class ICSFeatureExtractor:
    """
    Extracts ICS-specific features for improved anomaly detection.
    Combines temporal, protocol, physical, and network features.
    """
    
    def __init__(self, window_size=10):
        """
        Initialize feature extractor.
        
        Args:
            window_size (int): Temporal window for sequence features
        """
        self.window_size = window_size
        self.command_history = deque(maxlen=window_size)
        self.sensor_history = deque(maxlen=window_size)
        self.network_history = deque(maxlen=window_size)
        self.feature_names = []
    
    def extract_temporal_features(self, data):
        """
        Extract temporal sequence features.
        
        Args:
            data (dict): Current data point with 'timestamp', 'type', etc.
        
        Returns:
            dict: Temporal features
        """
        features = {}
        
        # Add to history
        self.command_history.append(data)
        
        if len(self.command_history) >= 2:
            # Inter-arrival time
            current_time = data.get('timestamp', 0)
            previous_time = self.command_history[-2].get('timestamp', 0)
            features['inter_arrival_time'] = current_time - previous_time
            
            # Command frequency analysis
            cmd_types = [cmd.get('type', 'unknown') for cmd in self.command_history]
            cmd_counts = Counter(cmd_types)
            
            # Command diversity (Shannon entropy)
            total = len(cmd_types)
            entropy = 0
            for count in cmd_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            features['command_diversity'] = entropy
            
            # Most frequent command ratio
            features['max_command_frequency'] = max(cmd_counts.values()) / total if total > 0 else 0
            
            # Unique command types in window
            features['unique_commands'] = len(cmd_counts)
            
            # Burst detection (commands in short time)
            if len(self.command_history) >= 5:
                recent_times = [cmd.get('timestamp', 0) for cmd in list(self.command_history)[-5:]]
                time_span = max(recent_times) - min(recent_times)
                features['burst_rate'] = 5 / max(time_span, 0.001)
            else:
                features['burst_rate'] = 0
            
            # Command repetition pattern
            last_cmd = cmd_types[-1]
            repeat_count = sum(1 for cmd in cmd_types[-5:] if cmd == last_cmd)
            features['command_repetition'] = repeat_count / min(5, len(cmd_types))
        else:
            # Default values for first command
            features['inter_arrival_time'] = 0
            features['command_diversity'] = 0
            features['max_command_frequency'] = 1.0
            features['unique_commands'] = 1
            features['burst_rate'] = 0
            features['command_repetition'] = 1.0
        
        return features
    
    def extract_protocol_features(self, packet_data):
        """
        Extract protocol-specific features.
        
        Args:
            packet_data (dict): Packet data with 'payload', 'header', 'function_code', etc.
        
        Returns:
            dict: Protocol features
        """
        features = {}
        
        # Packet size features
        payload = packet_data.get('payload', [])
        header = packet_data.get('header', [])
        
        features['packet_size'] = len(payload) if isinstance(payload, (list, bytes, str)) else 0
        features['header_size'] = len(header) if isinstance(header, (list, bytes, str)) else 0
        features['total_size'] = features['packet_size'] + features['header_size']
        features['payload_ratio'] = features['packet_size'] / max(features['total_size'], 1)
        
        # Payload entropy
        features['payload_entropy'] = self._calculate_entropy(payload) if payload else 0
        
        # Protocol flags
        features['has_error_flag'] = int(packet_data.get('error_flag', False))
        features['requires_response'] = int(packet_data.get('requires_response', False))
        features['is_broadcast'] = int(packet_data.get('broadcast', False))
        
        # Function code analysis
        function_code = packet_data.get('function_code', 0)
        features['function_code'] = function_code
        
        # Categorize operation type
        features['is_read_operation'] = int(function_code in [1, 2, 3, 4])
        features['is_write_operation'] = int(function_code in [5, 6, 15, 16])
        features['is_diagnostic'] = int(function_code in [7, 8, 11, 17])
        
        # Address and quantity features
        features['address'] = packet_data.get('address', 0)
        features['quantity'] = packet_data.get('quantity', 0)
        
        # Address range analysis
        features['is_low_address'] = int(features['address'] < 1000)
        features['is_high_address'] = int(features['address'] > 60000)
        
        return features
    
    def extract_physical_features(self, sensor_data, previous_sensor_data=None):
        """
        Extract physical process features.
        
        Args:
            sensor_data (dict): Current sensor readings
            previous_sensor_data (dict): Previous sensor readings
        
        Returns:
            dict: Physical features
        """
        features = {}
        
        if not sensor_data:
            return features
        
        # Raw sensor values
        for sensor, value in sensor_data.items():
            features[f'{sensor}_value'] = value
        
        # Rate of change
        if previous_sensor_data:
            for sensor, value in sensor_data.items():
                if sensor in previous_sensor_data:
                    rate = value - previous_sensor_data[sensor]
                    features[f'{sensor}_rate'] = rate
                    features[f'{sensor}_abs_rate'] = abs(rate)
        
        # Add to history
        self.sensor_history.append(sensor_data)
        
        # Statistical features over window
        if len(self.sensor_history) >= self.window_size:
            for sensor in sensor_data.keys():
                values = [data.get(sensor, 0) for data in self.sensor_history]
                
                features[f'{sensor}_mean'] = np.mean(values)
                features[f'{sensor}_std'] = np.std(values)
                features[f'{sensor}_min'] = np.min(values)
                features[f'{sensor}_max'] = np.max(values)
                features[f'{sensor}_range'] = np.max(values) - np.min(values)
                
                # Trend analysis (linear regression slope)
                if len(values) > 1:
                    x = np.arange(len(values))
                    trend = np.polyfit(x, values, 1)[0]
                    features[f'{sensor}_trend'] = trend
                else:
                    features[f'{sensor}_trend'] = 0
                
                # Volatility (coefficient of variation)
                mean_val = np.mean(values)
                if mean_val != 0:
                    features[f'{sensor}_volatility'] = np.std(values) / abs(mean_val)
                else:
                    features[f'{sensor}_volatility'] = 0
        
        # Cross-sensor correlations (if multiple sensors)
        if len(sensor_data) >= 2 and len(self.sensor_history) >= 5:
            sensor_names = list(sensor_data.keys())
            for i in range(len(sensor_names)):
                for j in range(i+1, len(sensor_names)):
                    s1, s2 = sensor_names[i], sensor_names[j]
                    vals1 = [data.get(s1, 0) for data in self.sensor_history]
                    vals2 = [data.get(s2, 0) for data in self.sensor_history]
                    
                    # Pearson correlation
                    if np.std(vals1) > 0 and np.std(vals2) > 0:
                        corr = np.corrcoef(vals1, vals2)[0, 1]
                        features[f'corr_{s1}_{s2}'] = corr
        
        return features
    
    def extract_network_features(self, connection_data):
        """
        Extract network-level features.
        
        Args:
            connection_data (dict): Network connection metadata
        
        Returns:
            dict: Network features
        """
        features = {}
        
        # Connection metadata
        features['source_port'] = connection_data.get('source_port', 0)
        features['dest_port'] = connection_data.get('dest_port', 0)
        
        # Check if using standard ICS ports
        standard_ports = [502, 20000, 102, 44818, 2222, 1911]
        features['is_standard_port'] = int(features['dest_port'] in standard_ports)
        features['is_high_port'] = int(features['source_port'] > 49152)
        
        # Traffic volume
        features['bytes_sent'] = connection_data.get('bytes_sent', 0)
        features['bytes_received'] = connection_data.get('bytes_received', 0)
        features['packets_sent'] = connection_data.get('packets_sent', 0)
        features['packets_received'] = connection_data.get('packets_received', 0)
        
        # Traffic ratios
        total_bytes = features['bytes_sent'] + features['bytes_received']
        if total_bytes > 0:
            features['bytes_sent_ratio'] = features['bytes_sent'] / total_bytes
        else:
            features['bytes_sent_ratio'] = 0
        
        # Connection duration and rates
        duration = connection_data.get('duration', 0)
        features['duration'] = duration
        
        if duration > 0:
            features['avg_packet_rate'] = features['packets_sent'] / duration
            features['avg_byte_rate'] = features['bytes_sent'] / duration
        else:
            features['avg_packet_rate'] = 0
            features['avg_byte_rate'] = 0
        
        # Average packet size
        if features['packets_sent'] > 0:
            features['avg_packet_size'] = features['bytes_sent'] / features['packets_sent']
        else:
            features['avg_packet_size'] = 0
        
        # Add to history
        self.network_history.append(connection_data)
        
        # Network behavior over time
        if len(self.network_history) >= 5:
            recent_bytes = [conn.get('bytes_sent', 0) for conn in self.network_history]
            recent_packets = [conn.get('packets_sent', 0) for conn in self.network_history]
            
            features['bytes_trend'] = recent_bytes[-1] - recent_bytes[0]
            features['packets_trend'] = recent_packets[-1] - recent_packets[0]
            features['traffic_volatility'] = np.std(recent_bytes)
        
        return features
    
    def extract_contextual_features(self, timestamp=None):
        """
        Extract contextual features (time of day, day of week, etc.).
        
        Args:
            timestamp (float or datetime): Timestamp of the data point
        
        Returns:
            dict: Contextual features
        """
        features = {}
        
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        
        # Time-based features
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = int(timestamp.weekday() >= 5)
        features['is_business_hours'] = int(8 <= timestamp.hour < 18)
        features['is_night'] = int(timestamp.hour < 6 or timestamp.hour >= 22)
        
        # Cyclical encoding for hour (24-hour cycle)
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        features['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
        
        return features
    
    def _calculate_entropy(self, data):
        """
        Calculate Shannon entropy of data.
        
        Args:
            data: Sequence of values
        
        Returns:
            float: Entropy value
        """
        if not data or len(data) == 0:
            return 0
        
        # Convert to list if needed
        if isinstance(data, (str, bytes)):
            data = list(data)
        
        counter = Counter(data)
        total = len(data)
        
        entropy = 0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def extract_all_features(self, packet_data=None, sensor_data=None, 
                           connection_data=None, previous_sensor_data=None):
        """
        Extract comprehensive feature set from all available data.
        
        Args:
            packet_data (dict): Packet/protocol data
            sensor_data (dict): Sensor readings
            connection_data (dict): Network connection data
            previous_sensor_data (dict): Previous sensor readings
        
        Returns:
            dict: All extracted features
        """
        all_features = {}
        
        # Temporal features (if packet data available)
        if packet_data:
            all_features.update(self.extract_temporal_features(packet_data))
            all_features.update(self.extract_protocol_features(packet_data))
            
            # Contextual features
            timestamp = packet_data.get('timestamp')
            all_features.update(self.extract_contextual_features(timestamp))
        
        # Physical features
        if sensor_data:
            all_features.update(self.extract_physical_features(sensor_data, previous_sensor_data))
        
        # Network features
        if connection_data:
            all_features.update(self.extract_network_features(connection_data))
        
        return all_features
    
    def extract_features_batch(self, data_list):
        """
        Extract features for a batch of data points.
        
        Args:
            data_list (list): List of data dictionaries
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        feature_list = []
        
        for i, data in enumerate(data_list):
            packet_data = data.get('packet', None)
            sensor_data = data.get('sensor', None)
            connection_data = data.get('connection', None)
            
            # Get previous sensor data if available
            if i > 0 and 'sensor' in data_list[i-1]:
                previous_sensor = data_list[i-1]['sensor']
            else:
                previous_sensor = None
            
            features = self.extract_all_features(
                packet_data=packet_data,
                sensor_data=sensor_data,
                connection_data=connection_data,
                previous_sensor_data=previous_sensor
            )
            
            feature_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_list)
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Store feature names
        self.feature_names = list(df.columns)
        
        return df
    
    def get_feature_names(self):
        """Get list of all feature names."""
        return self.feature_names
    
    def reset(self):
        """Reset history buffers."""
        self.command_history.clear()
        self.sensor_history.clear()
        self.network_history.clear()


if __name__ == "__main__":
    print("Testing ICS Feature Extractor...")
    
    # Create feature extractor
    extractor = ICSFeatureExtractor(window_size=5)
    
    # Test with synthetic data
    print("\n" + "="*60)
    print("Extracting features from sample data")
    print("="*60)
    
    # Sample packet data
    packet = {
        'timestamp': datetime.now().timestamp(),
        'type': 'read_holding_registers',
        'function_code': 3,
        'address': 100,
        'quantity': 10,
        'payload': b'test_payload_data',
        'header': b'header',
        'error_flag': False,
        'requires_response': True
    }
    
    # Sample sensor data
    sensor = {
        'temperature': 75.5,
        'pressure': 150.0,
        'flow_rate': 50.2,
        'valve_position': 45.0
    }
    
    # Sample network data
    network = {
        'source_port': 50123,
        'dest_port': 502,
        'bytes_sent': 1024,
        'bytes_received': 512,
        'packets_sent': 10,
        'packets_received': 5,
        'duration': 2.5
    }
    
    # Extract features for multiple time steps
    data_sequence = []
    for i in range(10):
        data_point = {
            'packet': {
                'timestamp': datetime.now().timestamp() + i,
                'type': ['read_coils', 'read_holding_registers', 'write_single_register'][i % 3],
                'function_code': [1, 3, 6][i % 3],
                'address': 100 + i * 10,
                'quantity': 5 + i,
                'payload': b'data' * (i + 1),
                'header': b'hdr',
                'error_flag': False,
                'requires_response': True
            },
            'sensor': {
                'temperature': 75.0 + np.random.randn() * 2,
                'pressure': 150.0 + np.random.randn() * 5,
                'flow_rate': 50.0 + np.random.randn() * 3,
                'valve_position': 45.0 + np.random.randn() * 2
            },
            'connection': {
                'source_port': 50000 + i,
                'dest_port': 502,
                'bytes_sent': 1000 + i * 100,
                'bytes_received': 500 + i * 50,
                'packets_sent': 10 + i,
                'packets_received': 5 + i,
                'duration': 2.0 + i * 0.5
            }
        }
        data_sequence.append(data_point)
    
    # Extract features for batch
    feature_df = extractor.extract_features_batch(data_sequence)
    
    print(f"\nExtracted {len(feature_df.columns)} features:")
    print(f"Feature names: {extractor.get_feature_names()[:10]}... (showing first 10)")
    
    print(f"\nFeature matrix shape: {feature_df.shape}")
    print(f"\nFirst few rows:")
    print(feature_df.head())
    
    print(f"\nFeature statistics:")
    print(feature_df.describe())
    
    print("\n" + "="*60)
    print("Feature extraction complete!")
    print("="*60)
