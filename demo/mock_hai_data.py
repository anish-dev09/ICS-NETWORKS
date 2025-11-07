"""
Mock HAI Dataset Generator
Generates realistic ICS sensor data for demo purposes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MockHAIData:
    """Generate mock HAI (Hardware-in-the-Loop Augmented ICS Security) dataset"""
    
    def __init__(self, n_samples=50000, attack_ratio=0.3, random_state=42):
        """
        Initialize mock data generator
        
        Args:
            n_samples (int): Number of samples to generate
            attack_ratio (float): Ratio of attack samples (0.0 to 1.0)
            random_state (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.attack_ratio = attack_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
        # HAI dataset has 78-86 sensor columns depending on version
        # Simulating industrial control system sensors
        self.sensor_types = {
            'P': 'Pressure',      # 20 sensors
            'F': 'Flow',          # 15 sensors
            'L': 'Level',         # 10 sensors
            'T': 'Temperature',   # 12 sensors
            'V': 'Valve',         # 15 sensors
            'PUMP': 'Pump'        # 8 sensors
        }
        
    def generate_sensor_columns(self):
        """Generate sensor column names matching HAI structure"""
        columns = ['timestamp']
        
        # Pressure sensors (P_xxx)
        for i in range(1, 21):
            columns.append(f'P_{i:03d}')
        
        # Flow sensors (F_xxx)
        for i in range(1, 16):
            columns.append(f'F_{i:03d}')
        
        # Level sensors (L_xxx)
        for i in range(1, 11):
            columns.append(f'L_{i:03d}')
        
        # Temperature sensors (T_xxx)
        for i in range(1, 13):
            columns.append(f'T_{i:03d}')
        
        # Valve position sensors (V_xxx)
        for i in range(1, 16):
            columns.append(f'V_{i:03d}')
        
        # Pump status sensors (PUMP_xxx)
        for i in range(1, 9):
            columns.append(f'PUMP_{i:03d}')
        
        # Add attack label
        columns.append('attack')
        
        return columns
    
    def generate_normal_data(self, n_samples):
        """Generate normal operating data"""
        data = {}
        
        # Timestamp
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        data['timestamp'] = [start_time + timedelta(seconds=i) for i in range(n_samples)]
        
        # Pressure sensors (bar) - normally distributed around operating pressure
        for i in range(1, 21):
            base_pressure = np.random.uniform(2.0, 8.0)
            data[f'P_{i:03d}'] = np.random.normal(base_pressure, 0.2, n_samples)
        
        # Flow sensors (L/min) - normally distributed
        for i in range(1, 16):
            base_flow = np.random.uniform(50.0, 200.0)
            data[f'F_{i:03d}'] = np.random.normal(base_flow, 5.0, n_samples)
        
        # Level sensors (%) - bounded between 30-70% normal operation
        for i in range(1, 11):
            base_level = np.random.uniform(40.0, 60.0)
            data[f'L_{i:03d}'] = np.clip(np.random.normal(base_level, 5.0, n_samples), 0, 100)
        
        # Temperature sensors (°C) - normally distributed
        for i in range(1, 13):
            base_temp = np.random.uniform(20.0, 80.0)
            data[f'T_{i:03d}'] = np.random.normal(base_temp, 2.0, n_samples)
        
        # Valve position sensors (%) - discrete states
        for i in range(1, 16):
            # Valves mostly at stable positions (0%, 50%, 100%)
            positions = np.random.choice([0.0, 50.0, 100.0], n_samples, p=[0.3, 0.4, 0.3])
            data[f'V_{i:03d}'] = positions + np.random.normal(0, 2.0, n_samples)
        
        # Pump status (0=off, 1=on) - binary
        for i in range(1, 9):
            data[f'PUMP_{i:03d}'] = np.random.choice([0.0, 1.0], n_samples, p=[0.3, 0.7])
        
        # Attack label (0 = normal)
        data['attack'] = np.zeros(n_samples, dtype=int)
        
        return pd.DataFrame(data)
    
    def generate_attack_data(self, n_samples):
        """Generate attack data with anomalous patterns"""
        data = {}
        
        # Timestamp
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        data['timestamp'] = [start_time + timedelta(seconds=i) for i in range(n_samples)]
        
        # Attack types: pressure spike, flow disruption, temperature anomaly, valve manipulation
        attack_types = np.random.choice(['pressure', 'flow', 'level', 'temperature', 'valve'], n_samples)
        
        # Pressure sensors - add spikes during pressure attacks
        for i in range(1, 21):
            base_pressure = np.random.uniform(2.0, 8.0)
            normal_vals = np.random.normal(base_pressure, 0.2, n_samples)
            # Add anomalies
            attack_mask = attack_types == 'pressure'
            normal_vals[attack_mask] += np.random.uniform(5.0, 15.0, attack_mask.sum())
            data[f'P_{i:03d}'] = normal_vals
        
        # Flow sensors - disruptions during flow attacks
        for i in range(1, 16):
            base_flow = np.random.uniform(50.0, 200.0)
            normal_vals = np.random.normal(base_flow, 5.0, n_samples)
            attack_mask = attack_types == 'flow'
            normal_vals[attack_mask] *= np.random.uniform(0.1, 0.3, attack_mask.sum())  # Drop to 10-30%
            data[f'F_{i:03d}'] = normal_vals
        
        # Level sensors - abnormal levels during level attacks
        for i in range(1, 11):
            base_level = np.random.uniform(40.0, 60.0)
            normal_vals = np.clip(np.random.normal(base_level, 5.0, n_samples), 0, 100)
            attack_mask = attack_types == 'level'
            normal_vals[attack_mask] = np.random.uniform(0, 20, attack_mask.sum())  # Dangerously low
            data[f'L_{i:03d}'] = normal_vals
        
        # Temperature sensors - spikes during temperature attacks
        for i in range(1, 13):
            base_temp = np.random.uniform(20.0, 80.0)
            normal_vals = np.random.normal(base_temp, 2.0, n_samples)
            attack_mask = attack_types == 'temperature'
            normal_vals[attack_mask] += np.random.uniform(30.0, 60.0, attack_mask.sum())
            data[f'T_{i:03d}'] = normal_vals
        
        # Valve position sensors - erratic during valve attacks
        for i in range(1, 16):
            positions = np.random.choice([0.0, 50.0, 100.0], n_samples, p=[0.3, 0.4, 0.3])
            normal_vals = positions + np.random.normal(0, 2.0, n_samples)
            attack_mask = attack_types == 'valve'
            normal_vals[attack_mask] = np.random.uniform(0, 100, attack_mask.sum())  # Erratic movement
            data[f'V_{i:03d}'] = normal_vals
        
        # Pump status - may be manipulated
        for i in range(1, 9):
            data[f'PUMP_{i:03d}'] = np.random.choice([0.0, 1.0], n_samples, p=[0.5, 0.5])
        
        # Attack label (1 = attack)
        data['attack'] = np.ones(n_samples, dtype=int)
        
        return pd.DataFrame(data)
    
    def generate_dataset(self):
        """Generate complete mock dataset with normal and attack samples"""
        n_attacks = int(self.n_samples * self.attack_ratio)
        n_normal = self.n_samples - n_attacks
        
        print(f"Generating {n_normal:,} normal samples and {n_attacks:,} attack samples...")
        
        # Generate both types
        normal_df = self.generate_normal_data(n_normal)
        attack_df = self.generate_attack_data(n_attacks)
        
        # Combine and shuffle
        df = pd.concat([normal_df, attack_df], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"✅ Generated {len(df):,} samples with {len(df.columns)} columns")
        print(f"   Normal: {(df['attack']==0).sum():,} | Attacks: {(df['attack']==1).sum():,}")
        
        return df


def generate_mock_hai_data(n_samples=50000, attack_ratio=0.3, random_state=42):
    """
    Convenience function to generate mock HAI data
    
    Args:
        n_samples (int): Number of samples
        attack_ratio (float): Ratio of attack samples
        random_state (int): Random seed
    
    Returns:
        pd.DataFrame: Mock HAI dataset
    """
    generator = MockHAIData(n_samples, attack_ratio, random_state)
    return generator.generate_dataset()


if __name__ == "__main__":
    # Test the generator
    print("Testing Mock HAI Data Generator...")
    df = generate_mock_hai_data(n_samples=1000, attack_ratio=0.3)
    print("\nDataset shape:", df.shape)
    print("\nFirst few columns:", df.columns[:10].tolist())
    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes.value_counts())
    print("\nAttack distribution:")
    print(df['attack'].value_counts())
