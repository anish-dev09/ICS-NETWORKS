"""
Sequence Generator for CNN Training

This module provides functionality to convert tabular ICS sensor data
into sequences suitable for CNN training.

Author: Anish
Date: November 6, 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SequenceGenerator:
    """
    Generate sequences from ICS sensor data for CNN training.
    
    Converts tabular time-series data into 3D sequences:
    - Input: DataFrame with sensor columns
    - Output: (samples, timesteps, sensors) array
    """
    
    def __init__(
        self, 
        window_size: int = 60,
        step: int = 1,
        scale: bool = True
    ):
        """
        Initialize sequence generator.
        
        Args:
            window_size: Number of timesteps per sequence (default: 60)
            step: Step size for sliding window (default: 1)
            scale: Whether to standardize features (default: True)
        """
        self.window_size = window_size
        self.step = step
        self.scale = scale
        self.scaler: Optional[StandardScaler] = StandardScaler() if scale else None
        self.sensor_cols: Optional[List[str]] = None
        
    def fit(self, df: pd.DataFrame, sensor_cols: Optional[List[str]] = None):
        """
        Fit the scaler on training data.
        
        Args:
            df: Training dataframe
            sensor_cols: List of sensor column names (if None, exclude 'attack*' columns)
        """
        # Identify sensor columns
        if sensor_cols is None:
            # Exclude attack labels, timestamp/time, and attack_ columns
            exclude_cols = ['timestamp', 'time', 'attack'] + [col for col in df.columns if col.startswith('attack_')]
            self.sensor_cols = [col for col in df.columns if col not in exclude_cols]
        else:
            self.sensor_cols = sensor_cols
            
        # Fit scaler
        if self.scale and self.scaler is not None:
            self.scaler.fit(df[self.sensor_cols])
            
        return self
    
    def transform(
        self, 
        df: pd.DataFrame,
        label_col: str = 'attack'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform dataframe into sequences.
        
        Args:
            df: Input dataframe
            label_col: Name of label column
            
        Returns:
            X: Sequences array (n_samples, window_size, n_sensors)
            y: Labels array (n_samples,)
        """
        # Get sensor data
        sensor_data = df[self.sensor_cols].values
        
        # Scale if needed
        if self.scale and self.scaler is not None:
            sensor_data = self.scaler.transform(df[self.sensor_cols])
        
        # Get labels
        labels = df[label_col].values if label_col in df.columns else None
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(sensor_data) - self.window_size + 1, self.step):
            # Extract window
            sequence = sensor_data[i:i + self.window_size]
            X_sequences.append(sequence)
            
            # Label is from the last timestep in the window
            if labels is not None:
                label = labels[i + self.window_size - 1]
                y_sequences.append(label)
        
        X = np.array(X_sequences)
        y = np.array(y_sequences) if labels is not None else None
        
        return X, y
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        sensor_cols: Optional[List[str]] = None,
        label_col: str = 'attack'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            sensor_cols: List of sensor column names
            label_col: Name of label column
            
        Returns:
            X: Sequences array (n_samples, window_size, n_sensors)
            y: Labels array (n_samples,)
        """
        self.fit(df, sensor_cols)
        return self.transform(df, label_col)
    
    def get_info(self) -> dict:
        """
        Get information about the sequence generator.
        
        Returns:
            Dictionary with configuration info
        """
        return {
            'window_size': self.window_size,
            'step': self.step,
            'scale': self.scale,
            'n_sensors': len(self.sensor_cols) if self.sensor_cols else None,
            'sensor_cols': self.sensor_cols
        }


def create_balanced_sequences(
    X: np.ndarray,
    y: np.ndarray,
    balance_ratio: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance sequences by undersampling the majority class.
    
    Args:
        X: Sequences array
        y: Labels array
        balance_ratio: Desired ratio of minority/majority (default: 0.5)
        random_state: Random seed
        
    Returns:
        X_balanced: Balanced sequences
        y_balanced: Balanced labels
    """
    np.random.seed(random_state)
    
    # Get indices for each class
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    
    n_attacks = len(attack_idx)
    n_normal_target = int(n_attacks / balance_ratio)
    
    # Undersample normal class
    if n_normal_target < len(normal_idx):
        normal_idx_sampled = np.random.choice(
            normal_idx, 
            size=n_normal_target, 
            replace=False
        )
    else:
        normal_idx_sampled = normal_idx
    
    # Combine indices
    selected_idx = np.concatenate([normal_idx_sampled, attack_idx])
    np.random.shuffle(selected_idx)
    
    return X[selected_idx], y[selected_idx]


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into train, validation, and test sets.
    
    Args:
        X: Sequences array
        y: Labels array
        train_size: Proportion for training (default: 0.7)
        val_size: Proportion for validation (default: 0.15)
        random_state: Random seed
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split points
    train_end = int(train_size * n_samples)
    val_end = int((train_size + val_size) * n_samples)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Test the sequence generator
    print("Testing SequenceGenerator...")
    
    # Create dummy data
    n_samples = 1000
    n_sensors = 10
    
    dummy_data = pd.DataFrame(
        np.random.randn(n_samples, n_sensors),
        columns=[f'sensor_{i}' for i in range(n_sensors)]
    )
    dummy_data['attack'] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Create sequences
    gen = SequenceGenerator(window_size=60, step=10)
    X, y = gen.fit_transform(dummy_data)
    
    print(f"✓ Input shape: {dummy_data.shape}")
    print(f"✓ Output X shape: {X.shape}")
    if y is not None:
        print(f"✓ Output y shape: {y.shape}")
        print(f"✓ Attack ratio: {y.sum() / len(y):.2%}")
        
        # Test balancing
        X_bal, y_bal = create_balanced_sequences(X, y)
        print(f"\n✓ Balanced X shape: {X_bal.shape}")
        print(f"✓ Balanced attack ratio: {y_bal.sum() / len(y_bal):.2%}")
        
        # Test splitting
        X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X, y)
        print(f"\n✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("\n✅ SequenceGenerator test passed!")
