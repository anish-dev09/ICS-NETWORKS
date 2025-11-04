"""
Data loading utilities for ICS datasets.
Supports multiple datasets: SWaT, WADI, Gas Pipeline, HAI, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ICSDataLoader:
    """
    Generic data loader for ICS datasets.
    """
    
    def __init__(self, dataset_name='swat', config=None):
        """
        Initialize data loader.
        
        Args:
            dataset_name (str): Name of dataset ('swat', 'wadi', 'gas_pipeline', 'hai')
            config (dict, optional): Configuration dictionary
        """
        self.dataset_name = dataset_name.lower()
        self.config = config
        
        if config is None:
            from src.utils.config_utils import load_config
            self.config = load_config()
    
    def load_data(self, data_type='train', nrows=None):
        """
        Load dataset based on type.
        
        Args:
            data_type (str): 'train' or 'test'
            nrows (int, optional): Number of rows to load (for testing)
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if self.dataset_name == 'swat':
            return self._load_swat(data_type, nrows)
        elif self.dataset_name == 'wadi':
            return self._load_wadi(data_type, nrows)
        elif self.dataset_name == 'gas_pipeline':
            return self._load_gas_pipeline(data_type, nrows)
        elif self.dataset_name == 'hai':
            return self._load_hai(data_type, nrows)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_swat(self, data_type='train', nrows=None):
        """Load SWaT dataset."""
        from src.utils.config_utils import get_data_path
        
        try:
            file_path = get_data_path('swat', data_type, self.config)
            
            print(f"Loading SWaT {data_type} data from: {file_path}")
            
            # SWaT specific loading
            df = pd.read_csv(file_path, nrows=nrows)
            
            # Convert timestamp column
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
            
            # Handle attack labels
            if 'Normal/Attack' in df.columns:
                df['label'] = (df['Normal/Attack'] == 'Attack').astype(int)
            elif 'Attack' in df.columns:
                df['label'] = df['Attack'].astype(int)
            
            print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ SWaT dataset not found. Please download it first.")
            print(f"   See docs/DATASET_GUIDE.md for instructions.")
            return None
    
    def _load_wadi(self, data_type='train', nrows=None):
        """Load WADI dataset."""
        from src.utils.config_utils import get_data_path
        
        try:
            file_path = get_data_path('wadi', data_type, self.config)
            
            print(f"Loading WADI {data_type} data from: {file_path}")
            
            df = pd.read_csv(file_path, nrows=nrows)
            
            # Convert timestamp
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            
            # Handle labels
            if 'Attack' in df.columns:
                df['label'] = df['Attack'].astype(int)
            
            print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ WADI dataset not found. Please download it first.")
            print(f"   See docs/DATASET_GUIDE.md for instructions.")
            return None
    
    def _load_gas_pipeline(self, data_type='train', nrows=None):
        """Load Gas Pipeline dataset."""
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'data' / 'raw' / 'gas_pipeline'
        
        try:
            # Gas Pipeline has different file naming
            if data_type == 'train':
                file_path = data_dir / 'normal_data.csv'
            else:
                file_path = data_dir / 'attack_data.csv'
            
            print(f"Loading Gas Pipeline {data_type} data from: {file_path}")
            
            df = pd.read_csv(file_path, nrows=nrows)
            
            # Handle labels based on data type
            if data_type == 'train':
                df['label'] = 0  # Normal
            else:
                # Check if label column exists
                if 'Attack' in df.columns:
                    df['label'] = df['Attack'].astype(int)
                elif 'label' not in df.columns:
                    df['label'] = 1  # All attack data
            
            print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Gas Pipeline dataset not found.")
            print(f"   Expected location: {data_dir}")
            print(f"   See docs/DATASET_GUIDE.md for download instructions.")
            return None
    
    def _load_hai(self, data_type='train', nrows=None):
        """Load HAI dataset."""
        from src.utils.config_utils import get_data_path
        
        try:
            file_path = get_data_path('hai', data_type, self.config)
            
            print(f"Loading HAI {data_type} data from: {file_path}")
            
            df = pd.read_csv(file_path, nrows=nrows)
            
            # HAI has 'attack' column
            if 'attack' in df.columns:
                df['label'] = (df['attack'] != 'Normal').astype(int)
            
            print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ HAI dataset not found. Please download it first.")
            print(f"   See docs/DATASET_GUIDE.md for instructions.")
            return None
    
    def get_sensor_columns(self, df):
        """
        Extract sensor/feature columns (exclude timestamp, labels, etc.).
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            list: List of sensor column names
        """
        exclude_cols = ['Timestamp', 'Date', 'Time', 'label', 'Normal/Attack', 
                       'Attack', 'attack', 'datetime']
        
        sensor_cols = [col for col in df.columns if col not in exclude_cols]
        
        return sensor_cols
    
    def split_features_labels(self, df):
        """
        Split dataframe into features and labels.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        sensor_cols = self.get_sensor_columns(df)
        
        X = df[sensor_cols].values
        
        if 'label' in df.columns:
            y = df['label'].values
        else:
            # If no label column, assume all normal
            y = np.zeros(len(df))
        
        return X, y
    
    def get_dataset_info(self, df):
        """
        Get summary information about the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            dict: Dataset information
        """
        sensor_cols = self.get_sensor_columns(df)
        
        info = {
            'total_samples': len(df),
            'num_features': len(sensor_cols),
            'feature_names': sensor_cols,
            'has_labels': 'label' in df.columns,
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        if 'label' in df.columns:
            info['num_normal'] = (df['label'] == 0).sum()
            info['num_attack'] = (df['label'] == 1).sum()
            info['attack_ratio'] = info['num_attack'] / info['total_samples']
        
        return info


def quick_load(dataset_name='swat', data_type='train', nrows=None):
    """
    Quick function to load dataset.
    
    Args:
        dataset_name (str): Dataset name
        data_type (str): 'train' or 'test'
        nrows (int, optional): Number of rows to load
    
    Returns:
        pd.DataFrame: Loaded data
    
    Example:
        >>> df = quick_load('swat', 'train', nrows=1000)
    """
    loader = ICSDataLoader(dataset_name)
    return loader.load_data(data_type, nrows)


if __name__ == "__main__":
    # Test loading
    print("=" * 60)
    print("Testing ICS Data Loader")
    print("=" * 60)
    
    datasets = ['swat', 'wadi', 'gas_pipeline', 'hai']
    
    for dataset in datasets:
        print(f"\n📊 Testing {dataset.upper()} dataset...")
        loader = ICSDataLoader(dataset)
        
        # Try to load small sample
        df = loader.load_data('train', nrows=100)
        
        if df is not None:
            print(f"✅ Successfully loaded {dataset}")
            info = loader.get_dataset_info(df)
            print(f"   Samples: {info['total_samples']}")
            print(f"   Features: {info['num_features']}")
            print(f"   Has labels: {info['has_labels']}")
        else:
            print(f"⚠️  {dataset} not available yet")
        
        print("-" * 60)
