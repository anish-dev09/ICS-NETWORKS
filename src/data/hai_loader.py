"""
HAI Dataset Loader and Preprocessor
Hardware-in-the-loop Augmented ICS Security Dataset
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HAIDataLoader:
    """
    Specialized loader for HAI (Hardware-in-the-loop Augmented ICS) dataset.
    """
    
    def __init__(self, version='21.03'):
        """
        Initialize HAI data loader.
        
        Args:
            version (str): HAI dataset version ('20.07', '21.03', '22.04', '23.05')
        """
        self.version = version
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'raw' / 'hai' / f'hai-{version}'
        
        # HAI dataset has 4 processes with different sensors
        self.process_names = ['P1', 'P2', 'P3', 'P4']
        
    def load_train_data(self, train_num=1, nrows=None):
        """
        Load HAI training data.
        
        Args:
            train_num (int): Training file number (1, 2, 3, etc.)
            nrows (int, optional): Number of rows to load
        
        Returns:
            pd.DataFrame: Training data
        """
        file_path = self.data_dir / f'train{train_num}.csv.gz'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_path}")
        
        print(f"Loading HAI training data from: {file_path}")
        
        # Read compressed CSV
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, nrows=nrows)
        
        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def load_test_data(self, test_num=1, nrows=None):
        """
        Load HAI test data (contains attacks).
        
        Args:
            test_num (int): Test file number (1, 2, 3, 4, 5)
            nrows (int, optional): Number of rows to load
        
        Returns:
            pd.DataFrame: Test data with attacks
        """
        file_path = self.data_dir / f'test{test_num}.csv.gz'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        print(f"Loading HAI test data from: {file_path}")
        
        # Read compressed CSV
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, nrows=nrows)
        
        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def get_sensor_columns(self, df):
        """
        Extract sensor feature columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            list: List of sensor column names
        """
        # HAI dataset structure: timestamp, attack, P1_*, P2_*, P3_*, P4_*
        exclude_cols = ['timestamp', 'attack']
        sensor_cols = [col for col in df.columns if col not in exclude_cols]
        
        return sensor_cols
    
    def split_features_labels(self, df):
        """
        Split dataframe into features (X) and labels (y).
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            tuple: (X, y) where X is features array and y is labels array
        """
        sensor_cols = self.get_sensor_columns(df)
        
        # Convert to numeric, ensuring all columns are float
        X = df[sensor_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        
        if 'attack' in df.columns:
            # HAI: attack column contains attack names ('Normal' or attack type)
            # It could be numeric (0/1) or string ('Normal'/attack name)
            if df['attack'].dtype == 'object':
                y = (df['attack'] != 'Normal').astype(int).values
            else:
                y = df['attack'].astype(int).values
        else:
            # If no attack column, assume all normal
            y = np.zeros(len(df))
        
        return X, y
    
    def get_attack_types(self, df):
        """
        Get unique attack types in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            list: List of unique attack types
        """
        if 'attack' in df.columns:
            attacks = df['attack'].unique()
            return [a for a in attacks if a != 'Normal']
        return []
    
    def get_dataset_info(self, df):
        """
        Get comprehensive dataset information.
        
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
            'has_labels': 'attack' in df.columns,
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        if 'attack' in df.columns:
            info['num_normal'] = (df['attack'] == 'Normal').sum()
            info['num_attack'] = (df['attack'] != 'Normal').sum()
            info['attack_ratio'] = info['num_attack'] / info['total_samples']
            info['attack_types'] = self.get_attack_types(df)
            info['num_attack_types'] = len(info['attack_types'])
        
        # Process-specific info
        for process in self.process_names:
            process_cols = [col for col in sensor_cols if col.startswith(f'{process}_')]
            info[f'{process}_sensors'] = len(process_cols)
        
        return info
    
    def get_process_data(self, df, process='P1'):
        """
        Extract data for a specific process.
        
        Args:
            df (pd.DataFrame): Input dataframe
            process (str): Process name ('P1', 'P2', 'P3', 'P4')
        
        Returns:
            pd.DataFrame: Data for specified process
        """
        process_cols = [col for col in df.columns if col.startswith(f'{process}_') or col in ['timestamp', 'attack']]
        return df[process_cols]
    
    def print_dataset_summary(self, info):
        """
        Print formatted dataset summary.
        
        Args:
            info (dict): Dataset info from get_dataset_info()
        """
        print("\n" + "="*70)
        print(f"HAI DATASET SUMMARY (Version {self.version})")
        print("="*70)
        print(f"Total Samples:       {info['total_samples']:,}")
        print(f"Total Features:      {info['num_features']}")
        print(f"Missing Values:      {info['missing_values']}")
        print(f"Memory Usage:        {info['memory_usage_mb']:.2f} MB")
        
        if info['has_labels']:
            print(f"\nAttack Information:")
            print(f"  Normal Samples:    {info['num_normal']:,}")
            print(f"  Attack Samples:    {info['num_attack']:,}")
            print(f"  Attack Ratio:      {info['attack_ratio']:.2%}")
            print(f"  Attack Types:      {info['num_attack_types']}")
            
            if info['attack_types']:
                print(f"\nAttack Types Found:")
                for i, attack in enumerate(info['attack_types'][:10], 1):  # Show first 10
                    print(f"    {i}. {attack}")
                if len(info['attack_types']) > 10:
                    print(f"    ... and {len(info['attack_types']) - 10} more")
        
        print(f"\nProcess Distribution:")
        for process in self.process_names:
            key = f'{process}_sensors'
            if key in info:
                print(f"  {process}: {info[key]} sensors")
        
        print("="*70 + "\n")


def quick_load_hai(version='21.03', data_type='train', num=1, nrows=None):
    """
    Quick function to load HAI dataset.
    
    Args:
        version (str): HAI version
        data_type (str): 'train' or 'test'
        num (int): File number
        nrows (int, optional): Number of rows to load
    
    Returns:
        pd.DataFrame: Loaded data
    
    Example:
        >>> df = quick_load_hai('21.03', 'train', 1, nrows=10000)
        >>> df = quick_load_hai('21.03', 'test', 1)
    """
    loader = HAIDataLoader(version)
    
    if data_type == 'train':
        return loader.load_train_data(num, nrows)
    elif data_type == 'test':
        return loader.load_test_data(num, nrows)
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Use 'train' or 'test'")


if __name__ == "__main__":
    print("="*70)
    print("Testing HAI Dataset Loader")
    print("="*70)
    
    # Test loading HAI data
    try:
        loader = HAIDataLoader('21.03')
        
        print("\n📊 Loading HAI training data (sample)...")
        train_df = loader.load_train_data(train_num=1, nrows=10000)
        
        print("\n📊 Getting dataset info...")
        train_info = loader.get_dataset_info(train_df)
        loader.print_dataset_summary(train_info)
        
        print("\n📊 Loading HAI test data (sample)...")
        test_df = loader.load_test_data(test_num=1, nrows=10000)
        
        print("\n📊 Getting test data info...")
        test_info = loader.get_dataset_info(test_df)
        loader.print_dataset_summary(test_info)
        
        print("\n✅ HAI Dataset Loader is working correctly!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure HAI dataset is downloaded in data/raw/hai/")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
