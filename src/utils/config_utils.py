"""
Utility functions for loading and managing configuration files.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file. 
                                    Defaults to configs/config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(dataset_name, file_type='train', config=None):
    """
    Get the full path to a dataset file.
    
    Args:
        dataset_name (str): Name of dataset ('swat', 'wadi', 'hai', etc.)
        file_type (str): 'train' or 'test'
        config (dict, optional): Configuration dictionary
    
    Returns:
        Path: Full path to dataset file
    """
    if config is None:
        config = load_config()
    
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / config['data']['raw_dir']
    
    dataset_config = config['data']['datasets'].get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in config")
    
    if file_type == 'train':
        filename = dataset_config['train_file']
    elif file_type == 'test':
        filename = dataset_config['test_file']
    else:
        raise ValueError(f"Invalid file_type: {file_type}")
    
    return raw_dir / dataset_name / filename


def get_results_path(result_type='models', config=None):
    """
    Get path to results directory.
    
    Args:
        result_type (str): Type of result ('models', 'metrics', 'plots')
        config (dict, optional): Configuration dictionary
    
    Returns:
        Path: Path to results directory
    """
    if config is None:
        config = load_config()
    
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / config['paths'][result_type]
    
    # Create directory if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)
    
    return results_path


def save_model(model, model_name, config=None):
    """
    Save a trained model to results directory.
    
    Args:
        model: Trained model object
        model_name (str): Name for the model file
        config (dict, optional): Configuration dictionary
    """
    import joblib
    
    model_path = get_results_path('models', config) / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path


def load_model(model_name, config=None):
    """
    Load a trained model from results directory.
    
    Args:
        model_name (str): Name of the model file
        config (dict, optional): Configuration dictionary
    
    Returns:
        Loaded model object
    """
    import joblib
    
    model_path = get_results_path('models', config) / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    return model
