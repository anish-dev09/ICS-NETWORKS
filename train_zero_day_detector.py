"""
Training Script for Zero-Day Attack Detection Ensemble
Demonstrates training and evaluation of the enhanced detection system.
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.baseline_detector import BaselineDetector
from src.models.ensemble_detector import ZeroDayEnsembleDetector
from src.data.hai_loader import HAILoader
from src.features.feature_engineering import engineer_features
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(data_path, sample_size=None):
    """
    Load and prepare HAI dataset for training.
    
    Args:
        data_path (str): Path to HAI dataset
        sample_size (int): Optional sample size for testing
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load HAI dataset
    loader = HAILoader(data_path)
    
    # Load train and test files
    print("\nLoading training data...")
    train_files = ['train1.csv', 'train2.csv']
    X_train_list = []
    y_train_list = []
    
    for train_file in train_files:
        try:
            X, y = loader.load_file(train_file)
            X_train_list.append(X)
            y_train_list.append(y)
            print(f"  ✓ Loaded {train_file}: {X.shape[0]} samples")
        except Exception as e:
            print(f"  ✗ Error loading {train_file}: {e}")
    
    print("\nLoading test data...")
    test_files = ['test1.csv', 'test2.csv']
    X_test_list = []
    y_test_list = []
    
    for test_file in test_files:
        try:
            X, y = loader.load_file(test_file)
            X_test_list.append(X)
            y_test_list.append(y)
            print(f"  ✓ Loaded {test_file}: {X.shape[0]} samples")
        except Exception as e:
            print(f"  ✗ Error loading {test_file}: {e}")
    
    # Combine datasets
    if X_train_list:
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
    else:
        raise ValueError("No training data loaded")
    
    if X_test_list:
        X_test = np.vstack(X_test_list)
        y_test = np.hstack(y_test_list)
    else:
        raise ValueError("No test data loaded")
    
    print(f"\nCombined dataset:")
    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  Training anomalies: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)")
    print(f"  Test anomalies: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)")
    
    # Sample if requested
    if sample_size and sample_size < len(X_train):
        print(f"\nSampling {sample_size} training samples for quick testing...")
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    if sample_size and sample_size < len(X_test):
        test_sample_size = min(sample_size // 2, len(X_test))
        print(f"Sampling {test_sample_size} test samples for quick testing...")
        indices = np.random.choice(len(X_test), test_sample_size, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    return X_train, X_test, y_train, y_test


def train_ensemble(X_train, y_train, X_test, y_test, 
                  enable_dl=True, enable_protocol=False, 
                  epochs=30, output_dir='results/zero_day'):
    """
    Train and evaluate zero-day detection ensemble.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        enable_dl (bool): Enable deep learning models
        enable_protocol (bool): Enable protocol validation
        epochs (int): Training epochs for DL models
        output_dir (str): Output directory for results
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print("TRAINING ZERO-DAY DETECTION ENSEMBLE")
    print("="*60)
    
    # Use only normal traffic for training (unsupervised learning)
    X_train_normal = X_train[y_train == 0]
    print(f"\nUsing {len(X_train_normal)} normal samples for training (unsupervised)")
    
    # Initialize ensemble
    ensemble = ZeroDayEnsembleDetector(
        input_dim=X_train.shape[1],
        enable_deep_learning=enable_dl,
        enable_protocol_validation=enable_protocol,
        sequence_length=10
    )
    
    # Train ensemble
    ensemble.fit(
        X_train_normal,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    metrics = ensemble.evaluate(X_test, y_test)
    ensemble.print_metrics(metrics)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'ensemble_confidence': metrics['ensemble_confidence'],
        'false_positive_rate': metrics['false_positive_rate'],
        'false_negative_rate': metrics['false_negative_rate']
    }])
    
    metrics_file = os.path.join(output_dir, 'ensemble_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n✓ Metrics saved to {metrics_file}")
    
    # Save individual detector metrics
    if 'individual_detector_metrics' in metrics:
        individual_df = pd.DataFrame(metrics['individual_detector_metrics']).T
        individual_file = os.path.join(output_dir, 'individual_detector_metrics.csv')
        individual_df.to_csv(individual_file)
        print(f"✓ Individual detector metrics saved to {individual_file}")
    
    # Demonstrate on sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Get some anomalous samples
    anomaly_indices = np.where(y_test == 1)[0][:5]
    normal_indices = np.where(y_test == 0)[0][:5]
    sample_indices = np.concatenate([anomaly_indices, normal_indices])
    
    sample_X = X_test[sample_indices]
    sample_y = y_test[sample_indices]
    
    result = ensemble.predict(sample_X, return_details=True)
    print(ensemble.explain_detection(result))
    
    print(f"\nActual labels: {sample_y}")
    print(f"Predicted:     {result['predictions']}")
    
    return metrics


def compare_with_baseline(X_train, y_train, X_test, y_test):
    """
    Compare ensemble with baseline detectors.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
    
    Returns:
        pd.DataFrame: Comparison results
    """
    print("\n" + "="*60)
    print("COMPARING WITH BASELINE DETECTORS")
    print("="*60)
    
    results = []
    
    # Use only normal data for training
    X_train_normal = X_train[y_train == 0]
    
    # Test baseline methods
    for method in ['zscore', 'iqr', 'isolation_forest']:
        print(f"\nTesting {method.upper()}...")
        
        detector = BaselineDetector(method=method, threshold=3.0)
        detector.fit(X_train_normal)
        
        metrics = detector.evaluate(X_test, y_test)
        metrics['method'] = method
        results.append(metrics)
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Train Zero-Day Attack Detection Ensemble')
    parser.add_argument('--data-path', type=str, default='data/raw/hai/hai-22.04',
                       help='Path to HAI dataset')
    parser.add_argument('--output-dir', type=str, default='results/zero_day',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs for deep learning models')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for quick testing (default: use all data)')
    parser.add_argument('--disable-dl', action='store_true',
                       help='Disable deep learning models')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare with baseline detectors')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ZERO-DAY ATTACK DETECTION TRAINING")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Deep learning: {'Disabled' if args.disable_dl else 'Enabled'}")
    print(f"Training epochs: {args.epochs}")
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            args.data_path,
            sample_size=args.sample_size
        )
        
        # Train ensemble
        metrics = train_ensemble(
            X_train, y_train, X_test, y_test,
            enable_dl=not args.disable_dl,
            enable_protocol=False,  # No protocol data in HAI dataset
            epochs=args.epochs,
            output_dir=args.output_dir
        )
        
        # Compare with baseline if requested
        if args.compare_baseline:
            comparison = compare_with_baseline(X_train, y_train, X_test, y_test)
            comparison_file = os.path.join(args.output_dir, 'baseline_comparison.csv')
            comparison.to_csv(comparison_file, index=False)
            print(f"\n✓ Baseline comparison saved to {comparison_file}")
            print("\nComparison Summary:")
            print(comparison[['method', 'accuracy', 'precision', 'recall', 'f1_score']])
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"✓ All results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
