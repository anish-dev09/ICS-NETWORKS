"""
Train CNN Model for ICS Intrusion Detection

This script trains a 1D-CNN model on prepared HAI sequences.

Author: Anish
Date: November 6, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.cnn_models import CNN1DDetector, compute_class_weights
import matplotlib.pyplot as plt


def load_cnn_data(data_dir='data/processed/cnn_sequences'):
    """
    Load preprocessed CNN sequences.
    
    Args:
        data_dir: Directory with preprocessed data
        
    Returns:
        Dictionary with X_train, y_train, etc.
    """
    data_dir = Path(data_dir)
    
    print(f"Loading CNN data from: {data_dir}")
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Load configuration
    with open(data_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    print(f"✓ Data loaded successfully!")
    print(f"  Train: {X_train.shape} - {np.sum(y_train)} attacks ({y_train.mean():.2%})")
    print(f"  Val:   {X_val.shape} - {np.sum(y_val)} attacks ({y_val.mean():.2%})")
    print(f"  Test:  {X_test.shape} - {np.sum(y_test)} attacks ({y_test.mean():.2%})")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'config': config
    }


def plot_training_history(history, output_path='results/plots/cnn_training_history.png'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Keras training history
        output_path: Path to save plot
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get metrics
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    available_metrics = [m for m in metrics if m in history.history]
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        # Plot training
        ax.plot(history.history[metric], label=f'Train {metric}', marker='o')
        
        # Plot validation if available
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label=f'Val {metric}', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs. Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {output_path}")
    plt.close()


def train_cnn_model(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    filters=[64, 128, 256],
    dense_units=[128, 64],
    dropout_rate=0.5
):
    """
    Train 1D-CNN model for ICS intrusion detection.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        filters: List of filters for Conv1D layers
        dense_units: List of units for dense layers
        dropout_rate: Dropout rate for regularization
    """
    print(f"\n{'='*70}")
    print("CNN Training Pipeline")
    print(f"{'='*70}\n")
    
    # Step 1: Load data
    print("Step 1: Loading Data")
    print(f"{'='*70}")
    data = load_cnn_data()
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, sensors)
    
    # Step 2: Build model
    print(f"\n{'='*70}")
    print("Step 2: Building CNN Model")
    print(f"{'='*70}")
    
    cnn = CNN1DDetector(
        input_shape=input_shape,
        filters=filters,
        kernel_size=3,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    model = cnn.build_model()
    
    print(f"\n✓ Model built successfully!")
    print(f"\nModel Architecture:")
    print(cnn.get_model_summary())
    print(f"\nTotal Parameters: {model.count_params():,}")
    
    # Step 3: Compute class weights
    print(f"\n{'='*70}")
    print("Step 3: Computing Class Weights")
    print(f"{'='*70}")
    
    class_weight = compute_class_weights(y_train)
    print(f"✓ Class weights computed:")
    print(f"  Normal (0): {class_weight[0]:.2f}")
    print(f"  Attack (1): {class_weight[1]:.2f}")
    
    # Step 4: Train model
    print(f"\n{'='*70}")
    print("Step 4: Training CNN Model")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    history = cnn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed!")
    print(f"  Time taken: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"  Epochs trained: {len(history.history['loss'])}")
    
    # Step 5: Evaluate on test set
    print(f"\n{'='*70}")
    print("Step 5: Evaluating CNN Model")
    print(f"{'='*70}")
    
    results = cnn.evaluate(X_test, y_test, threshold=0.5)
    cnn.print_metrics(results)
    
    # Step 6: Save model and results
    print(f"{'='*70}")
    print("Step 6: Saving Model and Results")
    print(f"{'='*70}\n")
    
    # Create output directories
    models_dir = Path('results/models')
    metrics_dir = Path('results/metrics')
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / 'cnn1d_detector.keras'
    cnn.save(str(model_path))
    
    # Save evaluation results
    results_df = pd.DataFrame([{
        'model': 'CNN1D',
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'roc_auc': results['roc_auc'],
        'threshold': results['threshold'],
        'training_time': training_time,
        'epochs': len(history.history['loss']),
        'n_params': model.count_params()
    }])
    
    results_path = metrics_dir / 'cnn_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save detailed configuration
    config = {
        'model_type': 'CNN1D',
        'input_shape': input_shape,
        'filters': filters,
        'dense_units': dense_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'total_parameters': int(model.count_params()),
        'training_time_seconds': training_time,
        'test_accuracy': float(results['accuracy']),
        'test_precision': float(results['precision']),
        'test_recall': float(results['recall']),
        'test_f1_score': float(results['f1_score']),
        'test_roc_auc': float(results['roc_auc'])
    }
    
    config_path = models_dir / 'cnn1d_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    print(f"✓ CNN model trained and saved successfully!")
    print(f"✓ Model: {model_path}")
    print(f"✓ Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"✓ Test Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"✓ Test F1-Score: {results['f1_score']:.4f}")
    print(f"{'='*70}\n")
    
    return cnn, results, history


if __name__ == "__main__":
    # Train CNN model
    print("\n🚀 Starting CNN Training...")
    
    cnn, results, history = train_cnn_model(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        filters=[64, 128, 256],
        dense_units=[128, 64],
        dropout_rate=0.5
    )
    
    print("\n✅ CNN training complete!")
    print("\nNext step: Run comparison with ML models")
