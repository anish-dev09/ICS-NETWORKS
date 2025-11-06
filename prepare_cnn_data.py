"""
Prepare CNN Data - Generate sequences from HAI dataset

This script loads the HAI dataset, creates sequences for CNN training,
and saves the processed data.

Author: Anish
Date: November 6, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.hai_loader import HAIDataLoader
from src.data.sequence_generator import (
    SequenceGenerator, 
    create_balanced_sequences,
    split_sequences
)


def prepare_cnn_data(
    window_size: int = 60,
    step: int = 10,
    balance_ratio: float = 0.5,
    train_size: float = 0.7,
    val_size: float = 0.15,
    max_samples: int = 20000
):
    """
    Prepare CNN sequences from HAI dataset.
    
    Args:
        window_size: Number of timesteps per sequence
        step: Step size for sliding window
        balance_ratio: Desired attack/normal ratio
        train_size: Proportion for training
        val_size: Proportion for validation
        max_samples: Maximum samples to load from HAI
    """
    print(f"\n{'='*70}")
    print("CNN Data Preparation Pipeline")
    print(f"{'='*70}\n")
    
    # Configuration
    print("Configuration:")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step}")
    print(f"  Balance ratio: {balance_ratio}")
    print(f"  Train/Val/Test split: {train_size}/{val_size}/{1-train_size-val_size}")
    print(f"  Max samples: {max_samples:,}")
    
    # Step 1: Load HAI dataset
    print(f"\n{'='*70}")
    print("Step 1: Loading HAI Dataset")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    loader = HAIDataLoader()
    
    # Load train data (mostly normal)
    print("\nLoading training data (train1.csv.gz)...")
    train_df = loader.load_train_data(
        train_num=1,
        nrows=max_samples
    )
    
    print(f"✓ Loaded {len(train_df):,} samples")
    print(f"  Attack ratio: {train_df['attack'].mean():.2%}")
    
    # Load test data (contains attacks)
    print("\nLoading test data (test1.csv.gz)...")
    test_df = loader.load_test_data(
        test_num=1,
        nrows=max_samples
    )
    
    print(f"✓ Loaded {len(test_df):,} samples")
    print(f"  Attack ratio: {test_df['attack'].mean():.2%}")
    
    # Combine datasets
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"\n✓ Total combined samples: {len(full_df):,}")
    print(f"  Attack ratio: {full_df['attack'].mean():.2%}")
    print(f"  Time taken: {time.time() - start_time:.2f}s")
    
    # Step 2: Create sequences
    print(f"\n{'='*70}")
    print("Step 2: Creating Sequences")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Initialize sequence generator
    generator = SequenceGenerator(
        window_size=window_size,
        step=step,
        scale=True
    )
    
    # Generate sequences
    print(f"\nGenerating sequences (window={window_size}, step={step})...")
    X, y = generator.fit_transform(full_df)
    
    print(f"✓ Generated {len(X):,} sequences")
    print(f"  X shape: {X.shape} (samples, timesteps, sensors)")
    print(f"  y shape: {y.shape}")
    print(f"  Attack ratio: {y.mean():.2%}")
    print(f"  Time taken: {time.time() - start_time:.2f}s")
    
    # Save generator info
    gen_info = generator.get_info()
    print(f"\nGenerator info:")
    print(f"  Sensors: {gen_info['n_sensors']}")
    print(f"  Scaling: {gen_info['scale']}")
    
    # Step 3: Balance data
    print(f"\n{'='*70}")
    print("Step 3: Balancing Data")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    print(f"\nBalancing to {balance_ratio:.0%} attack ratio...")
    X_balanced, y_balanced = create_balanced_sequences(
        X, y, 
        balance_ratio=balance_ratio
    )
    
    print(f"✓ Balanced dataset:")
    print(f"  Samples: {len(X_balanced):,} (from {len(X):,})")
    print(f"  Attack ratio: {y_balanced.mean():.2%}")
    print(f"  Normal samples: {np.sum(y_balanced == 0):,}")
    print(f"  Attack samples: {np.sum(y_balanced == 1):,}")
    print(f"  Time taken: {time.time() - start_time:.2f}s")
    
    # Step 4: Split data
    print(f"\n{'='*70}")
    print("Step 4: Splitting Data")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(
        X_balanced, y_balanced,
        train_size=train_size,
        val_size=val_size
    )
    
    print(f"\n✓ Data split:")
    print(f"  Train:      {X_train.shape} - {np.sum(y_train == 1):,} attacks ({y_train.mean():.2%})")
    print(f"  Validation: {X_val.shape} - {np.sum(y_val == 1):,} attacks ({y_val.mean():.2%})")
    print(f"  Test:       {X_test.shape} - {np.sum(y_test == 1):,} attacks ({y_test.mean():.2%})")
    print(f"  Time taken: {time.time() - start_time:.2f}s")
    
    # Step 5: Save data
    print(f"\n{'='*70}")
    print("Step 5: Saving Processed Data")
    print(f"{'='*70}")
    
    # Create directory
    output_dir = Path('data/processed/cnn_sequences')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    print(f"\nSaving to: {output_dir}")
    
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'y_train.npy', y_train)
    print(f"✓ Saved training data")
    
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'y_val.npy', y_val)
    print(f"✓ Saved validation data")
    
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test.npy', y_test)
    print(f"✓ Saved test data")
    
    # Save generator info
    import json
    with open(output_dir / 'generator_info.json', 'w') as f:
        json.dump(gen_info, f, indent=2)
    print(f"✓ Saved generator info")
    
    # Save configuration
    config = {
        'window_size': window_size,
        'step': step,
        'balance_ratio': balance_ratio,
        'train_size': train_size,
        'val_size': val_size,
        'max_samples': max_samples,
        'total_sequences': len(X_balanced),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'n_sensors': gen_info['n_sensors']
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved configuration")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"✓ Data preparation completed successfully!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Total sequences: {len(X_balanced):,}")
    print(f"✓ Ready for CNN training!")
    print(f"{'='*70}\n")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'config': config
    }


if __name__ == "__main__":
    # Run data preparation
    results = prepare_cnn_data(
        window_size=60,
        step=10,
        balance_ratio=0.5,
        train_size=0.7,
        val_size=0.15,
        max_samples=20000
    )
    
    print("\n✅ Data preparation complete! Ready to train CNN.")
    print("\nNext step: Run 'python train_cnn_model.py'")
