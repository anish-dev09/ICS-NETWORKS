"""
Test Feature Engineering on HAI Dataset

This script tests the feature engineering pipeline on real HAI data
and validates the created features.

Author: Anish
Date: November 6, 2025
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.hai_loader import HAIDataLoader
from src.features.feature_engineering import FeatureEngineer, FeatureSelector, create_features_pipeline

def main():
    print("=" * 80)
    print("FEATURE ENGINEERING TEST - HAI Dataset")
    print("=" * 80)
    print()
    
    # Initialize HAI data loader
    print("📊 Initializing HAI data loader...")
    hai_loader = HAIDataLoader(version='21.03')
    
    # Load training data (use smaller sample for testing)
    print("\n🔄 Loading HAI training data...")
    train_df = hai_loader.load_train_data()
    
    # Use first 5000 samples for quick testing
    print("   Using first 5000 samples for testing...")
    train_sample = train_df.head(5000).copy()
    
    # Get sensor columns
    sensor_cols = hai_loader.get_sensor_columns(train_sample)
    print(f"   Found {len(sensor_cols)} sensor columns")
    
    # Display original data shape
    print(f"\n📐 Original data shape: {train_sample.shape}")
    print(f"   Columns: {train_sample.shape[1]}")
    print(f"   Samples: {train_sample.shape[0]}")
    
    # Test 1: Feature Engineering
    print("\n" + "=" * 80)
    print("TEST 1: Feature Engineering")
    print("=" * 80)
    
    start_time = time.time()
    
    engineer = FeatureEngineer(window_sizes=[10, 30, 60])
    X_engineered = engineer.fit_transform(train_sample, sensor_cols=sensor_cols, include_original=True)
    
    eng_time = time.time() - start_time
    
    print(f"\n✅ Feature engineering completed in {eng_time:.2f} seconds")
    print(f"   Engineered features shape: {X_engineered.shape}")
    print(f"   Original features: {len(sensor_cols)}")
    print(f"   Total features created: {X_engineered.shape[1]}")
    print(f"   New features: {X_engineered.shape[1] - len(sensor_cols)}")
    
    # Show feature categories
    print("\n📊 Feature Categories:")
    feature_cols = X_engineered.columns.tolist()
    
    # Count feature types
    stat_features = [f for f in feature_cols if any(x in f for x in ['mean', 'std', 'min', 'max', 'range', 'median', 'skew', 'kurtosis', 'num_sensors'])]
    temporal_features = [f for f in feature_cols if 'rolling' in f]
    roc_features = [f for f in feature_cols if any(x in f for x in ['diff', 'roc'])]
    lag_features = [f for f in feature_cols if 'lag' in f]
    interaction_features = [f for f in feature_cols if 'ratio' in f]
    
    print(f"   - Original sensors: {len(sensor_cols)}")
    print(f"   - Statistical features: {len(stat_features)}")
    print(f"   - Temporal features: {len(temporal_features)}")
    print(f"   - Rate of change features: {len(roc_features)}")
    print(f"   - Lag features: {len(lag_features)}")
    print(f"   - Interaction features: {len(interaction_features)}")
    
    # Test 2: Feature Selection
    print("\n" + "=" * 80)
    print("TEST 2: Feature Selection")
    print("=" * 80)
    
    start_time = time.time()
    
    selector = FeatureSelector(variance_threshold=0.01, correlation_threshold=0.95)
    X_selected = selector.fit_transform(X_engineered)
    
    sel_time = time.time() - start_time
    
    print(f"\n✅ Feature selection completed in {sel_time:.2f} seconds")
    print(f"   Features before selection: {X_engineered.shape[1]}")
    print(f"   Features after selection: {X_selected.shape[1]}")
    print(f"   Features removed: {X_engineered.shape[1] - X_selected.shape[1]}")
    print(f"   Reduction: {(1 - X_selected.shape[1]/X_engineered.shape[1])*100:.1f}%")
    
    # Test 3: Complete Pipeline
    print("\n" + "=" * 80)
    print("TEST 3: Complete Pipeline")
    print("=" * 80)
    
    start_time = time.time()
    
    X_final, eng, sel = create_features_pipeline(
        train_sample,
        window_sizes=[10, 30, 60],
        include_original=True,
        apply_selection=True
    )
    
    pipeline_time = time.time() - start_time
    
    print(f"\n✅ Complete pipeline executed in {pipeline_time:.2f} seconds")
    print(f"   Final feature count: {X_final.shape[1]}")
    print(f"   Processing speed: {len(train_sample)/pipeline_time:.0f} samples/second")
    
    # Data quality checks
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)
    
    print(f"\n1. Missing values:")
    missing = X_final.isnull().sum().sum()
    print(f"   Total missing values: {missing}")
    print(f"   Status: {'✅ No missing values' if missing == 0 else '⚠️ Has missing values'}")
    
    print(f"\n2. Infinite values:")
    infinite = np.isinf(X_final.values).sum()
    print(f"   Total infinite values: {infinite}")
    print(f"   Status: {'✅ No infinite values' if infinite == 0 else '⚠️ Has infinite values'}")
    
    print(f"\n3. Feature statistics:")
    print(f"   Mean of features: {X_final.mean().mean():.4f}")
    print(f"   Std of features: {X_final.std().mean():.4f}")
    print(f"   Min of features: {X_final.min().min():.4f}")
    print(f"   Max of features: {X_final.max().max():.4f}")
    
    print(f"\n4. Feature variance:")
    variances = X_final.var()
    zero_var = (variances == 0).sum()
    low_var = (variances < 0.01).sum()
    print(f"   Zero variance features: {zero_var}")
    print(f"   Low variance features (<0.01): {low_var}")
    
    # Sample output
    print("\n" + "=" * 80)
    print("SAMPLE FEATURES")
    print("=" * 80)
    
    print("\nFirst 5 samples of selected features:")
    print(X_final.head())
    
    print("\nFeature names (first 20):")
    for i, fname in enumerate(X_final.columns[:20], 1):
        print(f"   {i:2d}. {fname}")
    
    if len(X_final.columns) > 20:
        print(f"   ... and {len(X_final.columns) - 20} more features")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Feature engineering successful!")
    print(f"   - Original sensors: {len(sensor_cols)}")
    print(f"   - Engineered features: {X_engineered.shape[1]}")
    print(f"   - Selected features: {X_final.shape[1]}")
    print(f"   - Total processing time: {pipeline_time:.2f}s")
    print(f"   - Memory usage: {X_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n🎯 Next Steps:")
    print("   1. Apply feature engineering to full training set")
    print("   2. Train ML models (Random Forest, XGBoost)")
    print("   3. Compare with baseline (82.77% accuracy)")
    print("   4. Analyze feature importance")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
