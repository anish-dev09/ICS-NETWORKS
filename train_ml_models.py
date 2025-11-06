"""
Train and Evaluate ML Models on HAI Dataset

This script trains Random Forest and XGBoost models on engineered features
and compares them with the baseline methods.

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
from src.features.feature_engineering import create_features_pipeline
from src.models.ml_models import MLDetector, train_and_compare_models
from imblearn.over_sampling import SMOTE

def main():
    print("=" * 80)
    print("ML MODELS TRAINING - HAI Dataset with Feature Engineering")
    print("=" * 80)
    print()
    
    # Configuration
    USE_SMOTE = True  # Use SMOTE to balance dataset
    USE_TEST_FOR_TRAINING = True  # Use test data (which has attacks) for supervised learning
    TRAIN_SAMPLE_SIZE = 15000  # Use 15K from test for training
    TEST_SAMPLE_SIZE = 5000  # Use 5K from test for validation
    
    # Initialize HAI data loader
    print("📊 Loading HAI dataset...")
    hai_loader = HAIDataLoader(version='21.03')
    
    # Strategy: Use test data for training since it contains attacks
    # Training data is normal-only, suitable for unsupervised methods
    print("\n🔄 Loading HAI test data (contains attacks for supervised learning)...")
    full_test_df = hai_loader.load_test_data(test_num=1, nrows=TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE)
    
    # Strategy: Use test data for training since it contains attacks
    # Training data is normal-only, suitable for unsupervised methods
    print("\n🔄 Loading HAI test data (contains attacks for supervised learning)...")
    full_test_df = hai_loader.load_test_data(test_num=1, nrows=TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE)
    
    # Get sensor columns
    sensor_cols = hai_loader.get_sensor_columns(full_test_df)
    
    # Split into training and validation sets
    train_df = full_test_df.iloc[:TRAIN_SAMPLE_SIZE].copy()
    test_df = full_test_df.iloc[TRAIN_SAMPLE_SIZE:].copy()
    
    # Prepare training data
    X_train_raw = train_df[sensor_cols].copy()
    y_train = train_df['attack'].copy()
    
    print(f"   Training samples: {len(X_train_raw)}")
    print(f"   Attack samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
    
    # Prepare test data  
    X_test_raw = test_df[sensor_cols].copy()
    y_test = test_df['attack'].copy()
    
    print(f"   Test samples: {len(X_test_raw)}")
    print(f"   Attack samples: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Feature Engineering
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    print("\n🔧 Engineering features for training data...")
    X_train_features, engineer, selector = create_features_pipeline(
        X_train_raw,
        y=y_train,
        window_sizes=[10, 30, 60],
        include_original=True,
        apply_selection=True
    )
    
    print(f"\n✅ Training features ready: {X_train_features.shape}")
    print(f"   Original columns: {X_train_raw.shape[1]}")
    print(f"   Engineered columns: {X_train_features.shape[1]}")
    
    print("\n🔧 Engineering features for test data...")
    X_test_engineered = engineer.transform(X_test_raw, include_original=True)
    
    if selector is not None:
        X_test_features = selector.transform(X_test_engineered)
    else:
        X_test_features = X_test_engineered
    
    print(f"✅ Test features ready: {X_test_features.shape}")
    
    # Handle class imbalance with SMOTE
    n_attack_samples = y_train.sum()
    
    if USE_SMOTE and n_attack_samples > 0:
        print("\n" + "="*80)
        print("HANDLING CLASS IMBALANCE WITH SMOTE")
        print("="*80)
        
        print(f"\n⚖️  Before SMOTE:")
        print(f"   Normal samples: {(y_train==0).sum()}")
        print(f"   Attack samples: {(y_train==1).sum()}")
        print(f"   Ratio: {(y_train==1).sum()/(y_train==0).sum()*100:.2f}%")
        
        smote = SMOTE(sampling_strategy=0.5, random_state=42)  # type: ignore
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_features, y_train)  # type: ignore
        
        print(f"\n⚖️  After SMOTE:")
        print(f"   Normal samples: {(y_train_balanced==0).sum()}")
        print(f"   Attack samples: {(y_train_balanced==1).sum()}")
        print(f"   Ratio: {(y_train_balanced==1).sum()/(y_train_balanced==0).sum()*100:.2f}%")
        
        X_train_final: pd.DataFrame = pd.DataFrame(X_train_balanced)
        y_train_final: pd.Series = pd.Series(y_train_balanced)
    else:
        if n_attack_samples == 0:
            print("\n" + "="*80)
            print("CLASS IMBALANCE HANDLING")
            print("="*80)
            print("\n⚠️  Training data contains ONLY normal samples (0 attacks)")
            print("   This is typical for ICS datasets - training on normal behavior only")
            print("   Using class_weight='balanced' in models to handle this")
        
        X_train_final: pd.DataFrame = X_train_features
        y_train_final: pd.Series = y_train
    
    # Train and Compare Models
    print("\n" + "="*80)
    print("TRAINING ML MODELS")
    print("="*80)
    
    models_config = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    }
    
    # Check if XGBoost is available
    try:
        import xgboost as xgb
        models_config['xgboost'] = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': 10
        }
    except ImportError:
        print("\n⚠️  XGBoost not installed. Training Random Forest only.")
        print("   Install with: pip install xgboost")
    
    results = train_and_compare_models(
        X_train_final, y_train_final,
        X_test_features, y_test,
        models_config
    )
    
    # Display Results Comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_df = results['comparison'].round(4)
    print("\n", comparison_df.to_string(index=False))
    
    # Feature Importance Analysis
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    for model_name, detector in results['models'].items():
        print(f"\n{'='*60}")
        print(f"Top 20 Features - {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            importance_df = detector.get_feature_importance(top_n=20)
            print(importance_df.to_string(index=False))
        except ValueError:
            print("Feature importance not available for this model")
    
    # Compare with Baseline
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    baseline_results = {
        'method': ['Z-Score', 'IQR', 'Isolation Forest'],
        'accuracy': [0.5937, 0.5098, 0.8277],
        'precision': [0.0566, 0.0425, 0.1056],
        'recall': [0.8963, 0.7963, 0.7204],
        'f1_score': [0.1064, 0.0806, 0.1842]
    }
    baseline_df = pd.DataFrame(baseline_results)
    
    print("\nBaseline Results (from Phase 3):")
    print(baseline_df.to_string(index=False))
    
    print("\nML Models Results:")
    ml_comparison = results['comparison'][['model', 'accuracy', 'precision', 'recall', 'f1_score']]
    print(ml_comparison.to_string(index=False))
    
    # Calculate improvement
    best_baseline_acc = baseline_df['accuracy'].max()
    best_ml_acc = results['comparison']['accuracy'].max()
    improvement = (best_ml_acc - best_baseline_acc) / best_baseline_acc * 100
    
    print(f"\n🎯 Performance Improvement:")
    print(f"   Best Baseline:  {best_baseline_acc:.4f} ({best_baseline_acc*100:.2f}%)")
    print(f"   Best ML Model:  {best_ml_acc:.4f} ({best_ml_acc*100:.2f}%)")
    print(f"   Improvement:    {improvement:+.2f}%")
    
    # Save Results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_dir = project_root / 'results' / 'metrics'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison
    comparison_file = results_dir / 'ml_models_comparison.csv'
    results['comparison'].to_csv(comparison_file, index=False)
    print(f"✅ Comparison saved to: {comparison_file}")
    
    # Save feature importance
    for model_name, detector in results['models'].items():
        try:
            importance_df = detector.get_feature_importance(top_n=50)
            importance_file = results_dir / f'feature_importance_{model_name}.csv'
            importance_df.to_csv(importance_file, index=False)
            print(f"✅ Feature importance saved: {importance_file}")
        except ValueError:
            pass
    
    # Save models
    models_dir = project_root / 'results' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, detector in results['models'].items():
        model_file = models_dir / f'{model_name}_detector.pkl'
        detector.save(str(model_file))
    
    # Save feature engineer and selector for consistent feature extraction
    import joblib
    engineer_file = models_dir / 'feature_engineer.pkl'
    selector_file = models_dir / 'feature_selector.pkl'
    joblib.dump(engineer, engineer_file)
    joblib.dump(selector, selector_file)
    print(f"✅ Feature engineer saved: {engineer_file}")
    print(f"✅ Feature selector saved: {selector_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✅ ML Models Training Complete!")
    print(f"   Models trained: {len(results['models'])}")
    print(f"   Training samples: {len(X_train_final)}")
    print(f"   Test samples: {len(X_test_features)}")
    print(f"   Features used: {X_train_final.shape[1]}")
    print(f"   Best accuracy: {best_ml_acc:.4f} ({best_ml_acc*100:.2f}%)")
    print(f"   Improvement over baseline: {improvement:+.2f}%")
    
    print("\n🎯 Next Steps:")
    print("   1. Analyze feature importance")
    print("   2. Fine-tune hyperparameters")
    print("   3. Try ensemble methods")
    print("   4. Move to Phase 6: Deep Learning (LSTM, Autoencoder)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()
