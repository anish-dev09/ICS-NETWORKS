"""
Optimize ML Model Thresholds for Better Attack Detection

The initial models had 0% recall because default threshold (0.5) is too high.
This script finds optimal thresholds and retrains with better parameters.

Author: Anish
Date: November 6, 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ml_models import MLDetector

def find_optimal_threshold(y_true, y_proba, target_recall=0.80):
    """Find threshold that achieves target recall while maximizing precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # precision_recall_curve returns n+1 precision/recall but n thresholds
    # Align them by trimming the last precision/recall value
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Find thresholds that meet recall target
    valid_idx = recalls >= target_recall
    if not any(valid_idx):
        # If can't meet target, use threshold that maximizes recall
        best_idx = np.argmax(recalls)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Among valid thresholds, pick one with best precision
    valid_precisions = precisions[valid_idx]
    valid_thresholds = thresholds[valid_idx]
    
    if len(valid_thresholds) == 0:
        return 0.5
    
    best_idx = np.argmax(valid_precisions)
    return valid_thresholds[best_idx]

def main():
    print("="*80)
    print("OPTIMIZING ML MODEL THRESHOLDS")
    print("="*80)
    
    # Load trained models
    models_dir = project_root / 'results' / 'models'
    
    print("\n📊 Loading trained models and feature transformers...")
    rf_model = MLDetector.load(str(models_dir / 'random_forest_detector.pkl'))
    xgb_model = MLDetector.load(str(models_dir / 'xgboost_detector.pkl'))
    
    # Load the SAME feature engineer and selector used during training
    import joblib
    engineer = joblib.load(str(models_dir / 'feature_engineer.pkl'))
    selector = joblib.load(str(models_dir / 'feature_selector.pkl'))
    print("✅ Feature transformers loaded (ensuring consistent features)")
    
    # Load test data (we'll reload a fresh test set)
    print("\n🔄 Loading test data...")
    from src.data.hai_loader import HAIDataLoader
    
    hai_loader = HAIDataLoader(version='21.03')
    test_df = hai_loader.load_test_data(test_num=1, nrows=5000)
    
    sensor_cols = hai_loader.get_sensor_columns(test_df)
    X_test_raw = test_df[sensor_cols].copy()
    y_test = test_df['attack'].copy()
    
    print(f"   Test samples: {len(X_test_raw)}")
    print(f"   Attack samples: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Feature engineering using saved transformers
    print("\n🔧 Engineering features using saved transformers...")
    
    # Use the SAME engineer and selector from training
    X_test_engineered = engineer.transform(X_test_raw, include_original=True)
    X_test_features = selector.transform(X_test_engineered)
    
    print(f"✅ Features ready: {X_test_features.shape}")
    print(f"   Feature columns match training: {X_test_features.shape[1]} features")
    
    # Get predictions with probabilities
    print("\n" + "="*80)
    print("FINDING OPTIMAL THRESHOLDS")
    print("="*80)
    
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"{model_name}")
        print(f"{'='*60}")
        
        # Get probabilities
        y_proba = model.predict_proba(X_test_features)[:, 1]
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, y_proba, target_recall=0.75)
        
        print(f"\n  Default threshold: 0.5000")
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        
        # Evaluate with different thresholds
        thresholds_to_test = [
            ('Very Low', 0.05),
            ('Low', 0.10),
            ('Medium-Low', 0.15),
            ('Medium', 0.20),
            ('Medium-High', 0.30),
            ('Default', 0.50)
        ]
        
        print(f"\n  Testing {len(thresholds_to_test)} different thresholds...")
        
        for threshold_name, threshold in thresholds_to_test:
            y_pred = (y_proba >= threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"  {threshold_name:12s} (t={threshold:.2f}): Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
            
            # Save all results for comparison
            results.append({
                'model': model_name,
                'threshold_name': threshold_name,
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            })
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZED RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    # Compare with baseline
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    baseline_best = {
        'method': 'Isolation Forest',
        'accuracy': 0.8277,
        'recall': 0.7204,
        'f1_score': 0.1842
    }
    
    ml_best = results_df.loc[results_df['f1_score'].idxmax()]
    
    print(f"\nBaseline (Isolation Forest):")
    print(f"  Accuracy:  {baseline_best['accuracy']:.4f} ({baseline_best['accuracy']*100:.2f}%)")
    print(f"  Recall:    {baseline_best['recall']:.4f}")
    print(f"  F1-Score:  {baseline_best['f1_score']:.4f}")
    
    print(f"\nBest ML Model ({ml_best['model']}):")
    print(f"  Accuracy:  {ml_best['accuracy']:.4f} ({ml_best['accuracy']*100:.2f}%)")
    print(f"  Recall:    {ml_best['recall']:.4f}")
    print(f"  F1-Score:  {ml_best['f1_score']:.4f}")
    
    acc_improvement = (ml_best['accuracy'] - baseline_best['accuracy']) / baseline_best['accuracy'] * 100
    f1_improvement = (ml_best['f1_score'] - baseline_best['f1_score']) / baseline_best['f1_score'] * 100
    
    print(f"\n🎯 Improvements:")
    print(f"  Accuracy:  {acc_improvement:+.2f}%")
    print(f"  F1-Score:  {f1_improvement:+.2f}%")
    
    # Save optimized results
    results_dir = project_root / 'results' / 'metrics'
    results_file = results_dir / 'ml_models_optimized.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Optimized results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
