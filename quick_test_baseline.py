"""
Quick Baseline Test on HAI Dataset
Test all baseline detectors and show immediate results
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.hai_loader import HAIDataLoader
from src.models.baseline_detector import BaselineDetector

print("=" * 80)
print(" 🔐 QUICK BASELINE TEST ON HAI DATASET")
print("=" * 80)

# Step 1: Load HAI Data
print("\n📊 Step 1: Loading HAI Dataset...")
print("-" * 80)

try:
    loader = HAIDataLoader('21.03')
    
    # Load training data (normal behavior)
    print("Loading training data (for fitting detectors)...")
    train_df = loader.load_train_data(train_num=1, nrows=50000)
    
    # Load test data (contains attacks)
    print("Loading test data (contains attacks)...")
    test_df = loader.load_test_data(test_num=1, nrows=20000)
    
    # Split features and labels
    X_train, y_train = loader.split_features_labels(train_df)
    X_test, y_test = loader.split_features_labels(test_df)
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Attack samples in test: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    
except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    sys.exit(1)

# Step 2: Test Baseline Detectors
print("\n" + "=" * 80)
print(" 🤖 Step 2: Testing Baseline Detectors")
print("=" * 80)

methods = ['zscore', 'iqr', 'isolation_forest']
results = []

for method in methods:
    print(f"\n{'='*80}")
    print(f" Testing: {method.upper()}")
    print(f"{'='*80}")
    
    try:
        # Initialize detector
        detector = BaselineDetector(method=method, threshold=3.0)
        
        # Train
        print(f"Training {method} detector...")
        start_time = time.time()
        detector.fit(X_train)
        train_time = time.time() - start_time
        print(f"✅ Training completed in {train_time:.2f} seconds")
        
        # Predict
        print(f"Running detection on test data...")
        start_time = time.time()
        y_pred = detector.predict(X_test)
        predict_time = time.time() - start_time
        print(f"✅ Detection completed in {predict_time:.2f} seconds")
        
        # Evaluate
        print(f"Evaluating performance...")
        metrics = detector.evaluate(X_test, y_test)
        
        # Print results
        detector.print_metrics(metrics)
        
        # Store results
        results.append({
            'method': method,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'fpr': metrics['false_positive_rate'],
            'tpr': metrics['true_positive_rate'],
            'train_time': train_time,
            'predict_time': predict_time
        })
        
    except Exception as e:
        print(f"❌ Error with {method}: {e}")
        import traceback
        traceback.print_exc()

# Step 3: Compare Results
print("\n" + "=" * 80)
print(" 📊 Step 3: BASELINE COMPARISON SUMMARY")
print("=" * 80)

if results:
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n📈 Performance Metrics:")
    print("-" * 80)
    print(f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['method']:<20} {row['accuracy']:<12.4f} {row['precision']:<12.4f} "
              f"{row['recall']:<12.4f} {row['f1_score']:<12.4f}")
    
    print("-" * 80)
    
    print("\n⚡ Performance Speed:")
    print("-" * 80)
    print(f"{'Method':<20} {'Train Time':<15} {'Predict Time':<15}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['method']:<20} {row['train_time']:<15.2f}s {row['predict_time']:<15.2f}s")
    
    print("-" * 80)
    
    print("\n🎯 Best Performers:")
    print("-" * 80)
    
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    
    print(f"Best Accuracy:  {best_accuracy['method']:<20} ({best_accuracy['accuracy']:.4f})")
    print(f"Best F1-Score:  {best_f1['method']:<20} ({best_f1['f1_score']:.4f})")
    print(f"Best Recall:    {best_recall['method']:<20} ({best_recall['recall']:.4f})")
    
    print("\n" + "=" * 80)
    
    # Save results
    results_path = project_root / 'results' / 'metrics'
    results_path.mkdir(parents=True, exist_ok=True)
    
    results_file = results_path / 'baseline_results_hai.csv'
    results_df.to_csv(results_file, index=False)
    
    print(f"\n💾 Results saved to: {results_file}")
    
else:
    print("\n❌ No results to display")

print("\n" + "=" * 80)
print(" ✅ QUICK BASELINE TEST COMPLETED!")
print("=" * 80)

print("\n📝 Next Steps:")
print("   1. Review the results above")
print("   2. Check saved results in: results/metrics/baseline_results_hai.csv")
print("   3. Run Jupyter notebook for detailed exploration")
print("   4. Start feature engineering based on insights")

print("\n🚀 To create detailed analysis notebook, run:")
print("   jupyter notebook notebooks/01_data_exploration.ipynb")

print("\n" + "=" * 80 + "\n")
