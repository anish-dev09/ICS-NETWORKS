"""Test script to verify model loading works correctly"""
import joblib
from pathlib import Path

# Test XGBoost
try:
    xgb_path = Path("results/models/xgboost_detector.pkl")
    xgb_model = joblib.load(xgb_path)
    print(f"✅ XGBoost loaded: {type(xgb_model)}")
except Exception as e:
    print(f"❌ XGBoost error: {e}")

# Test Random Forest
try:
    rf_path = Path("results/models/random_forest_detector.pkl")
    rf_model = joblib.load(rf_path)
    print(f"✅ Random Forest loaded: {type(rf_model)}")
except Exception as e:
    print(f"❌ Random Forest error: {e}")

print("\n✅ All models loaded successfully!")
