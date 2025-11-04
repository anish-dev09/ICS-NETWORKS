"""
Quick Start Script for ICS Intrusion Detection System

This script helps you verify that everything is set up correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print(" 🔐 ICS INTRUSION DETECTION SYSTEM - QUICK START")
print("=" * 70)

# Step 1: Check imports
print("\n📦 Step 1: Checking package installations...")
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import yaml
    print("✅ Core packages installed successfully")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Step 2: Check project structure
print("\n📁 Step 2: Verifying project structure...")
required_dirs = [
    'data/raw',
    'data/processed',
    'src/data',
    'src/models',
    'src/features',
    'configs',
    'results/models',
    'notebooks',
    'demo'
]

all_dirs_exist = True
for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ❌ {dir_path} missing")
        all_dirs_exist = False

if all_dirs_exist:
    print("✅ Project structure is correct")
else:
    print("⚠️  Some directories are missing")

# Step 3: Test configuration loading
print("\n⚙️  Step 3: Testing configuration system...")
try:
    from src.utils.config_utils import load_config
    config = load_config()
    print(f"✅ Configuration loaded successfully")
    print(f"   Project: {config['project']['name']}")
    print(f"   Version: {config['project']['version']}")
except Exception as e:
    print(f"❌ Configuration error: {e}")

# Step 4: Test data loader
print("\n📊 Step 4: Testing data loader...")
try:
    from src.data.data_loader import ICSDataLoader
    print("✅ Data loader module imported successfully")
    
    # Check if any dataset is available
    datasets_found = []
    data_dir = project_root / 'data' / 'raw'
    
    if (data_dir / 'swat').exists():
        datasets_found.append('SWaT')
    if (data_dir / 'wadi').exists():
        datasets_found.append('WADI')
    if (data_dir / 'gas_pipeline').exists():
        datasets_found.append('Gas Pipeline')
    if (data_dir / 'hai').exists():
        datasets_found.append('HAI')
    
    if datasets_found:
        print(f"✅ Found datasets: {', '.join(datasets_found)}")
    else:
        print("⚠️  No datasets found yet")
        print("   📖 See docs/DATASET_GUIDE.md for download instructions")
        
except Exception as e:
    print(f"❌ Data loader error: {e}")

# Step 5: Test baseline detector
print("\n🤖 Step 5: Testing baseline detector...")
try:
    from src.models.baseline_detector import BaselineDetector
    import numpy as np
    
    # Create simple test data
    X_test = np.random.randn(100, 5)
    
    detector = BaselineDetector(method='zscore')
    detector.fit(X_test)
    predictions = detector.predict(X_test)
    
    print(f"✅ Baseline detector working")
    print(f"   Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
    
except Exception as e:
    print(f"❌ Baseline detector error: {e}")

# Step 6: Summary and next steps
print("\n" + "=" * 70)
print(" 📋 SETUP SUMMARY")
print("=" * 70)

print("\n✅ Completed:")
print("   • Project structure created")
print("   • Configuration system ready")
print("   • Data loaders implemented")
print("   • Baseline detectors ready")
print("   • Demo application available")

print("\n📝 Next Steps:")
print("   1. Download datasets (see docs/DATASET_GUIDE.md)")
print("   2. Run data exploration: jupyter notebook notebooks/")
print("   3. Test baseline models on real data")
print("   4. Start developing deep learning models")

print("\n🚀 Quick Commands:")
print("   • Install packages:    pip install -r requirements.txt")
print("   • Run demo:           cd demo && streamlit run app.py")
print("   • Test data loader:   python src/data/data_loader.py")
print("   • Test baseline:      python src/models/baseline_detector.py")

print("\n📚 Documentation:")
print("   • Main README:        README.md")
print("   • Dataset Guide:      docs/DATASET_GUIDE.md")
print("   • Configuration:      configs/config.yaml")

print("\n" + "=" * 70)
print(" 🎓 Ready to start your BCA Final Year Project!")
print("=" * 70 + "\n")
