# 🔐 AI for Automated Intrusion Detection in ICS Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

## 📋 Project Overview

An AI-powered intrusion detection system for Industrial Control Systems (ICS) networks. This project implements advanced machine learning and deep learning techniques to detect various cyber-attacks on critical infrastructure systems like SCADA, water treatment plants, and power grids.

### 🎯 Key Features

- **Multi-Level Detection**: Process-level + Network-level anomaly detection
- **Advanced ML Models**: LSTM, Autoencoders, Random Forest, Isolation Forest
- **Real-time Monitoring**: Live dashboard with Streamlit
- **Explainable AI**: SHAP and LIME for interpretability
- **Multiple Attack Detection**: Sensor spoofing, command injection, DoS, replay attacks
- **Comprehensive Evaluation**: Precision, recall, F1-score, detection delay analysis

---

## 🏗️ Project Structure

```
ICS-NETWORKS/
│
├── data/                          # Datasets
│   ├── raw/                       # Original datasets (SWaT, WADI, Gas Pipeline)
│   └── processed/                 # Preprocessed data
│
├── src/                           # Source code
│   ├── data/                      # Data loading and preprocessing
│   ├── features/                  # Feature engineering
│   ├── models/                    # ML/DL models
│   ├── detection/                 # Detection algorithms
│   └── utils/                     # Utility functions
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── demo/                          # Live demo application
│   ├── app.py                     # Streamlit dashboard
│   └── mock_data.py              # Mock ICS data generator
│
├── configs/                       # Configuration files
│   └── config.yaml               # Main configuration
│
├── results/                       # Model outputs
│   ├── models/                    # Trained models
│   ├── metrics/                   # Evaluation metrics
│   └── plots/                     # Visualizations
│
├── tests/                         # Unit tests
│
├── docs/                          # Documentation
│   ├── DATASET_GUIDE.md          # How to obtain datasets
│   └── PROJECT_PLAN.md           # Detailed project plan
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/anish-dev09/ICS-NETWORKS.git
cd ICS-NETWORKS
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Get Datasets

Follow the detailed guide in [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) to obtain ICS datasets.

**Quick Start Options:**
- **Option A**: Request SWaT dataset (academic, free, 24-48h approval)
- **Option B**: Download Gas Pipeline dataset (public, immediate)
- **Option C**: Use HAI dataset (public, GitHub)

### 4. Run Demo Application

```bash
cd demo
streamlit run app.py
```

Visit `http://localhost:8501` to see the live dashboard.

---

## 📊 Datasets

This project uses real-world ICS datasets:

| Dataset | Type | Size | Sensors | Attack Types | Status |
|---------|------|------|---------|--------------|--------|
| **SWaT** | Water Treatment | 1.5 GB | 51 | 36 attacks | Request Required |
| **WADI** | Water Distribution | 2+ GB | 123 | 15 attacks | Request Required |
| **Gas Pipeline** | SCADA | 500 MB | Multiple | Various | Public |
| **HAI** | Multi-Process | 1 GB | 78 | 38 attacks | Public |

See [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) for download instructions.

---

## 🧠 Machine Learning Pipeline

### Phase 1: Data Preprocessing
- Missing value handling
- Normalization & scaling
- Time-window creation
- Imbalanced data handling (SMOTE)

### Phase 2: Feature Engineering
- **Statistical Features**: mean, std, min, max, skewness, kurtosis
- **Temporal Features**: rate of change, rolling statistics, EWMA
- **Correlation Features**: sensor correlation analysis
- **Domain-Specific**: physical constraint violations

### Phase 3: Model Development

#### Process-Level Models:
- **LSTM Networks**: Temporal pattern recognition
- **Autoencoders**: Reconstruction-based anomaly detection
- **GRU Networks**: Efficient sequence modeling
- **Isolation Forest**: Baseline anomaly detection

#### Network-Level Models:
- **Random Forest**: Traffic classification
- **XGBoost**: Gradient boosting for patterns
- **CNN**: Packet pattern recognition (advanced)

### Phase 4: Ensemble & Fusion
- Weighted voting
- Stacking
- Multi-level fusion (process + network)

---

## 🎯 Attack Types Detected

### 1. **Sensor Spoofing** 🎯 Primary Focus
- Manipulation of sensor readings
- False data injection
- Easy to visualize and explain

### 2. **Command Injection**
- Unauthorized control commands
- Actuator manipulation

### 3. **Denial of Service (DoS)**
- Network flooding
- Communication disruption

### 4. **Man-in-the-Middle**
- Data interception
- Command modification

### 5. **Replay Attacks**
- Recorded command replay
- Timing-based attacks

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Detection Delay**: Time to detect attack
- **False Positive Rate**: False alarm rate

---

## 🔬 Project Phases & Timeline

### ✅ Phase 1: Plan & Setup (Week 1)
- [x] Project structure created
- [x] Environment setup
- [ ] Dataset acquisition
- [ ] Literature review

### 🔄 Phase 2: Data & Baseline (Week 1-2)
- [ ] Data exploration
- [ ] Baseline models (Z-score, IQR, Isolation Forest)
- [ ] Initial evaluation

### 📋 Phase 3: Feature Engineering (Week 2)
- [ ] Statistical features
- [ ] Temporal features
- [ ] Feature selection

### 🤖 Phase 4: Model Development (Week 2-3)
- [ ] LSTM implementation
- [ ] Autoencoder implementation
- [ ] Random Forest for network data
- [ ] Hyperparameter tuning

### 🔗 Phase 5: Fusion & Explainability (Week 3)
- [ ] Ensemble methods
- [ ] SHAP/LIME integration
- [ ] Alert prioritization

### 🎥 Phase 6: Demo & Testbed (Week 3-4)
- [ ] Enhanced Streamlit dashboard
- [ ] Real-time detection
- [ ] OpenPLC integration (optional)

### 📝 Phase 7: Evaluation & Report (Week 4+)
- [ ] Comprehensive evaluation
- [ ] Documentation
- [ ] Final presentation
- [ ] Project report

---

## 🛠️ Technologies Used

- **Languages**: Python 3.8+
- **ML/DL**: TensorFlow, Keras, PyTorch, scikit-learn
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Explainability**: SHAP, LIME
- **Network Analysis**: Scapy, NetworkX
- **Experiment Tracking**: MLflow (optional)

---

## 📚 Key References

1. Goh et al. (2017) - "A Dataset to Support Research in the Design of Secure Water Treatment Systems" (SWaT)
2. Ahmed et al. (2016) - "WADI: A Water Distribution Testbed for Research in Cyber-Physical Systems"
3. Beaver et al. (2013) - "A Machine Learning Approach to ICS Network Intrusion Detection"
4. Kravchik & Shabtai (2018) - "Detecting Cyber Attacks in Industrial Control Systems Using Convolutional Neural Networks"

---

## 🎓 Academic Context

**Project Type**: BCA Final Year Project  
**Objective**: Develop production-grade AI system for ICS security  
**Timeline**: November 2025 - January 2026 (50% completion target: 2-3 weeks)  
**Future Scope**: Expandable for research publications and industry applications

---

## 📝 Usage Examples

### Training a Model

```python
from src.models.lstm_detector import LSTMDetector
from src.data.data_loader import load_swat_data

# Load data
X_train, y_train = load_swat_data('data/raw/swat/SWaT_Dataset_Normal_v1.csv')

# Initialize model
detector = LSTMDetector(input_dim=51, hidden_units=[128, 64])

# Train
detector.fit(X_train, y_train, epochs=50)

# Evaluate
metrics = detector.evaluate(X_test, y_test)
print(metrics)
```

### Running Detection

```python
from src.detection.realtime_detector import RealtimeDetector

# Initialize detector
detector = RealtimeDetector(model_path='results/models/lstm_best.h5')

# Detect on new data
predictions = detector.predict(new_data)
alerts = detector.generate_alerts(predictions)
```

---

## 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Anish Kumar**  
- GitHub: [@anish-dev09](https://github.com/anish-dev09)
- Project: BCA Final Year - ICS Security Research

---

## 🙏 Acknowledgments

- iTrust Centre for Research in Cyber Security, SUTD (for SWaT/WADI datasets)
- ICS Security Research Community
- Open-source contributors

---

## 📞 Contact & Support

For questions or collaboration:
- Open an issue on GitHub
- Email: [Your email]

---

## 🗺️ Roadmap

- [x] Project setup and structure
- [x] Dataset documentation
- [ ] Baseline implementation (Week 1)
- [ ] Deep learning models (Week 2-3)
- [ ] Real-time demo (Week 3-4)
- [ ] Final evaluation and report (Week 4+)
- [ ] Future: Publish research paper
- [ ] Future: Deploy as production system

---

**Status**: 🚧 In Active Development (Phase 1 Complete)

**Last Updated**: November 5, 2025
