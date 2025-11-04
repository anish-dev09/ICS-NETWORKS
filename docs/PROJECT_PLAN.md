# 🎯 Project Plan: AI for Automated Intrusion Detection in ICS Networks

**Student:** Anish Kumar  
**Project Type:** BCA Final Year Project  
**Timeline:** November 2025 - January 2026  
**Target:** 50% completion in 2-3 weeks

---

## 📅 Detailed Weekly Breakdown

### **WEEK 1: Foundation & Baseline** (Nov 5-11, 2025)

#### Day 1-2: Environment Setup & Data Acquisition ✅
- [x] Project structure created
- [x] Requirements.txt prepared
- [ ] Install Python packages
- [ ] Request SWaT dataset from iTrust
- [ ] Download Gas Pipeline dataset (public)
- [ ] Download HAI dataset (public)

**Deliverables:**
- Working environment
- At least one dataset downloaded
- All packages installed

---

#### Day 3-4: Data Exploration & Understanding
**Tasks:**
1. Create Jupyter notebook for data exploration
2. Load dataset(s)
3. Visualize sensor data
4. Understand normal behavior patterns
5. Identify attack patterns
6. Document findings

**Key Questions to Answer:**
- How many sensors/features?
- What are the value ranges?
- Are there missing values?
- How are attacks labeled?
- What types of attacks are present?
- What is the normal vs attack ratio?

**Deliverables:**
- `notebooks/01_data_exploration.ipynb`
- Data statistics document
- Visualization of sensor patterns

---

#### Day 5-7: Baseline Models Implementation
**Tasks:**
1. Implement and test Z-score detector
2. Implement and test IQR detector
3. Implement and test Isolation Forest
4. Compare all baseline methods
5. Document baseline performance

**Success Metrics:**
- Accuracy > 70% (baseline)
- F1-Score documented
- False Positive Rate < 20%

**Deliverables:**
- Working baseline detectors
- `notebooks/02_baseline_models.ipynb`
- Baseline evaluation report
- Performance comparison table

**📊 Week 1 Target: 15% Complete**

---

### **WEEK 2: Feature Engineering & Initial ML Models** (Nov 12-18, 2025)

#### Day 8-10: Feature Engineering
**Tasks:**
1. Extract statistical features (mean, std, min, max)
2. Create temporal features (rolling windows)
3. Compute sensor correlations
4. Implement rate of change features
5. Create feature selection pipeline
6. Handle imbalanced data (SMOTE)

**Features to Create:**
- **Statistical:** mean, std, min, max, median, skewness, kurtosis
- **Temporal:** rolling mean, rolling std, EWMA
- **Correlation:** sensor pair correlations
- **Domain-specific:** physical constraint violations

**Deliverables:**
- `src/features/feature_engineering.py`
- `notebooks/03_feature_engineering.ipynb`
- Feature importance analysis
- Processed dataset with features

---

#### Day 11-14: Machine Learning Models
**Tasks:**
1. **Random Forest Classifier**
   - Train on engineered features
   - Hyperparameter tuning
   - Feature importance analysis

2. **XGBoost Classifier**
   - Train and optimize
   - Compare with Random Forest

3. **Initial LSTM Model**
   - Prepare time-series data
   - Simple LSTM architecture
   - Basic training

**Success Metrics:**
- Accuracy > 85%
- Precision > 80%
- Recall > 75%
- F1-Score > 0.80

**Deliverables:**
- `src/models/random_forest_detector.py`
- `src/models/xgboost_detector.py`
- `src/models/lstm_detector.py`
- `notebooks/04_model_training.ipynb`
- Model comparison report

**📊 Week 2 Target: 35% Complete (Cumulative)**

---

### **WEEK 3: Deep Learning & Ensemble** (Nov 19-25, 2025)

#### Day 15-17: Advanced Deep Learning Models
**Tasks:**
1. **LSTM Network (Enhanced)**
   - Multi-layer LSTM
   - Bidirectional LSTM
   - Dropout regularization
   - Early stopping

2. **Autoencoder for Anomaly Detection**
   - Encoder-decoder architecture
   - Reconstruction error threshold
   - Train on normal data only

3. **GRU Networks** (if time permits)

**Architecture Details:**
```
LSTM: Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(32) → Output
Autoencoder: Input → 64 → 32 → 16 → 32 → 64 → Output
```

**Deliverables:**
- Enhanced LSTM model
- Autoencoder implementation
- Training history plots
- Model performance comparison

---

#### Day 18-21: Ensemble & Fusion
**Tasks:**
1. Implement weighted voting ensemble
2. Combine process-level + network-level detectors
3. Optimize ensemble weights
4. Implement alert prioritization
5. Add confidence scores

**Ensemble Strategy:**
- Process-level weight: 60%
- Network-level weight: 40%
- Threshold tuning for optimal F1

**Deliverables:**
- `src/detection/ensemble_detector.py`
- Ensemble evaluation results
- Alert prioritization system

**📊 Week 3 Target: 55% Complete (Cumulative)** ✅ **Project Goal Achieved**

---

### **WEEK 4+: Demo, Evaluation & Documentation** (Nov 26+)

#### Day 22-25: Enhanced Demo Application
**Tasks:**
1. Upgrade Streamlit dashboard
2. Integrate trained models
3. Add real-time prediction
4. Implement explainability (SHAP/LIME)
5. Create alert visualization
6. Add model comparison view

**Dashboard Features:**
- Live sensor monitoring
- Real-time anomaly detection
- Attack type classification
- Confidence scores
- Feature importance display
- Historical attack log

**Deliverables:**
- Enhanced demo application
- User guide for demo
- Demo video/screenshots

---

#### Day 26-28: Comprehensive Evaluation
**Tasks:**
1. Evaluate all models on test set
2. Generate performance metrics
3. Create confusion matrices
4. ROC and PR curves
5. Detection delay analysis
6. Compare with literature

**Metrics to Report:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- False Positive Rate
- True Positive Rate
- Detection Delay (seconds)
- Per-attack-type performance

**Deliverables:**
- Comprehensive evaluation report
- Performance comparison charts
- Results summary table

---

#### Day 29-35: Documentation & Presentation
**Tasks:**
1. Write project report
2. Create presentation slides
3. Prepare demo script
4. Document code
5. Create project video
6. Final code cleanup

**Report Structure:**
1. Abstract
2. Introduction & Literature Review
3. Methodology
4. System Architecture
5. Implementation Details
6. Results & Evaluation
7. Conclusion & Future Work
8. References

**Deliverables:**
- Final project report (20-30 pages)
- Presentation slides (15-20 slides)
- Demo video (5-10 minutes)
- Complete code documentation
- README updates

**📊 Week 4+ Target: 100% Complete**

---

## 🎯 Key Milestones

| Week | Milestone | Completion % | Key Deliverables |
|------|-----------|--------------|------------------|
| 1 | Baseline Working | 15% | Data loaded, baseline detectors tested |
| 2 | ML Models Ready | 35% | Feature engineering, RF, XGBoost, initial LSTM |
| 3 | DL & Ensemble | 55% | **TARGET ACHIEVED** - Advanced models, ensemble |
| 4 | Demo & Evaluation | 75% | Working demo, comprehensive evaluation |
| 5+ | Final Polish | 100% | Report, presentation, documentation |

---

## 🎓 Attack Types Focus

### Primary: **Sensor Spoofing** 🎯
- **Why:** Most impactful and easy to explain
- **Implementation:** Inject false sensor values
- **Detection:** Statistical deviation, reconstruction error
- **Real Example:** Tank level sensor shows full when empty

### Secondary: **Command Injection**
- **Why:** Critical for ICS security
- **Implementation:** Unauthorized actuator commands
- **Detection:** Sequence analysis, rule-based
- **Real Example:** Unauthorized valve opening

### Tertiary: **DoS Attack**
- **Why:** Network-level threat
- **Implementation:** Traffic flooding
- **Detection:** Packet rate analysis
- **Real Example:** Overwhelm SCADA communication

---

## 📊 Expected Performance Targets

### Baseline Models (Week 1)
- Accuracy: 70-75%
- F1-Score: 0.65-0.70
- FPR: 15-20%

### ML Models (Week 2)
- Accuracy: 85-90%
- F1-Score: 0.80-0.85
- FPR: 8-12%

### DL + Ensemble (Week 3)
- Accuracy: 92-95%
- F1-Score: 0.88-0.92
- FPR: 3-7%

### Final Target
- **Accuracy: > 93%**
- **F1-Score: > 0.90**
- **FPR: < 5%**
- **Detection Delay: < 10 seconds**

---

## 🚀 Daily Workflow

### Morning (2-3 hours)
1. Review yesterday's progress
2. Code implementation
3. Run experiments

### Afternoon (2-3 hours)
1. Analyze results
2. Document findings
3. Prepare next tasks

### Evening (1 hour)
1. Update progress tracking
2. Push code to GitHub
3. Plan tomorrow

---

## 📚 Key Resources

### Datasets
- SWaT: https://itrust.sutd.edu.sg/
- HAI: https://github.com/icsdataset/hai
- Gas Pipeline: http://www.ece.uah.edu/~thm0009/icsdatasets/

### Papers to Read
1. Goh et al. - SWaT Dataset Paper
2. Kravchik & Shabtai - CNN for ICS Detection
3. Beaver et al. - ML Approach to ICS IDS
4. Lemay & Fernandez - Survey on ICS Security

### Tools & Libraries
- Scikit-learn Documentation
- TensorFlow/Keras Tutorials
- SHAP Documentation
- Streamlit Gallery

---

## ✅ Success Criteria

### Minimum Viable Project (50% target)
- ✅ Working baseline detectors
- ✅ At least 2 ML models trained
- ✅ Initial LSTM implementation
- ✅ Basic ensemble method
- ✅ Preliminary evaluation results
- ✅ Working demo application

### Excellent Project (100% target)
- ✅ All above +
- ✅ Multiple DL models (LSTM, Autoencoder, GRU)
- ✅ Sophisticated ensemble
- ✅ Explainability features
- ✅ Comprehensive evaluation
- ✅ Professional documentation
- ✅ Publishable-quality report

---

## 🔄 Continuous Tasks

Throughout the project:
- **Daily:** Git commits with clear messages
- **Daily:** Update progress notes
- **Weekly:** Backup all code and data
- **Weekly:** Review and adjust timeline
- **Biweekly:** Meet with project supervisor

---

## 🎯 Week 1 Action Items (IMMEDIATE)

### TODAY (Day 1):
1. ✅ Project structure created
2. [ ] Install requirements: `pip install -r requirements.txt`
3. [ ] Request SWaT dataset from iTrust
4. [ ] Download Gas Pipeline dataset
5. [ ] Run quick_start.py to verify setup

### Tomorrow (Day 2):
1. [ ] Load first dataset
2. [ ] Create data exploration notebook
3. [ ] Understand data structure
4. [ ] Visualize normal behavior

### Day 3-4:
1. [ ] Statistical analysis of sensors
2. [ ] Identify attack patterns
3. [ ] Document data insights

### Day 5-7:
1. [ ] Implement baseline detectors
2. [ ] Run initial experiments
3. [ ] Document baseline results
4. [ ] Prepare for Week 2

---

## 📞 Support & Resources

**Need Help?**
- Review documentation in `docs/`
- Check configuration in `configs/config.yaml`
- Test with `quick_start.py`
- Review example notebooks

**Common Issues:**
- Package installation: Use `pip install -r requirements.txt`
- Dataset not found: Check `docs/DATASET_GUIDE.md`
- Config errors: Verify `configs/config.yaml`

---

## 🎓 Presentation Tips

### For Your BCA Project Defense:

1. **Start with Impact**
   - "ICS systems control critical infrastructure"
   - "Attacks can cause physical damage"
   - Show real-world incidents (e.g., Ukraine power grid)

2. **Explain Sensor Spoofing Simply**
   - Use water tank analogy
   - Show before/after graphs
   - Demonstrate impact

3. **Demo Live Detection**
   - Run demo application
   - Simulate attack in real-time
   - Show alert generation

4. **Show Technical Depth**
   - Architecture diagram
   - Model performance comparison
   - Feature importance plots

5. **Future Scope**
   - More attack types
   - Real deployment
   - Research publication

---

**Last Updated:** November 5, 2025  
**Status:** Phase 1 Complete ✅ | Ready for Week 1 Execution 🚀
