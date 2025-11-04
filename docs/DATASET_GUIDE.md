# ICS Intrusion Detection System - Dataset Guide

## Overview
This guide will help you obtain and set up ICS (Industrial Control Systems) datasets for training and testing your intrusion detection models.

## Primary Datasets (Recommended for BCA Project)

### 1. **SWaT Dataset** (Secure Water Treatment) - MOST RECOMMENDED
**Why:** Well-documented, real physical testbed data, multiple attack types, widely cited

**Access:**
- **Official Source:** iTrust, Singapore University of Technology and Design (SUTD)
- **Website:** https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
- **Request Process:**
  1. Go to: https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_datasets/
  2. Fill out the dataset request form
  3. Provide your academic email and project details
  4. Usually approved within 1-2 days for academic use
  5. **FREE for academic/research purposes**

**Dataset Details:**
- 11 days of normal operation data
- 4 days of attack data
- 51 sensors/actuators
- Attack types: SSPA (Single Stage Point Attack), MSPA (Multi Stage Point Attack)
- Sample rate: 1 second
- Size: ~1.5 GB

**Attack Types Included:**
- Sensor spoofing (easiest to explain!)
- Actuator manipulation
- Network DoS
- Command injection

---

### 2. **WADI Dataset** (Water Distribution) - Good Alternative
**Access:** Same as SWaT (iTrust SUTD)
- 16 days of normal operation
- 2 days of attacks
- 123 sensors/actuators
- Larger and more complex than SWaT

---

### 3. **Gas Pipeline Dataset** - PUBLIC & FREE (Best Backup)
**Why:** Publicly available, no registration required, good for getting started

**Access:**
- **GitHub:** https://github.com/icsresearch/gas-pipeline-dataset
- **Direct Download:** Available on Mississippi State's website
- **Direct Link:** http://www.ece.uah.edu/~thm0009/icsdatasets/

**Dataset Details:**
- Simulated gas pipeline SCADA system
- Multiple attack scenarios
- CSV format, easy to process
- Size: ~500 MB

---

### 4. **HAI Dataset** (Hardware-in-the-loop Augmented ICS) - PUBLIC
**Access:**
- **Website:** https://github.com/icsdataset/hai
- **Direct Download:** Available on GitHub
- **FREE and PUBLIC**

**Dataset Details:**
- 38 attacks on 4 different processes
- Process control + network traffic
- Well-labeled attack scenarios

---

## Alternative Public Datasets (If Above Not Available)

### 5. **Power System Attack Datasets**
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data
- **Status:** FREE, Immediate download

### 6. **BATADAL** (Battle of the Attack Detection Algorithms)
- **Source:** https://github.com/scy-phy/batadal
- **Status:** PUBLIC
- Water distribution network data

### 7. **Morris Worm ICS Dataset**
- **Source:** Various research repositories
- Can be used for network-level detection

---

## My Recommendation for Your Project

### **Phase 1 Strategy (Week 1):**

**Immediate Start (Today):**
1. Download **Gas Pipeline Dataset** (public, no wait)
   - Use this to build your pipeline and baseline models
   - Get familiar with data structure

**Parallel Request (Same Day):**
2. Request **SWaT Dataset** from iTrust
   - Fill form today (takes 5 minutes)
   - Usually approved in 24-48 hours
   - This will be your MAIN dataset for final results

### **Why This Approach?**
- ✅ Start working immediately with Gas Pipeline data
- ✅ Build and test your code while waiting for SWaT
- ✅ SWaT is industry-standard and will impress evaluators
- ✅ You'll have 2 datasets for comparison (stronger project)

---

## Attack Types to Focus On (Easy to Explain)

Based on your requirement for "impactful and easy to explain," I recommend:

### 1. **Sensor Spoofing Attack** ⭐ PRIMARY FOCUS
- **What:** Attacker manipulates sensor readings
- **Impact:** Operators see false data, make wrong decisions
- **Real Example:** "Level sensor shows tank is full when it's actually overflowing"
- **Easy to visualize:** Show graph of sensor before/during/after attack
- **Easy to explain:** "Fake readings to fool the system"

### 2. **Command Injection Attack** ⭐ SECONDARY
- **What:** Unauthorized commands sent to actuators
- **Impact:** Pumps/valves controlled by attacker
- **Real Example:** "Attacker opens valve when it should be closed"
- **Easy to explain:** "Hacker sends fake control commands"

### 3. **DoS (Denial of Service)** ⭐ GOOD ADDITION
- **What:** Flood network to prevent legitimate traffic
- **Impact:** Control system can't communicate
- **Easy to explain:** "Too many messages, system gets overwhelmed"

---

## Download Instructions

### For Gas Pipeline (Immediate):
```bash
# Method 1: Git Clone
git clone https://github.com/icsresearch/gas-pipeline-dataset.git

# Method 2: Direct download from browser
# Visit: http://www.ece.uah.edu/~thm0009/icsdatasets/
# Download the dataset files
```

### For SWaT (After Approval):
1. You'll receive download link via email
2. Download both Normal and Attack datasets
3. Place in `data/raw/swat/` folder

---

## Dataset Structure (After Download)

```
ICS NETWORKS/
└── data/
    └── raw/
        ├── swat/
        │   ├── SWaT_Dataset_Normal_v1.csv
        │   └── SWaT_Dataset_Attack_v0.csv
        ├── gas_pipeline/
        │   ├── normal_data.csv
        │   └── attack_data.csv
        └── wadi/
            ├── WADI_14days.csv
            └── WADI_attackdata.csv
```

---

## Next Steps (After Getting Data)

1. **Data Exploration** (I'll create notebook for this)
   - Understand sensor types
   - Visualize normal behavior
   - Identify attack patterns

2. **Data Preprocessing**
   - Handle missing values
   - Normalize data
   - Create time windows

3. **Feature Engineering**
   - Extract statistical features
   - Create correlation features
   - Temporal patterns

---

## Questions to Answer Before Requesting SWaT

For the iTrust request form, prepare these:
- **Purpose:** BCA Final Year Project - AI for Intrusion Detection
- **Institution:** [Your College Name]
- **Supervisor:** [Your Project Guide Name]
- **Timeline:** November 2025 - January 2026
- **Will you publish?** Mention it's for academic project (increases approval chance)

---

## Backup Plan (If Nothing Works)

If you face issues accessing datasets, I can help you:
1. Generate synthetic ICS data based on realistic models
2. Use your current `mock_data.py` but make it more sophisticated
3. Combine multiple small public datasets

---

## Contact & Support

- **iTrust Support:** itrust@sutd.edu.sg
- **HAI Dataset Issues:** Check GitHub issues

---

## Action Items for TODAY:

- [ ] Request SWaT dataset from iTrust (5 minutes)
- [ ] Download Gas Pipeline dataset (immediate)
- [ ] Install required Python packages
- [ ] Run data exploration notebook (I'll create this next)

Let me know once you've downloaded the Gas Pipeline dataset or requested SWaT!
