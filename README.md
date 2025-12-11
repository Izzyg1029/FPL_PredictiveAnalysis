"# FPL_PredictiveAnalysis" 

"#Cassidy Verifying Access"

# FPL Predictive Analysis – Feature Health Tree (Cassidy)

This branch implements the **Feature Health Tree**, which computes device-level
health indicators, flags, and risk scores for Ample device data (ZM1, UM3, MM3).
It also prepares the framework for multi-day analysis such as drift, variance,
and failure frequency calculations.

This document summarizes:
- Data inputs
- Feature engineering pipeline
- Device-specific features
- Risk score formulation
- Placeholder future features
- How to run the test script

---

## 📌 1. Overview

The Feature Health Tree converts raw Ample device exports into meaningful
health indicators and risk scores. Each device type (ZM1, UM3, MM3) receives
a tailored feature set based on its hardware characteristics:

- **ZM1 – Battery Powered Overhead**
- **UM3 – Underground**
- **MM3 – Line-Powered Overhead**

The feature pipeline is implemented in: feature_health/health_features.py
and is tested using:
test_health_features.py


---

## 📊 3. Required Input Data

### **A) Ample Device Export (CSV)**
Must contain:
- Serial  
- Device_Type  
- Last_Heard  
- LineCurrent  
- LineTemperatrue  
- BatteryLevel  
- BatteryLatestReport  
- Latitude, Longitude  

### **B) FPL Install-Date Sheet**
Must contain:
- Serial  
- InstallDate  

---

## ⚙️ 4. Feature Engineering Pipeline

The main entrypoint is:

```python
build_health_features(df_devices, install_df=None)

