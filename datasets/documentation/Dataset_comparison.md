# Dataset Comparison

## Overview

This project utilizes three publicly available human fall detection datasets:

1. **SisFall**
2. **MobiAct**
3. **UP-Fall**

Each dataset provides unique characteristics in terms of sensing modality, participant diversity, sampling frequency, and sensor placement. Combining these datasets improves the robustness and generalizability of machine learning and deep learning models for fall detection.

---

# Dataset Summary

| Feature               | SisFall      | MobiAct          | UP-Fall                             |
| --------------------- | ------------ | ---------------- | ----------------------------------- |
| Dataset Type          | Wearable IMU | Smartphone-Based | Multimodal Wearable + Environmental |
| Data Format           | TXT          | TXT              | CSV                                 |
| Number of Subjects    | 38           | 67               | 17                                  |
| Sampling Frequency    | 200 Hz       | ~20 Hz           | ~18–20 Hz                           |
| Accelerometer         | ✓            | ✓                | ✓                                   |
| Gyroscope             | ✓            | ✓                | ✓                                   |
| Orientation Sensor    | ✗            | ✓                | ✗                                   |
| Environmental Sensors | ✗            | ✗                | ✓                                   |
| Smartphone Sensors    | ✗            | ✓                | ✗                                   |
| ADL Activities        | ✓            | ✓                | ✓                                   |
| Fall Activities       | ✓            | ✓                | ✓                                   |

---

# Sensor Comparison

| Sensor           | SisFall | MobiAct | UP-Fall |
| ---------------- | ------- | ------- | ------- |
| Accelerometer    | ✓       | ✓       | ✓       |
| Gyroscope        | ✓       | ✓       | ✓       |
| Orientation      | ✗       | ✓       | ✗       |
| Light Sensor     | ✗       | ✗       | ✓       |
| Infrared Sensors | ✗       | ✗       | ✓       |
| Brain Sensor     | ✗       | ✗       | ✓       |

---

# Dataset Statistics (After Preprocessing)

| Dataset            |        Samples |
| ------------------ | -------------: |
| SisFall            |     15,858,929 |
| MobiAct            |      1,044,765 |
| UP-Fall            |        294,679 |
| **Merged Dataset** | **17,198,373** |

---

# Label Distribution (Merged Dataset)

| Label |    Samples |
| ----- | ---------: |
| ADL   | 11,517,277 |
| FALL  |  5,681,096 |

---

# Strengths of Each Dataset

## SisFall

* High sampling frequency (200 Hz)
* Includes both young and elderly participants
* Large number of sensor recordings
* Widely used benchmark for wearable fall detection

---

## MobiAct

* Smartphone-based data collection
* Large participant pool
* Multiple fall scenarios
* Suitable for mobile healthcare applications

---

## UP-Fall

* Multimodal sensing
* Multiple wearable sensor locations
* Environmental sensing (Infrared and Brain Sensors)
* Well suited for IoT and Ambient Assisted Living research

---

# Why Combine These Datasets?

Using multiple datasets provides several advantages:

* Increased diversity of participants and activities
* Improved robustness across different sensor configurations
* Better generalization to real-world scenarios
* Support for wearable, smartphone, and multimodal sensing environments
* Reduced dependence on a single benchmark dataset

---

# Preprocessing Pipeline

The preprocessing workflow implemented in this project is illustrated below.

```text
Raw Datasets
     │
     ├── SisFall
     ├── MobiAct
     └── UP-Fall
             │
             ▼
      Dataset Inspection
             │
             ▼
      Data Cleaning
             │
             ▼
 Missing Value Handling
             │
             ▼
 Label Standardization
             │
             ▼
 Binary Classification
      (ADL / FALL)
             │
             ▼
 Unified Master CSV Files
             │
             ▼
 Merged Dataset
 (17,198,373 Samples)
```

---

# Project Outputs

The preprocessing pipeline generates:

```text
datasets/processed/

├── SisFall/
│   └── sisfall_master.csv
│
├── MobiAct/
│   └── mobiact_master.csv
│
├── UP-Fall/
│   └── upfall_master.csv
│
└── merged/
    └── merged_dataset.csv
```

---

# Conclusion

By integrating SisFall, MobiAct, and UP-Fall into a unified preprocessing pipeline, this project creates a comprehensive benchmark dataset for binary fall detection (ADL vs FALL). The resulting merged dataset combines wearable, smartphone, and multimodal sensing data, making it suitable for traditional machine learning, deep learning, and future Agentic AI–based fall detection systems.
