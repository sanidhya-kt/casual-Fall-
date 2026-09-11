# MobiAct Dataset

## Overview

The **MobiAct** dataset is a publicly available smartphone-based human activity recognition and fall detection dataset. It was developed to support research in wearable computing, mobile healthcare, activity recognition, and automatic fall detection using smartphone sensors.

Unlike traditional wearable datasets, MobiAct utilizes the built-in sensors of a smartphone, making it highly suitable for real-world mobile applications.

---

# Dataset Characteristics

| Attribute          | Value                                       |
| ------------------ | ------------------------------------------- |
| Dataset Name       | MobiAct                                     |
| Dataset Type       | Smartphone Sensor Dataset                   |
| Domain             | Human Activity Recognition & Fall Detection |
| Data Format        | TXT                                         |
| Number of Subjects | 67                                          |
| Sensor Placement   | Smartphone (Pocket)                         |
| Sampling Frequency | ~20 Hz                                      |
| Sensors            | Accelerometer, Gyroscope, Orientation       |

---

# Sensor Information

The dataset contains measurements collected from smartphone sensors.

## Accelerometer

* X-axis
* Y-axis
* Z-axis

## Gyroscope

* X-axis
* Y-axis
* Z-axis

## Orientation Sensor

* Azimuth
* Pitch
* Roll

Each activity is recorded as three separate files:

* Accelerometer (`*_acc_*.txt`)
* Gyroscope (`*_gyro_*.txt`)
* Orientation (`*_ori_*.txt`)

---

# Activities

## Activities of Daily Living (ADL)

The dataset includes several daily activities such as:

* Walking
* Standing
* Sitting
* Lying
* Jogging
* Jumping
* Walking Upstairs
* Walking Downstairs
* Car Step In
* Car Step Out

---

## Fall Activities

The dataset contains four major fall categories.

| Code | Description                 |
| ---- | --------------------------- |
| FOL  | Forward Fall                |
| FKL  | Front Knee Fall             |
| BSC  | Backward Sitting Chair Fall |
| SDL  | Sideward Fall               |

Each fall type contains multiple trials for every participant.

---

# Folder Structure

```text
MobiAct/

├── sub1/
│   ├── ADL/
│   └── FALLS/
│
├── sub2/
│
├── ...
│
└── sub67/
```

Inside each activity folder:

```text
FOL/

├── FOL_acc_1_1.txt
├── FOL_gyro_1_1.txt
├── FOL_ori_1_1.txt
```

---

# Data Format

Example Accelerometer Record

```text
timestamp,x,y,z
1913880219000,0.90021986,-9.557653,-1.4939818
1914086499000,0.7565677,-9.5385,-1.13964
```

Each row represents one timestamp containing three-axis sensor measurements.

---

# Dataset Statistics (Project)

After preprocessing in this project:

| Statistic     | Value     |
| ------------- | --------- |
| Subjects      | 67        |
| Total Samples | 1,044,765 |
| ADL Samples   | 803,334   |
| Fall Samples  | 241,431   |

---

# Preprocessing Performed

The following preprocessing steps were applied:

* Directory traversal
* Sensor file parsing
* Accelerometer extraction
* Gyroscope extraction
* Orientation extraction
* Activity labeling
* Binary label generation (ADL / FALL)
* Dataset standardization
* Master CSV generation

Generated output:

```text
datasets/processed/MobiAct/mobiact_master.csv
```

---

# Advantages

* Smartphone-based dataset
* Large number of participants
* Multiple fall categories
* Suitable for mobile healthcare applications
* Frequently used benchmark dataset

---

# Limitations

* Controlled experimental environment
* Smartphone position may vary
* Lower sampling frequency than dedicated wearable IMUs

---

# Official Publication

Vavoulas et al.

**The MobiAct Dataset: Recognition of Activities of Daily Living Using Smartphones**

---

# Download

Download the original dataset from the official source provided by the dataset authors.

After downloading, place the dataset inside:

```text
datasets/raw/MobiAct/
```
