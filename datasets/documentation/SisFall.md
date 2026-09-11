# SisFall Dataset

## Overview

The **SisFall** dataset is a publicly available wearable sensor dataset developed for **human fall detection research**. It was created to evaluate machine learning and deep learning algorithms capable of distinguishing **Activities of Daily Living (ADLs)** from different types of human falls.

A unique characteristic of the dataset is that it includes data collected from both **young adults** and **elderly participants**, making it one of the most widely used benchmark datasets for fall detection.

---

# Dataset Characteristics

| Attribute          | Value                     |
| ------------------ | ------------------------- |
| Dataset Name       | SisFall                   |
| Dataset Type       | Wearable Sensor Dataset   |
| Domain             | Human Fall Detection      |
| Data Format        | TXT                       |
| Number of Subjects | 38                        |
| Young Adults       | 23                        |
| Elderly Subjects   | 15                        |
| Total Sensor Files | 4505                      |
| Sampling Frequency | 200 Hz                    |
| Sensor Placement   | Waist                     |
| Sensors            | Accelerometer + Gyroscope |

---

# Sensor Information

Each record contains inertial measurements collected using wearable IMU sensors.

## Accelerometer

* X-axis
* Y-axis
* Z-axis

## Gyroscope

* X-axis
* Y-axis
* Z-axis

The dataset stores nine sensor values for every timestamp corresponding to the recorded inertial measurements.

---

# Activities

The dataset contains two major categories.

## Activities of Daily Living (ADL)

Examples include:

* Walking
* Sitting
* Standing
* Running
* Climbing Stairs
* Descending Stairs
* Lying Down
* Picking Up Objects

---

## Fall Activities

Examples include:

* Forward Fall
* Backward Fall
* Sideward Fall
* Syncope Fall
* Falls During Walking

---

# Folder Structure

```text
SisFall/

├── SA01/
├── SA02/
├── ...
├── SA23/
├── SE01/
├── ...
└── SE15/
```

Each subject folder contains multiple TXT files corresponding to different activities and repeated trials.

---

# Data Format

Each sensor file contains timestamp-wise sensor readings.

Example:

```text
17,-179,-99,-18,-504,-352,76,-697,-279
15,-174,-90,-53,-568,-306,48,-675,-254
```

---

# Dataset Statistics (Project)

After preprocessing in this project:

| Statistic     | Value      |
| ------------- | ---------- |
| Subjects      | 38         |
| Sensor Files  | 4505       |
| Total Samples | 15,858,929 |
| ADL Samples   | 10,465,215 |
| Fall Samples  | 5,393,714  |

---

# Preprocessing Performed

The following preprocessing steps were performed:

* Dataset inspection
* Missing value validation
* File validation
* Label standardization
* Dataset statistics generation
* Conversion into a unified master CSV
* Binary label creation (ADL / FALL)

Generated file:

```text
datasets/processed/SisFall/sisfall_master.csv
```

---

# Advantages

* High sampling frequency (200 Hz)
* Includes elderly participants
* Large number of recordings
* Standard benchmark dataset
* Suitable for Machine Learning and Deep Learning

---

# Limitations

* Controlled laboratory environment
* Simulated falls
* Single wearable location (waist)

---

# Official Publication

Noury et al.

**SisFall: A Fall and Movement Dataset**

---

# Download

Download the original dataset from the official source provided by the dataset authors.

After downloading, place the dataset inside:

```text
datasets/raw/SisFall/
```
