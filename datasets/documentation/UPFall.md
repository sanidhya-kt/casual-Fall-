# UP-Fall Dataset

## Overview

The **UP-Fall** dataset is a publicly available **multimodal human fall detection dataset** developed by researchers at the University of Porto. Unlike traditional wearable datasets, UP-Fall combines wearable inertial sensors with environmental sensors, enabling research in wearable computing, Internet of Things (IoT), Ambient Assisted Living (AAL), and healthcare monitoring systems.

The dataset contains recordings of both **Activities of Daily Living (ADLs)** and **simulated fall events** performed by multiple participants.

---

# Dataset Characteristics

| Attribute          | Value                                 |
| ------------------ | ------------------------------------- |
| Dataset Name       | UP-Fall                               |
| Dataset Type       | Multimodal Fall Detection Dataset     |
| Domain             | Human Fall Detection                  |
| Data Format        | CSV                                   |
| Number of Subjects | 17                                    |
| Activities         | 11                                    |
| Sampling Frequency | ~18–20 Hz                             |
| Sensor Types       | Wearable IMUs + Environmental Sensors |

---

# Wearable Sensors

The dataset contains five wearable sensing locations.

| Sensor Location | Measurements                    |
| --------------- | ------------------------------- |
| Ankle           | Accelerometer, Gyroscope, Light |
| Belt            | Accelerometer, Gyroscope, Light |
| Pocket          | Accelerometer, Gyroscope, Light |
| Neck            | Accelerometer, Gyroscope, Light |
| Wrist           | Accelerometer, Gyroscope, Light |

---

# Environmental Sensors

The dataset also includes environmental sensing information.

* Brain Sensor
* Infrared Sensor 1
* Infrared Sensor 2
* Infrared Sensor 3
* Infrared Sensor 4
* Infrared Sensor 5
* Infrared Sensor 6

These additional sensors make UP-Fall suitable for multimodal and IoT-based healthcare applications.

---

# Activities

The dataset defines **11 activities**.

## Fall Activities

| Activity ID | Description                   |
| ----------- | ----------------------------- |
| 1           | Falling Forward (Using Hands) |
| 2           | Falling Forward (Using Knees) |
| 3           | Falling Backward              |
| 4           | Falling Sideward              |
| 5           | Falling While Sitting         |

---

## Activities of Daily Living (ADL)

| Activity ID | Description          |
| ----------- | -------------------- |
| 6           | Walking              |
| 7           | Standing             |
| 8           | Sitting              |
| 9           | Picking Up an Object |
| 10          | Jumping              |
| 11          | Laying Down          |

---

# Binary Label Mapping

For this project, the activities were converted into binary labels.

| Activities | Label |
| ---------- | ----- |
| 1–5        | FALL  |
| 6–11       | ADL   |

This binary mapping follows the activity definitions provided in the original UP-Fall publication and simplifies supervised learning for fall detection.

---

# Data Format

The original dataset is provided as a CSV file.

Example columns:

```text
TimeStamps
AnkleAccelerometer
AnkleAngularVelocity
AnkleLuminosity
RightPocketAccelerometer
RightPocketAngularVelocity
BeltAccelerometer
NeckAccelerometer
WristAccelerometer
BrainSensor
Infrared1
Infrared2
Infrared3
Infrared4
Infrared5
Infrared6
Subject
Activity
Trial
Tag
```

---

# Dataset Statistics (Project)

After preprocessing in this project:

| Statistic     | Value   |
| ------------- | ------- |
| Subjects      | 17      |
| Activities    | 11      |
| Total Samples | 294,679 |
| ADL Samples   | 248,728 |
| Fall Samples  | 45,951  |

---

# Preprocessing Performed

The following preprocessing pipeline was applied:

* Column renaming
* Missing value handling
* Duplicate removal
* Activity standardization
* Binary label generation
* Dataset validation
* Master CSV generation

Generated output:

```text
datasets/processed/UP-Fall/upfall_master.csv
```

---

# Advantages

* Multimodal dataset
* Multiple wearable sensor locations
* Environmental sensing support
* Suitable for IoT applications
* Useful for multimodal AI and healthcare systems

---

# Limitations

* Limited number of participants
* Controlled laboratory environment
* Simulated falls
* Lower sampling frequency compared with SisFall

---

# Official Publication

Martinez-Villaseñor et al.

**UP-Fall Detection Dataset: A Multimodal Approach**

---

# Download

Download the original dataset from the official source provided by the dataset authors.

After downloading, place the dataset inside:

```text
datasets/raw/UP-Fall/
```
