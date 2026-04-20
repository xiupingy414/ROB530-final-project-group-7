## 📌 ROB530 Final Project – Localization with Multiple Filters

This repository contains the implementation and evaluation of multiple state estimation methods for robot localization using the ANYmal Grand Tour dataset. The project focuses on comparing different filters in terms of accuracy, robustness, and computational efficiency.
A detailed description of the robot setup is available in the official documentation:
[ANYmal Grand Tour Setup](https://grand-tour.leggedrobotics.com/setup).
## Repository Structure

```text
ROB530-final-project-group-7/
│
├── EKF/
│   ├── extended_kalman_filter_s*.py
│   ├── ekf_trajectory_s*.npy
│   ├── gt_trajectory_s*.npy
│   ├── test_final_s*_v2.py
│   └── Readme.md
│
├── UKF/
│   ├── error.py
│   ├── scene*.png
│   ├── scene*_runtime_summary.txt
│   └── Readme.md
│
├── PF/
│   ├── PF_demo2.py
│   └── Readme.md
│
├── lidar/
│   ├── hesai_points_undistorted/
│   ├── KISS_ICP.py
│   ├── KISS_ICP_output.py
│   ├── mapping.py
│   └── animation.py
│
├── ri_ekf_results/
│   ├── RI_EKF_demo1.py
│   ├── ri_ekf_trajectory*.npy
│   ├── runtime_*.npz
│   ├── runtime_*.txt
├── README.md
└── WN26_ROB530_group7_poster.pdf
```

## 🚀 Overview
The goal of this project is to estimate the robot trajectory by fusing multiple sensor inputs:

- IMU → prediction  
- LiDAR (KISS-ICP) → pose estimation  
- Ground truth (prism) → evaluation  

We implement and compare:

- Extended Kalman Filter (EKF)  
- Unscented Kalman Filter (UKF)  
- Particle Filter (PF)  
- Invariant EKF (RI-EKF / InEKF)

## 📡 Dataset

We use the ANYmal Grand Tour dataset, which provides:

- Hesai LiDAR point clouds
- IMU measurements
- Ground truth trajectory

⚠️ Note:
Due to GitHub file size limits, raw dataset files are not included.
Please download them from the official dataset website [Hugging Face Dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/tree/main) and update the file paths accordingly.

## ⚙️ Environment Setup

### 1. Create Conda Environment

```bash
conda create -n na568 python=3.9
conda activate na568
```
> Note: The project was developed and tested in a Conda environment named `na568`.
> You may use a different environment name if preferred.

### 2. Install Required Packages
```bash
pip install numpy scipy matplotlib zarr open3d tqdm
```
### 3. Install KISS-ICP
KISS-ICP is required for LiDAR-based pose estimation.  

Option 1: Install via pip
```bash
pip install kiss-icp
```
Option 2: Install from source
```bash
git clone https://github.com/PRBonn/kiss-icp.git
cd kiss-icp
pip install -e .
```

## 🧭 Usage

This project is organized into multiple modules corresponding to different stages of the localization pipeline. Below is a guide on which files to use at each step.

---

### 1. LiDAR Processing

The LiDAR-related scripts are located in the `lidar/` folder.

- `lidar/hesai_points_undistorted/`  
  Raw LiDAR point cloud data.

- `lidar/KISS_ICP.py`  
  Generates mapping results and performs LiDAR-based registration.

- `lidar/KISS_ICP_output.py`  
  Extracts pose trajectories from LiDAR data. These poses are later used as correction inputs in the filters.

Run:

```bash
python lidar/KISS_ICP.py
python lidar/KISS_ICP_output.py
```

### 2. Extended Kalman Filter (EKF)

The EKF implementation is located in the `EKF/` folder.

- `extended_kalman_filter_s1.py`, `extended_kalman_filter_s2.py`, `extended_kalman_filter_s3.py`
EKF implementations for three different scenarios.
- `test_final_s1_v2.py`, `test_final_s2_v2.py`, `test_final_s3_v2.py`
Scripts used to run EKF for each scenario.
- `ekf_trajectory_s*.npy`
Saved EKF trajectory outputs.
- `gt_trajectory_s*.npy`
Ground truth trajectories for evaluation.

Run:

```bash
python EKF/test_final_s1_v2.py
python EKF/test_final_s2_v2.py
python EKF/test_final_s3_v2.py
```

### 3. Unscented Kalman Filter (UKF)

The UKF implementation is located in the `UKF/` folder.

- `test1_runtime_noplotwait.py`, `test2_runtime_noplotwait.py`, `test3_runtime_noplotwait.py`
Main UKF scripts for each scenario.
- `error.py`
Used for error analysis.
- `scene*.png`
Visualization results.
- `scene*_runtime_summary.txt`
Runtime summaries.

Run:

```bash
python UKF/test1_runtime_noplotwait.py
python UKF/test2_runtime_noplotwait.py
python UKF/test3_runtime_noplotwait.py
```

### 4. Particle Filter (PF)

The PF implementation is located in the `PF/` folder.

- `PF_demo2.py`
Main Particle Filter script.

Run:

```bash
python PF/PF_demo2.py
```

### 5. Invariant EKF (RI-EKF / InEKF)

The InEKF implementation is located in the `ri_ekf_results/` folder.

- `RI_EKF_demo1.py`
Main script for InEKF.
- `ri_ekf_trajectory*.npy`
Output trajectories.
- `runtime_*.npz, runtime_*.txt`
Runtime statistics.

Run:

```bash
python ri_ekf_results/RI_EKF_demo1.py
```

### 6. Notes
- Different scenarios (`s1`, `s2`, `s3`) correspond to different dataset sequences available in the 
[ANYmal Grand Tour dataset](https://grand-tour.leggedrobotics.com/dataset):  
`s1` → ETH-1, `s2` → SPX-1, `s3` → SNOW-1.
- Make sure dataset paths are correctly set before running any script.
- LiDAR results should be generated first, as they are used for correction in all filters.
