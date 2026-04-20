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
│
└── WN26_ROB530_group7_poster.pdf
```

## 🚀 Overview
The goal of this project is to estimate the robot trajectory by fusing multiple sensor inputs:

- IMU → prediction  
- LiDAR (KISS-ICP) → pose estimation  
- Odometry → additional correction  
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

### 4. Additional Notes
- Make sure the dataset paths in the scripts are updated correctly.
- KISS-ICP should be run before EKF/UKF/PF if LiDAR poses are required.
- Raw dataset files are not included in this repository due to GitHub size limits.
