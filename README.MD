# MLP Robot Arm Controller

A ROS2 package for controlling robot arms using Multi-Layer Perceptron (MLP) and LR networks trained to learn inverse kinematics.

## Overview

This project trains neural networks to predict joint angle changes (dq) given:
- Current end-effector position (x, y, z)
- Current joint angles (q1, q2, ..., qn)
- Desired displacement (dx, dy, dz)

Supports **3, 4, and 6 joint** robot arm configurations.

## Project Structure

```
├── src/                    # Training scripts
│   └── main_neural_network_mlp.py
│   └── main_linear_regression.py
│   └── export_scalers.py 		# To export scalers for ROS2 package
│   └── grid_search.py 			# Quick grid search script
│   └── hyperparameter_search.py 		# random hyperparameter search, used to select params for the grid search
├── mlp_controller/         # ROS2 package
│   ├── mlp_controller_node.py
│   └── mlp_target_sender.py
├── datasets/               # Training/test data
│   ├── train/
│   └── test/
├── models/                 # Trained models (.pth) and scalers (.pkl)
├── figures/                # Training curves and evaluation plots
└── launch/                 # ROS2 launch files
```

## Installation

### Dependencies
- ROS2
- PyTorch
- NumPy, Pandas, scikit-learn

### Install Python Libraries

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn
```

### Run Training Script

```bash
python src/main_neural_network_mlp.py datasets/train/reacher3_train_1.csv datasets/test/reacher3_test_1.csv
```

## Usage

### Launch Controller
```bash
ros2 launch mlp_controller mlp_controller.launch.py \
    njoints:=3 \
    model_path:=/path/to/mlp_model_3joints.pth \
    scaler_path:=/path/to/scalers_3joints.pkl
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `njoints` | 3 | Number of joints (3, 4, or 6) |
| `model_path` | - | Path to trained model |
| `scaler_path` | - | Path to scaler file |


### Send Target Positions
```bash
ros2 run mlp_controller mlp_target_sender
```
**Controls**: W/S (X), A/D (Y), Arrow Up/Down (Z), Q (quit)

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | JointState | Current joint positions |
| `/target_position` | PointStamped | Target end-effector position |
| `/arm_position_controller/commands` | Float64MultiArray | Joint commands |

