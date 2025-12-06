#!/usr/bin/env python3
"""
Utility script to export scalers from training data.
These scalers are required by the ROS2 node to properly normalize inputs/outputs.
"""

import numpy as np
import pandas as pd
import pickle
import sys
import os
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load training data and determine feature/target split."""
    data = pd.read_csv(file_path)
    
    if len(data.columns) == 10:
        # 3 joints: first 7 columns are features, last 3 are target
        X = data.iloc[:, :7].values
        y = data.iloc[:, 7:].values
        njoints = 3
    elif len(data.columns) == 14:
        # 4 joints: first 10 columns are features, last 4 are target
        X = data.iloc[:, :10].values
        y = data.iloc[:, 10:].values
        njoints = 4
    elif len(data.columns) == 18:
        # 6 joints: first 12 columns are features, last 6 are target
        X = data.iloc[:, :12].values
        y = data.iloc[:, 12:].values
        njoints = 6
    else:
        raise ValueError(f"Unexpected number of columns: {len(data.columns)}")
    
    return X, y, njoints


def export_scalers(train_file_path, output_path=None):
    """
    Fit scalers on training data and export them.
    
    Args:
        train_file_path: Path to training CSV file
        output_path: Output path for pickle file (optional)
    """
    print(f"Loading training data from {train_file_path}")
    X, y, njoints = load_data(train_file_path)
    
    print(f"Detected {njoints} joints")
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Fit scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X)
    scaler_y.fit(y)
    
    # Prepare output path
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(train_file_path)))
        output_path = os.path.join(base_dir, 'models', f'scalers_{njoints}joints.pkl')
    
    # Save scalers
    scalers = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'njoints': njoints,
        'input_size': X.shape[1],
        'output_size': y.shape[1],
    }
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"Scalers exported to {output_path}")
    print(f"  - Input mean: {scaler_X.mean_}")
    print(f"  - Input std: {scaler_X.scale_}")
    print(f"  - Output mean: {scaler_y.mean_}")
    print(f"  - Output std: {scaler_y.scale_}")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_scalers.py <train_csv_path> [output_path]")
        print("Example: python export_scalers.py datasets/train/reacher3_train_1.csv")
        sys.exit(1)
    
    train_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    export_scalers(train_path, output_path)
