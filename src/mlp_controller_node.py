#!/usr/bin/env python3
"""
ROS2 Node for MLP-based robot arm control.
Uses a trained MLP model to compute joint angle changes (dq) based on 
current end-effector position (x), joint angles (q), and desired displacement (dx).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener

import numpy as np
import torch
from torch import nn
import os
from ament_index_python.packages import get_package_share_directory


class MLPModel(nn.Module):
    """MLP Model architecture - must match the trained model."""
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, output_size)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.mish(self.fc1(x))
        x = self.mish(self.fc2(x))
        x = self.dropout2(x)
        x = self.mish(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class MLPControllerNode(Node):
    """ROS2 Node for MLP-based robot arm control."""
    
    def __init__(self):
        super().__init__('mlp_controller_node')
        
        # Declare parameters
        self.declare_parameter('njoints', 3)
        self.declare_parameter('model_path', '')
        self.declare_parameter('scaler_path', '')
        self.declare_parameter('position_threshold', 0.01)  # meters
        self.declare_parameter('velocity_threshold', 0.01)  # rad/s
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('max_dq_single', 0.1745)  # 10 degrees in radians
        self.declare_parameter('min_dq_sum', 0.1745)  # 10 degrees in radians
        
        # Get parameters
        self.njoints = self.get_parameter('njoints').value
        self.model_path = self.get_parameter('model_path').value
        self.scaler_path = self.get_parameter('scaler_path').value
        self.position_threshold = self.get_parameter('position_threshold').value
        self.velocity_threshold = self.get_parameter('velocity_threshold').value
        self.control_rate = self.get_parameter('control_rate').value
        self.max_dq_single = self.get_parameter('max_dq_single').value
        self.min_dq_sum = self.get_parameter('min_dq_sum').value
        
        # Joint limits
        self.joint_limits = self._get_joint_limits()
        
        # State variables
        self.q = None  # Current joint positions
        self.q_dot = None  # Current joint velocities
        self.x = None  # Current end-effector position [x, y, z]
        self.target = None  # Target position [x, y, z]
        self.is_moving = False
        
        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load model and scalers
        self._load_model()
        
        # TF2 for end-effector position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriber for joint states
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10)
        
        # Subscriber for target position
        self.target_sub = self.create_subscription(
            PointStamped, '/target_position',
            self.target_callback, 10)
        
        # Publisher for joint commands
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_position_controller/commands', 10)
        
        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate, self.control_loop)
        
        self.get_logger().info(f"MLP Controller Node initialized with {self.njoints} joints")
    
    def _get_joint_limits(self):
        """Get joint limits based on number of joints."""
        limits = []
        for i in range(self.njoints):
            if i == 0:
                limits.append((-np.pi, np.pi))
            else:
                limits.append((-1.8, 1.8))
        return limits
    
    def _load_model(self):
        """Load the trained MLP model and scalers."""
        if not self.model_path:
            self.get_logger().error("Model path not specified!")
            return
        
        # Determine input/output sizes based on number of joints
        # Input: [x, y, q1, ..., qn, dx, dy] for 2D or [x, y, z, q1, ..., qn, dx, dy, dz] for 3D
        # For 3 joints: 7 inputs (x, y, q1, q2, q3, dx, dy), 3 outputs (dq1, dq2, dq3)
        # For 4 joints: 10 inputs, 4 outputs
        # For 6 joints: 12 inputs, 6 outputs
        if self.njoints == 3:
            input_size = 7  # x, y, q1, q2, q3, dx, dy
        elif self.njoints == 4:
            input_size = 10  # x, y, z, q1, q2, q3, q4, dx, dy, dz
        elif self.njoints == 6:
            input_size = 12  # x, y, z, q1-q6, dx, dy, dz
        else:
            input_size = 2 + self.njoints + 2  # Default: 2D case
        
        output_size = self.njoints
        
        # Load model
        self.model = MLPModel(input_size, output_size).to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.get_logger().info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None
        
        # Load scalers if provided
        self.scaler_X = None
        self.scaler_y = None
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                import pickle
                with open(self.scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_X = scalers['scaler_X']
                    self.scaler_y = scalers['scaler_y']
                self.get_logger().info(f"Scalers loaded from {self.scaler_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load scalers: {e}")
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages."""
        arm_indices = [i for i, name in enumerate(msg.name) if 'arm_joint' in name]
        
        if len(arm_indices) < self.njoints:
            return
        
        self.q = [msg.position[arm_indices[i]] for i in range(self.njoints)]
        self.q_dot = [msg.velocity[arm_indices[i]] for i in range(self.njoints)]
    
    def target_callback(self, msg):
        """Callback for target position messages."""
        self.target = [msg.point.x, msg.point.y, msg.point.z]
        self.get_logger().info(f"New target received: {self.target}")
    
    def _update_ee_position(self):
        """Update end-effector position from TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tip_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            self.x = [
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ]
            return True
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")
            return False
    
    def _normalize_displacement(self, dt):
        """Normalize displacement vector to unit length."""
        norm = np.linalg.norm(dt)
        if norm < 1e-6:
            return np.zeros_like(dt)
        return dt / norm
    
    def _clip_dq(self, dq):
        """Clip joint angle changes to respect constraints."""
        dq = np.array(dq)
        
        # Clip individual joint changes to max_dq_single
        dq = np.clip(dq, -self.max_dq_single, self.max_dq_single)
        
        # Ensure sum of absolute changes is above minimum
        dq_sum = np.sum(np.abs(dq))
        if dq_sum < self.min_dq_sum and dq_sum > 1e-6:
            # Scale up to meet minimum
            scale = self.min_dq_sum / dq_sum
            dq = dq * scale
            # Re-clip after scaling
            dq = np.clip(dq, -self.max_dq_single, self.max_dq_single)
        
        return dq
    
    def _apply_joint_limits(self, q_target):
        """Apply joint limits to target positions."""
        q_target = np.array(q_target)
        for i in range(self.njoints):
            q_target[i] = np.clip(q_target[i], self.joint_limits[i][0], self.joint_limits[i][1])
        return q_target.tolist()
    
    def _is_arm_stationary(self):
        """Check if the arm has reached its target (low velocity)."""
        if self.q_dot is None:
            return False
        return all(abs(v) < self.velocity_threshold for v in self.q_dot)
    
    def _predict_dq(self, x, q, dx):
        """Use the learned function to predict joint angle changes."""
        if self.model is None:
            self.get_logger().error("Model not loaded!")
            return None
        
        # Build input features based on joint count
        if self.njoints == 3:
            # Input: [x, y, q1, q2, q3, dx, dy]
            features = [x[0], x[1]] + list(q) + [dx[0], dx[1]]
        elif self.njoints == 4:
            # Input: [x, y, z, q1, q2, q3, q4, dx, dy, dz]
            features = [x[0], x[1], x[2]] + list(q) + [dx[0], dx[1], dx[2]]
        elif self.njoints == 6:
            # Input: [x, y, z, q1-q6, dx, dy, dz]
            features = [x[0], x[1], x[2]] + list(q) + [dx[0], dx[1], dx[2]]
        else:
            features = [x[0], x[1]] + list(q) + [dx[0], dx[1]]
        
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Apply scaler if available
        if self.scaler_X is not None:
            features = self.scaler_X.transform(features)
        
        # Convert to tensor and predict
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            dq_scaled = self.model(features_tensor).cpu().numpy()
        
        # Inverse transform if scaler available
        if self.scaler_y is not None:
            dq = self.scaler_y.inverse_transform(dq_scaled)[0]
        else:
            dq = dq_scaled[0]
        
        return dq
    
    def position_control(self, q_target):
        """Publish joint angle commands."""
        msg = Float64MultiArray()
        msg.data = q_target
        self.arm_cmd_pub.publish(msg)
    
    def control_loop(self):
        """Main control loop."""
        # Check if we have all necessary data
        if self.target is None or self.q is None:
            return
        
        # Update end-effector position
        if not self._update_ee_position():
            return
        
        # Compute displacement to target
        dt = np.array(self.target) - np.array(self.x)
        dt_norm = np.linalg.norm(dt)
        
        # Check if we've reached the target
        if dt_norm < self.position_threshold:
            if self._is_arm_stationary():
                if self.is_moving:
                    self.get_logger().info("Target reached!")
                    self.is_moving = False
            return
        
        self.is_moving = True
        
        # Normalize displacement for prediction (dx outside the range of dataset values)
        dx = self._normalize_displacement(dt)
        
        # Scale dx to a reasonable magnitude for the model
        # The model was trained on small displacements
        scale_factor = min(dt_norm, 0.1)  # Cap at 10cm step
        dx = dx * scale_factor
        
        # Predict joint angle changes using learned function
        dq = self._predict_dq(self.x, self.q, dx)
        
        if dq is None:
            return
        
        # Apply constraints
        dq = self._clip_dq(dq)
        
        # Compute target joint angles
        q_target = np.array(self.q) + dq
        
        # Apply joint limits
        q_target = self._apply_joint_limits(q_target)
        
        # Send command
        self.position_control(q_target)
        
        self.get_logger().debug(
            f"x: {self.x}, target: {self.target}, dt_norm: {dt_norm:.4f}, dq: {dq}")


def main(args=None):
    rclpy.init(args=args)
    node = MLPControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
