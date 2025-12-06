#!/usr/bin/env python3
"""
ROS2 Node to send target positions using keyboard controls.
Use WASD for X/Y movement and Arrow Up/Down for Z movement.

Controls:
  W/S - Move forward/backward (X axis)
  A/D - Move left/right (Y axis)
  Up/Down Arrow - Move up/down (Z axis)
  Q - Quit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
import numpy as np
import sys
import termios
import tty
import select


class MLPTargetSender(Node):
    """Node to send target positions via keyboard control."""

    def __init__(self):
        super().__init__('mlp_target_sender')

        # Declare parameters
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_z', 0.0)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('step_size', 0.04)  # meters per keypress

        # Get parameters
        self.initial_x = self.get_parameter('initial_x').value
        self.initial_y = self.get_parameter('initial_y').value
        self.initial_z = self.get_parameter('initial_z').value
        self.frame_id = self.get_parameter('frame_id').value
        self.step_size = self.get_parameter('step_size').value

        # Current target position


        # Current end-effector position (from TF)
        self.current_ee_position = None

        # TF2 for end-effector position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        try:
            try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tip_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            self.target= [
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ]
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")
            self.target = [0.0, 0.0, 0.0]

        self._update_ee_position()
        # Publisher for target position
        self.target_pub = self.create_publisher(
            PointStamped, '/target_position', 10)

        # Publisher for visualization marker
        self.marker_pub = self.create_publisher(
            Marker, '/target_marker', 10)


        # Store terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)

        # Print instructions
        self.print_instructions()

        self.get_logger().info('MLP Target Sender initialized with keyboard control.')

    def print_instructions(self):
        """Print keyboard control instructions."""
        print("\n" + "="*60)
        print("    MLP Target Sender - Keyboard Control")
        print("="*60)
        print("\nControls:")
        print("  W      - Move forward  (+X)")
        print("  S      - Move backward (-X)")
        print("  A      - Move left     (+Y)")
        print("  D      - Move right    (-Y)")
        print("  ↑      - Move up       (+Z)")
        print("  ↓      - Move down     (-Z)")
        print("  ←/→    - Move left/right (Y)")
        print("  R      - Reset to initial position")
        print("  +/-    - Increase/decrease step size")
        print("  P      - Print position comparison")
        print("  Q      - Quit")
        print("\n" + "="*60)
        print(f"Step size: {self.step_size:.3f} m")
        print(f"Current target: X={self.target[0]:.3f}, Y={self.target[1]:.3f}, Z={self.target[2]:.3f}")
        print("="*60 + "\n")

    def get_key(self):
        """Get a single keypress without blocking."""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if key == '\x1b':
                key += sys.stdin.read(2)
            return key
        return None

    def restore_terminal(self):
        """Restore terminal settings."""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def process_key(self, key):
        """Process a keypress and update target position."""
        if key is None:
            return True

        updated = False

        # WASD controls for X/Y
        if key.lower() == 'w':
            self.target[0] += self.step_size
            updated = True
        elif key.lower() == 's':
            self.target[0] -= self.step_size
            updated = True
        elif key.lower() == 'a':
            self.target[1] += self.step_size
            updated = True
        elif key.lower() == 'd':
            self.target[1] -= self.step_size
            updated = True

        # Arrow keys for Z (Up/Down) and Y (Left/Right)
        elif key == '\x1b[A':  # Up arrow
            self.target[2] += self.step_size
            updated = True
        elif key == '\x1b[B':  # Down arrow
            self.target[2] -= self.step_size
            updated = True
        elif key == '\x1b[C':  # Right arrow
            self.target[1] -= self.step_size
            updated = True
        elif key == '\x1b[D':  # Left arrow
            self.target[1] += self.step_size
            updated = True
        # Reset position
        elif key.lower() == 'r':
            self.target = [self.initial_x, self.initial_y, self.initial_z]
            updated = True
            print("\nReset to initial position")
        # Adjust step size
        elif key == '+' or key == '=':
            self.step_size = min(0.1, self.step_size + 0.005)
            print(f"\nStep size: {self.step_size:.3f} m")
        elif key == '-':
            self.step__update_ee_positionsize = max(0.005, self.step_size - 0.005)
            print(f"\nStep size: {self.step_size:.3f} m")

        # Print position comparison
        elif key.lower() == 'p':
            self._update_ee_position()
            self._print_position_comparison()

        # Quit
        elif key.lower() == 'q' or key == '\x03':  # q or Ctrl+C
            print("\nQuitting...")
            return False

        if updated:
            self._update_ee_position()
            self._print_position_comparison()
        self.publish_target()
        return True

    def _update_ee_position(self):
        """Update end-effector position from TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tip_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            self.current_ee_position = [
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ]
            return True
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")
            return False

    def _print_position_comparison(self):
        """Print desired position, current position, and their difference."""
        print("\n" + "-"*60)
        print(f"\nDesired (Target):  X={self.target[0]:+.4f}, Y={self.target[1]:+.4f}, Z={self.target[2]:+.4f}")

        if self.current_ee_position is not None:
            ee = self.current_ee_position
            diff = [self.target[i] - ee[i] for i in range(3)]
            distance = np.sqrt(sum(d**2 for d in diff))

            print(f"\nCurrent (Actual):  X={ee[0]:+.4f}, Y={ee[1]:+.4f}, Z={ee[2]:+.4f}")
            print(f"\nDifference (Δ):    X={diff[0]:+.4f}, Y={diff[1]:+.4f}, Z={diff[2]:+.4f}")
            print(f"\nEuclidean Distance: {distance:.4f} m")
        else:
            print("\nCurrent (Actual):  [Waiting for TF data...]")
        print("-"*60)
        print("\n")

    def publish_target(self):
        """Publish the current target position and visualization marker."""
        # Publish target position
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.point.x = self.target[0]
        msg.point.y = self.target[1]
        msg.point.z = self.target[2]
        self.target_pub.publish(msg)

        # Publish visualization marker (sphere)
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = 'target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.target[0]
        marker.pose.position.y = self.target[1]
        marker.pose.position.z = self.target[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)

        # Publish arrow pointing to target
        arrow = Marker()
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.header.frame_id = self.frame_id
        arrow.ns = 'target_arrow'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = self.target[0]
        arrow.pose.position.y = self.target[1]
        arrow.pose.position.z = self.target[2] + 0.1
        arrow.pose.orientation.x = 0.707
        arrow.pose.orientation.y = 0.0
        arrow.pose.orientation.z = 0.0
        arrow.pose.orientation.w = 0.707
        arrow.scale.x = 0.08
        arrow.scale.y = 0.015
        arrow.scale.z = 0.015
        arrow.color.r = 1.0
        arrow.color.g = 0.2
        arrow.color.b = 0.2
        arrow.color.a = 1.0
        self.marker_pub.publish(arrow)


def main(args=None):
    rclpy.init(args=args)
    node = MLPTargetSender()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            key = node.get_key()
            if not node.process_key(key):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

