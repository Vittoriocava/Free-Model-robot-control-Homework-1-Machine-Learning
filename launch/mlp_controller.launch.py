"""
Launch file for the MLP Controller Node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('mlp_controller')
    
    # Default model paths using package share directory
    default_model_path = os.path.join(pkg_share, 'models', 'mlp_model_3joints_20251202_182913.pth')
    default_scaler_path = os.path.join(pkg_share, 'models', 'scalers_3joints.pkl')
    
    # Declare launch arguments
    njoints_arg = DeclareLaunchArgument(
        'njoints',
        default_value='3',
        description='Number of arm joints (3, 4, or 6)'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to the trained MLP model (.pth file)'
    )
    
    scaler_path_arg = DeclareLaunchArgument(
        'scaler_path',
        default_value=default_scaler_path,
        description='Path to the scaler pickle file'
    )
    
    position_threshold_arg = DeclareLaunchArgument(
        'position_threshold',
        default_value='0.01',
        description='Position threshold in meters to consider target reached'
    )
    
    velocity_threshold_arg = DeclareLaunchArgument(
        'velocity_threshold',
        default_value='0.01',
        description='Velocity threshold in rad/s to consider arm stationary'
    )
    
    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='10.0',
        description='Control loop rate in Hz'
    )
    
    # Create the node
    mlp_controller_node = Node(
        package='mlp_controller',
        executable='mlp_controller_node',
        name='mlp_controller_node',
        output='screen',
        parameters=[{
            'njoints': LaunchConfiguration('njoints'),
            'model_path': LaunchConfiguration('model_path'),
            'scaler_path': LaunchConfiguration('scaler_path'),
            'position_threshold': LaunchConfiguration('position_threshold'),
            'velocity_threshold': LaunchConfiguration('velocity_threshold'),
            'control_rate': LaunchConfiguration('control_rate'),
        }]
    )
    
    return LaunchDescription([
        njoints_arg,
        model_path_arg,
        scaler_path_arg,
        position_threshold_arg,
        velocity_threshold_arg,
        control_rate_arg,
        mlp_controller_node,
    ])
