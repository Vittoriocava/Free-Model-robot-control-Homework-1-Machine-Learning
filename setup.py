from setuptools import setup
import os
from glob import glob

package_name = 'mlp_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pth')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pkl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vittorio',
    maintainer_email='your_email@example.com',
    description='MLP-based robot arm controller using learned inverse kinematics',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mlp_controller_node = mlp_controller.mlp_controller_node:main',
            'mlp_target_sender = mlp_controller.mlp_target_sender:main',
        ],
    },
)
