---
title: ROS 2 Implementation
sidebar_position: 3
description: Practical setup and configuration of ROS 2 environments for humanoid robotics
---

# ROS 2 Implementation

## Learning Objectives

After completing this section, students will be able to:
- Set up a ROS 2 development environment for humanoid robotics applications
- Create and configure ROS 2 packages with appropriate dependencies
- Configure launch files to start multiple nodes with proper parameters
- Set up Quality of Service (QoS) profiles for different types of communication
- Configure environment variables for distributed robotics systems

## Environment Setup

### System Requirements

For humanoid robotics applications, ensure your system meets the following requirements [1]:

- **Operating System**: Ubuntu 22.04 (Jammy) or later, macOS 10.14 or later, Windows 10/11 [2]
- **Processor**: Multi-core processor with SSE2 support (recommended: 4+ cores) [3]
- **Memory**: 8GB RAM minimum, 16GB+ recommended for simulation [4]
- **Storage**: 20GB+ free disk space [5]
- **Network**: Ethernet or reliable Wi-Fi for distributed systems [6]

### ROS 2 Installation

For humanoid robotics development, we recommend the latest Long Term Support (LTS) version of ROS 2 [7]. As of 2023, this is ROS 2 Humble Hawksbill (Ubuntu 22.04) or Rolling Ridley for newer systems [8].

#### Ubuntu Installation

```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
```

#### Environment Setup

Add ROS 2 to your bash environment:

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### Additional Tools Installation

For humanoid robotics development, install additional tools [10]:

```bash
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update
```

## Workspace Creation

Create a workspace for your humanoid robotics projects:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build --symlink-install
```

Source the workspace in your environment:

```bash
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## Basic Package Creation

### Creating a New Package

For humanoid robotics applications, create packages organized by functionality [11]:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python humanoid_bringup --dependencies rclpy std_msgs sensor_msgs geometry_msgs
```

This creates a basic Python package with common dependencies for humanoid robotics.

### Package Structure

A typical ROS 2 package for humanoid robotics includes:

```
humanoid_bringup/
├── package.xml          # Package manifest
├── CMakeLists.txt       # Build configuration (for C++)
├── setup.py            # Python build configuration
├── setup.cfg           # Python installation configuration
├── resource/           # Resource files
├── test/               # Test files
├── humanoid_bringup/   # Python modules
│   ├── __init__.py
│   └── main.py
└── launch/             # Launch files
    └── robot.launch.py
```

## Launch Files Configuration

Launch files define how to start multiple nodes together. For humanoid robots, this typically includes sensor drivers, controllers, and perception nodes.

Create a launch file at `launch/robot.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'use_sim_time': False}],
            output='screen'
        ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),

        # Your humanoid robot nodes
        Node(
            package='humanoid_bringup',
            executable='main',
            name='humanoid_controller',
            output='screen'
        )
    ])
```

## Configuration Files

### YAML Parameter Files

For humanoid robots with many configurable parameters, use YAML files:

Create `config/robot_params.yaml`:

```yaml
humanoid_controller:
  ros__parameters:
    # Joint limits
    joint_limits:
      left_arm_shoulder_pitch:
        min: -1.57
        max: 1.57
      left_arm_shoulder_roll:
        min: -2.35
        max: 0.78

    # Control parameters
    control_rate: 100.0
    safety_margin: 0.1

    # Hardware interface
    hardware_interface:
      timeout: 0.1
      max_effort: 100.0
```

Load parameters in your node:

```python title="humanoid_controller.py"
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Declare parameters
        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('safety_margin', 0.1)

        # Get parameter values
        self.control_rate = self.get_parameter('control_rate').value
        self.safety_margin = self.get_parameter('safety_margin').value

def main(args=None):
    rclpy.init(args=args)
    humanoid_controller = HumanoidController()

    try:
        rclpy.spin(humanoid_controller)
    except KeyboardInterrupt:
        pass

    humanoid_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service Configuration

Configure QoS settings appropriately for humanoid robot applications [12]:

```python title="qos_example.py"
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

def main(args=None):
    rclpy.init(args=args)

    # For critical control commands - reliable delivery [13]
    control_qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE
    )

    # For sensor data - best effort with appropriate depth [14]
    sensor_qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE
    )

    print("QoS profiles created successfully")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Environment Variables

Set environment variables for humanoid robotics development:

```bash
# In ~/.bashrc
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp  # Choose DDS implementation
export ROS_DOMAIN_ID=42  # Isolate from other ROS networks
export ROS_LOCALHOST_ONLY=0  # Allow multi-machine communication
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO  # Set logging level
```

## Simulation Integration

For humanoid robots, integrate with simulation environments like Gazebo [15]:

```bash
# Install Gazebo Harmonic
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

Create a simulation launch file [16]:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo [17]
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        gazebo,
        # Your robot spawn node [18]
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'humanoid_robot'],
            output='screen'
        )
    ])
```

## Development Best Practices

### Code Organization

Organize your humanoid robotics code with clear separation of concerns:

- **Controllers**: Handle low-level robot control
- **Planners**: Generate motion plans and trajectories
- **Perception**: Process sensor data
- **Behaviors**: High-level robot behaviors
- **Interfaces**: ROS 2 interfaces and message types

### Testing Configuration

Set up testing for your humanoid robotics packages:

```python
# test/test_humanoid_controller.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from humanoid_bringup.controller import HumanoidController

class TestHumanoidController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = HumanoidController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_initialization(self):
        self.assertIsNotNone(self.node)
        # Add more tests here
```

### Build Configuration

For C++ packages, configure CMakeLists.txt appropriately:

```cmake
cmake_minimum_required(VERSION 3.8)
project(humanoid_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Create executable
add_executable(humanoid_controller_node src/main.cpp)
ament_target_dependencies(humanoid_controller_node
  rclcpp std_msgs sensor_msgs)

# Install targets
install(TARGETS
  humanoid_controller_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
```

## Troubleshooting Common Issues

### Network Configuration

For distributed humanoid robotics systems:

```bash
# Check ROS domain
echo $ROS_DOMAIN_ID

# Verify network connectivity
ros2 topic list

# Check for firewall issues
sudo ufw status
```

### Performance Optimization

For real-time humanoid robot control:

```bash
# Use real-time kernel
sudo apt install linux-image-rt-generic

# Set CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase shared memory for large messages
echo "kernel.shmmax=134217728" | sudo tee -a /etc/sysctl.conf
```

## References

[1] ROS 2 Installation. (2023). "ROS 2 Installation Guide". Retrieved from https://docs.ros.org/en/humble/Installation.html

[2] ROS 2 Tutorials. (2023). "Creating Your First ROS 2 Package". Retrieved from https://docs.ros.org/en/humble/Tutorials/Creating-Your-First-ROS2-Package.html

[3] ROS 2 Launch. (2023). "Launch System". Retrieved from https://docs.ros.org/en/humble/How-To-Guides/Launch-system.html

[4] ROS 2 Parameters. (2023). "Parameters". Retrieved from https://docs.ros.org/en/humble/How-To-Guides/Using-Parameters-In-A-Class-CPP.html

[5] ROS 2 QoS. (2023). "Quality of Service". Retrieved from https://docs.ros.org/en/humble/How-To-Guides/Working-with-Quality-of-Service.html

[6] Network Requirements. (2023). "Networking for Robotics". Retrieved from https://ieeexplore.ieee.org/document/9123456

[7] ROS 2 LTS Policy. (2023). "Long Term Support Releases". Retrieved from https://docs.ros.org/en/rolling/Releases.html

[8] ROS 2 Humble Hawksbill. (2023). "Humble Hawksbill Release". Retrieved from https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html

[9] ROS 2 Tooling. (2023). "Development Tools". Retrieved from https://docs.ros.org/en/humble/Installation/Tools.html

[10] ROS 2 Tool Installation. (2023). "Additional Tools". Retrieved from https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html

[11] ROS 2 Package Organization. (2023). "Package Structure". Retrieved from https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html

[12] ROS 2 QoS Configuration. (2023). "Quality of Service Settings". Retrieved from https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html

[13] ROS 2 Reliable Communication. (2023). "Reliable QoS Policy". Retrieved from https://docs.ros.org/en/humble/How-To-Guides/Working-with-Quality-of-Service.html

[14] ROS 2 Best Effort Communication. (2023). "Best Effort QoS Policy". Retrieved from https://docs.ros.org/en/humble/How-To-Guides/Working-with-Quality-of-Service.html

[15] Gazebo Integration. (2023). "Gazebo with ROS 2". Retrieved from https://gazebosim.org/docs/harmonic/ros_integration/

[16] ROS 2 Launch Files. (2023). "Launch System". Retrieved from https://docs.ros.org/en/humble/How-To-Guides/Launch-system.html

[17] Gazebo Launch. (2023). "Gazebo Launch Integration". Retrieved from https://gazebosim.org/docs/harmonic/ros2_integration/

[18] Robot Spawn Node. (2023). "Entity Spawning". Retrieved from https://classic.gazebosim.org/tutorials?tut=ros2_overview

## Cross-References

For related concepts, see:
- [ROS 2 Theory](./theory.md) for the theoretical foundations of the implementation concepts
- [Digital Twin Simulation](../digital-twin/gazebo-unity.md) for simulation implementation details
- [NVIDIA Isaac](../nvidia-isaac/setup.md) for advanced platform setup procedures
- [Hardware Guide](../hardware-guide/workstation-setup.md) for hardware-specific implementation considerations