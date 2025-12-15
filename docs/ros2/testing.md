---
title: ROS 2 Code Testing
sidebar_position: 8
description: Documentation for testing ROS 2 code examples in appropriate environments
---

# ROS 2 Code Testing

## Learning Objectives

After reviewing this testing documentation, students will be able to:
- Set up appropriate testing environments for ROS 2 code
- Execute and validate different types of ROS 2 communication patterns
- Use ROS 2 command-line tools for testing and debugging
- Create and run unit tests for ROS 2 nodes
- Document and report testing results effectively

## Overview

This document provides instructions for testing all code examples in the ROS 2 module of the Humanoid Robotics & Physical AI Course Book. Each code example should be tested in an appropriate ROS 2 environment to ensure functionality and educational value.

## Prerequisites

Before testing the code examples, ensure your environment is properly set up:

1. **ROS 2 Installation**: Install ROS 2 Humble Hawksbill (or latest LTS version)
2. **Development Environment**: Set up a workspace with proper sourcing
3. **Dependencies**: Install required packages (sensor_msgs, geometry_msgs, etc.)

## Testing Environment Setup

### Docker-based Testing Environment

For consistent testing across different systems, consider using a Docker container:

```bash
# Pull the official ROS 2 Humble image
docker pull osrf/ros:humble-desktop

# Run with appropriate volume mounting
docker run -it --name ros2-test-env \
  -v $(pwd):/workspace \
  osrf/ros:humble-desktop
```

### Native Environment Setup

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Create and source workspace
mkdir -p ~/test_ws/src
cd ~/test_ws
colcon build --symlink-install
source install/setup.bash
```

## Test Procedures

### 1. Publisher-Subscriber Test

**Files**: `humanoid_publisher.py`, `humanoid_subscriber.py`

**Test Steps**:
1. Launch the publisher in one terminal:
   ```bash
   ros2 run your_package humanoid_publisher
   ```
2. Launch the subscriber in another terminal:
   ```bash
   ros2 run your_package humanoid_subscriber
   ```
3. Verify that messages are published and received correctly
4. Check that joint positions are calculated as expected

**Expected Output**: Publisher should log "Publishing: [array of positions]", subscriber should log "Received robot status: [status]"

### 2. Service Test

**Files**: `walk_service_server.py`, `walk_service_client.py`

**Test Steps**:
1. Launch the service server:
   ```bash
   ros2 run your_package walk_service_server
   ```
2. Test the service from command line:
   ```bash
   ros2 service call /walk_to_target your_package/srv/WalkService "{target_x: 1.0, target_y: 2.0, target_theta: 0.0}"
   ```
3. Verify the service response

**Expected Output**: Service should return success response with actual position values

### 3. Action Test

**Files**: `walk_action_server.py`, `walk_action_client.py`

**Test Steps**:
1. Launch the action server:
   ```bash
   ros2 run your_package walk_action_server
   ```
2. Test the action from command line:
   ```bash
   ros2 action send_goal /walk_action your_package/action/Walk "{target_x: 1.0, target_y: 2.0, target_theta: 0.0}"
   ```
3. Verify action execution with feedback

**Expected Output**: Action should provide feedback during execution and return result upon completion

### 4. Parameter Test

**Files**: `parameter_node.py`

**Test Steps**:
1. Launch the parameter node:
   ```bash
   ros2 run your_package parameter_node
   ```
2. Test parameter changes from command line:
   ```bash
   ros2 param set /parameter_node control_rate 200
   ros2 param list
   ros2 param get /parameter_node control_rate
   ```
3. Verify parameter validation and changes

**Expected Output**: Parameters should be settable within valid ranges and return appropriate error messages for invalid values

### 5. Launch File Test

**Files**: Various `.launch.py` files

**Test Steps**:
1. Test launch file:
   ```bash
   ros2 launch your_package robot.launch.py
   ```
2. Verify all nodes start correctly
3. Check for any error messages

**Expected Output**: All configured nodes should start without errors

### 6. QoS Configuration Test

**Files**: `qos_example.py`

**Test Steps**:
1. Run the QoS example:
   ```bash
   ros2 run your_package qos_example
   ```
2. Verify different QoS profiles work as expected
3. Test with publisher/subscriber using different QoS settings

**Expected Output**: QoS profiles should be created successfully

### 7. Simulation Integration Test

**Files**: Simulation launch files

**Test Steps**:
1. Install Gazebo Harmonic:
   ```bash
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
   ```
2. Test simulation launch:
   ```bash
   ros2 launch your_package simulation.launch.py
   ```
3. Verify robot spawns correctly in Gazebo

**Expected Output**: Robot model should appear in Gazebo simulation

## Automated Testing

### Unit Tests

For nodes that include testable logic, create unit tests:

```python
# test/test_humanoid_controller.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from your_package.humanoid_controller import HumanoidController

class TestHumanoidController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = HumanoidController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_joint_validation(self):
        # Test joint validation logic
        valid_positions = [0.0, 0.5, -0.5]
        result = self.node.validate_joint_positions(valid_positions)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
```

Run tests with:
```bash
# From your package directory
python3 -m pytest test/
```

### Integration Tests

For testing communication between nodes:

```bash
# Use ros2 topic tools to verify communication
ros2 topic echo /joint_states --field position
ros2 topic list
ros2 node list
```

## Quality Assurance Checklist

Before marking examples as tested, verify:

- [ ] All Python code examples run without syntax errors
- [ ] ROS 2 nodes initialize and run properly
- [ ] Communication patterns (topics, services, actions) work as expected
- [ ] Parameters can be set and retrieved correctly
- [ ] Launch files start all intended nodes
- [ ] Error handling works appropriately
- [ ] Code follows ROS 2 best practices
- [ ] Examples are educational and clear

## Troubleshooting Common Issues

### Import Errors
- Ensure all dependencies are installed
- Verify package.xml includes all required dependencies
- Check that workspace is properly sourced

### Communication Issues
- Verify ROS_DOMAIN_ID is consistent
- Check network configuration for multi-machine setups
- Ensure QoS settings match between publishers and subscribers

### Build Issues
- Run `colcon build` to rebuild packages
- Check for missing dependencies in CMakeLists.txt/package.xml
- Verify correct Python version requirements

## Performance Testing

For real-time applications:
- Monitor CPU usage with `top` or `htop`
- Check message rate with `ros2 topic hz`
- Verify timing constraints are met

## Documentation Testing

Verify that:
- All code examples are properly formatted
- Comments explain the purpose of each section
- Examples build on previous concepts
- Error handling is demonstrated where appropriate

## Testing Report Template

After testing, document results in a report:

```
Test Report: ROS 2 Code Examples
Date: YYYY-MM-DD
Environment: ROS 2 Humble on Ubuntu 22.04
Tester: [Name]

Examples Tested:
- Publisher/Subscriber: PASSED/FAILED - [Notes]
- Services: PASSED/FAILED - [Notes]
- Actions: PASSED/FAILED - [Notes]
- Parameters: PASSED/FAILED - [Notes]
- Launch Files: PASSED/FAILED - [Notes]

Issues Found:
1. [Issue description and resolution]
2. [Issue description and resolution]

Recommendations:
- [Any suggestions for improvements]
```

## Cross-References

For related testing procedures, see:
- [Digital Twin Simulation Testing](../digital-twin/testing.md) for simulation-specific testing
- [NVIDIA Isaac Testing](../nvidia-isaac/best-practices.md) for platform-specific testing
- [Vision-Language-Action Testing](../vla-systems/implementation.md) for AI system testing
- [Capstone Project Testing](../capstone-humanoid/testing.md) for comprehensive system testing

## Conclusion

Testing ensures that all code examples in the ROS 2 module function correctly and provide educational value to students. Regular testing with updated ROS 2 distributions helps maintain the quality of the course material.