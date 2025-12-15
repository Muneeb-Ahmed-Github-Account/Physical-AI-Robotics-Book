---
title: Simulation Testing and Validation
sidebar_position: 7
description: Documentation for testing and validating simulation examples in appropriate environments
---

# Simulation Testing and Validation

## Overview

This document provides instructions for testing and validating all code examples in the Digital Twin Simulation module of the Humanoid Robotics & Physical AI Course Book. Each code example should be tested in an appropriate simulation environment to ensure functionality and educational value.

## Prerequisites

Before testing the simulation examples, ensure your environment is properly set up:

1. **ROS 2 Installation**: Install ROS 2 Humble Hawksbill (or latest LTS version)
2. **Simulation Platforms**: Install both Gazebo Harmonic and Unity Robotics packages
3. **Development Environment**: Set up a workspace with proper sourcing
4. **Dependencies**: Install required packages (sensor_msgs, geometry_msgs, joint_state_publisher, etc.)

### Environment Setup

```bash
# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop ros-humble-gazebo-ros-pkgs ros-humble-joint-state-publisher

# Create and build workspace
mkdir -p ~/simulation_ws/src
cd ~/simulation_ws
colcon build --symlink-install
source install/setup.bash
```

## Test Categories

### 1. Basic Publisher-Subscriber Test

**Files**: `examples.md` - Basic publisher/subscriber examples

**Test Steps**:
1. Create a test package:
   ```bash
   cd ~/simulation_ws/src
   ros2 pkg create --build-type ament_python simulation_test_pkg --dependencies rclpy std_msgs sensor_msgs
   ```
2. Copy the publisher code to `simulation_test_pkg/simulation_publisher.py`
3. Launch the publisher in one terminal:
   ```bash
   cd ~/simulation_ws
   source install/setup.bash
   ros2 run simulation_test_pkg simulation_publisher
   ```
4. Launch the subscriber in another terminal:
   ```bash
   cd ~/simulation_ws
   source install/setup.bash
   ros2 run simulation_test_pkg simulation_subscriber
   ```
5. Verify that messages are published and received correctly
6. Check that joint positions are calculated as expected

**Expected Output**: Publisher should log "Publishing: [array of positions]", subscriber should log "Received joint states: [positions]"

### 2. Service Test

**Files**: `examples.md` - Service server and client examples

**Test Steps**:
1. Create service definition file `srv/JointTrajectory.srv` in your package
2. Implement the service server code from examples
3. Launch the service server:
   ```bash
   ros2 run simulation_test_pkg joint_trajectory_server
   ```
4. Test the service from command line:
   ```bash
   ros2 service call /execute_trajectory simulation_test_pkg/srv/JointTrajectory "{'trajectory': {'positions': [0.0, 0.5, -0.5], 'velocities': [0.0, 0.0, 0.0]}}"
   ```
5. Verify the service response

**Expected Output**: Service should return success response with trajectory execution status

### 3. Action Test

**Files**: `examples.md` - Action server and client examples

**Test Steps**:
1. Create action definition file `action/Walk.action` in your package
2. Implement the action server code from examples
3. Launch the action server:
   ```bash
   ros2 run simulation_test_pkg walk_action_server
   ```
4. Test the action from command line:
   ```bash
   ros2 action send_goal /walk_to_target simulation_test_pkg/action/Walk "{target_x: 1.0, target_y: 2.0, target_theta: 0.0}"
   ```
5. Verify action execution with feedback

**Expected Output**: Action should provide feedback during execution and return result upon completion

### 4. Sensor Simulation Test

**Files**: `examples.md` - Sensor simulation examples

**Test Steps**:
1. Create a robot URDF with sensor definitions
2. Launch Gazebo with the robot model:
   ```bash
   ros2 launch gazebo_ros empty_world.launch.py
   ros2 run gazebo_ros spawn_entity.py -entity test_robot -file:///path/to/robot.urdf
   ```
3. Verify that sensor topics are being published:
   ```bash
   ros2 topic echo /camera/image_raw
   ros2 topic echo /lidar_scan
   ```
4. Check that sensor data is being published at expected rates

**Expected Output**: Sensor topics should have steady stream of data with appropriate message formats

### 5. Physics Simulation Test

**Files**: `simulation-basics.md` - Physics simulation examples

**Test Steps**:
1. Create a simple physics simulation scenario
2. Implement the physics validation code from examples
3. Run the simulation and verify physics parameters
4. Test collision detection and response
5. Validate that physics parameters match expected behavior

**Expected Output**: Objects should behave according to physical laws with appropriate collision response

### 6. QoS Configuration Test

**Files**: `examples.md` - QoS configuration examples

**Test Steps**:
1. Implement publishers with different QoS profiles:
   ```bash
   # Reliable communication for critical commands
   ros2 run simulation_test_pkg critical_command_publisher --qos-reliability reliable
   ```
   ```bash
   # Best effort for sensor data
   ros2 run simulation_test_pkg sensor_publisher --qos-reliability best_effort
   ```
2. Verify that messages are delivered according to QoS settings
3. Test with different history depths and durability settings

**Expected Output**: Reliable messages should be guaranteed delivery, best effort may drop messages under stress

## Automated Testing

### Unit Tests

For nodes that include testable logic, create unit tests:

```python
# test/test_simulation_nodes.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from simulation_test_pkg.joint_controller import JointController

class TestSimulationNodes(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = JointController()
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

    def test_trajectory_generation(self):
        # Test trajectory generation
        start_pos = [0.0, 0.0]
        end_pos = [1.0, 1.0]
        trajectory = self.node.generate_linear_trajectory(start_pos, end_pos, 10)
        self.assertEqual(len(trajectory), 10)
        self.assertEqual(trajectory[0], start_pos)
        self.assertEqual(trajectory[-1], end_pos)

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
# Use ros2 command line tools to verify communication
ros2 topic list
ros2 node list
ros2 service list
ros2 action list

# Echo specific topics to verify data
ros2 topic echo /joint_states --field position
ros2 topic hz /camera/image_raw
```

## Performance Testing

### Baseline Performance Metrics

For each simulation example, verify:

1. **CPU Usage**: Should not exceed 80% on a modern quad-core processor
2. **Memory Usage**: Should remain stable over extended runs
3. **Real-time Factor**: Should maintain >= 0.8 for interactive applications
4. **Message Rates**: Should match expected frequencies (e.g., 50Hz for joint states)

### Stress Testing

Test examples under stress conditions:

```bash
# Launch multiple instances to test resource usage
ros2 launch simulation_test_pkg multi_robot.launch.py num_robots:=5

# Monitor resource usage
htop
ros2 topic hz /joint_states
```

## Validation Checklist

### Before Testing
- [ ] Environment properly set up with ROS 2 and simulation tools
- [ ] All dependencies installed
- [ ] Workspace built and sourced
- [ ] Example code copied to test package

### During Testing
- [ ] Each example runs without errors
- [ ] Expected outputs are produced
- [ ] Performance metrics are within acceptable ranges
- [ ] No memory leaks or resource accumulation
- [ ] Proper error handling for edge cases

### After Testing
- [ ] Test results documented
- [ ] Performance metrics recorded
- [ ] Issues reported and tracked
- [ ] Examples validated for educational value

## Troubleshooting Common Issues

### Environment Issues
- **Problem**: "Command 'ros2' not found"
- **Solution**: Verify ROS 2 installation and environment setup: `source /opt/ros/humble/setup.bash`

- **Problem**: "Module not found" errors
- **Solution**: Ensure workspace is built and sourced: `cd ~/simulation_ws && colcon build && source install/setup.bash`

### Simulation Issues
- **Problem**: Gazebo fails to start
- **Solution**: Check graphics drivers and X11 forwarding if using SSH

- **Problem**: Physics simulation unstable
- **Solution**: Adjust time step and solver parameters in Gazebo configuration

### Communication Issues
- **Problem**: Nodes unable to communicate
- **Solution**: Check ROS_DOMAIN_ID, network configuration, and topic names

- **Problem**: High message latency
- **Solution**: Verify QoS settings and network configuration

## Quality Assurance

### Educational Value Assessment
- [ ] Examples clearly illustrate the concepts being taught
- [ ] Code is well-commented and educational
- [ ] Examples build progressively in complexity
- [ ] Error handling demonstrates good practices
- [ ] Examples are relevant to humanoid robotics

### Technical Accuracy Verification
- [ ] All technical claims are accurate and verifiable
- [ ] Code examples follow ROS 2 best practices
- [ ] Performance characteristics are realistic
- [ ] Safety considerations are addressed
- [ ] Examples are maintainable and extensible

## Reporting Results

After testing, document results in the following format:

```
Test Report: Digital Twin Simulation Examples
Date: YYYY-MM-DD
Environment: ROS 2 Humble on Ubuntu 22.04
Tester: [Name]

Examples Tested:
- Basic Publisher/Subscriber: PASSED - Messages exchanged correctly at 50Hz
- Service Implementation: PASSED - Requests responded to within 100ms
- Action Server: PASSED - Goals executed with proper feedback
- Sensor Simulation: PASSED - Data published at expected rates
- Physics Simulation: PASSED - Objects behave according to physical laws
- QoS Configuration: PASSED - Different profiles behave as expected

Performance Results:
- CPU Usage: Average 25%, Peak 45%
- Memory Usage: Stable at ~200MB
- Real-time Factor: 0.95+ for all examples
- Message Rates: Match expected frequencies

Issues Found:
1. Physics example occasionally shows instability with complex meshes - recommend simplifying collision geometries
2. Sensor noise parameters could be better documented with real-world values

Recommendations:
- Add more detailed comments explaining physics parameters
- Include more comprehensive error handling examples
- Add additional examples for multi-robot scenarios
```

## Continuous Integration

For maintaining simulation examples over time:

1. **Automated Builds**: Set up CI pipeline to build examples with each commit
2. **Unit Tests**: Include unit tests in CI pipeline
3. **Documentation Checks**: Verify code examples match documentation
4. **Performance Monitoring**: Track performance regressions over time

## Conclusion

Testing ensures that all code examples in the Digital Twin Simulation module function correctly and provide educational value to students. Regular testing with updated ROS 2 distributions helps maintain the quality of the course material.