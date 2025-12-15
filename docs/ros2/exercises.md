---
title: ROS 2 Exercises
sidebar_position: 6
description: Practical exercises for students to practice ROS 2 concepts in humanoid robotics
---

# ROS 2 Exercises

## Learning Objectives

After completing these exercises, students will be able to:
- Implement and test basic ROS 2 communication patterns
- Configure Quality of Service settings appropriately
- Create and manage launch files for multi-node systems
- Implement parameter management for dynamic reconfiguration
- Design action servers for complex humanoid robot behaviors

## Exercise 1: Basic Publisher-Subscriber Pattern

### Objective
Implement a simple publisher-subscriber system to understand the fundamental communication pattern in ROS 2.

### Instructions
1. Create a publisher node that publishes messages to a topic `/robot_status` containing a string message indicating the robot's current state (e.g., "idle", "moving", "stopping").
2. Create a subscriber node that listens to `/robot_status` and logs the received messages.
3. Use the `std_msgs/String` message type.
4. Set the publishing rate to 1 Hz.
5. Test the system by running both nodes and observing the communication.

### Learning Outcomes
- Understanding the publish-subscribe pattern
- Creating ROS 2 nodes in Python
- Working with standard message types
- Using timers in ROS 2 nodes

### Solution Reference
```python title="robot_status_publisher.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.statuses = ['idle', 'moving', 'stopping', 'charging']
        self.index = 0

    def timer_callback(self):
        msg = String()
        msg.data = self.statuses[self.index % len(self.statuses)]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    robot_status_publisher = RobotStatusPublisher()

    try:
        rclpy.spin(robot_status_publisher)
    except KeyboardInterrupt:
        pass

    robot_status_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python title="robot_status_subscriber.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotStatusSubscriber(Node):
    def __init__(self):
        super().__init__('robot_status_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received robot status: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    robot_status_subscriber = RobotStatusSubscriber()

    try:
        rclpy.spin(robot_status_subscriber)
    except KeyboardInterrupt:
        pass

    robot_status_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 2: Service Implementation

### Objective
Create a service server and client to understand request-response communication in ROS 2.

### Instructions
1. Define a custom service `SetPosition.srv` with request fields `float64 x`, `float64 y`, `float64 z` and response field `bool success`.
2. Implement a service server that simulates moving a robot to the requested position.
3. Implement a client that sends position requests to the server.
4. Add validation to ensure the requested position is within safe operational limits.
5. Test the system with various position requests.

### Learning Outcomes
- Creating custom service definitions
- Implementing service servers and clients
- Handling request-response communication
- Input validation in ROS 2 services

## Exercise 3: Quality of Service Configuration

### Objective
Experiment with different Quality of Service (QoS) profiles to understand their impact on communication.

### Instructions
1. Create a publisher that sends sensor data with different QoS profiles:
   - Reliable delivery with durability
   - Best effort delivery
   - Keep-all vs. keep-last history policies
2. Create corresponding subscribers that match the QoS profiles.
3. Simulate network conditions that might cause message loss.
4. Observe how different QoS profiles handle the simulated network issues.
5. Document the behavior differences between profiles.

### Learning Outcomes
- Understanding QoS policies in ROS 2
- Configuring different QoS profiles
- Matching QoS between publishers and subscribers
- Selecting appropriate QoS for different use cases

## Exercise 4: Launch File Configuration

### Objective
Create complex launch files to manage multiple ROS 2 nodes and their configurations.

### Instructions
1. Create a package with multiple nodes (publisher, subscriber, service server).
2. Create a launch file that starts all nodes with appropriate parameters.
3. Add launch arguments to control node behavior (e.g., publishing rate, logging level).
4. Use conditions to optionally start visualization tools like RViz.
5. Test the launch file with different argument combinations.

### Learning Outcomes
- Creating and organizing ROS 2 packages
- Writing launch files for complex systems
- Using launch arguments and conditions
- Managing multiple nodes in a single launch

## Exercise 5: Parameter Management

### Objective
Implement dynamic parameter configuration for a robot control system.

### Instructions
1. Create a node that declares parameters for robot control (e.g., maximum velocity, safety limits, control gains).
2. Implement parameter validation callbacks to ensure parameter values are within acceptable ranges.
3. Create a separate node that dynamically changes parameters at runtime.
4. Use ROS 2 command-line tools to view and modify parameters.
5. Observe how parameter changes affect the robot's behavior.

### Learning Outcomes
- Declaring and using parameters in ROS 2
- Implementing parameter validation
- Dynamic parameter reconfiguration
- Using ROS 2 parameter tools

## Exercise 6: Action Server Implementation

### Objective
Implement an action server for a complex humanoid robot behavior.

### Instructions
1. Define an action `WalkToGoal.action` with goal (target position), result (success/failure), and feedback (progress).
2. Implement an action server that simulates walking to a target location.
3. The server should provide feedback on progress during the "walk".
4. Implement an action client that sends goals to the server and handles feedback.
5. Test with various target positions and handle cancellation requests.

### Learning Outcomes
- Creating custom action definitions
- Implementing action servers and clients
- Handling long-running tasks with feedback
- Managing goal execution and cancellation

## Exercise 7: Multi-Node System Integration

### Objective
Design and implement a complete humanoid robot control system with multiple interconnected nodes.

### Instructions
1. Create nodes for:
   - Joint state publisher (simulated joint positions)
   - Robot state publisher (TF transforms)
   - Trajectory controller (receives trajectory goals)
   - Sensor data processor (processes IMU data)
2. Connect the nodes using appropriate topics, services, and parameters.
3. Create a launch file that starts all nodes with proper configuration.
4. Implement error handling and recovery mechanisms.
5. Test the integrated system with various scenarios.

### Learning Outcomes
- System-level ROS 2 design
- Node integration and communication
- Error handling in distributed systems
- Launch file organization for complex systems

## Exercise 8: Testing and Debugging

### Objective
Implement testing and debugging strategies for ROS 2 nodes.

### Instructions
1. Write unit tests for a simple ROS 2 node using Python's unittest framework.
2. Create integration tests that verify communication between multiple nodes.
3. Use ROS 2 tools like `ros2 topic echo`, `ros2 service call`, and `rqt_graph` for debugging.
4. Implement logging with appropriate severity levels.
5. Use `ros2 doctor` to diagnose common system issues.

### Learning Outcomes
- Writing unit and integration tests for ROS 2
- Using ROS 2 debugging tools
- Implementing proper logging
- System diagnosis and troubleshooting

## Exercise 9: Performance Optimization

### Objective
Optimize a ROS 2 system for performance and resource usage.

### Instructions
1. Create a system that processes high-frequency sensor data (100+ Hz).
2. Profile the system to identify performance bottlenecks.
3. Optimize message passing using appropriate QoS settings.
4. Implement message filtering to reduce unnecessary processing.
5. Use efficient data structures and algorithms for real-time processing.

### Learning Outcomes
- Performance profiling of ROS 2 systems
- Optimization techniques for real-time systems
- Efficient message passing strategies
- Resource management in ROS 2

## Exercise 10: Simulation Integration

### Objective
Integrate ROS 2 with a simulation environment for humanoid robot development.

### Instructions
1. Set up Gazebo simulation with a humanoid robot model.
2. Create ROS 2 nodes that interface with the simulation.
3. Implement sensor simulation (IMU, cameras, joint encoders).
4. Test robot behaviors in simulation before real-world deployment.
5. Compare simulation results with real hardware when available.

### Learning Outcomes
- ROS 2 simulation integration
- Robot model configuration
- Sensor simulation and processing
- Simulation-to-reality transfer

## Assessment Rubric

### Beginner Level (Covers Exercises 1-3)
- Successfully implements basic publisher-subscriber communication
- Creates simple services with appropriate request-response handling
- Configures basic QoS settings and understands their purpose

### Intermediate Level (Covers Exercises 4-7)
- Designs multi-node systems with proper interconnections
- Implements parameter management and dynamic reconfiguration
- Uses actions for long-running tasks with feedback
- Creates comprehensive launch files for system management

### Advanced Level (Covers Exercises 8-10)
- Implements comprehensive testing and debugging strategies
- Optimizes systems for performance and resource usage
- Integrates with simulation environments
- Demonstrates system-level understanding of ROS 2 architecture

## Cross-References

For related exercises and concepts, see:
- [Digital Twin Simulation Exercises](../digital-twin/gazebo-unity.md) for simulation-specific exercises
- [NVIDIA Isaac Exercises](../nvidia-isaac/examples.md) for advanced platform exercises
- [Vision-Language-Action Exercises](../vla-systems/implementation.md) for AI integration exercises
- [Capstone Humanoid Project](../capstone-humanoid/implementation.md) for comprehensive project exercises

## Hints for Implementation

1. **Start Simple**: Begin with basic implementations and gradually add complexity.
2. **Use Standard Tools**: Leverage ROS 2's built-in tools for debugging and testing.
3. **Follow Conventions**: Adhere to ROS 2 naming conventions and best practices.
4. **Validate Inputs**: Always validate inputs to prevent system failures.
5. **Document Code**: Include proper documentation and comments in your implementations.
6. **Test Incrementally**: Test each component individually before integration.
7. **Handle Errors**: Implement appropriate error handling and recovery mechanisms.

## Additional Resources

- ROS 2 Documentation: https://docs.ros.org/
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- ROS 2 Design Articles: https://design.ros2.org/
- ROS Answers: https://answers.ros.org/
- Robotics Stack Exchange: https://robotics.stackexchange.com/