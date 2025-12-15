---
title: Isaac Code Examples and Applications
sidebar_position: 4
description: Practical code examples and applications using NVIDIA Isaac for humanoid robotics
---

# Isaac Code Examples and Applications

## Learning Objectives

After completing this section, students will be able to:
- Implement GPU-accelerated perception pipelines using Isaac ROS [1]
- Create simulation environments for humanoid robot testing [2]
- Integrate Isaac Sim with real robot hardware [3]
- Deploy Isaac-based applications to humanoid robots [4]
- Optimize Isaac applications for real-time performance [5]
- Apply Isaac tools for sim-to-real transfer [6]
- Implement Isaac-based AI integration for humanoid systems [7]
- Configure Isaac for multi-robot coordination [8]
- Validate Isaac applications with comprehensive testing [9]
- Troubleshoot common Isaac application issues [10]

## Isaac Sim Examples

### Basic Humanoid Robot Simulation

#### Loading and Configuring a Humanoid Model

```python
# Example: Loading a humanoid robot model in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np

class HumanoidSimulationExample:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)

        # Load humanoid robot model (using a generic model for example)
        self.robot_path = "/World/HumanoidRobot"

        # Add robot to stage (in a real implementation, you would use your specific robot model)
        add_reference_to_stage(
            usd_path="path/to/your/humanoid/model.usd",  # Replace with actual path
            prim_path=self.robot_path
        )

        # Create articulation for the robot
        self.robot = self.world.scene.add(
            Articulation(
                prim_path=self.robot_path,
                name="humanoid_robot"
            )
        )

        # Set up camera view
        set_camera_view(eye=np.array([2.0, 2.0, 2.0]),
                       target=np.array([0.0, 0.0, 1.0]))

    def setup_physics(self):
        """Configure physics properties for humanoid simulation"""
        # Set gravity
        self.world.scene.set_gravity([0.0, 0.0, -9.81])

        # Configure physics solver parameters for humanoid stability
        physics_ctx = self.world.physics_sim_view
        physics_ctx.set_solver_type(0)  # TGS solver for better stability
        physics_ctx.set_position_iteration_count(8)
        physics_ctx.set_velocity_iteration_count(2)

        # Enable continuous collision detection for fast-moving parts
        physics_ctx.enable_ccd(True)

    def run_simulation(self):
        """Run the simulation with humanoid control"""
        self.world.reset()

        # Initialize physics
        self.setup_physics()

        # Simulation loop
        for i in range(1000):  # Run for 1000 steps
            # Reset every 100 steps to demonstrate control
            if i % 100 == 0:
                self.world.reset()

                # Apply initial configuration
                joint_positions = [0.0] * self.robot.num_dof
                self.robot.set_joint_positions(np.array(joint_positions))

            # Step the world
            self.world.step(render=True)

            # Print robot state periodically
            if i % 50 == 0:
                joint_positions = self.robot.get_joint_positions()
                joint_velocities = self.robot.get_joint_velocities()
                print(f"Step {i}: Joint positions: {joint_positions[:3]}...")  # Print first 3 joints

    def cleanup(self):
        """Clean up the simulation"""
        self.world.clear()

def main():
    sim = HumanoidSimulationExample()
    try:
        sim.run_simulation()
    finally:
        sim.cleanup()

if __name__ == "__main__":
    main()
```

### Isaac ROS Perception Pipeline Example

#### GPU-Accelerated Image Processing

```python
# Example: Isaac ROS image processing pipeline
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import time

class IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/front_camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            PointStamped,
            '/object_detection/result',
            10
        )

        # Isaac-specific GPU-accelerated processing parameters
        self.declare_parameter('use_gpu_acceleration', True)
        self.use_gpu = self.get_parameter('use_gpu_acceleration').value

        # Initialize Isaac perception components
        self.initialize_isaac_perception()

        self.get_logger().info('Isaac Perception Pipeline initialized')

    def initialize_isaac_perception(self):
        """Initialize Isaac-specific perception components"""
        if self.use_gpu:
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                from pycuda.compiler import SourceModule

                # Initialize CUDA context for Isaac acceleration
                self.cuda_ctx = cuda.Device(0).make_context()

                # Compile Isaac-specific CUDA kernels for image processing
                self.compile_isaac_kernels()

                self.get_logger().info('Isaac GPU acceleration enabled')
            except ImportError:
                self.get_logger().warn('CUDA not available, using CPU processing')
                self.use_gpu = False
        else:
            self.get_logger().info('Using CPU-based processing')

    def compile_isaac_kernels(self):
        """Compile Isaac-specific CUDA kernels"""
        # Example CUDA kernel for Isaac GPU-accelerated image processing
        cuda_code = """
        __global__ void detect_features_kernel(
            float* input_image,
            float* output_features,
            int width,
            int height
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                int idx = y * width + x;

                // Simple edge detection for example
                if (x > 0 && x < width-1 && y > 0 && y < height-1) {
                    float center = input_image[idx];
                    float left = input_image[y * width + (x-1)];
                    float right = input_image[y * width + (x+1)];
                    float top = input_image[(y-1) * width + x];
                    float bottom = input_image[(y+1) * width + x];

                    float gradient = fabsf(center - left) + fabsf(center - right) +
                                   fabsf(center - top) + fabsf(center - bottom);

                    output_features[idx] = gradient > 0.1f ? 1.0f : 0.0f;
                } else {
                    output_features[idx] = 0.0f;
                }
            }
        }
        """

        self.feature_module = SourceModule(cuda_code)
        self.feature_kernel = self.feature_module.get_function("detect_features_kernel")

    def image_callback(self, msg: Image):
        """Process incoming image with Isaac acceleration"""
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Normalize image to float32 for processing
            if cv_image.dtype == np.uint8:
                cv_image = cv_image.astype(np.float32) / 255.0

            # Apply Isaac-accelerated processing
            if self.use_gpu and hasattr(self, 'feature_kernel'):
                processed_features = self.isaac_feature_detection_gpu(cv_image)
            else:
                processed_features = self.cpu_feature_detection(cv_image)

            # Process results and publish
            result_point = self.extract_keypoint(processed_features, cv_image.shape)

            if result_point is not None:
                # Create and publish result
                detection_msg = PointStamped()
                detection_msg.header = Header()
                detection_msg.header.stamp = self.get_clock().now().to_msg()
                detection_msg.header.frame_id = msg.header.frame_id
                detection_msg.point.x = result_point[0]
                detection_msg.point.y = result_point[1]
                detection_msg.point.z = result_point[2]

                self.detection_pub.publish(detection_msg)

                self.get_logger().info(f'Detected feature at: ({result_point[0]:.2f}, {result_point[1]:.2f}, {result_point[2]:.2f})')

            processing_time = (time.time() - start_time) * 1000
            self.get_logger().info(f'Processing time: {processing_time:.2f}ms')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def isaac_feature_detection_gpu(self, image):
        """GPU-accelerated feature detection using Isaac CUDA"""
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        # Flatten image for GPU processing
        flat_image = image.flatten().astype(np.float32)

        # Allocate GPU memory
        input_gpu = cuda.mem_alloc(flat_image.nbytes)
        output_gpu = cuda.mem_alloc(flat_image.nbytes)

        # Copy image to GPU
        cuda.memcpy_htod(input_gpu, flat_image)

        # Configure kernel execution
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1], 1)

        # Execute kernel
        self.feature_kernel(
            input_gpu, output_gpu, np.int32(width), np.int32(height),
            block=block_size, grid=grid_size
        )

        # Copy result back to CPU
        result_flat = np.empty_like(flat_image)
        cuda.memcpy_dtoh(result_flat, output_gpu)

        # Clean up GPU memory
        del input_gpu
        del output_gpu

        # Reshape result
        return result_flat.reshape((height, width))

    def cpu_feature_detection(self, image):
        """CPU-based feature detection as fallback"""
        # Simple CPU implementation of feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        return edges.astype(np.float32) / 255.0

    def extract_keypoint(self, features, image_shape):
        """Extract keypoint from detected features"""
        height, width = image_shape[:2]

        # Find strongest feature response
        max_idx = np.argmax(features)
        max_y, max_x = np.unravel_index(max_idx, features.shape)

        # Convert to normalized coordinates
        norm_x = (max_x / width) * 2 - 1  # Normalize to [-1, 1]
        norm_y = (max_y / height) * 2 - 1  # Normalize to [-1, 1]
        norm_z = 0.0  # Assume feature is on image plane

        return (norm_x, norm_y, norm_z)

def main(args=None):
    rclpy.init(args=args)

    perception_pipeline = IsaacPerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac Visual SLAM Example

```python
# Example: Isaac Visual SLAM application
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np

class IsaacVisualSLAM(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # Isaac-specific SLAM parameters
        self.declare_parameter('enable_gpu_acceleration', True)
        self.declare_parameter('max_features', 2000)
        self.declare_parameter('min_match_distance', 30.0)
        self.declare_parameter('keyframe_threshold', 0.1)

        # Initialize GPU acceleration if available
        self.use_gpu = self.get_parameter('enable_gpu_acceleration').value
        self.initialize_gpu_acceleration()

        # Create subscribers for stereo camera
        self.left_image_sub = self.create_subscription(
            Image, '/stereo_camera/left/image_rect_color',
            self.left_image_callback, 10
        )

        self.right_image_sub = self.create_subscription(
            Image, '/stereo_camera/right/image_rect_color',
            self.right_image_callback, 10
        )

        self.left_info_sub = self.create_subscription(
            CameraInfo, '/stereo_camera/left/camera_info',
            self.left_info_callback, 10
        )

        # Create publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose', 10)

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # SLAM state variables
        self.prev_features = None
        self.prev_pose = np.eye(4)
        self.current_pose = np.eye(4)

        self.get_logger().info('Isaac Visual SLAM initialized')

    def initialize_gpu_acceleration(self):
        """Initialize GPU acceleration for visual SLAM"""
        if self.use_gpu:
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                import skcuda.linalg as culinalg
                import skcuda.misc as cumisc

                self.cuda_available = True
                self.get_logger().info('Isaac GPU acceleration enabled for SLAM')
            except ImportError:
                self.cuda_available = False
                self.get_logger().warn('CUDA not available for SLAM, using CPU fallback')
        else:
            self.cuda_available = False

    def left_image_callback(self, msg: Image):
        """Process left camera image for visual SLAM"""
        try:
            # Convert ROS image to OpenCV
            cv_bridge = CvBridge()
            cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Extract features using Isaac-accelerated methods
            if self.cuda_available:
                keypoints, descriptors = self.extract_features_gpu(cv_image)
            else:
                keypoints, descriptors = self.extract_features_cpu(cv_image)

            # Process SLAM if we have previous features
            if self.prev_features is not None:
                transformation = self.estimate_motion(
                    self.prev_features['descriptors'],
                    descriptors,
                    self.prev_features['keypoints'],
                    keypoints
                )

                if transformation is not None:
                    # Update pose
                    self.current_pose = self.current_pose @ transformation

                    # Publish odometry
                    self.publish_odometry(msg.header.stamp, msg.header.frame_id)

                    # Check if we should create a keyframe
                    translation_norm = np.linalg.norm(transformation[:3, 3])
                    if translation_norm > self.get_parameter('keyframe_threshold').value:
                        self.prev_features = {
                            'keypoints': keypoints,
                            'descriptors': descriptors
                        }

            else:
                # Store initial features
                self.prev_features = {
                    'keypoints': keypoints,
                    'descriptors': descriptors
                }

        except Exception as e:
            self.get_logger().error(f'Error in SLAM processing: {e}')

    def right_image_callback(self, msg: Image):
        """Process right camera image (for stereo depth estimation)"""
        # In a complete implementation, this would be used for stereo processing
        pass

    def extract_features_gpu(self, image):
        """GPU-accelerated feature extraction"""
        # This would use Isaac's GPU-accelerated feature extraction
        # For this example, we'll use a CPU fallback with a note about GPU acceleration
        self.get_logger().warn('GPU feature extraction not fully implemented in this example')
        return self.extract_features_cpu(image)

    def extract_features_cpu(self, image):
        """CPU-based feature extraction as fallback"""
        try:
            import cv2
            # Use ORB as an example feature detector
            orb = cv2.ORB_create(nfeatures=self.get_parameter('max_features').value)
            keypoints, descriptors = orb.detectAndCompute(image, None)

            if descriptors is not None:
                # Normalize descriptors for matching
                descriptors = descriptors.astype(np.float32)

            return keypoints or [], descriptors
        except ImportError:
            self.get_logger().error('OpenCV not available for feature extraction')
            return [], None

    def estimate_motion(self, prev_desc, curr_desc, prev_kp, curr_kp):
        """Estimate motion between two frames"""
        try:
            import cv2
            if prev_desc is None or curr_desc is None:
                return None

            # Use FLANN matcher for efficient matching
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                               key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(prev_desc, curr_desc, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Require minimum number of good matches
            min_matches = 10
            if len(good_matches) < min_matches:
                self.get_logger().warn(f'Insufficient matches: {len(good_matches)} < {min_matches}')
                return None

            # Extract matched points
            prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate essential matrix and decompose to get rotation/translation
            E, mask = cv2.findEssentialMat(curr_pts, prev_pts, focal=500.0, pp=(320, 240))
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts)

                # Create transformation matrix
                transformation = np.eye(4)
                transformation[:3, :3] = R
                transformation[:3, 3] = t.flatten()

                return transformation
            else:
                return None

        except ImportError:
            self.get_logger().error('OpenCV not available for motion estimation')
            return None

    def publish_odometry(self, stamp, frame_id):
        """Publish odometry information"""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = 'base_link'

        # Set pose
        odom_msg.pose.pose.position.x = self.current_pose[0, 3]
        odom_msg.pose.pose.position.y = self.current_pose[1, 3]
        odom_msg.pose.pose.position.z = self.current_pose[2, 3]

        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(self.current_pose[:3, :3])
        quat = r.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set twist (velocity estimation would go here)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom_msg)

        # Broadcast transform
        from geometry_msgs.msg import TransformStamped
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.current_pose[0, 3]
        t.transform.translation.y = self.current_pose[1, 3]
        t.transform.translation.z = self.current_pose[2, 3]

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)

    visual_slam = IsaacVisualSLAM()

    try:
        rclpy.spin(visual_slam)
    except KeyboardInterrupt:
        pass
    finally:
        visual_slam.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Apps Examples

### Isaac Navigation Example

```python
# Example: Isaac Navigation application for humanoid robots
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
import numpy as np

class IsaacHumanoidNavigator(Node):
    def __init__(self):
        super().__init__('isaac_humanoid_navigator')

        # Isaac Navigation parameters
        self.declare_parameter('planner_frequency', 10.0)
        self.declare_parameter('controller_frequency', 50.0)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('yaw_goal_tolerance', 0.1)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal',
            self.goal_callback, 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map',
            self.map_callback, 10
        )

        # Navigation state
        self.current_goal = None
        self.current_plan = None
        self.current_pose = None
        self.obstacles = []

        # Timers for planning and control
        self.planner_timer = self.create_timer(
            1.0 / self.get_parameter('planner_frequency').value,
            self.plan_callback
        )

        self.controller_timer = self.create_timer(
            1.0 / self.get_parameter('controller_frequency').value,
            self.control_callback
        )

        self.get_logger().info('Isaac Humanoid Navigator initialized')

    def goal_callback(self, msg: PoseStamped):
        """Handle new navigation goal"""
        self.current_goal = msg
        self.get_logger().info(f'New goal received: ({msg.pose.position.x}, {msg.pose.position.y})')

        # Trigger immediate replanning
        self.replan()

    def scan_callback(self, msg: LaserScan):
        """Process laser scan data for obstacle detection"""
        # Convert laser scan to obstacle positions
        angles = np.linspace(
            msg.angle_min, msg.angle_max, len(msg.ranges)
        )

        valid_ranges = []
        for i, range_val in enumerate(msg.ranges):
            if msg.range_min <= range_val <= msg.range_max:
                angle = angles[i]
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                valid_ranges.append((x, y))

        self.obstacles = valid_ranges

    def map_callback(self, msg: OccupancyGrid):
        """Process occupancy grid map"""
        # Store map information for path planning
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)

    def replan(self):
        """Replan path to current goal"""
        if self.current_goal is None or self.current_pose is None:
            return

        # In a real implementation, this would use Isaac's GPU-accelerated planners
        # For this example, we'll use a simple approach
        start = (self.current_pose.position.x, self.current_pose.position.y)
        goal = (self.current_goal.pose.position.x, self.current_goal.pose.position.y)

        # Simple path planning (in real implementation, use Isaac's planners)
        path = self.compute_simple_path(start, goal)

        if path:
            self.current_plan = path
            self.publish_path(path)
            self.get_logger().info(f'New path computed with {len(path)} waypoints')

    def compute_simple_path(self, start, goal):
        """Simple path computation (replace with Isaac planners)"""
        # This is a placeholder - in reality, Isaac would use GPU-accelerated planners
        # like A*, Dijkstra, or more advanced sampling-based planners

        # Create straight-line path for demonstration
        steps = 10
        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append((x, y))

        return path

    def plan_callback(self):
        """Periodic planning callback"""
        self.replan()

    def control_callback(self):
        """Periodic control callback"""
        if self.current_plan is None or len(self.current_plan) == 0:
            # Stop if no plan
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            return

        # Get next waypoint in plan
        next_waypoint = self.current_plan[0]

        # Calculate control command to reach next waypoint
        cmd_vel = self.compute_control_to_waypoint(next_waypoint)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Check if we've reached the waypoint
        if self.distance_to_waypoint(next_waypoint) < self.get_parameter('goal_tolerance').value:
            # Remove reached waypoint
            self.current_plan.pop(0)

            # If no more waypoints, we've reached the goal
            if len(self.current_plan) == 0:
                self.get_logger().info('Goal reached!')
                self.current_goal = None
                # Stop the robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)

    def compute_control_to_waypoint(self, waypoint):
        """Compute control command to reach a waypoint"""
        cmd_vel = Twist()

        if self.current_pose is None:
            return cmd_vel

        # Calculate desired direction
        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Calculate desired angle
        desired_yaw = np.arctan2(dy, dx)
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Simple proportional controller
        angular_error = self.normalize_angle(desired_yaw - current_yaw)

        # Set velocities
        max_linear = self.get_parameter('max_linear_speed').value
        cmd_vel.linear.x = min(max_linear * 0.8, max_linear * distance) if distance > 0.1 else 0.0
        cmd_vel.angular.z = angular_error * 1.0  # Proportional gain

        # Limit angular velocity
        max_angular = self.get_parameter('max_angular_speed').value
        cmd_vel.angular.z = max(-max_angular, min(max_angular, cmd_vel.angular.z))

        return cmd_vel

    def distance_to_waypoint(self, waypoint):
        """Calculate distance to a waypoint"""
        if self.current_pose is None:
            return float('inf')

        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        return np.sqrt(dx*dx + dy*dy)

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def publish_path(self, path):
        """Publish the computed path"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # Assuming map frame

        for x, y in path:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = path_msg.header.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)

    navigator = IsaacHumanoidNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Manipulation Example

### GPU-Accelerated Grasp Planning

```python
# Example: Isaac manipulation with GPU-accelerated grasp planning
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
import numpy as np

class IsaacGraspPlanner(Node):
    def __init__(self):
        super().__init__('isaac_grasp_planner')

        # Isaac manipulation parameters
        self.declare_parameter('grasp_approach_distance', 0.1)
        self.declare_parameter('grasp_grasp_distance', 0.02)
        self.declare_parameter('grasp_elevation_angle', 0.785)  # 45 degrees
        self.declare_parameter('num_grasp_candidates', 20)

        # Publishers and subscribers
        self.object_cloud_sub = self.create_subscription(
            PointCloud2, '/object_cloud',
            self.object_cloud_callback, 10
        )

        self.grasp_candidate_pub = self.create_publisher(
            Marker, '/grasp_candidates', 10
        )

        self.grasp_command_pub = self.create_publisher(
            Pose, '/grasp_pose', 10
        )

        # Initialize GPU acceleration for grasp planning
        self.initialize_gpu_grasp_planning()

        self.get_logger().info('Isaac Grasp Planner initialized')

    def initialize_gpu_grasp_planning(self):
        """Initialize GPU acceleration for grasp planning"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

            # CUDA kernel for grasp evaluation
            cuda_code = """
            __global__ void evaluate_grasps_kernel(
                float* points, int num_points,
                float* grasp_poses, int num_grasps,
                float* grasp_scores, int* grasp_validity
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx >= num_grasps) return;

                // Get grasp pose
                float pos_x = grasp_poses[idx * 7 + 0];
                float pos_y = grasp_poses[idx * 7 + 1];
                float pos_z = grasp_poses[idx * 7 + 2];

                // Evaluate grasp quality based on point cloud
                float score = 0.0f;
                int valid_points = 0;

                for (int i = 0; i < num_points; i++) {
                    float dx = points[i * 3 + 0] - pos_x;
                    float dy = points[i * 3 + 1] - pos_y;
                    float dz = points[i * 3 + 2] - pos_z;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                    if (dist < 0.05f) {  // Within grasp distance
                        score += 1.0f / (dist + 0.001f);  // Higher score for closer points
                        valid_points++;
                    }
                }

                grasp_scores[idx] = score;
                grasp_validity[idx] = (valid_points >= 5) ? 1 : 0;  // At least 5 points for validity
            }
            """

            self.grasp_module = SourceModule(cuda_code)
            self.evaluate_grasps_kernel = self.grasp_module.get_function("evaluate_grasps_kernel")

            self.gpu_available = True
            self.get_logger().info('GPU-accelerated grasp planning enabled')

        except ImportError:
            self.gpu_available = False
            self.get_logger().warn('CUDA not available for grasp planning, using CPU fallback')

    def object_cloud_callback(self, msg: PointCloud2):
        """Process object point cloud for grasp planning"""
        try:
            import struct
            # Convert PointCloud2 to numpy array (simplified)
            # In a real implementation, use sensor_msgs_py.point_cloud2.read_points
            points = self.pointcloud2_to_array(msg)

            if len(points) == 0:
                self.get_logger().warn('Empty point cloud received')
                return

            # Generate grasp candidates
            grasp_poses = self.generate_grasp_candidates(points)

            # Evaluate grasps using GPU acceleration
            if self.gpu_available:
                best_grasp = self.evaluate_grasps_gpu(points, grasp_poses)
            else:
                best_grasp = self.evaluate_grasps_cpu(points, grasp_poses)

            if best_grasp is not None:
                # Publish the best grasp
                self.grasp_command_pub.publish(best_grasp)

                # Visualize grasp candidates
                self.visualize_grasp_candidates(grasp_poses, best_grasp)

                self.get_logger().info('Published grasp pose for manipulation')

        except Exception as e:
            self.get_logger().error(f'Error in grasp planning: {e}')

    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array (simplified)"""
        # This is a simplified conversion - in practice use sensor_msgs_py.point_cloud2
        # For this example, return a dummy array
        return np.random.rand(100, 3).astype(np.float32) * 0.5  # 100 random points in 0.5m cube

    def generate_grasp_candidates(self, points):
        """Generate candidate grasp poses based on point cloud"""
        num_candidates = self.get_parameter('num_grasp_candidates').value

        # Find centroid of point cloud
        centroid = np.mean(points, axis=0)

        # Generate grasps around the centroid
        grasps = []
        for i in range(num_candidates):
            # Random offset from centroid
            offset = np.random.normal(0, 0.05, 3)  # 5cm random offset
            position = centroid + offset

            # Random orientation (simplified)
            roll = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi/4, np.pi/4)  # Limited pitch for humanoid
            yaw = np.random.uniform(-np.pi, np.pi)

            # Convert to quaternion
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)

            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            grasp_pose = np.array([position[0], position[1], position[2], x, y, z, w], dtype=np.float32)
            grasps.append(grasp_pose)

        return np.array(grasps, dtype=np.float32)

    def evaluate_grasps_gpu(self, points, grasp_poses):
        """GPU-accelerated grasp evaluation"""
        num_points = len(points)
        num_grasps = len(grasp_poses)

        # Flatten grasp poses
        grasp_poses_flat = grasp_poses.flatten()

        # Allocate GPU memory
        points_gpu = cuda.mem_alloc(points.nbytes)
        grasps_gpu = cuda.mem_alloc(grasp_poses_flat.nbytes)
        scores_gpu = cuda.mem_alloc(num_grasps * 4)  # float32
        validity_gpu = cuda.mem_alloc(num_grasps * 4)  # int32

        # Copy data to GPU
        cuda.memcpy_htod(points_gpu, points.astype(np.float32))
        cuda.memcpy_htod(grasps_gpu, grasp_poses_flat)

        # Configure kernel execution
        block_size = 256
        grid_size = (num_grasps + block_size - 1) // block_size

        # Execute kernel
        self.evaluate_grasps_kernel(
            points_gpu, np.int32(num_points),
            grasps_gpu, np.int32(num_grasps),
            scores_gpu, validity_gpu,
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        # Copy results back
        scores = np.empty(num_grasps, dtype=np.float32)
        validity = np.empty(num_grasps, dtype=np.int32)
        cuda.memcpy_dtoh(scores, scores_gpu)
        cuda.memcpy_dtoh(validity, validity_gpu)

        # Clean up GPU memory
        del points_gpu
        del grasps_gpu
        del scores_gpu
        del validity_gpu

        # Find best valid grasp
        valid_scores = scores * validity  # Zero out invalid grasps
        if np.any(validity > 0):
            best_idx = np.argmax(valid_scores)
            best_grasp = self.array_to_pose(grasp_poses[best_idx])
            return best_grasp

        return None

    def evaluate_grasps_cpu(self, points, grasp_poses):
        """CPU-based grasp evaluation as fallback"""
        best_score = -1
        best_grasp = None

        for i, grasp in enumerate(grasp_poses):
            score = 0
            valid_points = 0

            pos = grasp[:3]

            for point in points:
                dist = np.linalg.norm(point - pos)
                if dist < 0.05:  # Within grasp distance
                    score += 1.0 / (dist + 0.001)  # Higher score for closer points
                    valid_points += 1

            if valid_points >= 5:  # At least 5 points for validity
                if score > best_score:
                    best_score = score
                    best_grasp = self.array_to_pose(grasp)

        return best_grasp

    def array_to_pose(self, grasp_array):
        """Convert grasp array to Pose message"""
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position.x = grasp_array[0]
        pose.position.y = grasp_array[1]
        pose.position.z = grasp_array[2]
        pose.orientation.x = grasp_array[3]
        pose.orientation.y = grasp_array[4]
        pose.orientation.z = grasp_array[5]
        pose.orientation.w = grasp_array[6]
        return pose

    def visualize_grasp_candidates(self, grasp_poses, best_grasp):
        """Visualize grasp candidates in RViz"""
        # This would create visualization markers for RViz
        # For simplicity, just log the best grasp
        self.get_logger().info(f'Best grasp: ({best_grasp.position.x:.3f}, {best_grasp.position.y:.3f}, {best_grasp.position.z:.3f})')

def main(args=None):
    rclpy.init(args=args)

    grasp_planner = IsaacGraspPlanner()

    try:
        rclpy.spin(grasp_planner)
    except KeyboardInterrupt:
        pass
    finally:
        grasp_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Integration Examples

### Isaac Sim to Real Robot Transfer

```python
# Example: Isaac Sim-to-Real transfer pipeline
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class IsaacSimToRealTransfer(Node):
    def __init__(self):
        super().__init__('isaac_sim_to_real_transfer')

        # Isaac sim-to-real transfer parameters
        self.declare_parameter('domain_randomization_enabled', True)
        self.declare_parameter('parameter_variance', 0.1)
        self.declare_parameter('transfer_validation_threshold', 0.95)

        # Publishers and subscribers for sim and real robots
        self.sim_joint_pub = self.create_publisher(JointState, '/sim/joint_commands', 10)
        self.real_joint_pub = self.create_publisher(JointState, '/real/joint_commands', 10)

        self.sim_feedback_sub = self.create_subscription(
            JointState, '/sim/joint_states',
            self.sim_feedback_callback, 10
        )

        self.real_feedback_sub = self.create_subscription(
            JointState, '/real/joint_states',
            self.real_feedback_callback, 10
        )

        # Transfer validation publisher
        self.validation_pub = self.create_publisher(Float32MultiArray, '/transfer_validation', 10)

        # State tracking
        self.sim_states = {}
        self.real_states = {}
        self.sim_to_real_mapping = {}  # Maps sim joint names to real joint names

        self.get_logger().info('Isaac Sim-to-Real Transfer initialized')

    def sim_feedback_callback(self, msg: JointState):
        """Process simulation robot feedback"""
        self.sim_states = dict(zip(msg.name, msg.position))

        # Apply domain randomization to simulation parameters
        if self.get_parameter('domain_randomization_enabled').value:
            randomized_states = self.apply_domain_randomization(msg)
        else:
            randomized_states = msg

        # Transfer to real robot with appropriate mapping and corrections
        self.transfer_to_real_robot(randomized_states)

    def real_feedback_callback(self, msg: JointState):
        """Process real robot feedback"""
        self.real_states = dict(zip(msg.name, msg.position))

        # Validate sim-to-real transfer performance
        self.validate_transfer_performance()

    def apply_domain_randomization(self, joint_state_msg):
        """Apply domain randomization to simulation parameters"""
        # Randomize joint positions slightly
        variance = self.get_parameter('parameter_variance').value
        randomized_positions = []

        for pos in joint_state_msg.position:
            random_offset = np.random.normal(0, variance)
            randomized_positions.append(pos + random_offset)

        # Create new message with randomized values
        randomized_msg = JointState()
        randomized_msg.header = joint_state_msg.header
        randomized_msg.name = joint_state_msg.name
        randomized_msg.position = randomized_positions
        randomized_msg.velocity = [v + np.random.normal(0, variance*0.1) for v in joint_state_msg.velocity]
        randomized_msg.effort = [e + np.random.normal(0, variance*0.5) for e in joint_state_msg.effort]

        return randomized_msg

    def transfer_to_real_robot(self, sim_command):
        """Transfer simulation command to real robot with corrections"""
        real_command = JointState()
        real_command.header.stamp = self.get_clock().now().to_msg()
        real_command.header.frame_id = 'base_link'

        # Map simulation joints to real robot joints
        for sim_name, sim_pos in zip(sim_command.name, sim_command.position):
            if sim_name in self.sim_to_real_mapping:
                real_name = self.sim_to_real_mapping[sim_name]
                real_command.name.append(real_name)
                # Apply any sim-to-real corrections here
                corrected_pos = self.apply_corrections(sim_name, sim_pos)
                real_command.position.append(corrected_pos)

        # Publish command to real robot
        self.real_joint_pub.publish(real_command)

    def apply_corrections(self, joint_name, sim_position):
        """Apply sim-to-real corrections to joint positions"""
        # This would contain learned correction functions from sim-to-real transfer
        # For this example, apply simple corrections based on known differences

        # Example: Known offset for a specific joint
        if joint_name == 'left_hip_pitch':
            # Apply learned correction factor
            correction_factor = 0.98  # Learned from system identification
            offset = 0.02  # Fixed offset
            return sim_position * correction_factor + offset
        elif joint_name == 'right_knee_pitch':
            # Another learned correction
            correction_factor = 1.02
            offset = -0.01
            return sim_position * correction_factor + offset
        else:
            # Default correction
            return sim_position

    def validate_transfer_performance(self):
        """Validate sim-to-real transfer performance"""
        if not self.sim_states or not self.real_states:
            return

        # Calculate similarity between sim and real robot states
        similarities = []

        for joint_name in self.sim_states:
            if joint_name in self.real_states:
                sim_pos = self.sim_states[joint_name]
                real_pos = self.real_states[joint_name]

                # Calculate similarity (1.0 = identical, 0.0 = completely different)
                # Using a simple difference measure (in practice, use more sophisticated metrics)
                diff = abs(sim_pos - real_pos)
                similarity = max(0.0, 1.0 - diff)  # Simple linear similarity
                similarities.append(similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)

            # Publish validation result
            validation_msg = Float32MultiArray()
            validation_msg.data = [avg_similarity]
            self.validation_pub.publish(validation_msg)

            self.get_logger().info(f'Transfer validation: {avg_similarity:.3f}')

            # Check if transfer is performing adequately
            threshold = self.get_parameter('transfer_validation_threshold').value
            if avg_similarity < threshold:
                self.get_logger().warn(f'Transfer performance below threshold: {avg_similarity:.3f} < {threshold:.3f}')

def main(args=None):
    rclpy.init(args=args)

    transfer_node = IsaacSimToRealTransfer()

    try:
        rclpy.spin(transfer_node)
    except KeyboardInterrupt:
        pass
    finally:
        transfer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac AI Integration Example

### GPU-Accelerated Perception with Isaac

```python
# Example: Isaac AI integration for perception
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
import numpy as np

class IsaacAIPerception(Node):
    def __init__(self):
        super().__init__('isaac_ai_perception')

        # Isaac AI parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('max_objects', 10)

        # Publishers and subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_rect_color',
            self.rgb_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect',
            self.depth_callback, 10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info',
            self.camera_info_callback, 10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray, '/ai/detections', 10
        )

        # Initialize Isaac AI components
        self.initialize_isaac_ai()

        # Camera parameters
        self.camera_intrinsics = None

        self.get_logger().info('Isaac AI Perception initialized')

    def initialize_isaac_ai(self):
        """Initialize Isaac AI components"""
        try:
            import torch
            import torchvision.transforms as transforms

            # Initialize GPU acceleration if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load pre-trained model (in practice, load Isaac-specific models)
            # For this example, we'll use a placeholder
            self.model = None  # Would load actual model
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((416, 416))
            ])

            self.ai_initialized = True
            self.get_logger().info(f'Isaac AI initialized on {self.device}')

        except ImportError:
            self.ai_initialized = False
            self.device = torch.device('cpu')
            self.get_logger().warn('PyTorch not available, using CPU for AI processing')

    def rgb_callback(self, msg: Image):
        """Process RGB image for AI-based perception"""
        if not self.ai_initialized or self.camera_intrinsics is None:
            return

        try:
            # Convert ROS image to OpenCV
            from cv_bridge import CvBridge
            cv_bridge = CvBridge()
            cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Run AI inference
            if self.model is not None:
                detections = self.run_inference(cv_image)
            else:
                # Placeholder detections
                detections = self.placeholder_detection(cv_image)

            # Create and publish detection message
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'Error in AI perception: {e}')

    def depth_callback(self, msg: Image):
        """Process depth image to add 3D information to detections"""
        # In a complete implementation, this would be used to convert 2D detections to 3D positions
        pass

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics for 3D reconstruction"""
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)

    def run_inference(self, image):
        """Run AI inference on image"""
        # Preprocess image
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        # Run inference (placeholder)
        # In a real implementation, this would use Isaac's optimized inference
        with torch.no_grad():
            # Placeholder: return dummy detections
            # Real implementation would call self.model(input_tensor)
            return self.placeholder_detection(image)

    def placeholder_detection(self, image):
        """Placeholder detection function"""
        # This would be replaced with actual AI model inference
        # For demonstration, return random detections
        height, width = image.shape[:2]

        num_detections = np.random.randint(0, 4)  # 0-3 random detections
        detections = []

        for i in range(num_detections):
            # Random bounding box
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)

            # Random confidence
            confidence = np.random.uniform(0.5, 0.99)

            # Random class
            classes = ['person', 'chair', 'table', 'cup']
            class_name = np.random.choice(classes)

            detection = {
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'class': class_name
            }
            detections.append(detection)

        return detections

    def create_detection_message(self, detections, header):
        """Create vision_msgs/Detection2DArray message from detections"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection_msg = Detection2D()

            # Set bounding box
            bbox = det['bbox']
            detection_msg.bbox.center.x = bbox[0] + bbox[2] / 2  # center x
            detection_msg.bbox.center.y = bbox[1] + bbox[3] / 2  # center y
            detection_msg.bbox.size_x = bbox[2]  # width
            detection_msg.bbox.size_y = bbox[3]  # height

            # Set hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = det['class']
            hypothesis.score = det['confidence']
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        return detection_array

def main(args=None):
    rclpy.init(args=args)

    ai_perception = IsaacAIPerception()

    try:
        rclpy.spin(ai_perception)
    except KeyboardInterrupt:
        pass
    finally:
        ai_perception.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Launch Files

### Isaac Navigation Launch Example

```python title="isaac_navigation.launch.py"
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    enable_viz = LaunchConfiguration('enable_viz')
    declare_enable_viz_arg = DeclareLaunchArgument(
        'enable_viz',
        default_value='true',
        description='Enable visualization'
    )

    # Isaac Navigation nodes
    navigation_node = Node(
        package='isaac_navigation',
        executable='isaac_nav_node',
        name='isaac_humanoid_navigator',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('isaac_navigation'),
                'config',
                'nav_params.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('/cmd_vel', '/humanoid/cmd_vel'),
            ('/scan', '/humanoid/laser_scan'),
            ('/map', '/humanoid/map'),
            ('/move_base_simple/goal', '/humanoid/goal'),
        ],
        output='screen'
    )

    # Isaac Perception node
    perception_node = Node(
        package='isaac_perception',
        executable='isaac_perception_node',
        name='isaac_obstacle_detector',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'enable_gpu_processing': True}
        ],
        remappings=[
            ('/input_image', '/humanoid/front_camera/image_rect_color'),
            ('/obstacles', '/humanoid/obstacles'),
        ],
        output='screen'
    )

    # Isaac SLAM node (if needed)
    slam_node = Node(
        package='isaac_slam',
        executable='isaac_slam_node',
        name='isaac_vslam',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'enable_gpu_slam': True}
        ],
        remappings=[
            ('/camera/color/image_raw', '/humanoid/stereo_camera/left/image_rect_color'),
            ('/camera/depth/image_rect', '/humanoid/stereo_camera/depth/image_rect'),
            ('/map', '/humanoid/vslam_map'),
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static'),
        ],
        output='screen'
    )

    # RViz2 (if visualization enabled)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('isaac_navigation'),
            'rviz',
            'navigation.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(enable_viz)
    )

    return LaunchDescription([
        declare_use_sim_time_arg,
        declare_enable_viz_arg,
        navigation_node,
        perception_node,
        slam_node,
        rviz_node
    ])
```

## Isaac Configuration Files

### Isaac Parameter Configuration

```yaml title="config/nav_params.yaml"
isaac_humanoid_navigator:
  ros__parameters:
    # Controller parameters
    controller_frequency: 50.0
    min_x_velocity_threshold: 0.001
    max_x_velocity: 0.5
    min_y_velocity_threshold: 0.001
    max_y_velocity: 0.2
    max_theta_velocity: 1.0
    min_theta_velocity_threshold: 0.001

    # Goal checker parameters
    goal_checker.xy_goal_tolerance: 0.25
    goal_checker.yaw_goal_tolerance: 0.05
    goal_checker.stateful: True

    # Local planner (Trajectory follower)
    local_costmap:
      global_frame: odom
      robot_base_frame: base_link
      update_frequency: 5.0
      publish_frequency: 2.0
      footprint: "[[-0.325, -0.325], [-0.325, 0.325], [0.325, 0.325], [0.325, -0.325]]"
      resolution: 0.05
      inflation_radius: 0.55
      plugins: ["obstacle_layer", "inflation_layer"]

    local_costmap/obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0

    local_costmap/inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      enabled: True
      inflation_radius: 0.55
      cost_scaling_factor: 3.0
      inflate_unknown: False
      inflate_around_unknown: False