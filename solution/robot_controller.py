import sys
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSPresetProfiles
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from auro_interfaces.msg import StringWithPose, Item, ItemList
from assessment_interfaces.msg import HomeZone, RobotList, ItemHolder, ItemHolders
from tf_transformations import euler_from_quaternion
import angles
from enum import Enum
import random
import math

LINEAR_VELOCITY = 0.3  # Metres per second
ANGULAR_VELOCITY = 0.5  # Radians per second

TURN_LEFT = 1  # Positive angular velocity turns left
TURN_RIGHT = -1  # Negative angular velocity turns right

SCAN_THRESHOLD = 0.5  # Metres per second
# Array indexes for sensor sectors
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3

# Finite state machine (FSM) states
class State(Enum):
    FORWARD = 0
    SEARCHING = 1
    TURNING = 2
    COLLECTING = 3
    RETURN_HOME = 5
    ITEM_PICKED_UP = 6


class RobotController(Node):
    def __init__(self, robot_name):
        super().__init__('robot_controller')
        self.robot_name = robot_name

        # Class variables used to store persistent values between executions of callbacks and control loop
        self.state = State.FORWARD # Current FSM state
        self.pose = Pose() # Current pose (position and orientation), relative to the odom reference frame
        self.previous_pose = Pose() # Store a snapshot of the pose for comparison against future poses
        self.yaw = 0.0 # Angle the robot is facing (rotation around the Z axis, in radians), relative to the odom reference frame
        self.previous_yaw = 0.0 # Snapshot of the angle for comparison against future angles
        self.turn_angle = 0.0 # Relative angle to turn to in the TURNING state
        self.turn_direction = TURN_LEFT # Direction to turn in the TURNING state
        self.goal_distance = random.uniform(1.0, 2.0) # Goal distance to travel in FORWARD state
        self.scan_triggered = [False] * 4 # Boolean value for each of the 4 LiDAR sensor sectors. True if obstacle detected within SCAN_THRESHOLD
        self.items = ItemList()
        self.held_item = ItemHolders()

        self.item__holder_subscriber = self.create_subscription(
            ItemHolders,
            'robot1/item_holders',
            self.item_holder_callback,
            10
        )

        self.item_subscriber = self.create_subscription(
            ItemList,
            'robot1/items',
            self.item_callback,
            10
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            'robot1/odom',
            self.odom_callback,
            10)
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'robot1/scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value)
            
        # Subscribe to RobotSensor topics
        self.robot_list_subscriber = self.create_subscription(
            RobotList,
            'robot1/robots',
            self.robot_info_callback,
            10
        )

        self.home_zone_subscriber = self.create_subscription(
            HomeZone,
            'robot1/home_zone',
            self.home_zone_callback,
            QoSPresetProfiles.SENSOR_DATA.value)
        
        self.home_zone_detected = False
        self.home_zone_center = (0, 0)

        self.cmd_vel_publisher = self.create_publisher(Twist, 'robot1/cmd_vel', 10)
        # self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(StringWithPose, 'robot1/marker_input', 10)
        # ... (rest of the previous code remains unchanged)

        self.timer_period = 0.1 # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop)

    def robot_info_callback(self, msg):
        # Process the RobotList message
        for robot in msg.data:
            self.get_logger().info(f"Received robot information - x: {robot.x}, y: {robot.y}, size: {robot.size}")

    def item_callback(self, msg):
        self.items = msg

    def item_holder_callback(self, msg):
        self.get_logger().info("Item callback triggered")

        # Log the received message to check its structure
        self.get_logger().info(f"Received message: {msg}")

        # Process the ItemHolders message
        for item_holder in msg.data:
            self.get_logger().info(f"Processing item_holder: {item_holder}")

            if item_holder.robot_id == self.robot_name and item_holder.holding_item:
                # The current robot is holding an item

                # Add more debugging information
                self.get_logger().info(f"Robot {self.robot_name} is holding an item")

                # Stop the robot instantly
                stop_msg = Twist()
                self.cmd_vel_publisher.publish(stop_msg)

                # Update the state to indicate that the robot has picked up an item
                self.state = State.ITEM_PICKED_UP
                self.get_logger().info("State updated to ITEM_PICKED_UP")

                # Update the held_item variable
                self.held_item = self.items.data[0] if self.items.data else None
            else:
                # The robot is not holding an item, reset the held_item variable
                self.held_item = None
        
    def odom_callback(self, msg):
        self.pose = msg.pose.pose # Store the pose in a class variable

        (roll, pitch, yaw) = euler_from_quaternion([self.pose.orientation.x,
                                                    self.pose.orientation.y,
                                                    self.pose.orientation.z,
                                                    self.pose.orientation.w])
        
        self.yaw = yaw # Store the yaw in a class variable

    def scan_callback(self, msg):
        front_ranges = msg.ranges[331:359] + msg.ranges[0:30] # 30 to 331 degrees (30 to -30 degrees)
        left_ranges  = msg.ranges[31:90] # 31 to 90 degrees (31 to 90 degrees)
        back_ranges  = msg.ranges[91:270] # 91 to 270 degrees (91 to -90 degrees)
        right_ranges = msg.ranges[271:330] # 271 to 330 degrees (-30 to -91 degrees)

        # Store True/False values for each sensor segment, based on whether the nearest detected obstacle is closer than SCAN_THRESHOLD
        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD 
        self.scan_triggered[SCAN_LEFT]  = min(left_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK]  = min(back_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD

        self.get_logger().info(f"Scan Triggered: Front={self.scan_triggered[SCAN_FRONT]}, Left={self.scan_triggered[SCAN_LEFT]}, Back={self.scan_triggered[SCAN_BACK]}, Right={self.scan_triggered[SCAN_RIGHT]}")

    def home_zone_callback(self, msg):
        self.home_zone_detected = msg.visible
        self.home_zone_center = (msg.x, msg.y)


    def control_loop(self):
    # Send message to rviz_text_marker node
        marker_input = StringWithPose()
        marker_input.text = str(self.state)  # Visualize robot state as an RViz marker
        marker_input.pose = self.pose  # Set the pose of the RViz marker to track the robot's pose
        self.marker_publisher.publish(marker_input)

        match self.state:
            # ...

            case State.FORWARD:
                if self.scan_triggered[SCAN_FRONT]:
                    # Obstacle detected in front, switch to TURNING state
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info("Detected obstacle in front, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")
                    return

                if self.scan_triggered[SCAN_LEFT]:
                    # Obstacle detected to the left, switch to TURNING state
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(45, 90)
                    self.turn_direction = TURN_RIGHT  # Turn right to avoid the left obstacle
                    self.get_logger().info(f"Detected obstacle to the left, turning right by {self.turn_angle:.2f} degrees")
                    return

                if self.scan_triggered[SCAN_RIGHT]:
                    # Obstacle detected to the right, switch to TURNING state
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(45, 90)
                    self.turn_direction = TURN_LEFT  # Turn left to avoid the right obstacle
                    self.get_logger().info(f"Detected obstacle to the right, turning left by {self.turn_angle:.2f} degrees")
                    return

                # Continue moving forward
                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                # Update goal_distance based on distance traveled
                distance_traveled = math.sqrt((self.pose.position.x - self.previous_pose.position.x)**2 +
                                            (self.pose.position.y - self.previous_pose.position.y)**2)
                self.goal_distance -= distance_traveled

                # Check if the robot is holding an item
                if self.held_item is not None:
                    # If the robot is holding an item, update the state to indicate that the robot has picked up an item
                    self.state = State.ITEM_PICKED_UP     
                    print("this is working now")


                    #############################
                    # THIS BIT HERE IS ERRORING #
                    #############################





                    return

                # Check if the goal_distance is achieved
                if self.goal_distance <= 0.0:
                    # If the goal distance is achieved, switch to SEARCHING state
                    self.state = State.SEARCHING
                    self.get_logger().info(f"Goal distance achieved. Switching to SEARCHING state.")
                    return

            case State.SEARCHING:
                # Implement logic to search for red balls with number 5
                target_color = "red"
                target_value = 5

                # Filter the items based on color and value
                target_items = [item for item in self.items.data if item.colour == target_color and item.value == target_value]

                if target_items:
                    # If there are target items, choose the closest one and update the goal_distance and turn_angle
                    closest_item = min(target_items, key=lambda item: math.sqrt((item.x - self.pose.position.x)**2 + (item.y - self.pose.position.y)**2))
                    angle_to_item = math.atan2(closest_item.y - self.pose.position.y, closest_item.x - self.pose.position.x)
                    angle_difference = angles.normalize_angle(angle_to_item - self.yaw)

                    # If not aligned with the item, transition to TURNING state to align with the item
                    if abs(angle_difference) > math.radians(10):
                        self.state = State.TURNING
                        self.turn_angle = math.degrees(angle_difference)
                        self.turn_direction = TURN_LEFT if angle_difference > 0 else TURN_RIGHT
                        self.get_logger().info(f"Aligning with target item. Turning {'left' if self.turn_direction == TURN_LEFT else 'right'} by {self.turn_angle:.2f} degrees.")
                        return

                    # If aligned with the item, transition to COLLECTING state
                    self.goal_distance = math.sqrt((closest_item.x - self.pose.position.x)**2 + (closest_item.y - self.pose.position.y)**2)
                    self.state = State.COLLECTING
                    self.get_logger().info(f"Aligned with target item. Moving to COLLECTING state. Distance to item: {self.goal_distance:.2f} metres.")
                    return

            case State.TURNING:
                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)

                if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
                    if self.state == State.TURNING:
                        self.goal_distance = random.uniform(1.0, 2.0)
                        self.state = State.FORWARD
                        self.get_logger().info(f"Finished turning, driving forward by {self.goal_distance:.2f} metres")
                
            case State.COLLECTING:
                # Assuming there's a single item in the list (you may need to adjust based on your use case)
                if self.items.data:
                    # Robot is holding an item
                    self.held_item = self.items.data[0]

                    # Calculate the angle difference between the robot's current orientation and the item's position
                    angle_to_item = math.atan2(self.held_item.y - self.pose.position.y, self.held_item.x - self.pose.position.x)
                    angle_difference = angles.normalize_angle(angle_to_item - self.yaw)

                    if abs(angle_difference) > math.radians(10):  # Tolerance angle, adjust as needed
                        # If not aligned with the item, transition to TURNING state to align with the item
                        self.state = State.TURNING
                        self.turn_angle = math.degrees(angle_difference)
                        self.turn_direction = TURN_LEFT if angle_difference > 0 else TURN_RIGHT
                        self.get_logger().info(f"Aligning with item. Turning {'left' if self.turn_direction == TURN_LEFT else 'right'} by {self.turn_angle:.2f} degrees.")
                        return

                    # Check if the robot is close to the item (you may need to adjust the distance threshold)
                    if self.goal_distance <= 0.2:
                        # Assuming the item is collected, you can process it as needed
                        self.get_logger().info("Item collected and being held!")

                        # Stop the robot
                        msg = Twist()
                        self.cmd_vel_publisher.publish(msg)
                        self.get_logger().info("Robot has picked up an item. Stopping.")

                        # Set the state to indicate that the robot has successfully collected the item
                        self.state = State.ITEM_PICKED_UP
                        return

                    # Otherwise, drive forward towards the item
                    msg = Twist()
                    msg.linear.x = LINEAR_VELOCITY
                    self.cmd_vel_publisher.publish(msg)
                    self.get_logger().info(f"Aligned with item. Driving forward by {self.goal_distance:.2f} metres.")
                else:
                    # No item detected, transition to SEARCHING state or other appropriate state
                    self.state = State.SEARCHING
                    return


            # case State.RETURN_HOME:
            #     if self.home_zone_detected:
            #         # Stop the robot
            #         msg = Twist()
            #         self.cmd_vel_publisher.publish(msg)
            #         self.get_logger().info("Home zone detected. Stopping robot.")

            #         # If not facing home zone, turn towards it
            #         angle_to_home_zone = math.atan2(self.home_zone_center[1] - self.pose.position.y,
            #                                         self.home_zone_center[0] - self.pose.position.x)
            #         angle_difference = angles.normalize_angle(angle_to_home_zone - self.yaw)

            #         if abs(angle_difference) > math.radians(10):  # Tolerance angle, adjust as needed
            #             self.state = State.TURNING
            #             self.turn_angle = math.degrees(angle_difference)
            #             self.turn_direction = TURN_LEFT if angle_difference > 0 else TURN_RIGHT
            #             self.get_logger().info(f"Aligning with home zone. Turning {'left' if self.turn_direction == TURN_LEFT else 'right'} by {self.turn_angle:.2f} degrees.")
            #             return

            #         # Otherwise, drive forward towards home zone
            #         self.state = State.FORWARD
            #         self.goal_distance = random.uniform(1.0, 2.0)
            #         self.get_logger().info(f"Aligned with home zone. Driving forward by {self.goal_distance:.2f} metres.")
            #         return

            #     # Drive forward towards the home zone
            #     msg = Twist()
            #     msg.linear.x = LINEAR_VELOCITY
            #     self.cmd_vel_publisher.publish(msg)
            #     self.get_logger().info("Driving forward towards home zone.")
    
            case State.ITEM_PICKED_UP:
                # This state is entered only when an item is successfully picked up
                # Add any logic or actions you want to perform when the item is picked up
                self.get_logger().info("Robot has picked up an item. Performing actions in ITEM_PICKED_UP state.")

                # For example, you might want to update the held item information or perform additional tasks

                # Check if the robot actually picked up an item
                if self.held_item is not None:
                    # Transition to a new state or set the state based on your logic
                    self.state = State.FORWARD  # You can transition to another state after picking up the item
                    return
                else:
                    # If the robot didn't pick up an item, handle it accordingly (e.g., stay in the ITEM_PICKED_UP state or transition to another state)
                    self.get_logger().info("No item picked up. Handling it accordingly.")
                    # Add your logic here

            # Add any additional states and logic as needed

                            
    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    robot = RobotController('robot1')
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        robot.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()





    #item_callback not being triggered
    #need to add robot returning home
    #need to add robot targetting items
    #need to add mapping