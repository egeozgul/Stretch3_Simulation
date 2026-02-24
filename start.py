#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation
import stretch_body.robot as robot
import math

class FruitAlignmentNode(Node):
    def __init__(self):
        super().__init__('fruit_alignment_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # Stretch robot interface
        self.robot = robot.Robot()
        self.robot.startup()
        
        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera intrinsics (replace with your calibration)
        self.camera_matrix = np.array([
            [912.490478515625, 0.0, 643.6722412109375],
            [0.0, 912.8123779296875, 380.4475402832031],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                
        # Marker size in meters
        self.marker_size = 0.05  # 5cm marker
        
        # Alignment parameters
        self.alignment_threshold = 0.05  # 5 degrees
        self.fruit_detected = False
        self.fruit_id = 202  # ArUco marker ID for fruit
        
        self.get_logger().info("Fruit alignment node started")
    
    def image_callback(self, msg):
        """Detect ArUco marker and broadcast fruit frame"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            if ids is not None and self.fruit_id in ids:
                # Find the fruit marker
                idx = np.where(ids == self.fruit_id)[0][0]
                corner = corners[idx]
                
                # Estimate pose
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                # Broadcast fruit frame relative to camera
                self.broadcast_fruit_frame(rvec[0][0], tvec[0][0])
                self.fruit_detected = True
                
                # Draw detection for visualization
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                 rvec[0][0], tvec[0][0], 0.03)
                
                cv2.imshow("Fruit Detection", cv_image)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
    
    def broadcast_fruit_frame(self, rvec, tvec):
        """Broadcast fruit frame to TF tree"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_color_optical_frame'
        t.child_frame_id = 'fruit_marker'
        
        # Translation
        t.transform.translation.x = float(tvec[0])
        t.transform.translation.y = float(tvec[1])
        t.transform.translation.z = float(tvec[2])
        
        # Rotation (convert rodrigues to quaternion)
        rot_mat, _ = cv2.Rodrigues(rvec)
        quat = Rotation.from_matrix(rot_mat).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform
        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return None
    
    def calculate_angle_difference(self):
        try:
            fruit_odom = self.get_transform('odom', 'fruit_marker')
            if fruit_odom is None:
                return None
            ee_odom = self.get_transform('odom', 'link_grasp_center')
            if ee_odom is None:
                return None
            base_odom = self.get_transform('odom', 'base_link')
            if base_odom is None:
                return None
            fruit_x = fruit_odom.transform.translation.x
            fruit_y = fruit_odom.transform.translation.y
            ee_x = ee_odom.transform.translation.x
            ee_y = ee_odom.transform.translation.y
            base_x = base_odom.transform.translation.x
            base_y = base_odom.transform.translation.y
            angle_to_fruit = math.atan2(fruit_y - base_y, fruit_x - base_x)
            angle_to_ee = math.atan2(ee_y - base_y, ee_x - base_x)
            angle_diff = angle_to_fruit - angle_to_ee
            # Normalize to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            return angle_diff
        except Exception as e:
            self.get_logger().error(f"Angle calculation error: {e}")
            return None
    
    def rotate_base_to_align(self):
        """Rotate base to align end effector with fruit"""
        if not self.fruit_detected:
            self.get_logger().warn("No fruit detected yet")
            return False
        angle_diff = self.calculate_angle_difference()
        if angle_diff is None:
            self.get_logger().error("Could not calculate angle difference")
            return False
        angle_deg = math.degrees(angle_diff)
        self.get_logger().info(f"Angle difference: {angle_deg:.2f} degrees")
        if abs(angle_deg) < math.degrees(self.alignment_threshold):
            self.get_logger().info("Already aligned!")
            return True
        self.get_logger().info(f"Rotating base by {angle_deg:.2f} degrees")
        base_odom = self.get_transform('odom', 'base_link')
        current_quat = [
            base_odom.transform.rotation.x,
            base_odom.transform.rotation.y,
            base_odom.transform.rotation.z,
            base_odom.transform.rotation.w
        ]
        current_yaw = Rotation.from_quat(current_quat).as_euler('xyz')[2]
        target_yaw = current_yaw + angle_diff
        rotation_speed = 0.2  # rad/s
        rotation_time = abs(angle_diff) / rotation_speed
        
        if angle_diff > 0:
            self.robot.base.rotate_by(angle_diff, v_r=rotation_speed)
        else:
            self.robot.base.rotate_by(angle_diff, v_r=rotation_speed)
        self.robot.push_command()
        import time
        time.sleep(rotation_time + 0.5)
        
        # Verify alignment
        new_angle_diff = self.calculate_angle_difference()
        if new_angle_diff is not None:
            new_angle_deg = math.degrees(new_angle_diff)
            self.get_logger().info(f"After rotation, angle difference: {new_angle_deg:.2f} degrees")
            
            if abs(new_angle_deg) < math.degrees(self.alignment_threshold):
                self.get_logger().info("✓ Alignment successful!")
                return True
            else:
                self.get_logger().warn("Alignment incomplete, may need another iteration")
                return False
        
        return False
    
    def verify_alignment_in_base_frame(self):
        """Alternative: verify alignment by checking fruit Y-coordinate in base_link"""
        try:
            fruit_base = self.get_transform('base_link', 'fruit_marker')
            if fruit_base is None:
                return False
            
            fruit_y = fruit_base.transform.translation.y
            fruit_x = fruit_base.transform.translation.x
            
            self.get_logger().info(f"Fruit position in base_link: x={fruit_x:.3f}, y={fruit_y:.3f}")
            
            # Check if fruit is along +X axis (small Y component)
            if abs(fruit_y) < 0.05:  # 5cm tolerance
                self.get_logger().info("✓ Fruit is aligned with arm extension axis (+X)")
                return True
            else:
                self.get_logger().info(f"Fruit is {abs(fruit_y)*100:.1f}cm off-axis")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Verification error: {e}")
            return False
    
    def run_alignment_sequence(self):
        """Main alignment sequence"""
        rate = self.create_rate(10)  # 10 Hz
        
        self.get_logger().info("Waiting for fruit detection...")
        
        # Wait for fruit detection
        while not self.fruit_detected and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()
        
        self.get_logger().info("Fruit detected! Starting alignment...")
        
        # Attempt alignment (may need multiple iterations)
        max_iterations = 3
        for i in range(max_iterations):
            self.get_logger().info(f"Alignment attempt {i+1}/{max_iterations}")
            
            if self.rotate_base_to_align():
                # Double check in base frame
                if self.verify_alignment_in_base_frame():
                    self.get_logger().info("🎯 Alignment complete!")
                    return True
            
            # Small delay between attempts
            import time
            time.sleep(1.0)
        
        self.get_logger().warn("Alignment failed after max iterations")
        return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.robot.stop()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    
    node = FruitAlignmentNode()
    
    try:
        # Run alignment sequence
        success = node.run_alignment_sequence()
        
        if success:
            node.get_logger().info("Ready for IK and reaching!")
        else:
            node.get_logger().error("Alignment failed")
        
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()