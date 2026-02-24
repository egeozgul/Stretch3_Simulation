"""Navigation controller for moving robot to target positions.
Updated for Stretch 3 - outputs linear and angular velocities that are converted to wheel velocities.
"""

import math
import time
import numpy as np

# Navigation parameters
POSITION_TOLERANCE = 0.15  # meters
ALIGNMENT_THRESHOLD = math.radians(15)  # degrees
MAX_ANGLE_FOR_MOVEMENT = math.radians(45)  # degrees
DIRECTION_TOLERANCE = math.radians(5)  # degrees
# Max velocities limited by wheel actuator ctrlrange [-6, 6] rad/s
# With wheel radius 0.05m: max linear = 6 * 0.05 = 0.3 m/s
MAX_LINEAR_VEL = 0.3  # m/s (limited by Stretch 3 wheel actuator ctrlrange)
MAX_ANGULAR_VEL = 1.0  # rad/s (reasonable for Stretch 3)
K_P_ANGULAR = 2.0


class NavigationController:
    """Proportional controller for navigating to target positions."""
    
    def __init__(self):
        self.target_pos = None
        self.target_direction = None
        self.active = False
        self.reached = False
        self._was_aligned = False
        self._position_reached = False
        # Default tolerances
        self.turn_only_tolerance = math.radians(5.0)
        self.position_tolerance = POSITION_TOLERANCE
        self.direction_tolerance = DIRECTION_TOLERANCE
        # Timing tracking
        self._turn_only_start_time = None
        self._navigation_start_time = None
    
    def set_target(self, target_pos, target_direction=None, position_tolerance=0.05, direction_tolerance=None):
        """
        Set target position and optional direction.
        
        Args:
            target_pos: Target position [x, y]
            target_direction: Optional target direction in radians
            position_tolerance: Optional position tolerance in meters (default: 0.15)
            direction_tolerance: Optional direction tolerance in degrees (default: 5.0)
        """
        self.target_pos = np.array(target_pos[:2])
        self.target_direction = target_direction
        self.active = True
        self.reached = False
        self._was_aligned = False
        self._position_reached = False
        self._navigation_start_time = time.time()
        
        # Set tolerances: clamp to reasonable ranges
        if position_tolerance is not None:
            self.position_tolerance = max(0.01, min(5.0, float(position_tolerance)))
        else:
            self.position_tolerance = POSITION_TOLERANCE
        
        if direction_tolerance is not None:
            direction_tolerance_deg = max(0.1, min(180.0, float(direction_tolerance)))
            self.direction_tolerance = math.radians(direction_tolerance_deg)
        else:
            self.direction_tolerance = DIRECTION_TOLERANCE
    
    def set_turn_only_target(self, target_pos=None, delta_angle_degrees=None, target_angle_degrees=None):
        """
        Set target for turning only (no movement).
        
        Args:
            target_pos: Optional target position [x, y]. If None, uses target_angle_degrees
            delta_angle_degrees: Optional delta angle in degrees. Action completes when 
                                within this angle of target heading. Default: 5.0 degrees.
            target_angle_degrees: Optional absolute target angle in degrees (0-360). 
                                If specified, robot turns to this absolute heading.
        """
        if target_angle_degrees is not None:
            # Absolute angle mode
            self.target_pos = None
            self.target_direction = math.radians(float(target_angle_degrees) % 360.0)
        else:
            # Position-based mode
            self.target_pos = np.array(target_pos[:2]) if target_pos is not None else None
            self.target_direction = None
        
        self.active = True
        self.reached = False
        self._was_aligned = False
        self._position_reached = True
        self._turn_only_start_time = time.time()
        
        # Set tolerance: clamp to reasonable range and convert to radians
        if delta_angle_degrees is not None:
            delta_angle_degrees = max(0.1, min(180.0, float(delta_angle_degrees)))
            self.turn_only_tolerance = math.radians(delta_angle_degrees)
        else:
            self.turn_only_tolerance = math.radians(5.0)
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-π, π] range."""
        return math.atan2(math.sin(angle), math.cos(angle))
    
    @staticmethod
    def _quaternion_to_yaw(quat):
        """Extract yaw angle from quaternion [w, x, y, z]."""
        w, x, y, z = quat if len(quat) == 4 else (1.0, 0.0, 0.0, 0.0)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    def _calculate_angle_error(self, desired_angle, current_yaw):
        """Calculate angle error, choosing shortest rotation path."""
        return self._normalize_angle(desired_angle - current_yaw)
    
    def get_control(self, current_pos, current_quat):
        """
        Get control velocities to reach target.
        
        Returns:
            (linear_vel, angular_vel) tuple in (m/s, rad/s)
        """
        if not self.active or self.target_pos is None:
            return (0.0, 0.0)
        
        diff = self.target_pos - np.array(current_pos[:2])
        distance = np.linalg.norm(diff)
        current_yaw = self._quaternion_to_yaw(current_quat)
        
        if distance > 100.0:
            self.active = False
            return (0.0, 0.0)
        
        # Mark position as reached when within tolerance
        if not self._position_reached and distance < self.position_tolerance:
            self._position_reached = True
        
        # Phase 1: Navigate to position
        if not self._position_reached:
            desired_angle = math.atan2(diff[1], diff[0])
            angle_error = self._calculate_angle_error(desired_angle, current_yaw)
            angle_error_abs = abs(angle_error)
            
            angular_vel = np.clip(-K_P_ANGULAR * angle_error, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
            
            if angle_error_abs <= MAX_ANGLE_FOR_MOVEMENT:
                self._was_aligned = True
                if angle_error_abs <= ALIGNMENT_THRESHOLD:
                    linear_vel = MAX_LINEAR_VEL
                else:
                    speed_factor = 1.0 - (angle_error_abs / MAX_ANGLE_FOR_MOVEMENT) * 0.3
                    linear_vel = MAX_LINEAR_VEL * speed_factor
                # No negation needed for Stretch 3 - positive linear_vel moves forward
            else:
                self._was_aligned = False
                linear_vel = 0.0
        
        # Phase 2: Rotate to target direction (or turn towards target position/angle if turn-only)
        elif self._position_reached:
            is_turn_only = self._turn_only_start_time is not None
            
            # Handle go_to_anchor - complete immediately when position reached (no direction alignment)
            if not is_turn_only and self.target_direction is None:
                time_elapsed = time.time() - self._navigation_start_time if self._navigation_start_time else 0
                # Stop and complete when position reached, minimum time elapsed, and velocities are low
                if (time_elapsed >= 0.5 and 
                    distance < self.position_tolerance):
                    self.reached = True
                    self.active = False
                    return (0.0, 0.0)
                # Stop moving, don't turn
                return (0.0, 0.0)
            
            # Turn-only mode: determine desired angle
            if is_turn_only:
                if self.target_direction is not None:
                    # Absolute angle mode
                    desired_angle = self.target_direction
                elif self.target_pos is not None:
                    # Position-based mode
                    diff = self.target_pos - np.array(current_pos[:2])
                    desired_angle = math.atan2(diff[1], diff[0])
                else:
                    # Fallback
                    self.reached = True
                    self.active = False
                    return (0.0, 0.0)
            else:
                # Regular navigation with direction requirement
                desired_angle = self.target_direction
            
            angle_error = self._calculate_angle_error(desired_angle, current_yaw)
            angle_error_abs = abs(angle_error)
            
            angular_vel = np.clip(-K_P_ANGULAR * angle_error, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
            linear_vel = 0.0
            
            # Use turn_only_tolerance for turn-only mode, otherwise use configurable direction_tolerance
            tolerance = self.turn_only_tolerance if is_turn_only else self.direction_tolerance
            
            # Complete when: angle within tolerance, minimum time elapsed, and low velocities
            if angle_error_abs <= tolerance:
                if is_turn_only:  # Turn-only mode
                    time_elapsed = time.time() - self._turn_only_start_time if self._turn_only_start_time else 0
                    if time_elapsed >= 0.5 and abs(angular_vel) < 0.2:
                        self.reached = True
                        self.active = False
                        return (0.0, 0.0)
                else:  # Regular navigation (go_to_anchor with direction - should not reach here now)
                    time_elapsed = time.time() - self._navigation_start_time if self._navigation_start_time else 0
                    if (time_elapsed >= 0.5 and 
                        abs(linear_vel) < 0.1 and abs(angular_vel) < 0.2 and
                        distance < self.position_tolerance):
                        self.reached = True
                        self.active = False
                        return (0.0, 0.0)
        
        # Phase 3: Done
        else:
            self.reached = True
            self.active = False
            return (0.0, 0.0)
        
        return (linear_vel, angular_vel)
    
    def cancel(self):
        """Cancel current navigation."""
        self.active = False
        self.target_pos = None
        self.target_direction = None
        self.reached = False
        self._was_aligned = False
        self._position_reached = False
        self.turn_only_tolerance = math.radians(5.0)
        self.position_tolerance = POSITION_TOLERANCE
        self.direction_tolerance = DIRECTION_TOLERANCE
        self._turn_only_start_time = None
        self._navigation_start_time = None
    
    def is_active(self):
        """Check if navigation is active."""
        return self.active
    
    def has_reached(self):
        """Check if target has been reached."""
        return self.reached
