#!/usr/bin/env python3
"""Print MARL-style Stretch observations while the ROS sim is running."""

import argparse
import math
import os
import time

import mujoco
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from stretch_marl_observation import (
    DEFAULT_OBS_RADIUS,
    OBS_JOINT_NAMES,
    OBS_OBJECTS,
    get_observation_layout,
)


def yaw_from_ros_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MarlObservationPrinter(Node):
    def __init__(self, world_xml, obs_radius, rate_hz):
        super().__init__('marl_observation_printer')
        self.obs_radius = float(obs_radius)
        self.period = 1.0 / float(rate_hz)
        self.last_print = 0.0
        self.layout = get_observation_layout()
        self.model = mujoco.MjModel.from_xml_path(world_xml)
        self.object_positions = self._load_object_positions()
        self.chopped_status = {'tomato1': 0.0, 'lettuce1': 0.0, 'onion1': 0.0}
        self.robot_state = {
            '/stretch': {'odom': None, 'joints': {}},
            '/stretch2': {'odom': None, 'joints': {}},
        }

        for ns in self.robot_state:
            self.create_subscription(Odometry, f'{ns}/odom', self._odom_callback(ns), 10)
            self.create_subscription(JointState, f'{ns}/joint_states', self._joint_callback(ns), 10)
            self.create_subscription(String, f'{ns}/marl_chopped_status', self._chopped_callback, 10)

        self.timer = self.create_timer(self.period, self._print_observations)
        print(f'Layout ({len(self.layout)}): {self.layout}')

    def _load_object_positions(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        positions = {}
        for object_label, body_name, _ in OBS_OBJECTS:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                positions[object_label] = data.xpos[body_id].copy()
            else:
                positions[object_label] = None
        return positions

    def _odom_callback(self, namespace):
        def callback(msg):
            self.robot_state[namespace]['odom'] = msg
        return callback

    def _joint_callback(self, namespace):
        def callback(msg):
            joints = self.robot_state[namespace]['joints']
            for name, position in zip(msg.name, msg.position):
                joints[name] = float(position)
        return callback

    def _chopped_callback(self, msg):
        data = msg.data.strip()
        if ':' in data:
            object_name, value_text = data.split(':', 1)
        else:
            object_name, value_text = data, '1.0'
        object_name = object_name.strip()
        if object_name not in self.chopped_status:
            return
        try:
            self.chopped_status[object_name] = 1.0 if float(value_text.strip()) > 0.5 else 0.0
        except ValueError:
            return

    def _build_obs(self, namespace):
        state = self.robot_state[namespace]
        odom = state['odom']
        if odom is None:
            return None

        base_pos = odom.pose.pose.position
        yaw = yaw_from_ros_quat(odom.pose.pose.orientation)
        obs = [float(base_pos.x), float(base_pos.y), float(yaw)]

        joints = state['joints']
        for joint_name in OBS_JOINT_NAMES:
            obs.append(float(joints.get(joint_name, 0.0)))

        base_xy = np.array([base_pos.x, base_pos.y], dtype=np.float64)
        for object_label, _, has_chopped_status in OBS_OBJECTS:
            object_pos = self.object_positions.get(object_label)
            if object_pos is None:
                obs.extend([0.0, 0.0, 0.0])
                if has_chopped_status:
                    obs.append(0.0)
                continue

            distance = float(np.linalg.norm(object_pos[:2] - base_xy))
            if distance <= self.obs_radius:
                obs.extend([float(object_pos[0]), float(object_pos[1]), float(object_pos[2])])
                if has_chopped_status:
                    obs.append(float(self.chopped_status.get(object_label, 0.0)))
            else:
                obs.extend([0.0, 0.0, 0.0])
                if has_chopped_status:
                    obs.append(0.0)

        obs.append(1.0)
        return np.asarray(obs, dtype=np.float32)

    def _print_observations(self):
        now = time.time()
        if now - self.last_print < self.period:
            return
        self.last_print = now

        for namespace in ('/stretch', '/stretch2'):
            obs = self._build_obs(namespace)
            if obs is None:
                print(f'{namespace}: waiting for odom/joint_states...')
                continue
            values = ', '.join(f'{name}={value:.3f}' for name, value in zip(self.layout, obs))
            print(f'{namespace}: [{values}]')
        print('-' * 80)


def main():
    parser = argparse.ArgumentParser(description='Print live MARL observation vectors from ROS topics')
    parser.add_argument('--world', default='table_world.xml')
    parser.add_argument('--obs-radius', type=float, default=DEFAULT_OBS_RADIUS)
    parser.add_argument('--rate', type=float, default=1.0)
    args, ros_args = parser.parse_known_args()

    world_xml = args.world
    if not os.path.isabs(world_xml):
        world_xml = os.path.join(os.path.dirname(__file__), world_xml)

    rclpy.init(args=ros_args)
    node = MarlObservationPrinter(world_xml, args.obs_radius, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
