#!/usr/bin/env python3
"""Record MARL observations only when an interactive macro terminates.

After each macro termination event, 10 observations are collected at 100 ms
intervals and their mean is written as a single CSV row.
"""

import argparse
import csv
import json
import math
import os
import time
from collections import deque

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

OBS_SAMPLES = 10        # observations to average after each macro event
OBS_INTERVAL_S = 0.1   # seconds between samples
SETTLE_DELAY_S = 30.0  # wait this long after macro ends before collecting samples


def yaw_from_ros_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MacroObservationRecorder(Node):
    def __init__(self, world_xml, output_csv, obs_radius):
        super().__init__('macro_observation_recorder')
        self.obs_radius = float(obs_radius)
        self.output_csv = output_csv
        self.layout = get_observation_layout()
        self.model = mujoco.MjModel.from_xml_path(world_xml)
        self.object_positions = {}   # updated live from /sim/object_positions
        self.chopped_status = {'tomato1': 0.0, 'lettuce1': 0.0, 'onion1': 0.0}
        self.robot_state = {
            '/stretch': {'odom': None, 'joints': {}},
            '/stretch2': {'odom': None, 'joints': {}},
        }

        # Each entry: {'robot': str, 'action': str, 'success': bool,
        #              'timestamp': float, 'obs_list': []}
        self._pending: deque = deque()

        self.create_subscription(String, '/sim/object_positions', self._object_positions_callback, 10)

        for namespace in self.robot_state:
            self.create_subscription(Odometry, f'{namespace}/odom', self._odom_callback(namespace), 10)
            self.create_subscription(JointState, f'{namespace}/joint_states', self._joint_callback(namespace), 10)
            self.create_subscription(String, f'{namespace}/marl_chopped_status', self._chopped_callback, 10)
            self.create_subscription(
                String,
                f'{namespace}/marl_macro_terminated',
                self._macro_terminated_callback(namespace),
                10,
            )

        # Timer fires every OBS_INTERVAL_S to accumulate samples for pending events
        self.create_timer(OBS_INTERVAL_S, self._collection_tick)

        self.csv_file = open(self.output_csv, 'a', newline='')
        self.writer = csv.writer(self.csv_file)
        if os.path.getsize(self.output_csv) == 0:
            self.writer.writerow(['time', 'robot', 'macro_action', 'success'] + self.layout)
            self.csv_file.flush()

        print(f'Recording macro-terminal observations to {self.output_csv}')
        print(f'Observation layout ({len(self.layout)}): {self.layout}')
        print(f'Averaging {OBS_SAMPLES} samples at {OBS_INTERVAL_S*1000:.0f} ms intervals per event')

    def destroy_node(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        super().destroy_node()

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _load_object_positions(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        positions = {}
        for object_label, body_name, _ in OBS_OBJECTS:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            positions[object_label] = data.xpos[body_id].copy() if body_id >= 0 else None
        return positions

    def _collection_tick(self):
        """Called every OBS_INTERVAL_S. Adds one sample to each pending collection."""
        now = time.time()
        completed = []
        for entry in self._pending:
            if now < entry['collect_after']:
                continue  # still in settle window

            obs = self._build_obs(entry['robot'])
            if obs is not None:
                entry['obs_list'].append(obs)

            if len(entry['obs_list']) >= OBS_SAMPLES:
                completed.append(entry)

        for entry in completed:
            self._pending.remove(entry)
            mean_obs = np.mean(entry['obs_list'], axis=0)
            row = [f'{entry["timestamp"]:.6f}', entry['robot'],
                   entry['action'], int(entry['success'])]
            row.extend(f'{v:.8f}' for v in mean_obs)
            self.writer.writerow(row)
            self.csv_file.flush()
            print(f'Recorded {entry["robot"]} {entry["action"]} '
                  f'success={int(entry["success"])} '
                  f'(mean of {len(entry["obs_list"])} samples)')

    # ------------------------------------------------------------------ #
    # ROS callbacks
    # ------------------------------------------------------------------ #

    def _object_positions_callback(self, msg):
        try:
            data = json.loads(msg.data)
            for label, xyz in data.items():
                self.object_positions[label] = np.array(xyz, dtype=np.float64)
        except (json.JSONDecodeError, ValueError):
            pass

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

    def _macro_terminated_callback(self, namespace):
        def callback(msg):
            action_name, success = self._parse_macro_event(msg.data)
            now = time.time()
            self._pending.append({
                'robot':         namespace,
                'action':        action_name,
                'success':       success,
                'timestamp':     now,
                'collect_after': now + SETTLE_DELAY_S,
                'obs_list':      [],
            })
            print(f'Macro terminated: {namespace} {action_name} success={int(success)} '
                  f'— settling {SETTLE_DELAY_S:.1f}s then collecting {OBS_SAMPLES} samples…')
        return callback

    @staticmethod
    def _parse_macro_event(data):
        text = data.strip()
        if ':' not in text:
            return text, True
        action_name, success_text = text.split(':', 1)
        try:
            success = float(success_text.strip()) > 0.5
        except ValueError:
            success = success_text.strip().lower() in {'true', 'success', 'ok'}
        return action_name.strip(), success

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


def main():
    global OBS_SAMPLES, SETTLE_DELAY_S

    parser = argparse.ArgumentParser(description='Record observation rows at macro termination events')
    parser.add_argument('--world', default='table_world.xml')
    parser.add_argument('--output', default='macro_terminal_observations.csv')
    parser.add_argument('--obs-radius', type=float, default=DEFAULT_OBS_RADIUS)
    parser.add_argument('--samples', type=int, default=OBS_SAMPLES,
                        help='Number of obs samples to average after each macro event')
    parser.add_argument('--settle', type=float, default=SETTLE_DELAY_S,
                        help='Seconds to wait after macro ends before collecting samples')
    args, ros_args = parser.parse_known_args()

    OBS_SAMPLES = args.samples
    SETTLE_DELAY_S = args.settle

    world_xml = args.world
    if not os.path.isabs(world_xml):
        world_xml = os.path.join(os.path.dirname(__file__), world_xml)

    output_csv = args.output
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(os.getcwd(), output_csv)

    rclpy.init(args=ros_args)
    node = MacroObservationRecorder(world_xml, output_csv, args.obs_radius)
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
