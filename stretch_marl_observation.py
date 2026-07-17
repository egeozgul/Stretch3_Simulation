"""MARL observation helpers for the Stretch MuJoCo simulation."""

import mujoco
import numpy as np


OBS_OBJECTS = [
    ('tomato1', 'tomato1', True),
    ('lettuce1', 'lettuce1', True),
    ('onion1', 'onion1', True),
    ('plate1', 'plate1', False),
    ('cutting_board1', 'cutting_board1', False),
]

OBS_JOINT_NAMES = [
    'joint_lift',
    'joint_arm_l0',
    'joint_arm_l1',
    'joint_arm_l2',
    'joint_arm_l3',
    'joint_wrist_yaw',
]

DEFAULT_OBS_RADIUS = 1.0


def get_observation_layout():
    layout = ['base_x', 'base_y', 'base_yaw']
    layout.extend(OBS_JOINT_NAMES)
    for object_label, _, has_chopped_status in OBS_OBJECTS:
        layout.extend([
            f'{object_label}_x',
            f'{object_label}_y',
            f'{object_label}_z',
        ])
        if has_chopped_status:
            layout.append(f'{object_label}_chopped')
    layout.append('task_salad')
    return layout


def build_observation(sim_node, robot_label='/stretch', obs_radius=DEFAULT_OBS_RADIUS):
    pos, quat = (
        sim_node._get_r2_robot_pose()
        if robot_label == '/stretch2'
        else sim_node._get_robot_pose()
    )
    yaw = sim_node._yaw_from_quat(quat)
    obs = [float(pos[0]), float(pos[1]), float(yaw)]

    for joint_name in OBS_JOINT_NAMES:
        joint_pos, _ = (
            sim_node._get_r2_joint_state(joint_name)
            if robot_label == '/stretch2'
            else sim_node._get_joint_state(joint_name)
        )
        obs.append(float(joint_pos))

    for object_label, body_name, has_chopped_status in OBS_OBJECTS:
        object_pos = get_body_world_position(sim_node, body_name)
        if object_pos is None:
            obs.extend([0.0, 0.0, 0.0])
            if has_chopped_status:
                obs.append(0.0)
            continue
        distance = float(np.linalg.norm(object_pos[:2] - pos[:2]))
        if distance <= obs_radius:
            obs.extend([float(object_pos[0]), float(object_pos[1]), float(object_pos[2])])
            if has_chopped_status:
                obs.append(get_chopped_status(sim_node, object_label))
        else:
            obs.extend([0.0, 0.0, 0.0])
            if has_chopped_status:
                obs.append(0.0)

    obs.append(1.0)
    return np.asarray(obs, dtype=np.float32)


def get_body_world_position(sim_node, body_name):
    body_id = mujoco.mj_name2id(sim_node.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return sim_node.data.xpos[body_id].copy()


def get_chopped_status(sim_node, object_label):
    chopped_status = getattr(sim_node, 'marl_chopped_status', None)
    if isinstance(chopped_status, dict):
        return float(chopped_status.get(object_label, 0.0))
    return 0.0
