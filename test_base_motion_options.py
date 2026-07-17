#!/usr/bin/env python3
"""Sweep MuJoCo base-drive options for the Stretch simulation.

This script does not use ROS. It loads an XML, applies different wheel/base
drive strategies, and prints how far the robot moved plus whether it tipped.
"""

import argparse
import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np


WHEEL_BASE = 0.3407
WHEEL_RADIUS = 0.05


@dataclass
class Result:
    name: str
    dx: float
    dy: float
    distance: float
    yaw_delta: float
    roll: float
    pitch: float
    z_delta: float
    notes: str


def quat_to_euler_wxyz(quat):
    """Return roll, pitch, yaw from a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def name_to_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f'Missing {obj_type.name}: {name}')
    return obj_id


def actuator_ids(model):
    return {
        'left': name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_wheel_vel'),
        'right': name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_wheel_vel'),
    }


def joint_dof(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return None
    return int(model.jnt_dofadr[joint_id])


def configure_velocity_actuators(model, wheel_ids, kv=None, force_limit=None, ctrl_limit=None):
    for actuator_id in wheel_ids.values():
        if kv is not None:
            model.actuator_gainprm[actuator_id, 0] = kv
            model.actuator_biasprm[actuator_id, 2] = -kv
        if force_limit is not None:
            model.actuator_forcerange[actuator_id] = [-force_limit, force_limit]
            model.actuator_forcelimited[actuator_id] = 1
        if ctrl_limit is not None:
            model.actuator_ctrlrange[actuator_id] = [-ctrl_limit, ctrl_limit]
            model.actuator_ctrllimited[actuator_id] = 1


def set_wheel_friction(model, sliding=None, torsional=None, rolling=None):
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ''
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
        ) or ''
        if 'wheel' not in geom_name.lower() and 'wheel' not in body_name.lower():
            continue
        if sliding is not None:
            model.geom_friction[geom_id, 0] = sliding
        if torsional is not None:
            model.geom_friction[geom_id, 1] = torsional
        if rolling is not None:
            model.geom_friction[geom_id, 2] = rolling


def run_case(base_model, case, duration, linear_x, angular_z):
    model = copy.deepcopy(base_model)
    data = mujoco.MjData(model)
    base_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    wheels = actuator_ids(model)

    if case.get('kv') or case.get('force_limit') or case.get('ctrl_limit'):
        configure_velocity_actuators(
            model,
            wheels,
            kv=case.get('kv'),
            force_limit=case.get('force_limit'),
            ctrl_limit=case.get('ctrl_limit'),
        )
    if case.get('wheel_friction'):
        set_wheel_friction(model, **case['wheel_friction'])

    mujoco.mj_forward(model, data)
    start_pos = data.xpos[base_id].copy()
    start_roll, start_pitch, start_yaw = quat_to_euler_wxyz(data.xquat[base_id])

    v_left = linear_x - angular_z * WHEEL_BASE / 2.0
    v_right = linear_x + angular_z * WHEEL_BASE / 2.0
    omega_left = float(np.clip(v_left / WHEEL_RADIUS, -case.get('omega_limit', 6.0), case.get('omega_limit', 6.0)))
    omega_right = float(np.clip(v_right / WHEEL_RADIUS, -case.get('omega_limit', 6.0), case.get('omega_limit', 6.0)))

    base_x_dof = joint_dof(model, 'base_x')
    base_y_dof = joint_dof(model, 'base_y')
    base_yaw_dof = joint_dof(model, 'base_yaw')

    nsteps = max(1, int(duration / model.opt.timestep))
    for _ in range(nsteps):
        if case['type'] == 'wheel':
            data.ctrl[wheels['left']] = case['left_sign'] * omega_left
            data.ctrl[wheels['right']] = case['right_sign'] * omega_right
        elif case['type'] == 'base_qvel':
            if base_x_dof is not None:
                data.qvel[base_x_dof] = case.get('base_x_vel', 0.0)
            if base_y_dof is not None:
                data.qvel[base_y_dof] = case.get('base_y_vel', 0.0)
            if base_yaw_dof is not None:
                data.qvel[base_yaw_dof] = case.get('base_yaw_vel', 0.0)
        mujoco.mj_step(model, data)

    end_pos = data.xpos[base_id].copy()
    roll, pitch, yaw = quat_to_euler_wxyz(data.xquat[base_id])
    delta = end_pos - start_pos
    yaw_delta = math.atan2(math.sin(yaw - start_yaw), math.cos(yaw - start_yaw))

    notes = []
    if abs(roll - start_roll) > math.radians(5.0) or abs(pitch - start_pitch) > math.radians(5.0):
        notes.append('tips')
    if np.linalg.norm(delta[:2]) < 0.01:
        notes.append('nearly_stationary')
    return Result(
        name=case['name'],
        dx=float(delta[0]),
        dy=float(delta[1]),
        distance=float(np.linalg.norm(delta[:2])),
        yaw_delta=float(yaw_delta),
        roll=float(roll),
        pitch=float(pitch),
        z_delta=float(delta[2]),
        notes=','.join(notes) if notes else 'ok',
    )


def default_cases():
    return [
        {'name': 'wheel_current_code_left_neg_right_pos', 'type': 'wheel', 'left_sign': -1.0, 'right_sign': 1.0},
        {'name': 'wheel_both_positive', 'type': 'wheel', 'left_sign': 1.0, 'right_sign': 1.0},
        {'name': 'wheel_left_pos_right_neg', 'type': 'wheel', 'left_sign': 1.0, 'right_sign': -1.0},
        {'name': 'wheel_both_negative', 'type': 'wheel', 'left_sign': -1.0, 'right_sign': -1.0},
        {
            'name': 'wheel_current_sign_high_force',
            'type': 'wheel',
            'left_sign': -1.0,
            'right_sign': 1.0,
            'force_limit': 500.0,
        },
        {
            'name': 'wheel_current_sign_high_kv_force',
            'type': 'wheel',
            'left_sign': -1.0,
            'right_sign': 1.0,
            'kv': 500.0,
            'force_limit': 500.0,
        },
        {
            'name': 'wheel_current_sign_high_friction',
            'type': 'wheel',
            'left_sign': -1.0,
            'right_sign': 1.0,
            'wheel_friction': {'sliding': 3.0, 'torsional': 0.1, 'rolling': 0.01},
        },
        {'name': 'direct_planar_base_x_qvel', 'type': 'base_qvel', 'base_x_vel': 0.15},
        {'name': 'direct_planar_base_y_qvel', 'type': 'base_qvel', 'base_y_vel': 0.15},
        {'name': 'direct_planar_base_yaw_qvel', 'type': 'base_qvel', 'base_yaw_vel': 0.5},
    ]


def print_results(results):
    header = f"{'case':42} {'dx':>8} {'dy':>8} {'dist':>8} {'yaw_deg':>9} {'roll':>8} {'pitch':>8} {'z':>8} notes"
    print(header)
    print('-' * len(header))
    for result in results:
        print(
            f'{result.name:42} '
            f'{result.dx:8.4f} {result.dy:8.4f} {result.distance:8.4f} '
            f'{math.degrees(result.yaw_delta):9.2f} '
            f'{math.degrees(result.roll):8.2f} {math.degrees(result.pitch):8.2f} '
            f'{result.z_delta:8.4f} {result.notes}'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', default='table_world.xml', help='MuJoCo XML to load')
    parser.add_argument('--duration', type=float, default=2.0, help='Seconds to simulate per case')
    parser.add_argument('--linear', type=float, default=0.15, help='Commanded linear x velocity in m/s')
    parser.add_argument('--angular', type=float, default=0.0, help='Commanded angular z velocity in rad/s')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    print(f'Loaded {args.xml}: nq={model.nq}, nv={model.nv}, nu={model.nu}, timestep={model.opt.timestep}')
    print(f'Command: linear={args.linear:.3f} m/s, angular={args.angular:.3f} rad/s, duration={args.duration:.2f}s')
    print()

    results = [run_case(model, case, args.duration, args.linear, args.angular) for case in default_cases()]
    print_results(results)

    best = max(results, key=lambda item: item.distance)
    print()
    print(f'Best translation: {best.name} moved {best.distance:.4f} m')


if __name__ == '__main__':
    main()
