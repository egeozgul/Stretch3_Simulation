import numpy as np
import mujoco
import math
from navigation import NavigationController
JOINT_LIMITS = {
    'joint_lift': (0.0, 1.1),
    'joint_arm_l3': (0.0, 0.13),
    'joint_arm_l2': (0.0, 0.13),
    'joint_arm_l1': (0.0, 0.13),
    'joint_arm_l0': (0.0, 0.13),
    'joint_wrist_yaw': (-1.39, 4.42),
}


class IKSolver:
    def __init__(self, model, data, logger=None, name_prefix=''):
        self.model = model
        self.data = data
        self.nav_controller = NavigationController()
        self.logger=logger
        self.name_prefix = name_prefix

    def compute_ik(self, target_pos, max_iter=200, tol=0.015):

        target_pos = np.array(target_pos)
        joint_names = [
            'joint_lift',
            'joint_arm_l3',
            'joint_arm_l2',
            'joint_arm_l1',
            'joint_arm_l0',
            'joint_wrist_yaw',
        ]
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.name_prefix + name)
            for name in joint_names
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
            if self.logger:
                self.logger.error(f'Missing IK joints: {missing}')
            return False, np.zeros(len(joint_names))

        qpos_indices = [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids]
        qvel_indices = [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids]
        joint_limits = [JOINT_LIMITS[name] for name in joint_names]

        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos.copy()

    #    print("\n========== IK START ==========")
      #  print("Initial qpos:")
     #   print(f"  lift        : {ik_data.qpos[9]}")
      #  print(f"  arm joints  : {ik_data.qpos[10:14]}")
       # print(f"  wrist yaw   : {ik_data.qpos[14]}")
     #   print(f"Target pos    : {target_pos}")
       # print("================================\n")

        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.name_prefix + 'ee_site')
        if ee_site_id < 0:
            if self.logger:
                self.logger.error(f'Missing IK end-effector site: {self.name_prefix}ee_site')
            return False, np.zeros(len(joint_names))

        for i in range(max_iter):
            mujoco.mj_forward(self.model, ik_data)

            # Get end-effector position from SITE
            ee_pos = ik_data.site_xpos[ee_site_id].copy()
            error = target_pos - ee_pos
            error_norm = np.linalg.norm(error)

          #  print(f"EE position (ee_site)        : {ee_pos}")
           # print(f"Target position              : {target_pos}")
           # print(f"Error vector                 : {error}")
           # print(f"Error norm                   : {error_norm}")

            if error_norm < tol:
                if self.logger:
                    self.logger.info(f'IK converged in {i} iterations (error={error_norm:.3f}m)')
                
                return True, np.array([ik_data.qpos[idx] for idx in qpos_indices])
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, ee_site_id)
            
            J = np.column_stack([jacp[:3, idx] for idx in qvel_indices])

            #print("Jacobian J:")
            #print(J)

           
            damping = 0.01
            JJt = J @ J.T
            inv_term = np.linalg.inv(JJt + damping * np.eye(3))
            J_pinv = J.T @ inv_term
            dq = 0.25 * J_pinv @ error

            for qpos_idx, delta, (min_val, max_val) in zip(qpos_indices, dq, joint_limits):
                ik_data.qpos[qpos_idx] = np.clip(ik_data.qpos[qpos_idx] + delta, min_val, max_val)

        mujoco.mj_forward(self.model, ik_data)
        final_error = np.linalg.norm(target_pos - ik_data.site_xpos[ee_site_id])
        if self.logger:
            self.logger.warn(f'IK failed after {max_iter} iterations (error={final_error:.3f}m)')

        return False, np.array([ik_data.qpos[idx] for idx in qpos_indices])
    
    def align_with_target(self,pos,quat,tomato_name):

        current_yaw = self.nav_controller._quaternion_to_yaw(quat)
        '''fruit_body_id = mujoco.mj_name2id(
            self.model, 
            mujoco.mjtObj.mjOBJ_BODY, 
            'tomato2'
        )
        fruit_pos = self.data.xpos[fruit_body_id].copy()'''
        # Get the site instead of body
        fruit_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  f'{tomato_name}_site')

        if fruit_site_id >= 0:
            fruit_pos = self.data.site_xpos[fruit_site_id].copy()
            #self.get_logger().info(f'Actual fruit site position: {fruit_pos}')
        else:
            
            fruit_pos = np.array([-1.0, 5.54, 1.05])
        dx = fruit_pos[0] - pos[0]
        dy = fruit_pos[1] - pos[1]
        if self.logger:
            self.logger.info(f"dx: {dx}, dy: {dy}")
        desired_yaw = math.atan2(dy, dx)
        yaw_diff = (desired_yaw+1.5708) - current_yaw
        yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
        return yaw_diff, current_yaw, desired_yaw
