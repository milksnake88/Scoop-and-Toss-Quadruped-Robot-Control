import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1 import A1

class A1Custom(A1):
    # A1 with shoulder contact
    '''
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        #self.dof_vel_before = torch.zeros_like(self.dof_vel, dtype=torch.float, device=self.device, requires_grad=False)
        #self.rew_scales["acceleration"] = self.cfg["env"]["learn"]["accelerationRewardScale"]

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_a1_reward(
            # tensors
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
            self.dof_vel,
            self.dof_vel_before,
        )
    '''
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/a1_description/urdf/a1.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        a1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(a1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(a1_asset)
        self.dof_names = self.gym.get_asset_dof_names(a1_asset)

        extremity_name = "calf" if asset_options.collapse_fixed_joints else "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        front_feet_names = [s for s in feet_names if "FR" in s or "FL" in s]
        self.front_feet_indices = torch.zeros(len(front_feet_names), dtype=torch.long, device=self.device, requires_grad=False)

        knee_names = [s for s in body_names if "thigh" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)

        shoulder_names = [s for s in body_names if "hip" in s]
        self.shoulder_indices = torch.zeros(len(shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)

        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(a1_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.a1_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(env_ptr, a1_asset, start_pose, "a1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], knee_names[i])
        for i in range(len(front_feet_names)):
            self.front_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], front_feet_names[i])
        for i in range(len(shoulder_names)):
            self.shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], shoulder_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "base")

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_a1_reward(
            # tensors
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
            self.shoulder_indices,
        )
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_a1_reward(
    # tensors
    root_states,
    commands,
    torques,
    contact_forces,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length,
    shoulder_indices,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, Tensor) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
    total_reward = torch.clip(total_reward, 0., None)

    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    reset = reset | torch.any(torch.norm(contact_forces[:, shoulder_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset

'''
@torch.jit.script
def compute_a1_reward(
    # tensors
    root_states,
    commands,
    torques,
    contact_forces,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length,
    dof_vel,
    dof_vel_before,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # acceleration penalty
    acceleration = dof_vel - dof_vel_before
    rew_acceleration = torch.sum(torch.square(acceleration), dim=1) * rew_scales["acceleration"]
    dof_vel_before = dof_vel

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_acceleration
    total_reward = torch.clip(total_reward, 0., None)

    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset
'''



