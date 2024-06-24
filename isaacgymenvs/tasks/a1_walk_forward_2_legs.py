import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1 import A1

class A1WalkForward2Legs(A1):

  def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
    self.init = True
    super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    self.rew_scales["y_pos"] = self.cfg["env"]["learn"]["yPosRewardScale"]

    foot_names = [s for s in self.body_names if "foot" in s]
    front_foot_names = [s for s in foot_names if "FR" in s or "FL" in s]
    self.front_foot_indices = torch.zeros(len(front_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
    for i in range(len(front_foot_names)):
      self.front_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], front_foot_names[i])

    self.FL_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_hip")
    self.FR_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FR_hip")
    self.RL_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "RL_hip")
    self.RR_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "RR_hip")

    rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # The buffer has shape (num_rigid_bodies, 13)
    self.gym.refresh_rigid_body_state_tensor(self.sim)

    self.body_state = gymtorch.wrap_tensor(rigid_body_state) # torch.Size([53248, 13]) # 4096 * 13 = 53248
    self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3] # torch.Size([4096, 13, 3])

    self.commands_x[...] = torch.tensor(0.5, device=self.device).squeeze()

  def compute_reward(self, actions):
    self.cos_sim = self.compute_base_angle_reward()
    self.rew_buf[:], self.reset_buf[:] = compute_a1_reward(
        # tensors
        self.root_states,
        self.commands,
        self.torques,
        self.contact_forces,
        self.calf_indices,
        self.progress_buf,
        # Dict
        self.rew_scales,
        # other
        self.trunk_index,
        self.max_episode_length,
        self.front_foot_indices,
        self.thigh_indices,
        self.cos_sim,
    )

  def get_extra_dof_pos(self):
    self.named_extra_joint_angles = self.cfg["env"]["extraJointAngles"]
    self.extra_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

    for i in range(self.cfg["env"]["numActions"]):
      name = self.dof_names[i]
      angle = self.named_extra_joint_angles[name]
      self.extra_dof_pos[:, i] = angle

    self.init = False

    return self.extra_dof_pos

  def compute_base_angle_reward(self):
    FL_hip_pos = self.body_pos[:, self.FL_hip_index, :].squeeze().cpu().numpy()
    FR_hip_pos = self.body_pos[:, self.FR_hip_index, :].squeeze().cpu().numpy()
    RL_hip_pos = self.body_pos[:, self.RL_hip_index, :].squeeze().cpu().numpy()
    RR_hip_pos = self.body_pos[:, self.RR_hip_index, :].squeeze().cpu().numpy()

    midle_of_F_hips = (FL_hip_pos + FR_hip_pos) / 2
    midle_of_R_hips = (RL_hip_pos + RR_hip_pos) / 2

    up_vector = gymapi.Vec3(0., 0., 1.)

    if (self.num_envs == 1):
      np.expand_dims(midle_of_F_hips, axis=0)
      np.expand_dims(midle_of_R_hips, axis=0)

    cos_sim_list = []
    if (self.num_envs > 1):
      for i in range(self.num_envs):
        torso_vector = gymapi.Vec3(midle_of_F_hips[i, 0] - midle_of_R_hips[i, 0],
                                   midle_of_F_hips[i, 1] - midle_of_R_hips[i, 1],
                                   midle_of_F_hips[i, 2] - midle_of_R_hips[i, 2])
        cos_sim = (torso_vector.dot(up_vector)) / (torso_vector.length() * up_vector.length())
        cos_sim_list.append(cos_sim)
    else:
      torso_vector = gymapi.Vec3(midle_of_F_hips[0] - midle_of_R_hips[0],
                                 midle_of_F_hips[1] - midle_of_R_hips[1],
                                 midle_of_F_hips[2] - midle_of_R_hips[2])
      cos_sim = (torso_vector.dot(up_vector)) / (torso_vector.length() * up_vector.length())
      cos_sim_list.append(cos_sim)
    self.cos_sim = torch.tensor(cos_sim_list, device=self.device)
    return self.cos_sim

  def compute_observations(self):
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    super().compute_observations()

  def reset_idx(self, env_ids):
    # Randomization can happen only at reset time, since it can reset actor positions on GPU
    if self.randomize:
      self.apply_randomizations(self.randomization_params)

    if self.init:
      self.extra_dof_pos = self.get_extra_dof_pos()

    positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
    velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

    self.dof_pos[env_ids] = (self.default_dof_pos[env_ids] + self.extra_dof_pos[env_ids])* positions_offset
    self.dof_vel[env_ids] = velocities

    env_ids_int32 = env_ids.to(dtype=torch.int32)

    self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                 gymtorch.unwrap_tensor(self.initial_root_states),
                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    self.gym.set_dof_state_tensor_indexed(self.sim,
                                          gymtorch.unwrap_tensor(self.dof_state),
                                          gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 1


@torch.jit.script
def compute_a1_reward(
    # tensors
    root_states,
    commands,
    torques,
    contact_forces,
    calf_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    trunk_index,
    max_episode_length,
    front_foot_indices,
    thigh_indices,
    cos_sim,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_lin_vel = root_states[:, 7:10]
    base_position = root_states[:, 0:3]

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands - base_lin_vel), dim=1)
    rew_lin_vel_xyz = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]

    # y position tracking reward
    y_pos_error = torch.square(base_position[..., 1])
    rew_y_pos = y_pos_error * rew_scales["y_pos"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # base angle reward
    rew_base = cos_sim * 0.1

    total_reward = rew_lin_vel_xyz + rew_y_pos + rew_torque + rew_base
    total_reward = torch.clip(total_reward, 0., None)

    # reset agents
    reset = torch.norm(contact_forces[:, trunk_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, calf_indices, :], dim=2) > 1., dim=1)
    reset = reset | torch.any(torch.norm(contact_forces[:, thigh_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 2000  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset





