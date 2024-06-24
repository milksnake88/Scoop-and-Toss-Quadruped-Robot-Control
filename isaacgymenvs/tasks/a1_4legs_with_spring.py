import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1 import A1


class A14legsWithSpring(A1):

  def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
    super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    self.rew_scales["y_pos"] = self.cfg["env"]["learn"]["yPosRewardScale"]

    foot_names = [s for s in self.body_names if "foot" in s]

    front_foot_names = [s for s in foot_names if "FR" in s or "FL" in s]
    self.front_foot_indices = torch.zeros(len(front_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
    for i in range(len(front_foot_names)):
      self.front_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], front_foot_names[i])

    rear_foot_names = [s for s in foot_names if "RR" in s or "RL" in s]
    self.rear_foot_indices = torch.zeros(len(rear_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
    for i in range(len(rear_foot_names)):
      self.rear_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], rear_foot_names[i])

    self.FL_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_hip")
    self.FR_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FR_hip")
    self.RL_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "RL_hip")
    self.RR_hip_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "RR_hip")

    rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # The buffer has shape (num_rigid_bodies, 13)
    self.gym.refresh_rigid_body_state_tensor(self.sim)

    self.body_state = gymtorch.wrap_tensor(rigid_body_state) # torch.Size([53248, 13]) # env_nums(4096) * 13 = 53248
    self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3] # torch.Size([env_nums, 13, 3])

    self.desired_vel_x = 0.2
    self.commands_x[...] = torch.tensor(self.desired_vel_x, device=self.device).squeeze()

    self.env_origin = self.gym.get_env_origin(self.envs[0])

    self.curriculum_buf = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
    self.desired_sec = 10
    self.max_episode_length = self.max_episode_length_s * self.desired_sec
    self.max_step_return = 12
    self.max_return = self.max_step_return * self.max_episode_length

    # spring parameter
    # head(head는 trunk랑 이어져있어서 메뉴얼하게 pos지정해야 함)
    #self.head_pos_before = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    #self.head_pos_before[:, 0] = 0.25 # head_x
    #self.head_pos_before[:, 2] = 0.35 # head_z
    #self.anchor_pos_head_before = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    #self.anchor_pos_head_before[:, 2] = self.anchor_height
    #hip
    self.hip_pos_before = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    self.anchor_pos_hip_before = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

  def reset_idx(self, env_ids):
    # Randomization can happen only at reset time, since it can reset actor positions on GPU
    if self.randomize:
        self.apply_randomizations(self.randomization_params)
        self.create_randomized_ground(self.randomization_params)

    positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
    velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

    self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
    self.dof_vel[env_ids] = velocities

    env_ids_int32 = env_ids.to(dtype=torch.int32)

    self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                 gymtorch.unwrap_tensor(self.initial_root_states),
                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    self.gym.set_dof_state_tensor_indexed(self.sim,
                                          gymtorch.unwrap_tensor(self.dof_state),
                                          gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # self.commands_x[env_ids] = 0. # spring force version 3에서 초반 몇초는 target vel=0으로 설정했을 때 필요

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 1

    self.episode_reward[env_ids] = 0
    self.update_record[env_ids] = 1

    self.anchor_pos_head_before[env_ids, 0] = 0
    self.anchor_pos_head_before[env_ids, 1] = 0
    self.anchor_pos_head_before[env_ids, 2] = 0
    self.anchor_pos_x[env_ids] = 0.
    self.head_pos_before[env_ids, 0] = 0.25
    self.head_pos_before[env_ids, 1] = 0.
    self.head_pos_before[env_ids, 2] = 0.35

    self.init=True

  def pre_physics_step(self, actions):
    super().pre_physics_step(actions)

  def post_physics_step(self):
    super().post_physics_step()

    # apply_head_spring_damper_force
    head_pos = self.get_body_pos_in_global(self.trunk_index, 0.25, 0., 0.)
    anchor_pos_head = torch.zeros_like(head_pos, dtype=torch.float, device=self.device, requires_grad=False)
    anchor_pos_head[:, 2] = 1.6 # anchor_height: 0.56+0.25+0.7(두발로 완전히 일어섰을 때의 trunk 높이+trunk에서 head까지의 길이+얼만큼 높이 띄울까?)

    for env_idx in range(self.num_envs):
      if self.progress_buf[env_idx] < self.max_episode_length_s * 2:
        anchor_pos_head[env_idx, 0] = head_pos[env_idx, 0]
      else:
        self.anchor_pos_x[env_idx] += self.dt * self.desired_vel_x
        anchor_pos_head[env_idx, 0] = self.anchor_pos_x[env_idx]

    self.k = 36 ; rest_length_head = 0.4 ; c_head = 2

    self.apply_spring_damper_forces(head_pos, anchor_pos_head, self.head_pos_before, self.anchor_pos_head_before,
                                    self.k, rest_length_head, c_head)
    self.head_pos_before = head_pos
    self.anchor_pos_head_before = anchor_pos_head

    # apply_hip_spring_damper_force
    hip_pos = self.get_body_pos_in_global(self.trunk_index, -2.0, 0., 0.)
    anchor_pos_hip = hip_pos.clone()

    self.k = 36; rest_length_hip = 0.4; c_hip = 2

    for env_idx in range(self.num_envs):
      if self.progress_buf[env_idx] > self.max_episode_length_s * 2:
        anchor_pos_hip[env_idx, 0] += rest_length_hip
        if self.init==True:
            self.hip_pos_before = hip_pos
            self.anchor_pos_hip_before = anchor_pos_hip
        self.apply_spring_damper_forces(hip_pos, anchor_pos_hip, self.hip_pos_before, self.anchor_pos_hip_before,
                                        self.k, rest_length_hip, c_hip)
        self.init=False
        self.hip_pos_before = hip_pos
        self.anchor_pos_hip_before = anchor_pos_hip

    self.curriculum()

  def compute_observations(self):
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    super().compute_observations()

  def curriculum(self):
    if self.k < 1.2:
      pass
    else:
      self.normalized_return = self.episode_reward / self.max_return
      for env_idx in range(self.num_envs):
          if (self.update_record[env_idx] == 1) and (self.normalized_return[env_idx] > 0.8):
              self.curriculum_buf[env_idx] = 1

      if torch.count_nonzero(self.curriculum_buf) > self.num_envs * 0.8:
          env_ids = self.curriculum_buf.nonzero(as_tuple=False).squeeze(-1)
          self.k -= 1.2
          self.c -= 1/12
          self.curriculum_buf = 0 * self.curriculum_buf
          self.update_record[env_ids] = 0
          print("k is updated to: ", self.k)
          print("c is updated to: ", self.c)

  def get_body_pos_in_global(self, rigid_body_index, length_from_com_x, length_from_com_y, length_from_com_z):
    body_pos = self.body_pos[:, rigid_body_index, :]
    body_pos_in_local = torch.zeros_like(body_pos, dtype=torch.float, device=self.device, requires_grad=False)
    body_pos_in_local[..., 0] = length_from_com_x
    body_pos_in_local[..., 1] = length_from_com_y
    body_pos_in_local[..., 2] = length_from_com_z

    base_pos = self.root_states[:, 0:3].squeeze().cpu().numpy()
    base_ori = self.root_states[:, 3:7].squeeze().cpu().numpy()

    body_pos = torch.zeros_like(body_pos_in_local, dtype=torch.float, device=self.device, requires_grad=False)
    if (self.num_envs == 1):
      base_pos = base_pos.reshape((1, *base_pos.shape))
      base_ori = base_ori.reshape((1, *base_ori.shape))
    for i in range(self.num_envs):
      transform = gymapi.Transform(gymapi.Vec3(base_pos[i][0], base_pos[i][1], base_pos[i][2]),
                                   gymapi.Quat(base_ori[i][0], base_ori[i][1], base_ori[i][2], base_ori[i][3]))
      body_pos_in_global = transform.transform_point(gymapi.Vec3(body_pos_in_local[i][0], body_pos_in_local[i][1], body_pos_in_local[i][2]))
      body_pos[i] = torch.tensor([[body_pos_in_global.x, body_pos_in_global.y, body_pos_in_global.z]], dtype=torch.float, device=self.device, requires_grad=False)

    return body_pos

  def apply_spring_damper_forces(self, body_pos, anchor_pos, body_pos_before, anchor_pos_before, k, rest_length, c):
    delta_x = body_pos - anchor_pos
    current_length = torch.norm(delta_x, dim=1).unsqueeze(1)
    x = current_length - rest_length

    # 용수철 힘의 방향, 단위벡터
    dir_unit = delta_x / current_length

    # spring force
    spring_force = -1 * k * x * dir_unit

    # damper force
    body_vel = (body_pos - body_pos_before) / self.dt
    anchor_vel = (anchor_pos - anchor_pos_before) / self.dt

    delta_v = body_vel - anchor_vel
    temp = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
    for i in range(self.num_envs):
      temp[i] = torch.dot(delta_v[i], delta_x[i])

    damper_force = -1 * c * (temp/current_length) * dir_unit

    # mass spring damper model force
    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
    forces[:, self.trunk_index, :] = spring_force + damper_force

    force_positions = self.body_pos.clone()
    force_positions[:, self.trunk_index, :] = body_pos

    self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

    # spring 그리기
    body_pos_numpy = body_pos.squeeze().cpu().numpy()
    anchor_pos_numpy = anchor_pos.squeeze().cpu().numpy()
    if (self.num_envs == 1):
      body_pos_numpy = body_pos_numpy.reshape((1, *body_pos_numpy.shape))
      anchor_pos_numpy = anchor_pos_numpy.reshape((1, *anchor_pos_numpy.shape))
    line_v1 = [[body_pos_numpy[0][0], body_pos_numpy[0][1], body_pos_numpy[0][2]],
               [anchor_pos_numpy[0][0], anchor_pos_numpy[0][1], anchor_pos_numpy[0][2]]]
    self.gym.clear_lines(self.viewer)
    self.gym.add_lines(self.viewer, self.envs[0], 1, line_v1, [1., 0., 0.])

    '''
    # force 그리기
    head_pos_numpy = head_pos.squeeze().cpu().numpy()
    temp = spring_force + damper_force
    forces_numpy = temp.squeeze().cpu().numpy()
    if (self.num_envs == 1):
      head_pos_numpy = head_pos_numpy.reshape((1, *head_pos_numpy.shape))
      forces_numpy = forces_numpy.reshape((1, *forces_numpy.shape))
    line_v1 = [[head_pos_numpy[0][0], head_pos_numpy[0][1], head_pos_numpy[0][2]],
               [forces_numpy[0][0], forces_numpy[0][1], forces_numpy[0][2]]]
    self.gym.clear_lines(self.viewer)
    self.gym.add_lines(self.viewer, self.envs[0], 1, line_v1, [1., 0., 0.])
    '''

  def compute_base_angle_reward(self):
    FL_hip_pos = self.body_pos[:, self.FL_hip_index, :].squeeze().cpu().numpy()
    FR_hip_pos = self.body_pos[:, self.FR_hip_index, :].squeeze().cpu().numpy()
    RL_hip_pos = self.body_pos[:, self.RL_hip_index, :].squeeze().cpu().numpy()
    RR_hip_pos = self.body_pos[:, self.RR_hip_index, :].squeeze().cpu().numpy()

    middle_of_F_hips = (FL_hip_pos + FR_hip_pos) / 2
    middle_of_R_hips = (RL_hip_pos + RR_hip_pos) / 2

    up_vector = gymapi.Vec3(0., 0., 1.)
    cos_sim_list = []

    if (self.num_envs == 1):
      middle_of_F_hips = middle_of_F_hips.reshape((1, *middle_of_F_hips.shape))
      middle_of_R_hips = middle_of_R_hips.reshape((1, *middle_of_R_hips.shape))
    for i in range(self.num_envs):
      trunk_vector = gymapi.Vec3(middle_of_F_hips[i, 0] - middle_of_R_hips[i, 0],
                                 middle_of_F_hips[i, 1] - middle_of_R_hips[i, 1],
                                 middle_of_F_hips[i, 2] - middle_of_R_hips[i, 2])
      cos_sim = (trunk_vector.dot(up_vector)) / (trunk_vector.length() * up_vector.length())
      # print(up_vector.length())
      cos_sim_list.append(cos_sim)
    self.cos_sim = torch.tensor(cos_sim_list, device=self.device)

    return self.cos_sim

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
        self.rear_foot_indices,
        self.thigh_indices,
        self.max_episode_length_s,
        self.cos_sim,
        self.num_envs,
    )
    self.episode_reward += self.rew_buf

"""
  def apply_randomizations(self, dr_params):
        if "ground_friction" in dr_params and do_nonenv_randomize:
          dist = dr_params["ground_friction"]["distribution"]
          op_type = dr_params["ground_friction"]["operation"]
          if dist == "uniform":
            if op_type == "scaling":
              lo, hi = dr_params["ground_friction"]["range"]
              x = np.random.uniform(low=lo, high=hi, size=1)
              self.plane_static_friction *= x
              self.plane_dynamic_friction *= x
          self._create_ground_plane()
"""


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
    rear_foot_indices,
    thigh_indices,
    max_episode_length_s,
    cos_sim,
    num_envs,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, Tensor, Tensor, Tensor, int, Tensor, int) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_lin_vel = root_states[:, 7:10]
    base_position = root_states[:, 0:3]

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands - base_lin_vel), dim=1)

    # 0) 기존
    #rew_lin_vel_xyz = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_lin_vel_xyz = torch.exp(-lin_vel_error)

    # y position tracking reward
    y_pos_error = torch.square(base_position[..., 1])
    rew_y_pos = torch.exp(-y_pos_error)

    # torque penalty
    # rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]
    total_torque = torch.sum(torch.square(torques), dim=1)
    rew_torque = torch.exp(-total_torque)

    # base's raised angle reward
    cos_dis = 1-cos_sim
    rew_base_angle = torch.exp(-cos_dis)


    # 5) forefeet_noncontact
    front_feet_noncontact = 0 * rew_lin_vel_xyz
    for env_idx in range(num_envs):
      if not (torch.any(torch.norm(contact_forces[env_idx, front_foot_indices, :], dim=1) > 1., dim=0)):
        front_feet_noncontact[env_idx] = 1.
    rew_front_feet_noncontact = front_feet_noncontact

    #total_reward = rew_lin_vel_xyz + rew_y_pos + rew_torque + rew_base_angle
    total_reward = 2 * rew_lin_vel_xyz * rew_base_angle + rew_y_pos + rew_torque + 8 * rew_front_feet_noncontact
    total_reward = torch.clip(total_reward, 0., None)

    # reset agents
    reset = torch.norm(contact_forces[:, trunk_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, calf_indices, :], dim=2) > 1., dim=1)
    reset = reset | torch.any(torch.norm(contact_forces[:, thigh_indices, :], dim=2) > 1., dim=1)

    for env_idx in range(num_envs):
      if episode_lengths[env_idx] > max_episode_length_s * 3:
        reset[env_idx] = reset[env_idx] | torch.any(torch.norm(contact_forces[env_idx, front_foot_indices, :], dim=1) > 1., dim=0)

    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset




