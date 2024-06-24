import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1 import A1

from tensorboardX import SummaryWriter

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

    #self.writer = SummaryWriter()

    self.desired_vel_x = 0.1
    self.commands_x[...] = torch.tensor(self.desired_vel_x, device=self.device).squeeze()

    self.env_origin = self.gym.get_env_origin(self.envs[0])

    self.RL_hip_pos = self.body_pos[:, self.RL_hip_index, :]
    self.RR_hip_pos = self.body_pos[:, self.RR_hip_index, :]
    self.middle_of_R_hips = (self.RL_hip_pos + self.RR_hip_pos) / 2.0

    self.curriculum_buf = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
    self.desired_sec = 10
    self.max_episode_length = self.max_episode_length_s * self.desired_sec
    self.max_step_return = 12
    self.max_return = self.max_step_return * self.max_episode_length

    self.head_pos_in_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    self.head_pos_in_local[..., 0] = 0.25

    self.anchor_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    self.anchor_pos[:, 2] = 1.6 # 0.56+0.25+0.7(두발로 완전히 일어섰을 때의 trunk 높이+trunk에서 head까지의 길이+얼만큼 높이 띄울까?)
                                # reset에서도 수정해야함! -> 나중에 고치기
    self.k = 0
    self.rest_length = 0.4
    self.c = 0

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
        self.middle_of_R_hips,
    )
    self.episode_reward += self.rew_buf

  def reset_idx(self, env_ids):
    # Randomization can happen only at reset time, since it can reset actor positions on GPU
    if self.randomize:
        self.apply_randomizations(self.randomization_params)

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

    self.head_pos_before[env_ids, :] = torch.tensor([0.25, 0, 0.35], dtype=torch.float, device=self.device, requires_grad=False) # initial head_pos
    self.anchor_pos_before[env_ids, :] = torch.tensor([0.25, 0, 1.6], dtype=torch.float, device=self.device, requires_grad=False) # initial anchor_pos


  def pre_physics_step(self, actions):
    super().pre_physics_step(actions)

  def post_physics_step(self):
    super().post_physics_step()
    # self.set_commands()
    self.apply_spring_damper_forces()
    self.curriculum()

  def compute_observations(self):
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    super().compute_observations()

  def curriculum(self):
    if self.k < -22.0:
      pass
    else:
      self.normalized_return = self.episode_reward / self.max_return
      for env_idx in range(self.num_envs):
          if (self.update_record[env_idx] == 1) and (self.normalized_return[env_idx] > 0.85):
              self.curriculum_buf[env_idx] = 1

      if torch.count_nonzero(self.curriculum_buf) > self.num_envs * 0.85:
          env_ids = self.curriculum_buf.nonzero(as_tuple=False).squeeze(-1)
          self.k -= 2.0
          self.c -= 2/19
          self.curriculum_buf = 0 * self.curriculum_buf
          self.update_record[env_ids] = 0
          print("k is updated to: ", self.k)
          print("c is updated to: ", self.c)

  # spring force version 3에서 초반 몇초는 target vel=0으로 설정했을 때 필요
  """
  def set_commands(self):
    for env_idx in range(self.num_envs):
      if self.progress_buf[env_idx] > self.max_episode_length_s * 1.5:
        self.commands_x[env_idx] = self.desired_vel_x
  """

  # spring force version 2 : 스프링 힘이 위쪽으로만
  '''
  def apply_spring_damper_forces(self):
    head_pos = self.get_head_position()
    head_pos_z = head_pos[:, 2]

    anchor_to_head = head_pos_z - self.anchor_height
    current_length = torch.abs(anchor_to_head)
    force_dir_unit = anchor_to_head / current_length

    x = current_length - self.rest_length

    # spring force
    spring_force = -1*self.k*x*force_dir_unit # torch.size([env_nums])

    # damper force
    x_prime = (x-self.x_before) / self.dt
    damper_force = -1*self.c*x_prime
    self.x_before = x

    # mass spring damper model force
    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
    forces[:, self.trunk_index, 2] = spring_force + damper_force

    force_positions = self.body_pos.clone()
    force_positions[:, self.trunk_index, :] = head_pos
    self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

    # spring그리기
    head_pos_numpy = head_pos.squeeze().cpu().numpy()
    if (self.num_envs == 1):
      head_pos_numpy = head_pos_numpy.reshape((1, *head_pos_numpy.shape))
    line_v1 = [[head_pos_numpy[0][0], head_pos_numpy[0][1], head_pos_numpy[0][2]],
               [head_pos_numpy[0][0], head_pos_numpy[0][1], self.anchor_height]]
    self.gym.clear_lines(self.viewer)
    self.gym.add_lines(self.viewer, self.envs[0], 1, line_v1, [0., 0., 1.])
  '''

  def get_spring_position(self):
    base_pos = self.root_states[:, 0:3].squeeze().cpu().numpy()
    base_ori = self.root_states[:, 3:7].squeeze().cpu().numpy()

    head_pos = torch.zeros_like(self.head_pos_in_local, dtype=torch.float, device=self.device, requires_grad=False)
    if (self.num_envs == 1):
      base_pos = base_pos.reshape((1, *base_pos.shape))
      base_ori = base_ori.reshape((1, *base_ori.shape))
    for i in range(self.num_envs):
      transform = gymapi.Transform(gymapi.Vec3(base_pos[i][0], base_pos[i][1], base_pos[i][2]),
                                   gymapi.Quat(base_ori[i][0], base_ori[i][1], base_ori[i][2], base_ori[i][3]))
      head_pos_in_global = transform.transform_point(gymapi.Vec3(self.head_pos_in_local[i][0], self.head_pos_in_local[i][1], self.head_pos_in_local[i][2]))
      head_pos[i] = torch.tensor([[head_pos_in_global.x, head_pos_in_global.y, head_pos_in_global.z]],  dtype=torch.float, device=self.device, requires_grad=False)

    for env_idx in range(self.num_envs):
      if self.progress_buf[env_idx] < self.max_episode_length_s * 2:
        self.anchor_pos[env_idx, 0] = head_pos[env_idx, 0]
      else:
        self.anchor_pos[env_idx, 0] += self.dt * self.desired_vel_x
    #print("head_pos=", head_pos)
    #print("anchor_pos=", self.anchor_pos)
    return head_pos, self.anchor_pos

  # : spring force version 3: 위로만 당기는 것만 아니라 스프링 위쪽 점이 target vel로 앞으로 움직이도록 (이렇게하면 일어서는데 시간이 걸리니까 초반 몇초는 target vel=0)
  def apply_spring_damper_forces(self):
    head_pos, anchor_pos = self.get_spring_position()

    delta_x = head_pos - anchor_pos
    current_length = torch.norm(delta_x, dim=1).unsqueeze(1)
    x = current_length - self.rest_length

    # 용수철 힘의 방향, 단위벡터
    dir_unit = delta_x / current_length

    # spring force
    spring_force = -1 * self.k * x * dir_unit

    # damper force
    head_vel = (head_pos - self.head_pos_before) / self.dt
    self.head_pos_before = head_pos
    anchor_vel = (anchor_pos - self.anchor_pos_before) / self.dt
    self.anchor_pos_before = anchor_pos.clone()

    delta_v = head_vel - anchor_vel
    temp = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
    for i in range(self.num_envs):
      temp[i] = torch.dot(delta_v[i], delta_x[i])

    damper_force = -1 * self.c * (temp/current_length) * dir_unit

    # mass spring damper model force
    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
    forces[:, self.trunk_index, :] = spring_force + damper_force
    #print("spring forces= ", spring_force+damper_force)
    force_positions = self.body_pos.clone()
    force_positions[:, self.trunk_index, :] = head_pos
    #print("body_position= ", self.body_pos)
    self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

    # spring 그리기

    head_pos_numpy = head_pos.squeeze().cpu().numpy()
    anchor_pos_numpy = anchor_pos.squeeze().cpu().numpy()
    if (self.num_envs == 1):
      head_pos_numpy = head_pos_numpy.reshape((1, *head_pos_numpy.shape))
      anchor_pos_numpy = anchor_pos_numpy.reshape((1, *anchor_pos_numpy.shape))
    line_v1 = [[head_pos_numpy[0][0], head_pos_numpy[0][1], head_pos_numpy[0][2]],
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
    middle_of_R_hips,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, Tensor, Tensor, Tensor, int, Tensor, int, Tensor) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_lin_vel = root_states[:, 7:10]
    base_position = root_states[:, 0:3]

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands - base_lin_vel), dim=1)


    # 0) 기존
    #rew_lin_vel_xyz = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_lin_vel_xyz = torch.exp(-lin_vel_error/0.5)

    '''
    # 1) 앞발의 컨택이 있는 경우에는 0(네 발로 걷는 거 방지), 그렇지 않은 경우에는 원래대로 계산하도록
    for env_idx in range(num_envs):
      if (torch.any(torch.norm(contact_forces[env_idx, front_foot_indices, :], dim=1) > 1., dim=0)):
        rew_lin_vel_xyz[env_idx] = 0.

    # 2) base의 높이가 일정 이하일 경우에는 0, 일정높이보다 높은 경우에는 원래대로 계산하도록
    for env_idx in range(num_envs):
      if (base_position[env_idx, 2] < 0.335):
        rew_lin_vel_xyz[env_idx] = 0.
    '''

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

    # base height reward
    rew_base_height = base_position[..., 2] * 0.15

    '''
    # 3) 앞발 contact penalty추가
    front_foot_contact_force = torch.sum(torch.norm(contact_forces[:, front_foot_indices, :], dim=2), dim=1)
    contact_penalty = 0. * front_foot_contact_force
    for env_idx in range(num_envs):
      if episode_lengths[env_idx] > max_episode_length_s * 5:
        contact_penalty[env_idx] = -0.1 * front_foot_contact_force[env_idx]
    '''

    '''
    # 4) 뒷발 contact penalty추가
    rear_feet_contact = torch.all(torch.norm(contact_forces[:, rear_foot_indices, :], dim=2) > 1., dim=1)
    rear_feet_contact_force = torch.sum(torch.norm(contact_forces[:, rear_foot_indices, :], dim=2), dim=1) * rear_feet_contact
    rear_feet_contact_penalty = -0.05 * rear_feet_contact_force

    '''

    # 5) forefeet_noncontact
    front_feet_noncontact = 0 * rew_lin_vel_xyz
    for env_idx in range(num_envs):
      if not (torch.any(torch.norm(contact_forces[env_idx, front_foot_indices, :], dim=1) > 1., dim=0)):
        front_feet_noncontact[env_idx] = 1.
    rew_front_feet_noncontact = front_feet_noncontact

    # 뒷다리 hip joint
    rew_hip_height = middle_of_R_hips[..., 2] * 0.15

    #total_reward = rew_lin_vel_xyz + rew_y_pos + rew_torque + rew_base_angle
    total_reward = 5 * rew_lin_vel_xyz * rew_base_angle + rew_y_pos + rew_torque + 5 * rew_front_feet_noncontact
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





