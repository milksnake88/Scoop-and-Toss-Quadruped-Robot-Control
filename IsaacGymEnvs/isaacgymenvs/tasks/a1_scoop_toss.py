import numpy as np
import os
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1_with_shovel_env import A1WithShovelEnv

class A1ScoopToss(A1WithShovelEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame

    def compute_reward(self):
        # landing reward
        rew_landing = torch.zeros_like(self.landing, dtype=torch.float, device=self.device, requires_grad=False)
        for env_idx in range(self.num_envs):
            if (torch.norm(self.contact_forces[env_idx, self.bed_bottom_index, :]) >0.):
                self.landing[env_idx] += 1
                rew_landing[env_idx] = 100

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1)

        # distance reward
        bed_position = self.get_transformed_position(point_local=self.bed_position_local) 
        prop_position = self.prop_root_states[:, 0:3] 
        bed_prop_distance = torch.norm(prop_position - bed_position, dim=1)
        rew_bed_prop_distance = torch.exp(-bed_prop_distance/0.25)

        # throwing up reward
        box_position_upward = self.prop_root_states[:, 2]
        rew_box_position_upward = box_position_upward

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)

        # joint acceleration penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)

        # picking reward
        shovel_bottom_contact_forces = torch.norm(self.contact_forces[:, self.shovel_bottom_index, :], dim=1)
        prop_contact_forces = torch.norm(self.contact_forces[torch.arange(self.num_envs), self.prop_index, :], dim=1)
        contact_differences = torch.sqrt(torch.square(shovel_bottom_contact_forces-prop_contact_forces))
        rew_prop_is_picked = (shovel_bottom_contact_forces>0.) & (prop_contact_forces>0.) & (contact_differences<0.05)
        picked_ids = rew_prop_is_picked.nonzero(as_tuple=False).squeeze(-1)
        self.picked[picked_ids] += 1
        rew_hip_pos = torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices[2:4]] - self.default_dof_pos[:, self.hip_joint_indices[2:4]]), dim=1)
        # total
        total_reward = rew_landing + (30 * rew_bed_prop_distance * rew_box_position_upward)\
                       -0.001 * rew_action_rate -0.0001 * rew_joint_acc -0.00003 * rew_torque

        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        # reset agents
        contact_force_norms = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1)
        calf_contact_force_norms = torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2)
        thigh_contact_force_norms = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        shovel_contact = torch.norm(self.contact_forces[:, self.shovel_index, :], dim=-1)
        prop_contact = torch.norm(self.contact_forces[:, self.prop_index, :], dim=-1)
        margin = 0.1
  
        reset = (contact_force_norms > 1.) | \
                torch.any(calf_contact_force_norms > 1., dim=1) | \
                torch.any(thigh_contact_force_norms > 1., dim=1) | \
                (bed_contact_force_norm > 120.)

        shovel_landing_reset = self.landing.bool() & shovel_contact.bool()
        reset = reset | shovel_landing_reset

        prop_reset = (self.progress_buf > self.reset_progress_s / self.dt) & \
                    (prop_contact > 0.) & \
                    ~self.landing.bool() & \
                    ~bed_contact_force_norm.bool()

        reset = reset | prop_reset

        # landing 조건 처리
        landing_reset = (self.landing > 5 / self.dt)
        reset = reset | landing_reset

        # randing_cnt 업데이트 (조건에 맞는 인덱스만 증가)
        for env_idx in range(self.num_envs):
            if landing_reset[env_idx]:
                self.randing_cnt[env_idx] += 1

        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset