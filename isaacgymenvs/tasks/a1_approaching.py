
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1_with_shovel import A1WithShovel
from isaacgymenvs.tasks.a1_with_shovel_passive_joint import A1WithShovelPassiveJoint
from isaacgymenvs.tasks.a1_without_shovel import A1WithoutShovel
from isaacgymenvs.tasks.a1_with_shovel_passive_joint_with_camera import A1WithShovelPassiveJointWithCamera
from isaacgymenvs.tasks.a1_with_shovel_throwing import A1WithShovelThrowing

class A1Approaching(A1WithShovelThrowing):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame
        self.commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands[:, 0] = 0.3
        self.feet_air_time = torch.zeros(self.num_envs, self.foot_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.foot_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.prop_num = 0


    def compute_reward(self):
         # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1)

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = action_rate

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = joint_acc

        # angular velocity penalty
        # a1_ang_vel = torch.norm(self.a1_root_states[:, 10:13], dim=1)
        # rew_ang_vel = -0.000003 * a1_ang_vel

        base_quat = self.a1_root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])

        # reward tracking goal vel
        target_pos_rel = self.prop_root_states[:, :3] - self.rb_states[:, self.shovel_index, 0:3]
        #target_pos_rel = self.prop_root_states[:, :3] - self.a1_root_states[:, 0:3]
        target_pos_rel_norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = target_pos_rel / (target_pos_rel_norm + 1e-5)
        #cur_vel = self.rb_states[:, self.shovel_index, 7:9]
        cur_vel = self.a1_root_states[:, 7:9]
        rew_tracking_goal_vel = torch.minimum(torch.sum(target_vec_norm[:, :2] * cur_vel, dim=-1), self.commands[:, 0] + 1e-5)
        rew_into_10cm = (target_pos_rel_norm < 0.12).squeeze()

        # reward cos similarity
        cur_vel_norm = torch.norm(cur_vel, dim=-1, keepdim=True)
        cur_vel_normed = cur_vel / cur_vel_norm
        # 목표 벡터와 현재 속도 벡터의 코사인 유사도 계산
        rew_cos_similarity = torch.sum(target_vec_norm[:, :2] * cur_vel_normed, dim=1)

        # reward tracking yaw
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        _, pitch, yaw = euler_from_quaternion(base_quat)
        rew_tracking_yaw = torch.exp(-torch.abs(target_yaw - yaw))

        # reward ang vel xy
        rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        # reward hip pos
        rew_hip_pos = torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - self.default_dof_pos[:, self.hip_joint_indices]), dim=1)

        # reward_feet_air_time(self)
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.foot_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt

        #total_reward = 3.5 * rew_tracking_goal_vel + 1.0 * rew_tracking_yaw\
        #                     -0.004 * rew_hip_pos\
        #       - 0.000003 * rew_torque -0.001 * rew_action_rate -0.0002 * rew_joint_acc + 1.5 * self.dt * rew_airTime

        total_reward = 3.5 * rew_tracking_goal_vel + 1.0 * rew_tracking_yaw\
                -0.00001 * rew_torque -0.003 * rew_action_rate -0.01 * rew_joint_acc

        rew_approach = (target_pos_rel_norm < 0.05).squeeze().bool()


        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()
        shovel_contact = torch.norm(self.contact_forces[:, self.shovel_index, :], dim=-1)

        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()
        reset = reset | shovel_contact.bool()
        reset = reset | (target_pos_rel_norm < 0.05).squeeze().bool()

        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset


