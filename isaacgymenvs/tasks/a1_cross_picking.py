import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1_with_shovel import A1WithShovel
from isaacgymenvs.tasks.a1_with_shovel_n_camera import A1WithShovelnCamera
from isaacgymenvs.tasks.a1_with_shovel_dagger_passive_joint import A1WithShovelDaggerPassiveJoint



class A1CrossPicking(A1WithShovelnCamera):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.FL_shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame

        self.middle_of_shovel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.middle_of_shovel[:, 0] = 0.2025 + 0.045# 원래 shovel joint pos값(urdf): 0.1805+0.022 조정: +0.04
        self.middle_of_shovel[:, 1] = 0.1308 # 원래: 0.047+0.0838
        self.middle_of_shovel[:, 2] = -0.323 -0.1023 # 원래: -0.123-0.2


    def draw_lines(self, start, end):
        start_np = start.squeeze().cpu().numpy()
        end_np = end.squeeze().cpu().numpy()

        line_vertices = [[start_np[0], start_np[1], start_np[2]],
                         [end_np[0], end_np[1], end_np[2]]]
        num_lines = 1
        line_colors = [1., 0., 0.]

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices, line_colors)


    def compute_reward(self):
        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * -0.000025

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = -0.0009 * action_rate

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = -0.00009 * joint_acc

        base_quat = self.a1_root_states[:, 3:7]
        base_ang_vel = quat_rotate_inverse(base_quat, self.a1_root_states[:, 10:13])
        base_ang_vel_z_square = torch.sqrt(torch.square(base_ang_vel[:, 2]))
        rew_base_ang_vel_z = -0.00002 * base_ang_vel_z_square

        # distance reward
        shovel_bottom_pos = self.rb_states[:, self.FL_shovel_index, 0:3]
        prop_root_pos = self.prop_root_states[:, 0:3]
        shovel_to_prop_dis = torch.norm(prop_root_pos - shovel_bottom_pos, dim=1)
        rew_shovel_to_prop_dis = torch.exp(-shovel_to_prop_dis)
        #print(rew_shovel_to_prop_dis)
        if self.num_envs==1:
            self.draw_lines(shovel_bottom_pos, prop_root_pos)

        # picking policy
        shovel_bottom_contact_forces = torch.norm(self.contact_forces[:, self.shovel_bottom_index, :], dim=1)
        prop_contact_forces = torch.norm(self.contact_forces[:, self.prop_index, :], dim=1)
        contact_differences = torch.sqrt(torch.square(shovel_bottom_contact_forces-prop_contact_forces))
        #rew_prop_is_picked = (shovel_bottom_contact_forces>0.) & (prop_contact_forces>0.) & (contact_differences<0.01)
        rew_prop_is_picked = (shovel_bottom_contact_forces>0.) & (prop_contact_forces>0.) & (contact_differences<0.5)
        #print(rew_prop_is_picked)
        #print(contact_differences)
        total_reward = 50 * rew_prop_is_picked + rew_torque + rew_joint_acc + rew_action_rate + 10 * rew_shovel_to_prop_dis
        #print(total_reward)
        #print(shovel_bottom_contact_forces)
        #print(prop_contact_forces)
        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        # reset agents
        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1000.
        if reset[0]:
            print("trunk reset = ", torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1))
        #reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        #reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        #if reset[0]:
        #    print('thigh reset')
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()
        if reset[0]:
            print("bed reset")
        for env_idx in range(self.num_envs):
            if self.progress_buf[env_idx] > 10./ self.dt:
                if (torch.norm(self.contact_forces[env_idx, self.prop_index, :]) > 0.) & ~rew_prop_is_picked[env_idx]:
                    reset[env_idx] = True
                    print("ddddd")
        time_out = self.progress_buf >= self.max_episode_length/2 - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset



