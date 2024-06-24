import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.a1_with_shovel_n_camera import A1WithShovelnCamera


class A1Locomotion(A1WithShovelnCamera):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.FL_foot_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_foot")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.FL_calf_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_calf")

        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.057 # bed joint position in base frame
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame
        self.forward_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.forward_vector[:, 0] = 1.
        self.middle_of_shovel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.middle_of_shovel[:, 0] = 0.04 + 0.022
        self.middle_of_shovel[:, 2] = -0.08 - 0.123
        self.middle_of_shovel_norm = torch.norm(self.middle_of_shovel, dim=1)
        self.shovel_forward_vector = self.middle_of_shovel / self.middle_of_shovel_norm.unsqueeze(1)
        #print(self.shovel_forward_vector)

        self.FL_foot_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_foot")
        self.FR_foot_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FR_foot")


    def draw_lines(self, start, end):
        # type: (Tensor, Tensor) -> None

        start_np = start.squeeze().cpu().numpy()
        end_np = end.squeeze().cpu().numpy()

        line_vertices = [[start_np[0], start_np[1], start_np[2]],
                         [end_np[0], end_np[1], end_np[2]]]
        num_lines = 1
        line_colors = [1., 0., 0.]

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices, line_colors)


    def get_global_position(self, base_states, point_local=None, vector_local=None):
        base_pos = base_states[:, 0:3].squeeze().cpu().numpy()
        base_ori = base_states[:, 3:7].squeeze().cpu().numpy()

        global_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        if (self.num_envs == 1):
            base_pos = base_pos.reshape((1, *base_pos.shape))
            base_ori = base_ori.reshape((1, *base_ori.shape))
        for i in range(self.num_envs):
            transform = gymapi.Transform(gymapi.Vec3(base_pos[i][0], base_pos[i][1], base_pos[i][2]),
                                         gymapi.Quat(base_ori[i][0], base_ori[i][1], base_ori[i][2], base_ori[i][3]))
            if point_local is not None:
                transformed_position = transform.transform_point(gymapi.Vec3(point_local[i][0], point_local[i][1], point_local[i][2]))
                global_position[i] = torch.tensor([[transformed_position.x, transformed_position.y, transformed_position.z]],  dtype=torch.float, device=self.device, requires_grad=False)
            if vector_local is not None:
                transformed_vector = transform.transform_vector(gymapi.Vec3(vector_local[i][0], vector_local[i][1], vector_local[i][2]))
                global_position[i] = torch.tensor([[transformed_vector.x, transformed_vector.y, transformed_vector.z]],  dtype=torch.float, device=self.device, requires_grad=False)

        return global_position

    def compute_reward(self):
        # 일직선 reward
        FL_foot_pos = self.rb_states[:, self.FL_foot_index, 0:3] # z값은 신경안씀 (안써도 되겠지.....?)
        FR_foot_pos = self.rb_states[:, self.FR_foot_index, 0:3]
        prop_root_pos = self.prop_root_states[:, 0:3]
        a1_root_pos = self.a1_root_states[:, 0:3]

        FL_foot_to_FR_foot_vec = (FR_foot_pos-FL_foot_pos)/torch.norm((FR_foot_pos-FL_foot_pos), dim=1).unsqueeze(1)
        FL_foot_to_a1_root_vec = (a1_root_pos-FL_foot_pos)/torch.norm(((a1_root_pos-FL_foot_pos)), dim=1).unsqueeze(1)
        prop_root_to_FL_foot_vec = (FL_foot_pos-prop_root_pos)/torch.norm((FL_foot_pos-prop_root_pos), dim=1).unsqueeze(1)
        prop_root_to_FR_foot_vec = (FR_foot_pos-prop_root_pos)/torch.norm((FR_foot_pos-prop_root_pos), dim=1).unsqueeze(1)

        FL_foot_to_FR_foot_vec[:, 2] = 0
        FL_foot_to_a1_root_vec[:, 2] = 0
        prop_root_to_FL_foot_vec[:, 2] = 0
        prop_root_to_FR_foot_vec[:, 2] = 0

        cos_of_foots_n_base = torch.einsum('ij,ij->i', FL_foot_to_FR_foot_vec, FL_foot_to_a1_root_vec)
        cos_of_foots_n_prop = torch.einsum('ij,ij->i', prop_root_to_FL_foot_vec, prop_root_to_FR_foot_vec)

        rew_cos_of_foots_n_base = -cos_of_foots_n_base + 1
        rew_cos_of_foots_n_prop = -cos_of_foots_n_prop + 1

        #print("1", rew_cos_of_foots_n_base)
        #print("2", rew_cos_of_foots_n_prop)
        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * -0.000025

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = -0.0009 * action_rate

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = -0.00009 * joint_acc


        # distance reward
        a1_root_position = self.a1_root_states[:, 0:3]
        prop_root_position = self.prop_root_states[:, 0:3]
        a1_prop_distance = torch.norm(a1_root_position-prop_root_position, dim=1)
        rew_a1_prop_distance = torch.exp(-a1_prop_distance)

        """
        # a1 facing to prop reward

        a1_prop_vec = (prop_root_position-a1_root_position)/a1_prop_distance.unsqueeze(1)
        #print(a1_prop_vec)
        a1_facing_vec = self.get_global_position(self.a1_root_states, vector_local=self.forward_vector)
        #print(a1_facing_vec)
        cos_of_direction = torch.einsum('ij,ij->i', a1_prop_vec, a1_facing_vec)
        rew_facing = cos_of_direction + 1
        #print("cos", rew_facing)
        """
        shovel_states = self.rb_states[:, self.shovel_index, :]
        shovel_facing_vec = self.get_global_position(shovel_states, vector_local=self.shovel_forward_vector)

        # shovel facing
        shovel_pos = shovel_states[:, 0:3]
        shovel_to_prop_dis = torch.norm(prop_root_pos - shovel_pos, dim=1)
        shovel_to_prop_vec = (prop_root_pos - shovel_pos) / shovel_to_prop_dis.unsqueeze(1)

        shovel_to_prop_vec[:, 2] = 0
        shovel_facing_vec[:, 2] = 0

        shovel_pos[:, 2] = 0

        FL_calf_states = self.rb_states[:, self.FL_calf_index, :]
        shovel_middle_pos = self.get_global_position(FL_calf_states, point_local=self.middle_of_shovel)
        shovel_prop_distance = torch.norm(shovel_middle_pos-prop_root_position, dim=1)
        shovel_prop_vec = (prop_root_position-shovel_middle_pos) / shovel_prop_distance
        #print("norm=", torch.norm(shovel_prop_vec, dim=1))
        self.draw_lines(shovel_middle_pos, shovel_middle_pos+5*shovel_prop_vec)

        #middle_of_shovel = self.get_global_position(shovel_states, point_local=self.middle_of_shovel)
        #self.draw_lines(shovel_pos, middle_of_shovel)
        cos_similarity = torch.einsum('ij,ij->i', shovel_to_prop_vec, shovel_facing_vec)
        rew_cos_similarity = cos_similarity + 1
        #print("3", rew_cos_similarity)
        total_reward = 5 * rew_a1_prop_distance + rew_torque + rew_action_rate + rew_joint_acc

        #total_reward =  rew_cos_of_foots_n_prop * rew_cos_of_foots_n_base + rew_torque + rew_action_rate + rew_joint_acc
        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        #print(torch.norm(self.contact_forces[:, self.shovel_bottom_index, :], dim=1))
        # reset agents
        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        #reset = reset | (torch.norm(self.contact_forces[:, self.bed_index, :], dim=1) > 1.).bool()
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()

        time_out = self.progress_buf >= self.max_episode_length/5 - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset



