
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

class A1Approaching(A1WithoutShovel):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.prop_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.prop_handles[0], "prop")
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame

    def get_global_position(self, point_local=None, vector_local=None):
        base_pos = self.a1_root_states[:, 0:3].squeeze().cpu().numpy()
        base_ori = self.a1_root_states[:, 3:7].squeeze().cpu().numpy()

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
        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * -0.000025

        # action rate penalty
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        rew_action_rate = -0.0009 * action_rate

        # joint acceleration penalty
        joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        rew_joint_acc = -0.00009 * joint_acc

        # angular velocity penalty
        a1_ang_vel = torch.norm(self.a1_root_states[:, 10:13], dim=1)
        rew_ang_vel = -0.000003 * a1_ang_vel

        # velocity tracking error(robot approaching policy)
        a1_root_pos = self.a1_root_states[:, 0:3]
        prop_root_pos = self.prop_root_states[:, 0:3]
        a1_to_prop_dis = torch.norm(prop_root_pos - a1_root_pos, dim=1)
        a1_to_prop_vec = (prop_root_pos - a1_root_pos) / a1_to_prop_dis.unsqueeze(1)
        commands = 0.5 * a1_to_prop_vec
        #print("commands", commands)
        a1_lin_vel = self.a1_root_states[:, 7:10]
        #print("a1_lin_vel", a1_lin_vel)
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - a1_lin_vel[:, :2]), dim=1)
        #print("lin_vel_error", lin_vel_error)
        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25)

        # lateral deviation
        base_quat = self.a1_root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, a1_lin_vel)
        rew_lateral_deviation = torch.sqrt(torch.square(base_lin_vel[:, 1]))

        # rew_energy
        rew_energy = torch.sum(torch.square(self.torques*self.dof_vel),dim=1)
        total_reward = 30 * torch.exp(-a1_to_prop_dis/2) + rew_torque + rew_action_rate + rew_joint_acc
        print(total_reward)
        #total_reward = -1. * lin_vel_error -1. * rew_lateral_deviation -1e-5 * rew_energy + 5.
        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        reset = torch.norm(self.contact_forces[:, self.trunk_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()

        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset
