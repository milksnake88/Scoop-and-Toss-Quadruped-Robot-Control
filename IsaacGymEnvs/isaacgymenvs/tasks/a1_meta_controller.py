import numpy as np
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import matplotlib.pyplot as plt

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from torch.nn.functional import normalize

from isaacgymenvs.tasks.a1_meta_controller_env import A1MetaControllerEnv

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

class A1MetaController(A1MetaControllerEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame
        self.bed_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed")
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.shovel_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel_bottom")
        self.bed_position_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.bed_position_local[:, 2] = 0.06 # bed joint position in base frame
        self.counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands[:, 0] = 0.2
        self.num_props = self.num_boxes


    def compute_reward(self):
        rew_landing = self._reward_landing()
        rew_experts = self._reward_experts()

        total_reward = rew_experts + 100 * rew_landing

        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        self.check_termination()

    def check_termination(self):
        # reset agents
        trunk_reset = torch.norm(self.a1_contact_forces[:, self.trunk_index, :], dim=1) > 1.

        calf_reset = torch.any(torch.norm(self.a1_contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        thigh_reset = torch.any(torch.norm(self.a1_contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.a1_contact_forces[:, self.bed_index, :], dim=1)
        bed_contact_reset = (bed_contact_force_norm > 120.).bool()
        reset = trunk_reset
        reset = trunk_reset | calf_reset | thigh_reset | bed_contact_reset

        self.num_landing = (self.landing >= 1).sum(dim=-1)

        self.total_num_landing = (self.landing > 10 / self.dt).sum(dim=-1)

        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        reset = self.num_landing == self.num_props
        reset = reset | time_out

        self.reset_buf[:] = reset


    ############## rewards ##############

    def _reward_alive(self):
        rew_alive = 0.1
        return rew_alive

    def _reward_landing(self):
        # compute_box_position_along_bed
        bed_normal_vector = self.get_transformed_position(self.a1_root_states, vector_local=self.base_up_vector).unsqueeze(1)
        bed_normal_vector_expanded = bed_normal_vector.expand(-1, self.num_props, -1)
        bed_pos = self.rb_states[:, self.bed_bottom_index, 0:3].unsqueeze(1)
        bed_pos_expanded = bed_pos.expand(-1, self.num_props, -1)
        prop_root_pos = self.prop_root_states[:, :, 0:3]
        bed_to_box_vector = prop_root_pos - bed_pos_expanded
        distance = torch.norm(bed_to_box_vector, dim=-1)

        box_position_in_normal_direction = torch.einsum('ijk,ijk->ij', bed_to_box_vector, bed_normal_vector_expanded)
        box_position_in_plane_direction = torch.sqrt(torch.square(distance)-torch.square(box_position_in_normal_direction))

        normal_condition = (0<box_position_in_normal_direction) & (box_position_in_normal_direction<0.1)
        plane_condition = box_position_in_plane_direction < 0.12
        landing = (normal_condition & plane_condition).float()
        rew_landing = torch.sum(landing, dim=-1)
        self.landing += landing
        self.landing_now = landing
        return rew_landing

    def _reward_distance(self):
        min_distance_indices, closest_prop_pos = self.get_closest_prop_position()
        bed_bottom_position = self.rb_states[:, self.bed_bottom_index, 0:3]
        bed_prop_distance = torch.norm(closest_prop_pos - bed_bottom_position, dim=-1)
        rew_distance = torch.exp(-bed_prop_distance/0.25)
        return rew_distance


    def _reward_picking(self):
        min_distance_indices, closest_prop_pos = self.get_closest_prop_position()
        shovel_bottom_contact_force = torch.norm(self.a1_contact_forces[:, self.shovel_bottom_index, :], dim=1)
        closest_prop_contact_force = torch.norm(self.prop_contact_forces[torch.arange(self.num_envs), min_distance_indices], dim=1)
        shovel_prop_contact_differences = torch.sqrt(torch.square(shovel_bottom_contact_force-closest_prop_contact_force))
        shovel_prop_contact = (shovel_bottom_contact_force>0.) & (closest_prop_contact_force>0.) & (shovel_prop_contact_differences<0.01)

        return shovel_prop_contact


    def _reward_experts(self):
        # distance reward
        min_distance_indices, closest_prop_pos = self.get_closest_prop_position()
        bed_bottom_position = self.rb_states[:, self.bed_bottom_index, 0:3]
        bed_prop_distance = torch.norm(closest_prop_pos - bed_bottom_position, dim=-1)
        rew_bed_prop_distance = torch.exp(-bed_prop_distance/0.25)

        # box upward reward
        rew_closest_box_upward = closest_prop_pos[:, 2]

        # action rate reward

        # reward tracking goal vel
        target_pos_rel = closest_prop_pos[:, :3] - self.rb_states[:, self.shovel_bottom_index, 0:3]
        target_pos_rel_norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = target_pos_rel / (target_pos_rel_norm + 1e-5)
        cur_vel = self.rb_states[:, self.shovel_index, 7:9]
        rew_tracking_goal_vel = torch.minimum(torch.sum(target_vec_norm[:, :2] * cur_vel, dim=-1), self.commands[:, 0] + 1e-5)

        base_quat = self.a1_root_states[:, 3:7]
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        _, pitch, yaw = euler_from_quaternion(base_quat)
        rew_tracking_yaw = torch.exp(-torch.abs(target_yaw - yaw))

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)

        # joint acceleration penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1)

        rew_picking = self._reward_picking()
        rew_picking_throwing = 15 * rew_picking + (30 * rew_bed_prop_distance * rew_closest_box_upward)\
                       -0.001 * rew_action_rate -0.0001 * rew_joint_acc -0.00003* rew_torque

        return rew_picking_throwing