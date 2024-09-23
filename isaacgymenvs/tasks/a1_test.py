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

from isaacgymenvs.tasks.a1_with_shovel import A1WithShovel
from isaacgymenvs.tasks.a1_with_shovel_passive_joint import A1WithShovelPassiveJoint
from isaacgymenvs.tasks.a1_with_shovel_dagger_passive_joint import A1WithShovelDaggerPassiveJoint
from isaacgymenvs.tasks.a1_with_shovel_passive_joint_with_camera import A1WithShovelPassiveJointWithCamera
from isaacgymenvs.tasks.a1_with_shovel_dagger_fixed_joint import A1WithShovelDaggerFixedJoint
from isaacgymenvs.tasks.a1_MoE_passive_joint_two_actions import A1MoEPassiveJointTwoActions
from isaacgymenvs.tasks.a1_MoE import A1MoE
from isaacgymenvs.tasks.a1_MoE_passive_joint_two_actions_test import A1MoEPassiveJointTwoActionsTest

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

class A1Test(A1MoEPassiveJointTwoActionsTest):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")
        self.counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.shovel_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "FL_shovel")
        self.counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands[:, 0] = 0.3

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

    def visualize_object_map(self, object_map, pos_local, pos_global, env_idx=1):
        linspace = torch.linspace(-self.env_space*2, self.env_space*2, self.grid_size, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(linspace, linspace)
        grid_x_np = grid_x.cpu().numpy()
        grid_y_np = grid_y.cpu().numpy()
        object_map_np = object_map.cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.contourf(grid_x_np, grid_y_np, object_map_np[env_idx], cmap='viridis', levels=100)
        for pos in pos_local[env_idx]:
            plt.plot(pos[0].item(), pos[1].item(), 'ro')  # 물체 위치를 빨간 점으로 표시
        for pos in pos_global[env_idx]:
            plt.plot(pos[0].item(), pos[1].item(), 'bo')
        plt.colorbar(label='Object Influence')
        plt.title(f'Environment {env_idx + 1} Object Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.show()

    def compute_reward(self):
        rew_alive = self._reward_alive()
        rew_landing = self._reward_landing()
        rew_picking = self._reward_picking()
        rew_experts = self._reward_experts()
        #print(rew_experts)

        total_reward = self.rew_scales["landing"] * rew_landing + self.rew_scales["alive"] * rew_alive + self.rew_scales["picking"] * rew_picking

        total_reward = torch.clip(total_reward, 0., None)
        self.rew_buf[:] = total_reward.detach()

        self.check_termination()

    def check_termination(self):
        # reset agents
        reset = torch.norm(self.a1_contact_forces[:, self.trunk_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.a1_contact_forces[:, self.calf_indices, :], dim=2) > 1., dim=1)
        reset = reset | torch.any(torch.norm(self.a1_contact_forces[:, self.thigh_indices, :], dim=2) > 1., dim=1)
        bed_contact_force_norm = torch.norm(self.a1_contact_forces[:, self.bed_index, :], dim=1)
        reset = reset | (bed_contact_force_norm > 120.).bool()

        """
        min_distance_indices, closest_prop_pos = self.get_closest_prop_position()
        shovel_bottom_pos = self.rb_states[:, self.shovel_bottom_index, 0:3]

        closest_prop_contact_force = torch.norm(self.prop_contact_forces[torch.arange(self.num_envs), min_distance_indices], dim=1)
        bed_prop_contact_differences = torch.sqrt(torch.square(bed_contact_force_norm-closest_prop_contact_force))
        bed_prop_contact = (bed_contact_force_norm>0.) & (closest_prop_contact_force>0.) & (bed_prop_contact_differences<0.01)

        distance = torch.norm(closest_prop_pos - shovel_bottom_pos, dim=-1)

        distance_condition = distance < 0.15
        contact_force_condition = closest_prop_contact_force > 0.
        landing_condition = ~self.landing[torch.arange(self.num_envs), min_distance_indices].bool()

        for env_idx in range(self.num_envs):
            if distance_condition[env_idx]:
                self.counter[env_idx] += 1
            else:
                self.counter[env_idx] = 0

        count_condition = self.counter > 0.7 / self.dt
        throwing_condition = distance_condition & contact_force_condition & landing_condition & count_condition & ~bed_prop_contact
        reset = reset | throwing_condition
        """

        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        reset = reset | time_out

        self.reset_buf[:] = reset


    ############## rewards ##############

    def _reward_alive(self):
        rew_alive = 1.
        return rew_alive

    def _reward_landing(self):
        # compute_box_position_along_bed
        bed_normal_vector = self.get_transformed_position(vector_local=self.base_up_vector).unsqueeze(1)
        bed_normal_vector_expanded = bed_normal_vector.expand(-1, self.num_boxes, -1)
        bed_pos = self.rb_states[:, self.bed_bottom_index, 0:3].unsqueeze(1)
        bed_pos_expanded = bed_pos.expand(-1, self.num_boxes, -1)
        prop_root_pos = self.prop_root_states[:, :, 0:3]
        bed_to_box_vector = prop_root_pos - bed_pos_expanded
        distance = torch.norm(bed_to_box_vector, dim=-1)

        box_position_in_normal_direction = torch.einsum('ijk,ijk->ij', bed_to_box_vector, bed_normal_vector_expanded)
        box_position_in_plane_direction = torch.sqrt(torch.square(distance)-torch.square(box_position_in_normal_direction))

        #self.draw_lines(bed_pos_expanded[0][0], bed_pos_expanded[0][0]+bed_normal_vector_expanded[0][0]*box_position_in_normal_direction[0][0])

        normal_condition = (0<box_position_in_normal_direction) & (box_position_in_normal_direction<0.05)
        plane_condition = box_position_in_plane_direction < 0.12
        landing = (normal_condition & plane_condition).float()
        rew_landing = torch.sum(landing, dim=-1)
        self.landing += landing

        return rew_landing

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

        rew_picking = self._reward_picking()
        rew_picking_throwing = 15 * rew_picking + (30 * rew_bed_prop_distance * rew_closest_box_upward)
        rew_tracking = 3.5 * rew_tracking_goal_vel + 1.0 * rew_tracking_yaw

        #print("t", rew_tracking)
        #print("p", rew_picking_throwing)

        return rew_picking_throwing + rew_tracking

