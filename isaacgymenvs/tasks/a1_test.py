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

class A1Test(A1MoEPassiveJointTwoActions):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.base_up_vector = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_up_vector[:, 2] = 1. # bed joint position in base frame
        self.bed_bottom_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "bed_bottom")

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

        total_reward = self.rew_scales["landing"] * rew_landing + self.rew_scales["alive"] * rew_alive

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

        min_distance_indices, closest_prop_pos = self.get_closest_prop_position()
        shovel_bottom_pos = self.rb_states[:, self.shovel_bottom_index, 0:3]
        distance = torch.norm(closest_prop_pos[:, 0:2] - shovel_bottom_pos[:, 0:2], dim=-1)

        distance_condition = distance < 0.08
        contact_force_condition = torch.norm(self.prop_contact_forces[torch.arange(self.num_envs), min_distance_indices], dim=1) > 0.
        landing_condition = ~self.landing[torch.arange(self.num_envs), min_distance_indices].bool()

        throwing_condition = distance_condition & contact_force_condition & landing_condition

        reset = reset | throwing_condition

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
