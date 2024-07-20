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
from isaacgymenvs.tasks.a1_MoE import A1MoE

class A1Test(A1MoE):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

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

        time_out = self.progress_buf >= self.max_episode_length / 10- 1  # no terminal reward for time-outs
        reset = time_out

        self.reset_buf[:] = reset



