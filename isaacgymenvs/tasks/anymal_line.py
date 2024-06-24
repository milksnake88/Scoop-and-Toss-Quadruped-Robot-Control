import torch
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.anymal import Anymal

class AnymalLine(Anymal):

  def compute_observations(self):
    self.draw_lines()
    super().compute_observations()

  def draw_lines(self):
    base_pos, base_ori = self.get_base_pose()
    transform = gymapi.Transform(gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2]),
                                 gymapi.Quat(base_ori[0], base_ori[1], base_ori[2], base_ori[3]))
    commands = self.commands.squeeze().cpu().numpy()
    commands_transformed = transform.transform_point(gymapi.Vec3(commands[0], commands[1], 0.0))
    line_vertices = [[base_pos[0], base_pos[1], base_pos[2]],
                     [commands_transformed.x, commands_transformed.y, commands_transformed.z]]
    num_lines = 1
    line_colors = [1.0, 0.0, 0.0]

    self.gym.clear_lines(self.viewer)
    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices, line_colors)

  def get_base_pose(self):
    base_pos = self.root_states[:, 0:3].squeeze().cpu().numpy()
    base_ori = self.root_states[:, 3:7].squeeze().cpu().numpy()
    return base_pos, base_ori

