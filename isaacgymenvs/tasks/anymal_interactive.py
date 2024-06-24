import torch
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.anymal import Anymal

class AnymalInteractive(Anymal):

  def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
    super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    self.keyboard_input_reset()

  def set_viewer(self):
    super().set_viewer()
    if self.headless == False:
      self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "right")
      self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "left")
      self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
      self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "down")
      self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ENTER, "enter")

  def render(self, mode="rgb_array"):
    super().render(mode="rgb_array")
    # check for keyboard events
    for evt in self.gym.query_viewer_action_events(self.viewer):
      if evt.action == "left" and evt.value > 0:
        self.keyboard_input_y += 1
      elif evt.action == "right" and evt.value > 0:
        self.keyboard_input_y -= 1
      elif evt.action == "up" and evt.value > 0:
        self.keyboard_input_x += 1
      elif evt.action == "down" and evt.value > 0:
        self.keyboard_input_x -= 1

  def reset_idx(self, env_ids):
    super().reset_idx(env_ids)
    self.commands[env_ids] = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    self.keyboard_input_reset()

  def compute_observations(self):
    self.draw_lines()

    env_num = torch.arange(self.num_envs, device=self.device)

    keyboard_input_scales = 0.2
    self.commands_x[env_num] = torch.tensor(keyboard_input_scales*self.keyboard_input_x, device=self.device).squeeze()
    self.commands_y[env_num] = torch.tensor(keyboard_input_scales*self.keyboard_input_y, device=self.device).squeeze()

    super().compute_observations()

  def draw_lines(self):
    base_pos, base_ori = self.get_base_pose()
    transform = gymapi.Transform(gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2]),
                                 gymapi.Quat(base_ori[0], base_ori[1], base_ori[2], base_ori[3]))
    keyboard_input_transformed = transform.transform_point(gymapi.Vec3(self.keyboard_input_x, self.keyboard_input_y, 0.0))
    line_vertices = [[base_pos[0], base_pos[1], base_pos[2]],
                     [keyboard_input_transformed.x, keyboard_input_transformed.y, keyboard_input_transformed.z]]
    num_lines = 1
    line_colors = [1.0, 0.0, 0.0]

    self.gym.clear_lines(self.viewer)
    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices, line_colors)

  def get_base_pose(self):
    base_pos = self.root_states[:, 0:3].squeeze().cpu().numpy()
    base_ori = self.root_states[:, 3:7].squeeze().cpu().numpy()
    return base_pos, base_ori

  def keyboard_input_reset(self):
    self.keyboard_input_x = 0.0
    self.keyboard_input_y = 0.0
    self.keyboard_input_done = False

