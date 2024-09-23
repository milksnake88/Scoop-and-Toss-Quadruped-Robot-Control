import yaml
from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch
from torch import nn
import numpy as np


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class PpoPlayerContinuousMoETwoActions(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        self.params = params
        #expert1_cfg_path = os.path.join('runs', params['experts']['expert1']['name'], 'config.yaml')
        #expert2_cfg_path = os.path.join('runs', params['experts']['expert2']['name'], 'config.yaml')
        expert1_cfg_path = params['experts']['expert1']['config']
        expert2_cfg_path = params['experts']['expert2']['config']

        e1_cfg = self.load_experts(expert1_cfg_path)
        e2_cfg = self.load_experts(expert2_cfg_path)

        self.e1_params = e1_cfg['train']['params']
        self.e2_params = e2_cfg['train']['params']

        # load experts model
        self.e1_model = self.load_experts_model(self.e1_params)
        self.e2_model = self.load_experts_model(self.e2_params)

        e1_checkpoint = self.params['experts']['expert1']['checkpoint']
        e2_checkpoint = self.params['experts']['expert2']['checkpoint']

        self.restore_experts(self.e1_model, e1_checkpoint)
        self.restore_experts(self.e2_model, e2_checkpoint)

        self.shovel_to_prop_dis = torch.tensor([[0.]], device=self.device, requires_grad=False)

    def load_experts(self, config_path):
        with open(config_path, 'r') as stream:
            return yaml.safe_load(stream)

    def load_experts_model(self, params):
        builder = model_builder.ModelBuilder()
        network = builder.load(params)
        # TODO: expert cfg에 추가
        actions_num = 12 # params['experts']['env_info']['actions_num']
        obs_shape = (49, ) #self.obs_shape
        num_agents = 1 # params['experts']['env_info']['agent']
        value_size = 1 # params['experts']['env_info']['value_size']
        normalize_value = params['config']['normalize_value']
        normalize_input = params['config']['normalize_input']
        config = {
            'actions_num' : actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : num_agents,
            'value_size': value_size,
            'normalize_value': normalize_value,
            'normalize_input': normalize_input,
        }
        model = network.build(config)
        model.to(self.device)
        model.eval()
        return model


    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu[:24]
        else:
            current_action = action[:24]
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore_experts(self, model, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()

    def get_expert_action(self, expert, obs, shovel_to_prop_dis, is_deterministic=False):
        # if len(self.obs.size()) > len(self.obs_shape):
        #    self.has_batch_dimension = True
        processed_obs = self._preproc_obs(obs)
        closest_prop_pos = processed_obs[:, 0:3]
        proprioception = processed_obs[:, 3:] #TODO: cfg로 받기(map size 달라질수도 있음)
        num_envs = proprioception.shape[0]
        if shovel_to_prop_dis.shape[0] != num_envs:
            shovel_to_prop_dis = shovel_to_prop_dis.repeat(num_envs, 1)
        obs_for_experts = torch.cat((closest_prop_pos, shovel_to_prop_dis, proprioception), dim=-1)
        expert.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs_for_experts,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = expert(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if is_deterministic:
            current_action =  mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            actions_low = self.actions_low[:12]
            actions_high = self.actions_high[:12]
            return rescale_actions(actions_low, actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                # Split action into weights and position
                weights1 = action[:, 0].unsqueeze(-1)
                weights2 = action[:, 1].unsqueeze(-1)
                #print("1", weights1)
                #print("2", weights2)
                e1_action = self.get_expert_action(self.e1_model, obses, self.shovel_to_prop_dis)
                e2_action = self.get_expert_action(self.e2_model, obses, self.shovel_to_prop_dis)

                mixtured_action = weights1*e1_action + weights2*e2_action

                obses, r, done, info = self.env_step(self.env, mixtured_action)
                cr += r
                steps += 1

                self.shovel_to_prop_dis = info["shovel_to_prop_dis"]

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)
