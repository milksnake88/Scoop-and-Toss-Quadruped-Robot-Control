import os, yaml
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import model_builder

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
import time
from torch import optim
import torch

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

class A2CDiscreteMoETwoActionsAgent(a2c_common.DiscreteA2CBase):

    def __init__(self, base_name, params):
        a2c_common.DiscreteA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        self.actions_num = 2
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.optimizer = optim.AdamW(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape,
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device,
                'num_agents' : self.num_agents,
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors,
                'num_actions' : self.actions_num,
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config,
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

        self.params = params
        expert1_cfg_path = params['experts']['expert1']['config']
        expert2_cfg_path = params['experts']['expert2']['config']

        e1_cfg = self.load_experts(expert1_cfg_path)
        e2_cfg = self.load_experts(expert2_cfg_path)

        self.e1_params = e1_cfg['train']['params']
        self.e2_params = e2_cfg['train']['params']

        # load experts model
        self.e1_model = self.load_run_model(self.e1_params)
        self.e2_model = self.load_pick_throw_model(self.e2_params)

        e1_checkpoint = self.params['experts']['expert1']['checkpoint']
        e2_checkpoint = self.params['experts']['expert2']['checkpoint']

        self.restore_experts(self.e1_model, e1_checkpoint)
        self.restore_experts(self.e2_model, e2_checkpoint)

        self.shovel_to_prop_dis = torch.tensor([[0.15]], device=self.ppo_device, requires_grad=False)
        self.pos_of_prop_wrt_a1_base = torch.tensor([[0.34, 0.13, 0]], device=self.ppo_device, requires_grad=False)

    def load_experts(self, config_path):
        with open(config_path, 'r') as stream:
            return yaml.safe_load(stream)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore_experts(self, model, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def get_masked_action_values(self, obs, action_masks):
        processed_obs = self._preproc_obs(obs['obs'])
        action_masks = torch.BoolTensor(action_masks).to(self.ppo_device)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'action_masks' : action_masks,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                input_dict = {
                    'is_train': False,
                    'states' : obs['states'],
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.is_multi_discrete:
            action_masks = torch.cat(action_masks, dim=-1)
        res_dict['action_masks'] = action_masks
        return res_dict

    def train_actor_critic(self, input_dict):
        self.set_train()
        self.calc_gradients(input_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        return self.train_result

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs' : obs_batch,
        }
        if self.use_action_masks:
            batch_dict['action_masks'] = input_dict['action_masks']

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length
            batch_dict['bptt_len'] = self.bptt_len
            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)


            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy = losses[0], losses[1], losses[2]
            loss = a_loss + 0.5 *c_loss * self.critic_coef - entropy * self.entropy_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            kl_dist = 0.5 * ((old_action_log_probs_batch - action_log_probs)**2)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel() # / sum_mask
            else:
                kl_dist = kl_dist.mean()

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, kl_dist,self.last_lr, lr_mul)

    def train_actor_critic(self, input_dict):
        self.set_train()
        self.calc_gradients(input_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        return self.train_result


    def get_run_action(self, expert, obs, is_deterministic=False):
        #if len(self.obs.size()) > len(self.obs_shape):
        #    self.has_batch_dimension = True
        processed_obs = self._preproc_obs(obs['obs'])
        obs_for_experts = processed_obs
        expert.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs_for_experts,
            'rnn_states' : self.rnn_states
        }
        with torch.no_grad():
            res_dict = expert(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        #if self.has_batch_dimension == False:
        #    current_action = torch.squeeze(current_action.detach())

        """
        if self.clip_actions:
            actions_low = self.actions_low[:12]
            actions_high = self.actions_high[:12]
            return rescale_actions(actions_low, actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
        """
        return current_action
      
    def get_pick_throw_action(self, expert, obs, is_deterministic=False):
        #if len(self.obs.size()) > len(self.obs_shape):
        #    self.has_batch_dimension = True
        processed_obs = self._preproc_obs(obs['obs'])
        obs_for_experts = processed_obs
        expert.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs_for_experts,
            'rnn_states' : self.rnn_states
        }
        with torch.no_grad():
            res_dict = expert(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        #if self.has_batch_dimension == False:
        #    current_action = torch.squeeze(current_action.detach())

        """
        if self.clip_actions:
            actions_low = self.actions_low[:12]
            actions_high = self.actions_high[:12]
            return rescale_actions(actions_low, actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
        """
        return current_action

    def load_run_model(self, params):
        builder = model_builder.ModelBuilder()
        network = builder.load(params)
        # TODO: expert cfg에 추가
        actions_num = 12 # params['experts']['env_info']['actions_num']
        obs_shape = (49, ) #(289, )
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

    def load_pick_throw_model(self, params):
        builder = model_builder.ModelBuilder()
        network = builder.load(params)
        # TODO: expert cfg에 추가
        actions_num = 12 # params['experts']['env_info']['actions_num']
        obs_shape = (49, ) #(289, )
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
    
    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_cmd_vel(self):
        # 주어진 텐서
        d = self.shovel_to_prop_dis.squeeze(-1)  # (num_envs,) -> 로봇과 물체 간 거리
        p = self.pos_of_prop_wrt_a1_base       # (num_envs, 3) -> 물체 위치 (x, y, z)

        """
        # 파라미터 설정
        V_MAX = 0.7                             # 최대 속도 (m/s)
        D_NORM = 5.0                            # 속도 정규화 기준 거리 (m)
        D_MIN = 0.1                             # 최소 임계 거리 (m)

        # 방향 단위 벡터 계산 (물체 위치가 (x, y, z)인데 x, y만 사용)
        # 3차원 벡터에서 x, y만 사용하여 2D 방향 벡터를 계산
        dir_vec = p[:, :2] / d.unsqueeze(-1)  # (num_envs, 2) 크기, 각 환경별 2D 방향 벡터

        # 속도 크기 계산 (선형 스케일링 + 클램핑)
        speed = torch.zeros_like(d)  # (num_envs,)
        mask_min = d <= D_MIN
        mask_norm = d >= D_NORM

        # 데드존 적용 (D_MIN 이하에서는 속도 0)
        speed[mask_min] = 0.0

        # 정규화 거리(D_NORM)까지 선형적으로 속도 계산
        speed[~mask_min & ~mask_norm] = V_MAX * (d[~mask_min & ~mask_norm] / D_NORM)

        # D_NORM 이상이면 최대 속도로 설정
        speed[mask_norm] = V_MAX

        """

        # 파라미터 설정
        V_FIXED = 0.3                             # 고정 속도 (m/s)

        # 방향 단위 벡터 계산 (물체 위치가 (x, y, z)인데 x, y만 사용)
        # 3차원 벡터에서 x, y만 사용하여 2D 방향 벡터를 계산
        dir_vec = p[:, :2] / d.unsqueeze(-1)  # (num_envs, 2) 크기, 각 환경별 2D 방향 벡터

        # 고정 속도 크기
        speed = torch.full_like(d, V_FIXED)  # 모든 환경에 대해 고정된 속도 0.3 (m/s)
                
        # 최종 타겟 속도 벡터 계산
        target_velocity = dir_vec * speed.unsqueeze(-1)  # (num_envs, 2) 크기
        
        return target_velocity

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            actions = res_dict['actions']
            res_dict['actions'] = actions.unsqueeze(-1)
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            actions = res_dict['actions'] # torch.Size([num_envs, 2])

            # Split action into weights and position
            weights1 = actions
            weights2 = 1 - weights1
            #cmd_vel = self.get_cmd_vel()

            e1_action = self.get_run_action(self.e1_model, self.obs)
            e2_action = self.get_pick_throw_action(self.e2_model, self.obs)

            mixtured_action = weights1*e1_action + weights2*e2_action

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(mixtured_action)
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            self.shovel_to_prop_dis = infos["shovel_to_prop_dis"]
            self.pos_of_prop_wrt_a1_base = infos["pos_of_prop_wrt_a1_base"]

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict