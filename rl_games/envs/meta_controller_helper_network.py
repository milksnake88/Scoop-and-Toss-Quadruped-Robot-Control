from rl_games.common import object_factory
from rl_games.common.extensions.distributions import CategoricalMasked
import numpy as np
from rl_games.algos_torch import torch_ext
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

import torch
from torch import nn
import torch.nn.functional as F

def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)

class MetaControllerHelperNet(nn.Module):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        self.import_objectFactory_builder()
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)
        self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

        self.load(params)
        self.controller_actor_mlp = nn.Sequential()
        self.controller_critic_mlp = nn.Sequential()
        self.helper_actor_mlp = nn.Sequential()
        self.helper_critic_mlp = nn.Sequential()

        controller_mlp_input_shape = 48
        helper_mlp_input_shape = input_shape[0]

        if len(self.controller_units) == 0:
            controller_out_size = controller_mlp_input_shape
            helper_out_size = helper_mlp_input_shape
        else:
            controller_out_size = self.controller_units[-1]
            helper_out_size = self.helper_units[-1]

        controller_mlp_args = {
          'input_size' : controller_mlp_input_shape,
          'units' : self.controller_units,
          'activation' : self.activation,
          'norm_func_name' : self.normalization,
          'dense_func' : torch.nn.Linear,
          'd2rl' : self.is_d2rl,
          'norm_only_first_layer' : self.norm_only_first_layer
        }

        self.controller_actor_mlp = self._build_sequential_mlp(**controller_mlp_args)
        print("controller_actor_mlp \n", self.controller_actor_mlp)
        if self.separate:
            self.controller_critic_mlp = self._build_sequential_mlp(**controller_mlp_args)

        helper_mlp_args = {
          'input_size' : helper_mlp_input_shape,
          'units' : self.helper_units,
          'activation' : self.activation,
          'norm_func_name' : self.normalization,
          'dense_func' : torch.nn.Linear,
          'd2rl' : self.is_d2rl,
          'norm_only_first_layer' : self.norm_only_first_layer
        }

        self.helper_actor_mlp = self._build_sequential_mlp(**helper_mlp_args)
        print("helper_actor_mlp \n", self.helper_actor_mlp)
        if self.separate:
            self.helper_critic_mlp = self._build_sequential_mlp(**helper_mlp_args)

        self.value = torch.nn.Linear(helper_out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        # controller: discrete space
        self.logits = torch.nn.Linear(controller_out_size, 2)

        # helper: continuous space
        self.mu = torch.nn.Linear(helper_out_size, actions_num)
        self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
        mu_init = self.init_factory.create(**self.space_config['mu_init'])
        self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
        sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = torch.nn.Linear(helper_out_size, actions_num)

        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_cnn:
            cnn_init = self.init_factory.create(**self.cnn['initializer'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

    def is_rnn(self):
        return False

    def forward(self, obs_dict):
        obs = obs_dict['obs'] #torch.Size([num_envs, 4144(64*64+48)])
        states = obs_dict.get('rnn_states', None)
        dones = obs_dict.get('done', None)
        bptt_len = obs_dict.get('bptt_len', 0)
        e1_action = obs_dict.get('e1_action', None)
        e2_action = obs_dict.get('e2_action', None)
        is_train = obs_dict.get('is_train', True)
        action_masks = obs_dict.get('action_masks', None)
        prev_actions = obs_dict.get('prev_actions', None)

        controller_obs = obs[:, 12:]
        controller_out = self.controller_actor_mlp(controller_obs)
        logits = self.logits(controller_out)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        categorical = CategoricalMasked(logits=logits, masks=action_masks)
        action = categorical.sample().long()

        weigths1 = action.unsqueeze(-1)
        weigths2 = 1 - weigths1

        helper_out = obs
        helper_out = self.helper_actor_mlp(helper_out)
        value = self.value_act(self.value(helper_out))

        if self.is_continuous:
            mu = self.mu_act(self.mu(helper_out))
            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(helper_out))
            states = weigths1
            return mu, mu*0 + sigma, value, states

    def _build_sequential_mlp(self, input_size, units, activation, dense_func, d2rl, norm_only_first_layer=False, norm_func_name = None):
        print('build mlp:', input_size)
        in_size = input_size
        layers = []
        need_norm = True
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(self.activations_factory.create(activation))

            if not need_norm:
                continue
            if norm_only_first_layer and norm_func_name is not None:
                need_norm = False
            if norm_func_name == 'layer_norm':
                layers.append(torch.nn.LayerNorm(unit))
            elif norm_func_name == 'batch_norm':
                layers.append(torch.nn.BatchNorm1d(unit))
            in_size = unit

        return nn.Sequential(*layers)


    def load(self, params):
        self.separate = params.get('separate', False)
        self.controller_units = params['mlp']['controller']['units']
        self.helper_units = params['mlp']['helper']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
        self.value_activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', None)
        self.has_rnn = 'rnn' in params
        self.has_space = 'space' in params
        self.central_value = params.get('central_value', False)
        self.joint_obs_actions_config = params.get('joint_obs_actions', None)

        if self.has_space:
            self.is_multi_discrete = 'multi_discrete'in params['space']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous'in params['space']
            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False

        if self.has_rnn:
            self.rnn_units = params['rnn']['units']
            self.rnn_layers = params['rnn']['layers']
            self.rnn_name = params['rnn']['name']
            self.rnn_ln = params['rnn'].get('layer_norm', False)
            self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
            self.rnn_concat_input = params['rnn'].get('concat_input', False)

        if 'cnn' in params:
            self.has_cnn = True
            self.cnn = params['cnn']
            self.permute_input = self.cnn.get('permute_input', True)
        else:
            self.has_cnn = False

    def import_objectFactory_builder(self):
        self.activations_factory = object_factory.ObjectFactory()
        self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
        self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
        self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
        self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
        self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
        self.activations_factory.register_builder('swish', lambda **kwargs : nn.SiLU(**kwargs))
        self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
        self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
        self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())

        self.init_factory = object_factory.ObjectFactory()
        #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
        self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
        self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
        self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
        self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
        self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(torch_ext.variance_scaling_initializer,**kwargs))
        self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
        self.init_factory.register_builder('kaiming_normal', lambda **kwargs : _create_initializer(nn.init.kaiming_normal_,**kwargs))
        self.init_factory.register_builder('orthogonal', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
        self.init_factory.register_builder('default', lambda **kwargs : nn.Identity())


from rl_games.algos_torch.network_builder import NetworkBuilder

class MetaControllerHelperNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return MetaControllerHelperNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

