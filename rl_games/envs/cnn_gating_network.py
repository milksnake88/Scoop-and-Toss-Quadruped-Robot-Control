from rl_games.common import object_factory
#from torchsummary import summary
import numpy as np
from rl_games.algos_torch import torch_ext
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

import torch
from torch import nn
import torch.nn.functional as F

def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)

class CnnGatingNet(nn.Module):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        self.import_objectFactory_builder()
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size') #TODO: cfg로 받기
        self.num_seqs = num_seqs = kwargs.pop('num_seqs')

        self.load(params)
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        self.num_channels = self.cnn['num_channels']
        self.image_width = self.cnn['image_width']
        self.image_height = self.cnn['image_height']
        self.image_size = self.num_channels * self.image_width * self.image_height

        self.cnn_input_shape = (self.num_channels, self.image_width, self.image_height)
        #print(cnn_input_shape)
        cnn_args = {
          'ctype': self.cnn['type'],
          'input_shape': self.cnn_input_shape, # channels, width, height
          'convs': self.cnn['convs'],
          'activation': self.cnn['activation'],
          'norm_func_name': self.normalization,
        }
        self.actor_cnn = self._build_cnn2d(**cnn_args) # 딕셔너리 언팩킹
        print("actor_cnn: \n", self.actor_cnn)

        cnn_output_size = self._calc_cnn_output_size(self.cnn_input_shape, self.actor_cnn)
        #print("ssssssssssssssssssss", cnn_output_size)
        self.mlp_input_size = cnn_output_size + (input_shape[0] - self.image_size)

        if len(self.units) == 0:
            out_size = self.mlp_input_size
        else:
            out_size = self.units[-1]

        mlp_args = {
          'input_size' : self.mlp_input_size,
          'units' : self.units,
          'activation' : self.activation,
          'norm_func_name' : self.normalization,
          'dense_func' : torch.nn.Linear,
          'd2rl' : self.is_d2rl,
          'norm_only_first_layer' : self.norm_only_first_layer
        }

        self.actor_mlp = self._build_sequential_mlp(**mlp_args)
        print("actor_mlp \n", self.actor_mlp)
        self.value = torch.nn.Linear(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

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
        num_envs = obs.shape[0]
        states = obs_dict.get('rnn_states', None)
        seq_length = obs_dict.get('seq_length', 1)
        dones = obs_dict.get('done', None)
        bptt_len = obs_dict.get('bptt_len', 0)

        obs_depth_images = obs[:, :self.image_size].view(num_envs, self.num_channels, self.image_width, self.image_height) #torch.Size([num_envs, 1, 64, 64])
        obs_robot_states = obs[:, self.image_size:] #torch.Size([num_envs, 48])

        out = obs_depth_images
        out = self.actor_cnn(out)
        out = out.flatten(1) #torch.Size([num_envs, 1024(cnn output size)])
        #print(out.cpu().numpy())
        #print("norm", torch.norm(out, dim=1))
        out = torch.cat((out, obs_robot_states), dim=1) #torch.Size([num_envs, 1072(cnn output size + 48)])
        out = self.actor_mlp(out) #torch.Size([num_envs, 64(mlp output size)]
        value = self.value(out)
        # t = value.size()


        #print("cnn_network: ", summary(self.actor_cnn, self.cnn_input_shape))
        #print("mlp_network: ", self.actor_mlp)

        if self.central_value:
            return value, states

        if self.is_continuous:
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
            return mu, mu*0 + sigma, value, states

    def _build_sequential_mlp(self, input_size, units, activation, dense_func, d2rl, norm_only_first_layer=False, norm_func_name = None):
        #print('build mlp:', input_size)
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


    def _calc_cnn_output_size(self, cnn_input_shape, cnn_layers=None):
        if cnn_layers is None:
            assert(len(cnn_input_shape) == 1)
            return cnn_input_shape[0]
        else:
            return nn.Sequential(*cnn_layers)(torch.rand(1, *(cnn_input_shape))).flatten(1).data.size(1)

    def _build_cnn2d(self, ctype, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None):
        in_channels = input_shape[0]
        layers = []
        for conv in convs:
          layers.append(conv_func(in_channels=in_channels,
                                  out_channels=conv['filters'],
                                  kernel_size=conv['kernel_size'],
                                  stride=conv['strides'],
                                  padding=conv['padding']))
          conv_func=torch.nn.Conv2d
          act = self.activations_factory.create(activation)
          layers.append(act)
          in_channels = conv['filters']
          if norm_func_name == 'layer_norm':
              layers.append(torch_ext.LayerNorm2d(in_channels))
          elif norm_func_name == 'batch_norm':
              layers.append(torch.nn.BatchNorm2d(in_channels))
          if conv['pooling']['type'] == 'max':
              layers.append(torch.nn.MaxPool2d(kernel_size=conv['pooling']['kernel_size'],
                                               stride=conv['pooling']['strides'],
                                               padding=conv['pooling']['padding']))
          elif conv['pooling']['type'] == 'avg':
              layers.append(torch.nn.AdaptiveAvgPool2d(output_size=conv['pooling']['output_size']))
        return nn.Sequential(*layers)

    def load(self, params):
        self.separate = params.get('separate', False)
        self.units = params['mlp']['units']
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

class CnnGatingNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return CnnGatingNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
