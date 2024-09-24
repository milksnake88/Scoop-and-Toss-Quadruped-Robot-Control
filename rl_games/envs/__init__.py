

from rl_games.envs.connect4_network import ConnectBuilder
from rl_games.envs.test_network import TestNetBuilder
from rl_games.algos_torch import model_builder
from rl_games.envs.cnn_mlp_network import CNNMLPNetBuilder
from rl_games.envs.cnn_gating_network import CnnGatingNetBuilder
from rl_games.envs.gating_network import GatingNetBuilder
from rl_games.envs.gating_network_two_actions import GatingNetTwoActionsBuilder
from rl_games.envs.meta_controller_helper_network import MetaControllerHelperNetBuilder

model_builder.register_network('connect4net', ConnectBuilder)
model_builder.register_network('testnet', TestNetBuilder)
model_builder.register_network('cnn_mlp_network', CNNMLPNetBuilder)
model_builder.register_network('cnn_gating_network', CnnGatingNetBuilder)
model_builder.register_network('gating_network', GatingNetBuilder)
model_builder.register_network('gating_network_two_actions', GatingNetTwoActionsBuilder)
model_builder.register_network('meta_controller_helper_network', MetaControllerHelperNetBuilder)
