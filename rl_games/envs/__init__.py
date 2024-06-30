

from rl_games.envs.connect4_network import ConnectBuilder
from rl_games.envs.test_network import TestNetBuilder
from rl_games.algos_torch import model_builder
from rl_games.envs.cnn_mlp_network import CNNMLPNetBuilder

model_builder.register_network('connect4net', ConnectBuilder)
model_builder.register_network('testnet', TestNetBuilder)
model_builder.register_network('cnn_mlp_network', CNNMLPNetBuilder)
