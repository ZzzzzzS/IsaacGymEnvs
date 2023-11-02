import torch.nn as nn
import torch
import time
from rl_games.interfaces.base_algorithm import BaseAlgorithm
from rl_games.common import vecenv
from utils.rlgames_utils import RLGPUEnv

from abc import ABC
from abc import abstractmethod, abstractproperty


class MyDaggerAgent(BaseAlgorithm):
    def __init__(self, base_name, config):
        pass

    def device(self):
        pass

    def clear_stats(self):
        pass

    def train(self):
        pass

    def train_epoch(self):
        pass

    def get_full_state_weights(self):
        pass

    def set_full_state_weights(self, weights):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
