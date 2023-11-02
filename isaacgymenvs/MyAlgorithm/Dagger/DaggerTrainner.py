from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class DaggerAgent:
    pass


class ObsStorage:
    pass


class DaggerTrainer:
    def __init__(self, actor: DaggerAgent, storage: ObsStorage,
                 obsv_dim: int, act_dim: int,
                 num_learning_ep=4, mini_batch_size=2048,
                 device=None, lr=5e-4) -> None:

        self.actor = actor
        self.storage = storage
