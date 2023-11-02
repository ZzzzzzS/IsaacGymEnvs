import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from MyAlgorithm.ppo.helper import RLRunningMeanStd
from MyAlgorithm.ppo.AgentNets import MLP


class Actor:
    def __init__(self, architecture, distribution, device='cpu', input_norm=None):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None
        self.input_norm: RLRunningMeanStd = input_norm

    def sample(self, obs):
        if self.input_norm is not None:
            obs = self.input_norm(obs)
        self.action_mean = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        if (self.input_norm is not None):
            obs = self.input_norm(obs)
        self.action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs: torch.Tensor):
        return self.architecture.architecture(obs.to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(
            self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu', input_norm=None, value_norm=None):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
        self.input_norm: RLRunningMeanStd = input_norm
        self.value_norm: RLRunningMeanStd = value_norm

    def predict(self, obs):
        if self.input_norm is not None:
            obs = self.input_norm(obs)
        value = self.architecture.architecture(obs).detach()
        if self.value_norm is not None:
            value = self.value_norm(value, denorm=True).detach()
        return value

    def evaluate(self, obs):
        if self.input_norm is not None:
            obs = self.input_norm(obs)
        value = self.architecture.architecture(obs)
        # if self.value_norm is not None:
        #    value = self.value_norm(value, denorm=True)
        return value

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        # self.fast_sampler = fast_sampler
        # self.fast_sampler.seed(seed)
        self.samples = torch.zeros([size, dim], dtype=torch.float32)
        self.logprob = torch.zeros(size, dtype=torch.float32)
        self.std_np = self.std

    def update(self):
        self.std_np = self.std

    def sample(self, logits):
        log_std = logits*0+self.std_np
        # exp_std = torch.exp(log_std)
        fast_sampler = Normal(logits, log_std)
        self.samples = fast_sampler.sample()
        self.logprob = fast_sampler.log_prob(self.samples).sum(dim=1)
        # self.logprob = self.neglogp(self.samples, logits, exp_std, log_std)
        # self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples, self.logprob

    def evaluate(self, logits, outputs):
        log_std = logits*0+self.std
        # exp_std = torch.exp(log_std)
        distribution = Normal(logits, log_std)
        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        # actions_log_prob = self.neglogp(outputs, logits, exp_std, log_std)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std: torch.Tensor):
        current_std = self.std.detach()
        # new_std = torch.max(current_std, torch.log(min_std.detach())).detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
