import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import numpy as np
from .network import BasicA2CNetwork


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std, log_distrib=False):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.log_distrib = log_distrib

    def sample(self, logits):
        log_std = logits*0+self.std
        if self.log_distrib:
            log_std = torch.exp(log_std)

        sampler = Normal(logits, log_std)
        samples = sampler.sample()
        logprob = sampler.log_prob(samples).sum(dim=1)
        # self.logprob = self.neglogp(self.samples, logits, exp_std, log_std)
        return samples, logprob, self.std

    def evaluate(self, logits, outputs):
        log_std = logits*0+self.std
        if self.log_distrib:
            log_std = torch.exp(log_std)
        distribution = Normal(logits, log_std)
        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        # actions_log_prob = self.neglogp(outputs, logits, exp_std, log_std)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy, self.std

    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)

    def enforce_minimum_std(self, min_std: torch.Tensor):
        # new_std = torch.max(current_std, torch.log(min_std.detach())).detach()
        if self.log_distrib:
            min_std = torch.log(min_std)
        new_std = torch.max(self.std, min_std.detach()).detach()
        self.std.data = new_std


class A2CModule_Continuous(nn.Module):
    def __init__(self, A2CNet: BasicA2CNetwork,
                 init_std=1.0, log_distrib=False,
                 device="cpu", rnn=False, seq_len=None) -> None:
        super().__init__()
        self.device = device
        self.net = A2CNet
        self.distribution = MultivariateGaussianDiagonalCovariance(
            self.act_shape, init_std, log_distrib)

        self.rnn = rnn
        if self.rnn:
            self.seq_len = seq_len

    def sample(self, obsv: Tensor, state=None):
        self.eval()
        with torch.no_grad():
            act_mu, value, next_state = self.net(obsv, state)
            act, logprob, std = self.distribution.sample(act_mu)

        if self.rnn:
            next_state = next_state.detach()
            state = state.detach()

        state_dict = {
            'observations': obsv.detach(),
            'actions': act.detach(),
            'actions_mu': act_mu.detach(),
            'actions_std': std,
            'actions_logprob': logprob.detach(),
            'value': value.detach(),
            'next_state': next_state,
            'current_state': state
        }
        return state_dict

    def evaluate(self, obsv: Tensor, prev_action: Tensor, states: Tensor = None, dones: Tensor = None):
        self.train()
        if not self.rnn:
            act_mu, value = self.__evaluate_mlp(obsv)
        else:
            act_mu, value = self.__evaluate_rnn(
                obsv, states, dones)

        actions_log_prob, entropy, act_std = self.distribution.evaluate(
            act_mu, prev_action)

        state_dict = {
            'actions_mu': act_mu,
            'actions_std': act_std,
            'actions_logprob': actions_log_prob,
            'entropy': entropy,
            'value': value
        }
        return state_dict

    def parameters(self, recurse: bool = True):
        return [*self.net.parameters(), *self.distribution.parameters()]

    def get_default_rnn_states(self):
        return self.net.get_default_rnn_states()

    def get_value_norm(self):
        return self.net.get_value_norm()

    def is_rnn(self):
        return self.net.is_rnn()

    @property
    def obs_shape(self):
        return self.net.obsv_shape

    @property
    def act_shape(self):
        return self.net.act_shape

    @property
    def value_shape(self):
        return self.net.value_shape

    @property
    def rnn_states_shape(self):
        return self.net.rnn_states_shape

    def __evaluate_mlp(self, obsv: Tensor):
        act_mu, value, _ = self.net(obsv)
        return act_mu, value

    def __evaluate_rnn(self, obsv: Tensor, states: Tensor = None, dones: Tensor = None):
        batch_sz = obsv.shape[0]
        actor_num = obsv.shape[1]
        seq_num = batch_sz//self.seq_len
        assert batch_sz % self.seq_len == 0
        obsv = obsv.reshape(seq_num, self.seq_len, actor_num, -1)
        obsv = obsv.transpose(0, 1).contiguous()
        if dones is not None:
            dones = dones.reshape(seq_num, self.seq_len, actor_num, -1)
            dones = dones.transpose(0, 1).contiguous()
            not_done = 1.0-dones

        assert states.shape[0] == seq_num
        act_mu: Tensor = torch.zeros([self.seq_len, seq_num, actor_num,
                                     self.act_shape], dtype=torch.float32).to(self.device)
        value: Tensor = torch.zeros(
            [self.seq_len, seq_num, actor_num, self.value_shape], dtype=torch.float32).to(self.device)

        next_states = states
        for i in range(self.seq_len):
            if dones is not None:
                next_states = not_done[i, :] * \
                    next_states  # TODO:check if correct

            act_mu[i, :], value[i, :], next_states = self.net(
                obsv[i, :], next_states)

        act_mu = act_mu.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()

        act_mu = act_mu.reshape(batch_sz, actor_num, -1)
        value = value.reshape(batch_sz, actor_num, -1)
        return act_mu, value
