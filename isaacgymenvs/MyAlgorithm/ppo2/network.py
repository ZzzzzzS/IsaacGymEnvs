from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
from .helper import RLRunningMeanStd

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'swish': nn.SiLU,
    'gelu': nn.GELU,
    'softplus': nn.Softplus,
    'None': nn.Identity
}


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        # self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, input_obs):
        return self.architecture(input_obs)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        [torch.nn.init.constant_(module.bias, 0) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MLP_NO_OUT(nn.Module):
    def __init__(self, input_size, shape, actionvation_fn=nn.ELU):
        super(MLP_NO_OUT, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        # self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [shape[-1]]

    def forward(self, input_obs):
        return self.architecture(input_obs)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        [torch.nn.init.constant_(module.bias, 0) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class BasicA2CNetwork(nn.Module):
    def __init__(self, obsv_dim, act_dim, value_dim, device, rnn=False, input_norm=False, value_norm=False, rnn_states_dim=None):
        super().__init__()
        self.obsv_dim = obsv_dim
        self.act_dim = act_dim
        self.value_dim = value_dim
        self.device = device
        self.input_norm = RLRunningMeanStd(
            self.obsv_dim) if input_norm else None
        self.value_norm = RLRunningMeanStd(
            self.value_dim) if value_norm else None
        self.rnn = rnn
        self.rnn_states_dim = rnn_states_dim

    def forward(self, obsv, state=None):
        if self.input_norm is not None:
            obsv = self.input_norm(obsv)

        act, value, next_state = self.inference(obsv, state)

        if self.value_norm is not None:
            value = self.value_norm(value, denorm=True)

        return act, value, next_state

    @abstractmethod
    def inference(self, obsv, state=None):
        pass

    @abstractmethod
    def predict(self, obsv, state=None):
        pass

    @abstractmethod
    def action(self, obsv, state=None):
        pass

    @abstractmethod
    def get_default_rnn_states(self):
        pass

    @property
    def obsv_shape(self):
        return self.obsv_dim

    @property
    def act_shape(self):
        return self.act_dim

    @property
    def value_shape(self):
        return self.value_dim

    @property
    def rnn_states_shape(self):
        return self.rnn_states_dim if self.rnn else None

    def is_rnn(self):
        return self.rnn

    def get_value_norm(self):
        return self.value_norm

    @staticmethod
    @abstractmethod
    def create(obsv_dim, act_dim, value_dim, input_norm, value_norm, device, net_config):
        pass


class seperate_symmetry_MLP_A2CNetwork(BasicA2CNetwork):
    def __init__(self, obsv_dim, act_dim, value_dim, hidden_layer, activation_fn=nn.ELU, device="cpu", input_norm=None, value_norm=None):
        super().__init__(obsv_dim, act_dim, value_dim, device, False,
                         input_norm, value_norm, None)
        self.actor_net = MLP(hidden_layer, activation_fn,
                             self.obsv_dim, self.act_dim)
        self.critic_net = MLP(hidden_layer, activation_fn,
                              self.obsv_dim, self.value_dim)

    def inference(self, obsv, state=None):
        act_mu = self.actor_net(obsv)
        value = self.critic_net(obsv)
        return act_mu, value, None

    @staticmethod
    def create(obsv_dim, act_dim, value_dim, input_norm, value_norm, device, net_config):
        hidden_layer = net_config["mlp"]["units"]
        activation_fn = ACTIVATION_MAP.get(net_config["mlp"]["activation"])
        net = seperate_symmetry_MLP_A2CNetwork(obsv_dim=obsv_dim,
                                               act_dim=act_dim,
                                               value_dim=value_dim,
                                               hidden_layer=hidden_layer,
                                               activation_fn=activation_fn,
                                               device=device,
                                               input_norm=input_norm,
                                               value_norm=value_norm)

        return net
