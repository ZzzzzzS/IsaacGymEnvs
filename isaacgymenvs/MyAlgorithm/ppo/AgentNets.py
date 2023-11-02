import torch
import torch.nn as nn
import numpy as np

import rl_games.common.layers.recurrent as rl_rnn


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


class TeacherEnc_07638(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass


class StdentConv_07638(nn.Module):
    pass


class StudentEnv_07638(nn.Module):
    pass
