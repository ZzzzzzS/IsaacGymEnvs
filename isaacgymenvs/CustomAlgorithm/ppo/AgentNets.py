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


class TeacherEnc_RMA(nn.Module):
    def __init__(self, actionvation_fn=nn.ELU) -> None:
        super().__init__()
        mu = nn.Sequential([nn.Linear(17, 256),
                            actionvation_fn(),
                            nn.Linear(256, 128),
                            actionvation_fn(),
                            nn.Linear(128, 8)])
        pi = nn.Sequential([nn.Linear(50, 256),
                            actionvation_fn(),
                            nn.Linear(256, 128),
                            actionvation_fn(),
                            nn.Linear(128, 12)])

    def forward(self, obs):
        zt = obs[:, 42:]
        xt_at = obs[:, :42]
        et = self.mu(zt)
        pi_input = torch.cat((xt_at, et), dim=1)
        act = self.pi(pi_input)
        return act


class Student_Enc_RMA(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = input_size*tsteps
        self.output_shape = output_size

        if tsteps == 50:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32,
                          kernel_size=8, stride=4), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32,
                          kernel_size=5, stride=1), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 10:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32,
                          kernel_size=4, stride=2), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32,
                          kernel_size=2, stride=1), nn.LeakyReLU(),
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 20:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32,
                          kernel_size=6, stride=2), nn.LeakyReLU(),
                nn.Conv1d(in_channels=32, out_channels=32,
                          kernel_size=4, stride=2), nn.LeakyReLU(),
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        else:
            raise NotImplementedError()

    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output


class Student_RMA(nn.Module):
    def __init__(self, actionvation_fn=nn.ELU) -> None:
        super().__init__()
