import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class BasicLogger:
    def __init__(self, log_dir, model=None):
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        # if model is not None: TODO: add model
        # obsv = torch.rand(model.obs_shape)
        # self.writer.add_graph(model, obsv)

        self.log_buffer = {}

    def __del__(self):
        self.writer.close()

    def append_log_info(self, info):
        self.log_buffer.update(info)

    def log(self, it):
        for key, value in self.log_buffer.items():
            self.writer.add_scalar(key, value, it)
        self.log_buffer.clear()


class ConfigurationSaver:
    def __init__(self, log_dir, task_name):
        self.tsk_name = task_name
        self._data_dir = log_dir + '/' + task_name+"_" + \
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        self._model_dir = self.data_dir+"/nn/"
        self._summaries_dir = self.data_dir+"/summaries/"
        os.makedirs(self._model_dir)
        os.makedirs(self._summaries_dir)

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def summary_dir(self):
        return self._summaries_dir

    def gen_model_save_name(self, ep_num, fmt=".pt"):
        ep_model_name = self.tsk_name+"_ep_"+str(ep_num)+fmt
        last_model_name = self.tsk_name+fmt
        return {
            "ep_model_name": self.model_dir+ep_model_name,
            "last_model_name": self.model_dir+last_model_name
        }


def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    try:
        url = tb.launch()
        print("Tensorboard session created: "+url)
    except:
        print("\033[0;37;41mFailed to create tensorboard session!\033[0m")


class DefaultRewardsShaper:
    def __init__(self, scale_value=1, shift_value=0, min_val=-np.inf, max_val=np.inf, log_val=False, is_torch=True):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.log_val = log_val
        self.is_torch = is_torch

        if self.is_torch:
            self.log = torch.log
            self.clip = torch.clamp
        else:
            self.log = np.log
            self.clip = np.clip

    def __call__(self, reward):
        # orig_reward = reward
        reward = reward + self.shift_value
        reward = reward * self.scale_value

        reward = self.clip(reward, self.min_val, self.max_val)

        if self.log_val:
            reward = self.log(reward)
        return reward


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 0
        self.mean: torch.Tensor = torch.zeros(shape).to(device)
        self.S = torch.zeros(shape).to(device)
        self.std = torch.sqrt(self.S).to(device)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class RewardScaling:
    def __init__(self, shape, gamma, device):
        self.device = device
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape, device=self.device)
        self.R = torch.zeros(self.shape)

    def __call__(self, x, enable=True):
        if (not enable):
            return x

        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = torch.zeros(self.shape).to(self.device)


class RLRunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RLRunningMeanStd, self).__init__()
        print('RLRunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(
            in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(
            in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        with torch.no_grad():
            delta = batch_mean - mean
            tot_count = count + batch_count

            new_mean = mean + delta * batch_count / tot_count
            m_a = var * count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
            new_var = M2 / tot_count
            new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input: torch.Tensor, denorm=False, mask=None):
        tensor_size = input.shape
        input = input.view(-1, self.insize)
        if self.training:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count,
                                                                                                       mean, var, input.size()[0])

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view(
                    [1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * \
                y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / \
                    torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y.reshape(tensor_size)
