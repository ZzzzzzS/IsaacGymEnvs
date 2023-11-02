from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage
from CustomAlgorithm.ppo.helper import RLRunningMeanStd


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 bounds_loss_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 learning_rate_schedule='adaptive',
                 desired_kl=0.01,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 truncate_grads=True,
                 adv_norm=True,
                 value_norm=None):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env,
                                      actor.obs_shape, critic.obs_shape, actor.action_shape, device, adv_norm)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.bounds_loss_coef = bounds_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.value_norm: RLRunningMeanStd = value_norm

        # Log
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule
        self.truncate_grads = truncate_grads

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor.sample(actor_obs)
        return self.actions

    def step(self, value_obs, rews, dones):
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, self.actor.action_mean, self.actor.distribution.std_np, rews, dones,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        last_values = self.critic.predict(value_obs)

        # Learning step
        self.storage.compute_returns(
            last_values, self.critic, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step(
            log_this_iteration)
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()
        self.writer.add_scalar(
            'PPO/value_loss', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar(
            'PPO/surrogate_loss', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar(
            'PPO/bound_loss', variables["bound_loss"], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std',
                               mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate',
                               self.learning_rate, variables['it'])
        self.writer.add_scalar(
            'PPO/advangate', self.storage.adv_mean, variables['it'])
        self.writer.add_scalar(
            'PPO/mean KL', variables['kl_mean'], variables['it'])

    def log_other(self, update, reward, rew_ep):
        self.writer.add_scalar("Agent/average step reward", reward, update)
        self.writer.add_scalar("Agent/average episode reward", rew_ep, update)

    def _train_step(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(
                    actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Adjusting the learning rate using KL divergence
                mu_batch = self.actor.action_mean
                sigma_batch = self.actor.distribution.std

                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.no_grad():

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(
                                1e-6, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(
                                1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch -
                                  torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.value_norm is not None:
                    self.value_norm.train()
                    current_values_batch = self.value_norm(
                        current_values_batch)
                    # value_batch = self.value_norm(value_batch)
                    returns_batch = self.value_norm(returns_batch)

                if self.use_clipped_value_loss:
                    value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                      self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (
                        value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(
                        value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                if self.value_norm is not None:
                    self.value_norm.eval()

                # bound loss
                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(mu_batch - soft_bound, 0.0)**2
                mu_loss_low = torch.clamp_max(mu_batch + soft_bound, 0.0)**2
                bound_loss = (mu_loss_low + mu_loss_high).sum(axis=-1).mean()

                # all loss
                loss = surrogate_loss + self.value_loss_coef * \
                    value_loss*0.5 - self.entropy_coef * \
                    entropy_batch.mean() + self.bounds_loss_coef*bound_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    nn.utils.clip_grad_norm_(
                        [*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                if log_this_iteration:
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
