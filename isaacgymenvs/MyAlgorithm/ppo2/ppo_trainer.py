import torch
import torch.nn as nn
import torch.optim as optim
from .storage import RolloutStorage
from .helper import RLRunningMeanStd
from .module import A2CModule_Continuous


class ppo_trainer:
    def __init__(self,
                 A2CModule,
                 num_envs,
                 num_total_epoches,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 rnn=False,
                 seq_len=0,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 bounds_loss_coef=0.0,
                 learning_rate=5e-4,
                 learning_rate_schedule='adaptive',
                 max_grad_norm=0.5,
                 desired_kl=0.01,
                 use_clipped_value_loss=True,
                 device='cpu',
                 truncate_grads=True,
                 adv_norm=True):

        # env parameters
        self.num_total_epoches = num_total_epoches
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.rnn = rnn
        self.seq_len = seq_len
        self.device = device
        self.num_update = 0

        # PPO components
        self.A2CModule: A2CModule_Continuous = A2CModule

        self.storage = RolloutStorage(num_envs=self.num_envs,
                                      num_transitions_per_env=self.num_transitions_per_env,
                                      obs_shape=self.A2CModule.obs_shape,
                                      actions_shape=self.A2CModule.act_shape,
                                      device=self.device,
                                      adv_norm=adv_norm,
                                      rnn=self.rnn,
                                      seq_len=self.seq_len,
                                      rnn_states_shape=self.A2CModule.rnn_states_shape)

        self.batch_sampler = self.storage.mini_batch_generator_inorder
        self.optimizer = optim.Adam(
            self.A2CModule.parameters(), lr=learning_rate)

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
        self.value_norm: RLRunningMeanStd = self.A2CModule.get_value_norm()

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule
        self.truncate_grads = truncate_grads

        # temps
        self.step_state_dict = None

    def act(self, obsv, state=None):
        with torch.no_grad():
            self.step_state_dict = self.A2CModule.sample(obsv, state)
        return self.step_state_dict["actions"], self.step_state_dict["value"], self.step_state_dict["next_state"]

    def step(self, rews, dones):
        obsv = self.step_state_dict["observations"]
        action = self.step_state_dict["actions"]
        action_mu = self.step_state_dict["actions_mu"]
        action_std = self.step_state_dict["actions_std"]
        action_logprob = self.step_state_dict["actions_logprob"]
        value = self.step_state_dict["value"]
        state = self.step_state_dict["current_state"]
        self.storage.add_transitions(obsv=obsv, actions=action,
                                     value=value, mu=action_mu,
                                     sigma=action_std, rewards=rews.detach(),
                                     dones=dones.detach(), actions_log_prob=action_logprob,
                                     states=state)

    def update(self, obs, state=None):
        last_values = self.A2CModule.sample(obs, state)["value"]

        # Learning step
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        log_info = self._train_step()
        self.storage.clear()
        self.num_update += 1
        return log_info

    def _train_step(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_bound_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0

        for epoch in range(self.num_learning_epochs):
            for obsv_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, states_batch \
                    in self.batch_sampler(self.num_mini_batches):

                state_dict = self.A2CModule.evaluate(
                    obsv_batch, actions_batch, states_batch)

                actions_log_prob_batch = state_dict["actions_logprob"]
                entropy_batch = state_dict["entropy"]
                value_batch = state_dict["value"]
                # Adjusting the learning rate using KL divergence
                mu_batch = state_dict["actions_mu"]
                sigma_batch = state_dict["actions_std"]

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
                elif self.schedule == 'linear':
                    raise NotImplementedError()

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
                        current_values_batch).detach()
                    returns_batch = self.value_norm(returns_batch).detach()
                    value_batch = self.value_norm(value_batch)
                    self.value_norm.eval()

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
                        self.A2CModule.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += entropy_batch.mean().item()
                mean_bound_loss += bound_loss.item()
                mean_kl += kl_mean.item()

            num_updates = self.num_learning_epochs * self.num_mini_batches
            log_info_dict = {
                'ppo/value_loss': mean_value_loss/num_updates,
                'ppo/actor_loss': mean_surrogate_loss/num_updates,
                'ppo/entropy_loss': mean_entropy_loss/num_updates,
                'ppo/bound_loss': mean_bound_loss/num_updates,
                'agent/mean_kl': mean_kl/num_updates,
                'ppo/learning_rate': self.learning_rate
            }

        return log_info_dict
