import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device, adv_norm=True):
        self.device = device
        # Core
        self.critic_obs = torch.zeros(
            [num_transitions_per_env, num_envs, *critic_obs_shape], dtype=torch.float32).to(self.device)
        self.actor_obs = torch.zeros(
            [num_transitions_per_env, num_envs, *actor_obs_shape], dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.actions = torch.zeros(
            [num_transitions_per_env, num_envs, *actions_shape], dtype=torch.float32).to(self.device)
        self.dones = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.long).to(self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.values = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.returns = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.mu = torch.zeros([num_transitions_per_env, num_envs,
                              *actions_shape], dtype=torch.float32).to(self.device)
        self.sigma = torch.zeros(
            [num_transitions_per_env, num_envs, *actions_shape], dtype=torch.float32).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.adv_norm = adv_norm
        self.step = 0
        self.adv_mean = None

    def add_transitions(self, actor_obs, critic_obs, actions, mu, sigma, rewards, dones, actions_log_prob):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step] = critic_obs
        self.actor_obs[self.step] = actor_obs
        self.actions[self.step] = actions
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        self.rewards[self.step] = rewards.reshape(-1, 1)
        self.dones[self.step] = dones.reshape(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, critic, gamma, lam):
        with torch.no_grad():
            self.values = critic.predict(self.critic_obs)

        advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + next_is_not_terminal * \
                gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.adv_mean = self.advantages.mean()
        if (self.adv_norm):
            self.advantages = (self.advantages - self.advantages.mean()
                               ) / (self.advantages.std() + 1e-8)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs.view(-1,
                                                  *self.actor_obs.size()[2:])[indices]
            critic_obs_batch = self.critic_obs.view(
                -1, *self.critic_obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            sigma_batch = self.sigma.view(-1, self.sigma.size(-1))[indices]
            mu_batch = self.mu.view(-1, self.mu.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob.view(-1, 1)[
                indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]
            yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.actor_obs.view(-1, *self.actor_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.critic_obs.view(-1, *self.critic_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.sigma.view(-1, self.sigma.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.mu.view(-1, self.mu.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob.view(-1, 1)[batch_id *
                                                  mini_batch_size:(batch_id+1)*mini_batch_size]
