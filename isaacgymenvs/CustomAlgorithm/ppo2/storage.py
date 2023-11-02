import torch


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device, adv_norm=True, rnn=False, seq_len=1, rnn_states_shape=1):
        self.device = device
        self.rnn = rnn
        self.num_seq = num_transitions_per_env
        self.rnn_states_shape = 1
        self.seq_len = 1
        if self.rnn:
            self.seq_len = seq_len  # rnn sequence lenth
            self.num_seq = num_transitions_per_env//self.seq_len
            self.rnn_states_shape = rnn_states_shape
            assert num_transitions_per_env % self.seq_len == 0  # 判断是否整除
        # Core
        self.obs = torch.zeros(
            [num_transitions_per_env, num_envs, obs_shape], dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.actions = torch.zeros(
            [num_transitions_per_env, num_envs, actions_shape], dtype=torch.float32).to(self.device)
        self.values = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)  # NOTE: currently only support value_dim=1
        self.dones = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.long).to(self.device)

        self.states = torch.zeros(
            [self.num_seq, num_envs, self.rnn_states_shape], dtype=torch.float32).to(self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.returns = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros(
            [num_transitions_per_env, num_envs, 1], dtype=torch.float32).to(self.device)
        self.mu = torch.zeros([num_transitions_per_env, num_envs,
                              actions_shape], dtype=torch.float32).to(self.device)
        self.sigma = torch.zeros(
            [num_transitions_per_env, num_envs, actions_shape], dtype=torch.float32).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.adv_norm = adv_norm
        self.step = 0
        self.adv_mean = None

    def add_transitions(self, obsv, actions, value, mu, sigma, rewards, dones, actions_log_prob, states=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step] = obsv
        self.actions[self.step] = actions
        self.values[self.step] = value
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        self.rewards[self.step] = rewards.reshape(-1, 1)
        self.dones[self.step] = dones.reshape(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        if self.rnn:  # rnn states
            if self.step % self.seq_len == 0:
                self.states[self.step/self.seq_len] = states

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
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

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        mini_batch_size_mb = mini_batch_size//self.seq_len
        assert mini_batch_size % self.seq_len == 0

        for batch_id in range(num_mini_batches):
            yield self.obs.view(-1, *self.obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.sigma.view(-1, self.sigma.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.mu.view(-1, self.mu.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob.view(-1, 1)[batch_id *
                                                  mini_batch_size:(batch_id+1)*mini_batch_size],\
                self.states.view(-1, self.states.size(-1))[
                batch_id*mini_batch_size_mb:(batch_id+1)*mini_batch_size_mb, :]
