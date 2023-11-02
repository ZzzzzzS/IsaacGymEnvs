import torch
import torch.nn as nn
import rl_games.algos_torch.network_builder as rl_net_builder
import rl_games.algos_torch.model_builder as rl_model_builder
from MyAlgorithm.ppo.AgentNets import MLP, MLP_NO_OUT

'''
class MyMLPBuilder(rl_net_builder.NetworkBuilder):
    def __init__(self, **kwargs):  # 被迫在屎山上拉屎
        super().__init__(**kwargs)

    def load(self, param):
        self.param = param

    def build(self, name, **kwargs):
        net = MyMLPBuilder.Network(self.param, **kwargs)
        return net

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class Network(rl_net_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            self.action_num = kwargs.pop('actions_num')
            self.input_shape = kwargs.pop('input_shape')[0]
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            rl_net_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            if (self.separate):
                self.critic_mlp = MLP(
                    self.hiddenlayer, self.activation, self.input_shape, 1)
                self.actor_mlp = MLP(
                    self.hiddenlayer, self.activation, self.input_shape, self.action_num)
                print('critic_mlp', self.critic_mlp)
                print('actor_mlp', self.actor_mlp)

            else:
                self.actor_mlp = MLP_NO_OUT(
                    self.hiddenlayer, self.activation, self.input_shape)
                self.actor_out = nn.Linear(
                    self.actor_mlp.output_shape[0], self.action_num)
                self.critic_out = nn.Linear(self.actor_mlp.output_shape[0], 1)

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, 'bias', None) is not None:
                        torch.nn.init.zeros_(m.bias)

            # only support fixed sigma now
            self.sigma = nn.Parameter(torch.ones(
                self.action_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            self.sigma_init(self.sigma)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            state = None  # do NOT support rnn

            if self.separate:
                mu = self.actor_mlp(obs)
                value = self.critic_mlp(obs)
                sigma = mu*0.0+self.sigma
                return mu, sigma, value, state
            else:
                a_out = self.actor_mlp(obs)
                mu = self.actor_out(a_out)
                sigma = mu*0.0+self.sigma
                value = self.critic_out(a_out)
                return mu, sigma, value, state

        def is_separate_critic(self):
            return self.separate

        def get_value_layer(self):
            return self.critic_out if self.separate else self.critic_mlp

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def load(self, param):
            self.separate = param.get('separate', False)
            self.hiddenlayer = param['mlp']['units']
            self.activation = self.activations_factory._builders.get(
                param['mlp']['activation'])
            self.sigma_init = self.init_factory.create(
                **param['space']['continuous']['sigma_init'])
            self.initializer = param['mlp']['initializer']
'''


class MyMLPBuilder(rl_net_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        rl_net_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(rl_net_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            rl_net_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            in_mlp_shape = input_shape[0]
            if len(self.units) == 0:
                out_size = input_shape[0]
            else:
                out_size = self.units[-1]

            # self.actor_mlp = build_sequential_mlp(
            #    in_mlp_shape, self.units, self.activation)
            # print('actor_mlp', self.actor_mlp)
            self.actor_mlp = MLP_NO_OUT(
                in_mlp_shape, self.units, self.activation)
            print('actor_mlp', self.actor_mlp)
            if self.separate:
                # self.critic_mlp = build_sequential_mlp(
                #    in_mlp_shape, self.units, self.activation)
                # print('critic_mlp', self.critic_mlp)
                self.critic_mlp = MLP_NO_OUT(
                    in_mlp_shape, self.units, self.activation)
                print('critic_mlp', self.critic_mlp)

            self.value = torch.nn.Linear(out_size, self.value_size)

            self.mu = torch.nn.Linear(out_size, actions_num)
            mu_init = self.init_factory.create(
                **self.space_config['mu_init'])

            sigma_init = self.init_factory.create(
                **self.space_config['sigma_init'])

            self.sigma = nn.Parameter(torch.zeros(
                actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            mu_init(self.mu.weight)
            sigma_init(self.sigma)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            if self.separate:
                a_out = c_out = obs
                a_out = a_out.contiguous().view(a_out.size(0), -1)
                c_out = c_out.contiguous().view(c_out.size(0), -1)

                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

                value = self.value(c_out)

                mu = self.mu(a_out)

                sigma = mu*0.0+self.sigma

                return mu, sigma, value, states
            else:
                out = obs

                out = self.actor_mlp(out)
                value = self.value(out)

                mu = self.mu(out)
                sigma = mu*0.0+self.sigma
                return mu, sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = self.activations_factory._builders.get(
                params['mlp']['activation'])
            self.initializer = params['mlp']['initializer']
            self.has_space = 'space' in params

            if self.has_space:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']

    def build(self, name, **kwargs):
        net = MyMLPBuilder.Network(self.params, **kwargs)
        print('net', net)
        return net
