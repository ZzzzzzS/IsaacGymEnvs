import torch.nn as nn
import torch
import time
from CustomAlgorithm.ppo.AgentNets import MLP
import CustomAlgorithm.ppo.ppo as ppo_algo
import CustomAlgorithm.ppo.module as ppo_model
from rl_games.interfaces.base_algorithm import BaseAlgorithm
from rl_games.common import vecenv
from utils.rlgames_utils import RLGPUEnv
import CustomAlgorithm.ppo.helper as helper


class MyPPOAgent(BaseAlgorithm):
    def __init__(self, base_name, params, play=False):

        # basic params
        self.param = params
        self.config = params['config']
        self.network_model = params['network']
        self.base_name = base_name
        self.task_name = self.config["name"]
        self.env_name = self.config["env_name"]
        self.num_envs = self.config["num_actors"]
        self.device_name = torch.device(
            self.config['device'] if torch.cuda.is_available() else 'cpu')
        self.seed = self.param["seed"]
        torch.manual_seed(self.seed)

        # create env
        self.env: RLGPUEnv = vecenv.create_vec_env(
            self.env_name, self.num_envs)
        self.env_info = self.env.get_env_info()
        self.ob_dim = self.env_info["observation_space"].shape[0]
        self.act_dim = self.env_info["action_space"].shape[0]

        # input value norm
        self.input_norm = helper.RLRunningMeanStd(
            self.ob_dim).to(self.device_name) if self.config["normalize_input"] else None
        self.value_norm = helper.RLRunningMeanStd(
            1).to(self.device_name) if self.config["normalize_value"] else None

        # create agent
        self.update = 0
        self.obsv = None
        self.action = None

        self.n_steps = self.config["max_epochs"]
        self.horizon_length = self.config["horizon_length"]
        self.steps_per_ep = self.horizon_length*self.num_envs

        self.actor_net = ppo_model.MLP(
            self.network_model["mlp"]["units"], nn.ELU, self.ob_dim, self.act_dim)
        print("Actor", self.actor_net)
        self.critic_net = ppo_model.MLP(
            self.network_model["mlp"]["units"], nn.ELU, self.ob_dim, 1)
        print("Critic", self.critic_net)

        mlp_init = nn.Identity()
        for m in self.actor_net.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        for m in self.critic_net.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)

        self.actor_model = ppo_model.MultivariateGaussianDiagonalCovariance(
            self.act_dim, self.num_envs, self.network_model["space"]["continuous"]["sigma_init"]["val"], 42)

        self.actor = ppo_model.Actor(
            self.actor_net, self.actor_model, self.device_name, self.input_norm)
        self.critic = ppo_model.Critic(
            self.critic_net, self.device_name, self.input_norm, self.value_norm)

        # create saver
        self.saver = helper.ConfigurationSaver("./runs", self.task_name)
        self.save_frequency = self.config["save_frequency"]
        self.save_best_after = self.config["save_best_after"]

        # create algorithm
        num_mini_batches = self.steps_per_ep//self.config["minibatch_size"]
        if not play:
            assert num_mini_batches != 0
        self.ppo = ppo_algo.PPO(actor=self.actor,
                                critic=self.critic,
                                num_envs=self.num_envs,
                                num_transitions_per_env=self.config["horizon_length"],
                                num_learning_epochs=self.config["mini_epochs"],
                                num_mini_batches=num_mini_batches,
                                clip_param=self.config["e_clip"],
                                gamma=self.config["gamma"],
                                lam=self.config["tau"],
                                value_loss_coef=self.config["critic_coef"],
                                entropy_coef=self.config["entropy_coef"],
                                learning_rate=self.config["learning_rate"],
                                learning_rate_schedule=self.config["lr_schedule"],
                                max_grad_norm=self.config["grad_norm"],
                                desired_kl=self.config["kl_threshold"],
                                use_clipped_value_loss=self.config["clip_value"],
                                device=self.device_name,
                                log_dir=self.saver.summary_dir,
                                shuffle_batch=False,
                                truncate_grads=self.config["truncate_grads"],
                                adv_norm=self.config["normalize_advantage"],
                                value_norm=self.value_norm)

        # reward shaper
        self.use_reward_norm = self.config.get("reward_scaling", False)
        self.reward_shaper = self.config["reward_shaper"]
        self.reward_norm = helper.RewardScaling(
            [self.num_envs], self.config["gamma"], self.device_name)
        self.value_bootstrap = self.config.get("value_bootstrap", False)
        self.gamma = self.config["gamma"]

        # open tensorboard
        helper.tensorboard_launcher(self.saver.summary_dir)

    def device(self):
        return self.device_name

    def clear_stats(self):
        pass

    def train(self):
        # init
        self.obsv = self.env.reset()["obs"]
        self.reward_norm.reset()

        # update
        for update in range(self.n_steps+1):
            self.update = update
            start = time.time()
            average_dones, average_ll_performance, avg_rew_ep = self.train_epoch()
            end = time.time()
            print('----------------------------------------------------')
            print('{:>6}th iteration'.format(self.update))
            print('{:<40} {:>6}'.format("reward per step: ",
                  '{:0.10f}'.format(average_ll_performance)))
            print('{:<40} {:>6}'.format(
                "dones: ", '{:0.6f}'.format(average_dones)))
            print('{:<40} {:>6}'.format(
                "reward per ep: ", '{:0.6f}'.format(avg_rew_ep)))
            print('{:<40} {:>6}'.format(
                "time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
            print('{:<40} {:>6}'.format(
                "fps: ", '{:6.0f}'.format(self.steps_per_ep / (end - start))))
            print('----------------------------------------------------\n')

        print("MAX EPOCH REACHED!")
        self.save_weight()

    def play_step(self):
        self.action = self.ppo.act(self.obsv)
        return self.env.step(self.action)

    def train_epoch(self):
        reward_ll_sum = 0
        done_sum = 0

        # play step
        self.set_eval()
        for step in range(self.horizon_length):
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            new_obsv, rewards, dones, infos = self.play_step()

            shaped_rewards = self.reward_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                rest_value = self.critic.predict(self.obsv).squeeze(-1)
                shaped_rewards += self.gamma*rest_value*infos['time_outs']

            scaled_rewards = self.reward_norm(
                shaped_rewards, self.use_reward_norm)

            self.ppo.step(self.obsv, scaled_rewards, dones)

            self.obsv = new_obsv["obs"]

            done_sum = done_sum+torch.sum(dones)
            reward_ll_sum = reward_ll_sum+torch.sum(rewards)

        avg_rew = float(reward_ll_sum)/self.steps_per_ep
        avg_done = float(done_sum)/self.steps_per_ep
        avg_rew_ep = float(
            reward_ll_sum)/float(done_sum) if (float(done_sum) != 0.0) else float(0)

        # train step
        if_log = True if self.update % 5 == 0 else False
        self.set_train()

        self.ppo.update(actor_obs=self.obsv, value_obs=self.obsv,
                        log_this_iteration=if_log, update=self.update)
        self.actor.update()
        self.actor.distribution.enforce_minimum_std(
            (torch.ones(self.act_dim)*0.2).to(self.device_name))

        if if_log:
            self.ppo.log_other(self.update, avg_rew, avg_rew_ep)

        if_save = False if self.update < self.save_best_after else (True if (
            self.update-self.save_best_after) % self.save_frequency == 0 else False)
        if if_save:
            self.save_weight()

        return avg_done, avg_rew, avg_rew_ep

    def get_full_state_weights(self):
        pass

    def set_full_state_weights(self, weights):
        pass

    def get_weights(self):
        weights_dict = {
            'actor_architecture_state_dict': self.actor.architecture.state_dict(),
            'actor_distribution_state_dict': self.actor.distribution.state_dict(),
            'critic_architecture_state_dict': self.critic.architecture.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict()
        }
        if self.input_norm is not None:
            weights_dict['input_norm_dict'] = self.input_norm.state_dict()
        if self.value_norm is not None:
            weights_dict['value_norm_dict'] = self.value_norm.state_dict()
        return weights_dict

    def set_weights(self, weights):
        if weights == "":
            raise Exception(
                "\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
        print("\nRetraining from the checkpoint:", weights+"\n")

        checkpoint = torch.load(weights)
        self.actor.architecture.load_state_dict(
            checkpoint['actor_architecture_state_dict'])
        self.actor.distribution.load_state_dict(
            checkpoint['actor_distribution_state_dict'])
        self.critic.architecture.load_state_dict(
            checkpoint['critic_architecture_state_dict'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.input_norm is not None:
            self.input_norm.load_state_dict(checkpoint['input_norm_dict'])
        if self.value_norm is not None:
            self.value_norm.load_state_dict(checkpoint['value_norm_dict'])

    def save_weight(self):
        weights_dict = self.get_weights()
        save_path = self.saver.gen_model_save_name(self.update)
        print("save weight:", save_path["last_model_name"])
        torch.save(weights_dict, save_path["last_model_name"])
        print("save weight:", save_path["ep_model_name"])
        torch.save(weights_dict, save_path["ep_model_name"])

    def load_weight(self):
        model_path = self.saver.model_dir+self.task_name+".pt"
        self.set_weights(model_path)

    # callback function by rl_games runner.py
    def restore(self, fn, set_epoch=True):
        self.set_weights(fn)

    def set_train(self):
        self.actor_model.train()
        self.actor_net.train()
        self.critic_net.train()
        if self.input_norm is not None:
            self.input_norm.train()

    def set_eval(self):
        self.actor_model.eval()
        self.actor_net.eval()
        self.critic_net.eval()
        if self.input_norm is not None:
            self.input_norm.eval()
