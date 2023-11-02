import torch
import time
from rl_games.interfaces.base_algorithm import BaseAlgorithm
from rl_games.common import vecenv
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
from .helper import ConfigurationSaver, RewardScaling, tensorboard_launcher, BasicLogger
from .ppo_trainer import ppo_trainer
from .algo_maps import NET_MAP, MODULE_MAP
from .module import A2CModule_Continuous
from .network import BasicA2CNetwork


class online_algorithm_agent(BaseAlgorithm):
    def __init__(self, base_name, params):

        # basic params
        self.update = 0
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
        self.input_norm = self.config["normalize_input"]
        self.value_norm = self.config["normalize_value"]

        # trainning params
        self.n_steps = self.config["max_epochs"]
        self.horizon_length = self.config["horizon_length"]
        self.steps_per_ep = self.horizon_length*self.num_envs
        self.gamma = self.config["gamma"]

        # create agent
        self.net: BasicA2CNetwork = NET_MAP[self.network_model["name"]].create(self.ob_dim,
                                                                               self.act_dim, 1,  # TODO: Add support of value dim
                                                                               self.input_norm,
                                                                               self.value_norm,
                                                                               self.device_name,
                                                                               self.network_model)

        self.model: A2CModule_Continuous = MODULE_MAP[self.param["model"]["name"]](A2CNet=self.net,
                                                                                   init_std=self.network_model["space"][
                                                                                       "continuous"]["sigma_init"]["val"],
                                                                                   log_distrib=False,
                                                                                   device=self.device_name,
                                                                                   rnn=self.net.is_rnn(),
                                                                                   seq_len=self.config.get("seq_len", 4))
        self.model.to(self.device_name)

        # create saver
        self.saver = ConfigurationSaver("./runs", self.task_name)
        self.save_frequency = self.config["save_frequency"]
        self.save_best_after = self.config["save_best_after"]

        # create algorithm
        num_mini_batches = self.steps_per_ep//self.config["minibatch_size"]
        assert num_mini_batches != 0

        if self.param['algo']['name'] == 'ppo2':  # currently only support ppo2
            self.trainer = ppo_trainer(A2CModule=self.model,
                                       num_envs=self.num_envs,
                                       num_total_epoches=self.n_steps,
                                       num_transitions_per_env=self.config["horizon_length"],
                                       num_learning_epochs=self.config["mini_epochs"],
                                       num_mini_batches=num_mini_batches,
                                       rnn=self.net.is_rnn(),
                                       seq_len=self.config.get("seq_len", 4),
                                       clip_param=self.config["e_clip"],
                                       gamma=self.gamma,
                                       lam=self.config["tau"],
                                       value_loss_coef=self.config["critic_coef"],
                                       entropy_coef=self.config["entropy_coef"],
                                       bounds_loss_coef=self.config["bounds_loss_coef"],
                                       learning_rate=self.config["learning_rate"],
                                       learning_rate_schedule=self.config["lr_schedule"],
                                       max_grad_norm=self.config["grad_norm"],
                                       desired_kl=self.config["kl_threshold"],
                                       use_clipped_value_loss=self.config["clip_value"],
                                       device=self.device_name,
                                       truncate_grads=self.config["truncate_grads"],
                                       adv_norm=self.config["normalize_advantage"])
        else:
            raise NotImplementedError()

        # reward shaper
        self.use_reward_norm = self.config.get("reward_scaling", False)
        self.reward_shaper = self.config["reward_shaper"]
        self.reward_norm = RewardScaling(
            [self.num_envs], self.config["gamma"], self.device_name)
        self.value_bootstrap = self.config.get("value_bootstrap", False)

        # open tensorboard
        tensorboard_launcher(self.saver.summary_dir)
        self.logger = BasicLogger("runs", self.model)

        # temp variables
        self.obsv = None
        self.action = None
        self.value = None
        self.rnn = self.model.is_rnn()
        if self.rnn:
            self.states = self.model.get_default_rnn_states()

    def device(self):
        return self.device_name

    def clear_stats(self):
        pass

    def train(self):
        # init
        obs = self.env.reset()
        if "states" in obs:
            self.obsv = torch.cat((obs["obs"], obs["states"]), dim=-1)
        else:
            self.obsv = obs["obs"]

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

            if self.update % 5 == 0:
                performance_info = {
                    "performance/fps": self.steps_per_ep / (end - start),
                    "performance/time elapsed": end - start
                }
                self.logger.append_log_info(performance_info)
                self.logger.log(self.update)

        print("MAX EPOCH REACHED!")
        del self.logger
        self.save_weight()

    def play_one_step(self):
        # agent action
        if self.rnn:
            self.action, self.value, self.states = self.trainer.act(
                self.obsv, self.states)
        else:
            self.action, self.value, _ = self.trainer.act(self.obsv)

        # env action
        new_obsv, rewards, dones, infos = self.env.step(self.action)

        # update new observation and states
        if "states" in new_obsv:
            self.obsv = torch.cat(
                (new_obsv["obs"], new_obsv["states"]), dim=-1)
        else:
            self.obsv = new_obsv["obs"]

        if self.rnn:
            not_done = 1.0-dones
            self.states = self.states*not_done

        return rewards, dones, infos

    def train_epoch(self):
        reward_ll_sum = 0
        done_sum = 0

        # play step
        for step in range(self.horizon_length):
            # play one step
            rewards, dones, infos = self.play_one_step()

            # process and record data
            # reward shape
            shaped_rewards = self.reward_shaper(rewards)

            # if self.value_bootstrap and 'time_outs' in infos:
            # TODO: add value bootstrap support
            # rest_value = self.critic.predict(self.obsv).squeeze(-1)
            # shaped_rewards += self.gamma*rest_value*infos['time_outs']

            scaled_rewards = self.reward_norm(
                shaped_rewards, self.use_reward_norm)

            # record steps
            self.trainer.step(scaled_rewards, dones)

            # record performance
            done_sum = done_sum+torch.sum(dones)
            reward_ll_sum = reward_ll_sum+torch.sum(rewards)

        avg_rew = float(reward_ll_sum)/self.steps_per_ep
        avg_done = float(done_sum)/self.steps_per_ep
        avg_rew_ep = float(
            reward_ll_sum)/float(done_sum) if (float(done_sum) != 0.0) else float(0)

        # train step
        log_info = self.trainer.update(
            obs=self.obsv, state=self.states if self.rnn else None)

        self.model.distribution.enforce_minimum_std(
            (torch.ones(self.act_dim)*0.2).to(self.device_name))

        if self.update % 5 == 0:
            self.logger.append_log_info(log_info)
            reward_info = {
                "agent/averange_step_reward": avg_rew,
                "agent/averange_episode_reward": avg_rew_ep
            }
            self.logger.append_log_info(reward_info)

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
            'model_state_dict': self.model.state_dict(),
            'trainer_state_dict': self.trainer.optimizer.state_dict()
        }
        return weights_dict

    def set_weights(self, weights):
        if weights == "":
            raise Exception(
                "\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
        print("\nRetraining from the checkpoint:", weights+"\n")

        checkpoint = torch.load(weights)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(
            checkpoint["trainer_state_dict"])

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
