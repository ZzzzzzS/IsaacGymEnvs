import torch.nn as nn
import torch
import time
import MyAlgorithm.ppo.ppo as ppo_algo
import MyAlgorithm.ppo.module as ppo_model
import MyAlgorithm.ppo.PPOAgent as ppo_agent
from rl_games.common import vecenv
from utils.rlgames_utils import RLGPUEnv
import MyAlgorithm.ppo.helper as helper
from MyAlgorithm.fake_base_player import BasePlayer


class MyPPOPlayer(BasePlayer):
    def __init__(self, params):
        self.agent = ppo_agent.MyPPOAgent('', params, True)
        self.horizon_length = params["config"]["horizon_length"]
        self.update = 0

    def restore(self, fn: str):
        self.agent.restore(fn)

    @property
    def device(self):
        return self.agent.device

    def set_weights(self, weights: str):
        self.agent.set_weights(weights)

    def run(self):
        self.agent.obsv = self.agent.env.reset()["obs"]
        while True:
            start = time.time()
            average_dones, average_ll_performance = self.run_step()
            end = time.time()
            print('----------------------------------------------------')
            print('{:>6}th iteration'.format(self.update))
            print('{:<40} {:>6}'.format("average ll reward: ",
                  '{:0.10f}'.format(average_ll_performance)))
            print('{:<40} {:>6}'.format(
                "dones: ", '{:0.6f}'.format(average_dones)))
            print('{:<40} {:>6}'.format(
                "time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
            print('{:<40} {:>6}'.format(
                "fps: ", '{:6.0f}'.format(self.agent.steps_per_ep / (end - start))))
            print('----------------------------------------------------\n')

    def run_step(self):
        reward_ll_sum = 0
        done_sum = 0
        for step in range(self.horizon_length):
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            new_obsv, rewards, dones, infos = self.agent.play_step()

            shaped_obsv = new_obsv["obs"]

            self.agent.obsv = shaped_obsv
            done_sum = done_sum+torch.sum(dones)
            reward_ll_sum = reward_ll_sum+torch.sum(rewards)

        avg_rew = float(reward_ll_sum)/self.agent.steps_per_ep
        avg_done = float(done_sum)/self.agent.steps_per_ep
        return avg_done, avg_rew
