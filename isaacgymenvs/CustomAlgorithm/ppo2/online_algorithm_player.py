from isaacgymenvs.CustomAlgorithm.fake_base_player import BasePlayer
from .online_algorithm_agent import online_algorithm_agent
import torch
import time


class online_algorithm_player(BasePlayer):
    def __init__(self, params):
        super().__init__(params)
        self.agent = online_algorithm_agent('play', params)
        self.horizon_length = params["config"]["horizon_length"]

    def restore(self, fn: str):
        self.agent.restore(fn)

    def device(self):
        return self.agent.device()

    def set_weights(self, weights: str):
        self.agent.set_weights(weights)

    def run(self):
        obs = self.agent.env.reset()
        if "states" in obs:
            self.agent.obsv = torch.cat((obs["obs"], obs["states"]), dim=-1)
        else:
            self.agent.obsv = obs["obs"]

        while True:
            start = time.time()
            average_dones, average_ll_performance = self.run_step()
            end = time.time()
            print('----------------------------------------------------')
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
            rewards, dones, infos = self.agent.play_one_step()
            done_sum = done_sum+torch.sum(dones)
            reward_ll_sum = reward_ll_sum+torch.sum(rewards)

        avg_rew = float(reward_ll_sum)/self.agent.steps_per_ep
        avg_done = float(done_sum)/self.agent.steps_per_ep
        return avg_done, avg_rew
