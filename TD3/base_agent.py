import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from replay_buffer.PER import PER
from abc import ABC, abstractmethod
from environment_wrapper.racecar_env import CarRacingRewardCoefficient


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = np.ones(dim) * std if std else np.ones(dim) * 0.1

    def reset(self):
        pass

    def generate(self):
        return np.random.normal(self.mu, self.std)


class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (
            self.x
            + self.theta * (self.mean - self.x) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        return self.x


class TD3BaseAgent(ABC):
    def __init__(self, config):
        # env
        self.scenario = config["scenario"]

        # hyperparameters
        ## car racing reward
        self.car_racing_reward_coeff = CarRacingRewardCoefficient(
            accumulated_progress_coeff=config["accumulated_progress_coeff"],
            velocity_coeff=config["velocity_coeff"],
            position_coeff=config["position_coeff"],
            collision_coeff=config["collision_coeff"],
        )
        ## action
        self.motor_magnitude = config["motor_magnitude"]
        ## NN
        self.backbone = config["backbone"]
        ## training
        self.training_steps = int(config["training_steps"])
        self.total_episode = int(config["total_episode"])
        self.lra = config["lra"]
        self.lrc = config["lrc"]
        self.batch_size = int(config["batch_size"])
        ### TD3 only
        self.warmup_steps = config["warmup_steps"]
        self.gamma = config["discount_factor_gamma"]
        self.tau = config["soft_update_tau"]
        self.update_freq = config["update_freq"]
        self.PER = config["PER"]

        # training
        self.training_steps = int(config["training_steps"])

        # hardware
        self.device = torch.device(
            config["device"] if torch.cuda.is_available() else "cpu"
        )

        # eval
        self.eval_interval = config["eval_interval"]
        self.eval_episode = config["eval_episode"]

        # log
        self.writer = SummaryWriter(config["logdir"])

        # global variables
        self.total_time_step = 0
        if self.PER:
            self.replay_buffer = PER(int(config["replay_buffer_capacity"]))
        else:
            self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))

    @abstractmethod
    def decide_agent_actions(self, state, sigma=0.0):
        ### TODO ###
        # based on the behavior (actor) network and exploration noise

        return NotImplementedError

    def update(self):
        # update the behavior networks
        self.update_behavior_network()
        # update the target networks
        if self.total_time_step % self.update_freq == 0:
            self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
            self.update_target_network(
                self.target_critic_net1, self.critic_net1, self.tau
            )
            self.update_target_network(
                self.target_critic_net2, self.critic_net2, self.tau
            )

    @abstractmethod
    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        ### TODO ###
        # calculate the loss and update the behavior network

        return NotImplementedError

    @staticmethod
    def update_target_network(target_net, net, tau):
        # update target network by "soft" copying from behavior network
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_((1 - tau) * target.data + tau * behavior.data)

    def train(self):
        for episode in range(self.total_episode):
            total_reward = 0
            state, infos = (
                self.env.reset(test=True) if episode % 5 == 0 else self.env.reset()
            )
            self.noise.reset()

            frac = 1.0 - episode / self.total_episode
            lrnow = 0.9 * frac * self.lra + 0.1 * self.lra
            self.actor_opt.param_groups[0]["lr"] = lrnow
            self.critic_opt1.param_groups[0]["lr"] = lrnow
            self.critic_opt2.param_groups[0]["lr"] = lrnow

            for t in range(10000):
                if self.total_time_step < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    # exploration degree
                    sigma = max(0.1 * (1 - episode / self.total_episode), 0.01)
                    action = self.decide_agent_actions(state, sigma=sigma)

                next_state, reward, terminates, truncates, _ = self.env.step(action)
                if self.PER:
                    self.replay_buffer.append(
                        state,
                        action,
                        [reward],
                        next_state,
                        [int(terminates)],
                        error=reward,
                    )
                else:
                    self.replay_buffer.append(
                        state, action, [reward], next_state, [int(terminates)]
                    )
                if self.total_time_step >= self.warmup_steps:
                    self.update()

                self.total_time_step += 1
                total_reward += reward
                state = next_state
                if terminates or truncates:
                    self.writer.add_scalar(
                        "Train/Episode Reward", total_reward, self.total_time_step
                    )
                    print(
                        "Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}".format(
                            self.total_time_step, episode + 1, t, total_reward
                        )
                    )

                    break

                if self.total_time_step >= self.training_steps:
                    return

            if (episode + 1) % self.eval_interval == 0:
                # save model checkpoint
                avg_score = self.evaluate()
                self.save(
                    os.path.join(
                        self.writer.log_dir,
                        f"model_{self.total_time_step}_{int(avg_score)}.pth",
                    )
                )
                self.writer.add_scalar(
                    "Evaluate/Episode Reward", avg_score, self.total_time_step
                )

    def evaluate(self):
        print("==============================================")
        print("Evaluating...")
        all_rewards = []
        for episode in range(self.eval_episode):
            total_reward = 0
            state, infos = self.test_env.reset(test=True)
            for t in range(10000):
                action = self.decide_agent_actions(state)
                next_state, reward, terminates, truncates, _ = self.test_env.step(
                    action
                )
                total_reward += reward
                state = next_state
                if terminates or truncates:
                    print(
                        "Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}".format(
                            episode + 1, t, total_reward
                        )
                    )
                    all_rewards.append(total_reward)
                    break

        avg = sum(all_rewards) / self.eval_episode
        print(f"average score: {avg}")
        print("==============================================")
        return avg

    # save model
    def save(self, save_path):
        torch.save(
            {
                "actor": self.actor_net.state_dict(),
                "critic1": self.critic_net1.state_dict(),
                "critic2": self.critic_net2.state_dict(),
            },
            save_path,
        )

    # load model
    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net1.load_state_dict(checkpoint["critic1"])
        self.critic_net2.load_state_dict(checkpoint["critic2"])

    # load model weights and evaluate
    def load_and_evaluate(self, load_path):
        self.load(load_path)
        self.evaluate()
