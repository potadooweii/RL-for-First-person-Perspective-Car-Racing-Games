import torch
import torch.nn as nn
import numpy as np
from ppo.base_agent import PPOBaseAgent
from models.PPO_model import PPONet
from environment_wrapper.racecar_env import CarRacingEnvironment


class AtariPPOAgent(PPOBaseAgent):
    def __init__(self, config):
        super(AtariPPOAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        self.env = CarRacingEnvironment(
            self.scenario,
            test=False,
            discrete_action=~self.continuous,
            motor_magnitude=self.motor_magnitude,
            reward_coeff=self.car_racing_reward_coeff,
        )
        self.test_env = CarRacingEnvironment(
            self.scenario,
            test=True,
            discrete_action=~self.continuous,
            motor_magnitude=self.motor_magnitude,
            reward_coeff=self.car_racing_reward_coeff,
        )

        channel_dim = self.env.observation_space.shape[0]
        state_dim = self.env.observation_space.shape[1]
        action_dim = self.env.action_space.shape[0] if self.continuous else self.env.action_space.n

        self.net = PPONet(
            channel_dim,
            state_dim,
            action_dim,
            backbone=self.backbone,
            motor_magnitude=self.motor_magnitude,
            continuous=self.continuous,
        )
        self.net.to(self.device)

        self.optim = torch.optim.AdamW(self.net.parameters(), lr=self.lr)

    def decide_agent_actions(self, observation, eval=False):
        ### TODO ###
        # add batch dimension in observation
        # get action, value, logp from net

        observation = torch.tensor(
            np.asarray(observation), dtype=torch.float, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action, action_logprob, value, _ = self.net(observation, eval=eval)

        return (
            action.detach().cpu().numpy(),
            value.detach().cpu().numpy(),
            action_logprob.detach().cpu().numpy(),
        )

    def update(self):
        loss_counter = 0.0001
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(
            self.discount_factor_gamma, self.discount_factor_lambda
        )
        sample_count = len(batches["action"])
        batch_index = np.random.permutation(sample_count)

        observation_batch = {}
        for key in batches["observation"]:
            observation_batch[key] = batches["observation"][key][batch_index]
        action_batch = batches["action"][batch_index]
        return_batch = batches["return"][batch_index]
        adv_batch = batches["adv"][batch_index]
        v_batch = batches["value"][batch_index]
        logp_pi_batch = batches["logp_pi"][batch_index]

        for _ in range(self.update_count):
            for start in range(0, sample_count, self.batch_size):
                ob_train_batch = {}
                for key in observation_batch:
                    ob_train_batch[key] = observation_batch[key][
                        start : start + self.batch_size
                    ]
                ac_train_batch = action_batch[start : start + self.batch_size]
                return_train_batch = return_batch[start : start + self.batch_size]
                adv_train_batch = adv_batch[start : start + self.batch_size]
                v_train_batch = v_batch[start : start + self.batch_size]
                logp_pi_train_batch = logp_pi_batch[start : start + self.batch_size]

                ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
                ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
                ac_train_batch = torch.from_numpy(ac_train_batch)
                ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
                adv_train_batch = torch.from_numpy(adv_train_batch)
                adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
                logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
                logp_pi_train_batch = logp_pi_train_batch.to(
                    self.device, dtype=torch.float32
                )
                return_train_batch = torch.from_numpy(return_train_batch)
                return_train_batch = return_train_batch.to(
                    self.device, dtype=torch.float32
                )
                v_train_batch = torch.from_numpy(v_train_batch)
                v_train_batch = v_train_batch.to(self.device, dtype=torch.float32)

                # calculate loss and update network
                _, action_logprob, value, entropy = self.net(
                    ob_train_batch, old_action=ac_train_batch.squeeze()
                )
                logp_pi_train_batch = logp_pi_train_batch.squeeze()

                # calculate policy loss
                ratio = torch.exp(action_logprob - logp_pi_train_batch)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                surrogate_loss = (
                    -torch.min(ratio * adv_train_batch, clipped_ratio * adv_train_batch)
                ).mean()

                # calculate value loss
                value = value.view(-1)
                v_loss = (value - return_train_batch) ** 2
                v_loss = v_loss.mean()

                # calculate total loss
                entropy_loss = entropy.mean()
                loss = (
                    surrogate_loss
                    - self.entropy_coefficient * entropy_loss
                    + self.value_coefficient * v_loss
                )

                # update network
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
                self.optim.step()

                total_surrogate_loss += surrogate_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                total_loss += loss.item()
                loss_counter += 1

        self.writer.add_scalar(
            "PPO/Loss", total_loss / loss_counter, self.total_time_step
        )
        self.writer.add_scalar(
            "PPO/Surrogate Loss",
            total_surrogate_loss / loss_counter,
            self.total_time_step,
        )
        self.writer.add_scalar(
            "PPO/Value Loss", total_v_loss / loss_counter, self.total_time_step
        )
        self.writer.add_scalar(
            "PPO/Entropy", total_entropy / loss_counter, self.total_time_step
        )
        print(
            f"Loss: {total_loss / loss_counter}  \
			Surrogate Loss: {total_surrogate_loss / loss_counter}  \
			Value Loss: {total_v_loss / loss_counter}  \
			Entropy: {total_entropy / loss_counter}  \
			"
        )
