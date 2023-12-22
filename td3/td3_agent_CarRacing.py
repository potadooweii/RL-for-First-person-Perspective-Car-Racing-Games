import torch
import torch.nn as nn
import numpy as np
from td3.base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.racecar_env import CarRacingEnvironment
import random
from td3.base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(self.scenario, test=False)
		self.test_env = CarRacingEnvironment(self.scenario, test=True)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_shape[1], self.env.action_space.shape[0], self.env.observation_shape[0])
		self.critic_net1 = CriticNetSimple(self.env.observation_shape[1], self.env.action_space.shape[0], self.env.observation_shape[0])
		self.critic_net2 = CriticNetSimple(self.env.observation_shape[1], self.env.action_space.shape[0], self.env.observation_shape[0])
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_shape[1], self.env.action_space.shape[0], self.env.observation_shape[0])
		self.target_critic_net1 = CriticNetSimple(self.env.observation_shape[1], self.env.action_space.shape[0], self.env.observation_shape[0])
		self.target_critic_net2 = CriticNetSimple(self.env.observation_shape[1], self.env.action_space.shape[0], self.env.observation_shape[0])
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.AdamW(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.AdamW(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.AdamW(self.critic_net2.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise
		self.noise = GaussianNoise(self.env.action_space.shape[0])
		self.max_action = torch.tensor([0.5, 1], device=self.device)
		self.min_action = torch.tensor([0.1, -1], device=self.device)
		self.noise_clip_rate = config['noise_clip_rate']
	
	def decide_agent_actions(self, state, sigma=0):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		state = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(0)

		with torch.no_grad():
			action = self.actor_net(state)

		if sigma != 0:
			self.noise.std = sigma
			exploration_noise = torch.tensor(self.noise.generate(), device=self.device).clamp(
				self.noise_clip_rate * self.min_action,
				self.noise_clip_rate * self.max_action
			)
			action = (action + exploration_noise).clamp(self.min_action, self.max_action)

		return action.squeeze(0).cpu().numpy()
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		if self.PER:
			idxs, state, action, reward, next_state, done, is_weights = self.replay_buffer.sample(self.batch_size, self.device)
		else:
			state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = torch.tensor(self.noise.generate(), dtype=torch.float32, device=self.device).clamp(
				self.noise_clip_rate * self.min_action,
				self.noise_clip_rate * self.max_action
			)
			next_action = (
				self.target_actor_net(next_state) + noise
			).clamp(self.min_action, self.max_action)

			# Compute the target Q value
			target_Q = torch.min(
				self.target_critic_net1(next_state, next_action),
				self.target_critic_net2(next_state, next_action),
			).detach()
			target_Q = reward + (1 - done) * self.gamma * target_Q

		## Update Critic ##
		# critic loss & critic loss function
		if self.PER:
			criterion = self.weighted_mse
			q_value1 = self.critic_net1(state, action)
			critic_loss1 = criterion(q_value1, target_Q, is_weights)
			q_value2 = self.critic_net2(state, action)
			critic_loss2 = criterion(q_value2, target_Q, is_weights)
			errors1 = np.abs((q_value1 - target_Q).detach().cpu().numpy())
			self.replay_buffer.batch_update(idxs, errors1)
		else:
			criterion = nn.MSELoss()
			q_value1 = self.critic_net1(state, action)
			critic_loss1 = criterion(q_value1, target_Q)
			q_value2 = self.critic_net2(state, action)
			critic_loss2 = criterion(q_value2, target_Q)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_net2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		## Delayed Actor(Policy) Updates ##
		if self.total_time_step % self.update_freq == 0:
			## update actor ##
			# actor loss
			# select action a from behavior actor network (a is different from sample transition's action)
			# get Q from behavior critic network, mean Q value -> objective function
			# maximize (objective function) = minimize -1 * (objective function)
			action = self.actor_net(state)
			actor_loss = -self.critic_net1(state, action).mean()
			# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
	
	def weighted_mse(self, expected, targets, is_weights):
		"""Custom loss function that takes into account the importance-sampling weights."""
		td_error = expected - targets
		weighted_squared_error = is_weights * td_error * td_error
		return torch.sum(weighted_squared_error) / torch.numel(weighted_squared_error)
