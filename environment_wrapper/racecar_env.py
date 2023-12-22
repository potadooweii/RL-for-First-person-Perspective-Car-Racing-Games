from collections import deque
import cv2

from racecar_gym.env import RaceEnv
import numpy as np

class CarRacingEnvironment:
	def __init__(self, scenario = "austria_competition", render_mode: str = "rgb_array_birds_eye", test=False, N_frame:int = 4, frame_skip: int = 4, continuous_action: bool = True):
		
		#
		self.test = test
		self.scenario = scenario # circle_cw_competition_collisionStop, austria_competition
		reset_when_collision = True if "austria" in scenario else False
		if self.test:
			self.env = RaceEnv(scenario=scenario, render_mode='rgb_array_birds_eye', reset_when_collision=reset_when_collision)
		else:
			self.env = RaceEnv(scenario=scenario, render_mode=render_mode, reset_when_collision=reset_when_collision)
		
		self.discrete_to_continuous_action_map = {
			0: [1, 0], # accelerate forward
			1: [0.3, -0.3], # slowly turn left
			2: [0.3, 0.3], # slowly turn right
			3: [0.1, -1], # turn left
			4: [0.1, 1], # turn right
			5: [-1, 0] # brake
		}

		#
		self.continuous_action = continuous_action
		if self.continuous_action:
			self.action_space = self.env.action_space
		else:
			self.action_space = np.array(list(self.discrete_to_continuous_action_map.keys()))
		self.observation_space = self.env.observation_space
		self.observation_shape = np.array([4,64,64])
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)
		self.skip_flag = False # skip once every 2 frames

		#
		self.accumulated_progress = 0

	def step(self, action):
		if not self.continuous_action:
			action = self.discrete_to_continuous_action_map[action]

		obs, reward, terminates, truncates, info = self.env.step(action)
		delta_progress = reward
		self.accumulated_progress += delta_progress
		original_terminates = terminates
		self.ep_len += 1

		collision_penalty = 0.5 if info['wall_collision'] else 0
		velocity_reward = np.sum(info['velocity'][:3] ** 2)
		reward = delta_progress + 0.1 * self.accumulated_progress + 1e-5 * velocity_reward + 2e-3 * info['obstacle'] - collision_penalty
		reward = -0.001 if delta_progress == 0 else reward

		obs = self.resize_obs(obs)
		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		if self.skip_flag:
			self.skip_flag = False
		else:
			self.frames.append(obs)
			self.skip_flag = True
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = delta_progress
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			# terminates = original_terminates

		return obs, reward, terminates, truncates, info
	
	def reset(self, test: bool = False):
		if test:
			obs, info = self.env.reset()
		else:
			obs, info = self.env.reset(options=dict(mode='random'))

		self.ep_len = 0
		# init reward
		self.accumulated_progress = 0

		obs = self.resize_obs(obs)
		# frame stacking
		self.skip_flag = False
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def resize_obs(self, obs):
		ret = np.transpose(obs, (1,2,0))
		ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY) # 128x128
		ret = cv2.resize(ret, (64, 64), interpolation=cv2.INTER_AREA) # 64x64
		return ret

	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = CarRacingEnvironment(scenario="circle_cw_competition_collisionstop", continuous_action=False)
	obs, info = env.reset()
	done = 0
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = 0
		obs, reward, terminates, truncates, info = env.step(action)
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = 1

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
