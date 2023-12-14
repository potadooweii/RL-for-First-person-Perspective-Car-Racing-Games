from collections import deque
import cv2

from racecar_gym.env import RaceEnv
import numpy as np

class CarRacingEnvironment:
	def __init__(self, scenario = "austria_competition", render_mode: str = "rgb_array_birds_eye", test=False, N_frame:int = 4, frame_skip: int = 4):
		
		#
		self.test = test
		self.scenario = scenario
		reset_when_collision = True if "austria" in scenario else False
		if self.test:
			self.env = RaceEnv(scenario=scenario, render_mode='rgb_array_birds_eye', reset_when_collision=reset_when_collision)
		else:
			self.env = RaceEnv(scenario=scenario, render_mode=render_mode, reset_when_collision=reset_when_collision)
		
		#
		self.action_space = self.env.action_space
		self.original_observation_space = self.env.observation_space
		self.observation_shape = np.array([4,32,32])
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)

	def step(self, action):
		obs, reward, terminates, truncates, info = self.env.step(action)
		original_reward = reward
		original_terminates = terminates
		self.ep_len += 1

		collision_penalty = 0.2 * int(info['wall_collision'])
		velocity_reward = np.sum(info['velocity'] ** 2)
		reward = original_reward + 1e-4 * velocity_reward + 1e-3 * info['obstacle'] - collision_penalty

		obs = self.resize_obs(obs)
		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)


		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = original_reward
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			# terminates = original_terminates

		return obs, reward, terminates, truncates, info
	
	def reset(self, test: bool = False):
		obs, info = self.env.reset(options=dict(mode='random'))
		self.ep_len = 0

		obs = self.resize_obs(obs)
		# frame stacking
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def resize_obs(self, obs):
		ret = np.transpose(obs, (1,2,0))
		ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY) # 128x128
		ret = cv2.resize(ret, (32, 32), interpolation=cv2.INTER_AREA) # 32x32
		return ret

	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

# if __name__ == '__main__':
# 	env = CarRacingEnvironment(scenario="circle_cw_competition_collisionstop")
# 	obs, info = env.reset()
# 	done = 0
# 	total_reward = 0
# 	total_length = 0
# 	t = 0
# 	while not done:
# 		t += 1
# 		action = env.action_space.sample()
# 		action[0] = 1
# 		obs, reward, terminates, truncates, info = env.step(action)
# 		total_reward += reward
# 		total_length += 1
# 		env.render()
# 		if terminates or truncates:
# 			done = 1

# 	print("Total reward: ", total_reward)
# 	print("Total length: ", total_length)
# 	env.close()
