from collections import deque
import cv2

from gymnasium import spaces
from racecar_gym.env import RaceEnv
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class CarRacingRewardCoefficient:
    accumulated_progress_coeff: float = 0.1
    velocity_coeff: float = 1e-5
    position_coeff: float = 2e-3
    collision_coeff: float = 0.5
    

class CarRacingEnvironment:
    def __init__(
        self,
        scenario="austria_competition_collisionStop",
        test: bool = False,
        frame_stack_count: int = 4,
        obs_size: int = 64,
        frame_skip_count: int = 1,
        discrete_action: bool = True,
        motor_magnitude: float = 0.03,
        reward_coeff: Optional[CarRacingRewardCoefficient] = None,
    ):
        # Init Environment
        self.test = test
        legal_scenarios = [
            "circle_cw_competition_collisionstop",
            "austria_competition",
            "austria_competition_collisionstop",
        ]
        assert (
            scenario.lower() in legal_scenarios
        ), f"only {legal_scenarios} are available."
        self.scenario = scenario.lower()

        if self.test:
            reset_when_collision = True if "austria" in self.scenario else False
            self.env = RaceEnv(
                scenario=scenario,
                render_mode="rgb_array_birds_eye",
                reset_when_collision=reset_when_collision,
            )
        else:
            self.env = RaceEnv(
                scenario=scenario,
                render_mode="rgb_array_birds_eye",
                reset_when_collision=False,
            )

        # Rescale Observation
        self.frame_stack_count = frame_stack_count
        orignal_obs_size = self.env.observation_space.shape[-1]
        assert (
            obs_size <= orignal_obs_size
        ), f"The max size of obs_size is {orignal_obs_size}"
        self.obs_size = obs_size
        self.observation_space = spaces.Box(
            0,
            255,
            shape=(self.frame_stack_count, self.obs_size, self.obs_size),
            dtype=np.uint8,
        )
        self.frame_skip_count = frame_skip_count

        # Init action space
        self.max_motor_magnitude = motor_magnitude
        self.discrete_action = discrete_action
        if self.discrete_action:
            self.action_space = spaces.Discrete(2)
            self._discrete_action_to_direction = {
                0: np.array([motor_magnitude, 1]),
                1: np.array([motor_magnitude, -1]),
            }
        else:
            self.action_space = spaces.Box(
                0.01, self.max_motor_magnitude, shape=(2,), dtype=np.float32
            )

        # Init Reward function
        self.reward_coeff = CarRacingRewardCoefficient() if reward_coeff is None else reward_coeff

        # Global variables
        self.episode_len = 0
        self.frames = deque(maxlen=self.frame_stack_count)
        self.accumulated_progress = 0

    def step(self, action):
        if self.discrete_action:
            action = self._discrete_action_to_direction[action]

        total_reward = 0
        total_progress = 0
        for _ in range(self.frame_skip_count):
            obs, reward, terminates, truncates, info = self.env.step(action)

            # Calculate reward
            delta_progress = reward
            total_progress += delta_progress
            self.accumulated_progress += delta_progress
            reward = self._calculate_reward(delta_progress, info, self.reward_coeff)

            total_reward += reward
            self.episode_len += 1

        total_reward = 0 if total_progress == 0 else total_reward
        total_reward = total_progress if self.test else total_reward

        # Frame Stacking
        obs = self._resize_obs(obs)
        # # save image for debugging
        # filename = "images/image" + str(self.episode_len) + ".jpg"
        # cv2.imwrite(filename, obs)
        self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)

        return obs, total_reward, terminates, truncates, info

    def reset(self, test: bool = False):
        # Init Env & Global Variables
        obs, info = (
            self.env.reset() if test else self.env.reset(options=dict(mode="random"))
        )
        obs = self._resize_obs(obs)
        self.frames = deque(
            [obs for _ in range(self.frames.maxlen)], maxlen=self.frame_stack_count
        )
        obs = np.stack(self.frames, axis=0)

        self.episode_len = 0
        self.accumulated_progress = 0

        return obs, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def _resize_obs(self, obs):
        ret = np.transpose(obs, (1, 2, 0))
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)  # 128x128
        ret = cv2.resize(
            ret, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA
        )
        return ret

    def _calculate_reward(self, delta_progress: float, info: dict, coefficients: CarRacingRewardCoefficient):
        progress_reward = delta_progress + coefficients.accumulated_progress_coeff * self.accumulated_progress
        position_reward = coefficients.position_coeff * info["obstacle"]
        velocity_reward = coefficients.velocity_coeff * np.sum(info["velocity"][0] ** 2)
        collision_penalty = coefficients.collision_coeff * int(info["wall_collision"])
        
        return (
            progress_reward
            + velocity_reward
            + position_reward
            - collision_penalty
        )


# if __name__ == '__main__':
# 	env = CarRacingEnvironment(scenario="circle_cw_competition_collisionStop", discrete_action=True)
# 	print("observation space: ", env.observation_space)
# 	print("action space: ",env.action_space)
# 	obs, info = env.reset()

# 	done = False
# 	total_reward = 0
# 	total_length = 0
# 	while not done:
# 		action = 0
# 		obs, reward, terminates, truncates, info = env.step(action)
# 		total_reward += reward
# 		total_length += 1
# 		if terminates or truncates:
# 			done = True

# 	print("Total reward: ", total_reward)
# 	print("Total length: ", total_length)
# 	env.close()
