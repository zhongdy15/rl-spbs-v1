import gym
import numpy as np
from gym import spaces


class DisabledWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, disabled_action_idx: int = 4, disabled_default_value = 0):
        super().__init__(env)
        self.disabled_action_idx = disabled_action_idx
        self.disabled_default_value = disabled_default_value

    def step(self, action):
        disabled_action = np.copy(action)
        disabled_action[self.disabled_action_idx] = self.disabled_default_value
        return self.env.step(disabled_action)
