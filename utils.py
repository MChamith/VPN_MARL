import gym
import numpy as np
import cv2

class PreprocessAtari(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Resize the image to (84, 84)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('frostbite.jpg', obs)
        return np.expand_dims(obs, axis=-1)  # Add channel dimension


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break
        # print('returning total reward ' + str(reward))
        return obs, total_reward, terminated, truncated, info
