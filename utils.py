import gym
import numpy as np
import cv2
import torch


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
        self.frames = []  # To store the frames

    def step(self, action):
        total_reward = 0
        self.frames = []  # Clear the frames at the beginning of each step
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = np.transpose(obs, (2, 0, 1))
            self.frames.append(obs)  # Store the frame
            total_reward += reward

            if terminated or truncated:
                break

        # Stack the frames to return them together
        stacked_obs = np.stack(self.frames, axis=0)
        stacked_obs = torch.tensor(stacked_obs, dtype=torch.float32)
        # print('stacked obs shape ' + str(stacked_obs.shape))
        return stacked_obs, total_reward, terminated, truncated, info
