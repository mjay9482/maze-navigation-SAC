import gymnasium as gym 
import numpy as np 
from gymnasium import ObservationWrapper

class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)  

    def reset(self):
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        return obs, info 
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.process_observation(obs)
        return obs, reward, done, truncated, info

    def process_observation(self, obs):
        obs_map = obs['observation']
        obs_achieved_goal = obs['achieved_goal']
        obs_desired_goal = obs['desired_goal']
        
        obs_concat = np.concatenate((obs_map, obs_achieved_goal, obs_desired_goal))
        
        return obs_concat
