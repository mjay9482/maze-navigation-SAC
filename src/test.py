import numpy as np
import gymnasium as gym
import gymnasium_robotics
from model import *
from agent import Agent
from config import Config
from replay_buffer import ReplayBuffer
from gym_robotics_custom import RoboGymObservationWrapper

if __name__ == "__main__":
    gym.register_envs(gymnasium_robotics)
    
    env_name = Config["env"]
    
    # Simplified maze map - smaller and less complex
    simplified_maze = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    env = gym.make(
        env_name, 
        max_episode_steps=Config["max_episode_steps"], 
        render_mode="human",
        maze_map=simplified_maze)

    env = RoboGymObservationWrapper(env)
    
    obs, info = env.reset()
    obs_size = obs.shape[0]
    
    agent = Agent(
        obs_size, env.action_space,
        gamma=Config["gamma"], tau=Config["tau"], alpha=Config["alpha"],
        target_update_interval=Config["target_update_interval"],
        hidden_size=Config["hidden_size"], learning_rate=Config["learning_rate"],
        exploration_scaling_factor=Config["exploration_scaling_factor"]
    )
    
    agent.load_checkpoint(evaluate=True)
    agent.test(env=env, episodes=10, max_episode_steps=500)
    env.close()