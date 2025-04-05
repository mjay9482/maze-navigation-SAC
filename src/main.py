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
    
    env = gym.make(
        env_name, 
        max_episode_steps=Config["max_episode_steps"], 
        maze_map=[[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1]], 
        # render_mode="human"
    )
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
    
    memory = ReplayBuffer(
        Config["replay_buffer_size"], input_size=obs_size, n_action=env.action_space.shape[0]
    )
    
    agent.train(env = env, env_name = env_name, memory = memory, episodes = Config["episodes"],
                batch_size = Config["batch_size"], updates_per_step = Config["updates_per_step"],
                summary_writer_name=f"maze_map={Config['alpha']}_lr={Config['learning_rate']}_hs={Config['hidden_size']}_a={Config['alpha']}_phase_1",
                max_episode_steps= Config["max_episode_steps"])
    
    L_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    max_episode_steps_phase_2 = 500 
    
    env = gym.make(
        env_name, 
        max_episode_steps= max_episode_steps_phase_2, 
        maze_map=L_map
        # render_mode="human"
    )
    env = RoboGymObservationWrapper(env)

    agent.train(env = env, env_name = env_name, memory = memory, episodes = 500,
        batch_size = Config["batch_size"], updates_per_step = Config["updates_per_step"],
        summary_writer_name=f"maze_map={Config['alpha']}_lr={Config['learning_rate']}_hs={Config['hidden_size']}_a={Config['alpha']}_phase_2",
        max_episode_steps= max_episode_steps_phase_2)
    
    env.close()