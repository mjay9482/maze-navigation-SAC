
Config = {
    "env": "PointMaze_UMaze-v3", 
    "warmup": 40, 
    "batch_size": 32,  
    "learning_rate": 0.008, 
    "alpha": 0.1,
    "gamma": 0.99,  
    "tau": 0.005,  
    "replay_buffer_size": int(1e6),  
    "exploration_scaling_factor": 1.5,  
    "episodes": 100,
    "max_episode_steps": 500, 
    "target_update_interval" : 1,
    "updates_per_step" : 4,
    "hidden_size" : 512,
    "LOG_SIGN_MIN" : -2,
    "LOG_SIGN_MAX" : 20, 
    "epsilon" : 1e-6
}

