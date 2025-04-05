import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.distributions import Normal
from config import Config

LOG_SIGN_MIN = Config["LOG_SIGN_MIN"]
LOG_SIGN_MAX = Config["LOG_SIGN_MAX"]
epsilon = Config["epsilon"]

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name="critic_network"):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init)
        self.to(self.device)  
        
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)  
        q1_val = self.q1(xu)
        q2_val = self.q2(xu)
        return q1_val, q2_val  

    def save_checkpoint(self):
        try:
            torch.save(self.state_dict(), self.checkpoint_file)
            print(f"Checkpoint saved at {self.checkpoint_file}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self):
        try:
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
            print(f"Checkpoint loaded from {self.checkpoint_file}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name="actor_network"):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_layer = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init)
        self.to(self.device)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        if action_space is None: 
            self.action_scale = torch.tensor(1.0, device=self.device)
            self.action_bias = torch.tensor(0.0, device=self.device)
        else:
            self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, device=self.device)
            self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0, device=self.device)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIGN_MIN, max=LOG_SIGN_MAX) 
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()  
        normal = Normal(mean, std)

        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)  
        action = y_t * self.action_scale + self.action_bias 

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def save_checkpoint(self):
        try:
            torch.save(self.state_dict(), self.checkpoint_file)
            print(f"Checkpoint saved at {self.checkpoint_file}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self):
        try:
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
            print(f"Checkpoint loaded from {self.checkpoint_file}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            

class PredictiveModel(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name="predictive_network"):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_inputs)
        )

        self.apply(weights_init)
        self.to(self.device)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)
    
    def save_checkpoint(self):
        try:
            torch.save(self.state_dict(), self.checkpoint_file)
            print(f"Checkpoint saved at {self.checkpoint_file}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self):
        try:
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
            print(f"Checkpoint loaded from {self.checkpoint_file}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")