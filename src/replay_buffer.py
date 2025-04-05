import numpy as np 

class ReplayBuffer: 
    
    def __init__(self, max_size, input_size, n_action):
        
        self.m_size = max_size 
        self.m_c = 0 
        self.state_mem = np.zeros((self.m_size, input_size), dtype = np.float32)
        self.next_state_mem = np.zeros((self.m_size, input_size), dtype = np.float32)
        self.action_mem = np.zeros((self.m_size, n_action), dtype = np.float32)
        self.reward_mem = np.zeros(self.m_size, dtype = np.float32)
        self.terminal_mem = np.zeros(self.m_size, dtype = np.float32)
        
    def store_transitions(self, state, action, reward, next_state, done):
        
        idx = self.m_c % self.m_size 
        
        self.state_mem[idx] = state
        self.next_state_mem[idx] = next_state 
        self.action_mem[idx] = action 
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = done 
        
        self.m_c += 1
        
    def can_sample(self, batch_size):
        
        if self.m_c > (batch_size*5):
            return True
        else:
            return False
    
    def sample_buffer(self, batch_size):
        
        max_mem = min(self.m_c , self.m_size)

        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_mem[batch]
        next_states = self.next_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]
        
        return states, actions, rewards, next_states, dones