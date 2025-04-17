from collections import deque
import random 
import torch

class ReplayBuffer: 
    def __init__(self, max: int = 10000):
        self.memory = deque(maxlen=max)
        
    def push(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int): 
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self): 
        return len(self.memory)
    