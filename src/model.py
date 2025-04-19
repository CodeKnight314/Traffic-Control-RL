import torch 
import torch.nn as nn 

class DQN(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int): 
        super().__init__()
        
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        ])
        
    def forward(self, x: torch.Tensor): 
        return self.net(x)
    
    def load_weights(self, path: str): 
        self.load_state_dict(torch.load(path))
        
    def save_weights(self, path: str): 
        torch.save(self.state_dict(), path)
        
class DuelDQN(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int): 
        super().__init__()
        self.input = nn.Sequential(*[
            nn.Linear(input_dim, 128), 
            nn.ReLU(inplace=True)
        ])
        
        self.main = nn.Sequential(*[
            nn.Linear(128, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), 
            nn.ReLU(inplace=True), 
        ])
        
        self.value = nn.Sequential(*[ 
            nn.Linear(128, 1)
        ])
        
        self.advantage = nn.Sequential(*[
            nn.Linear(128, output_dim)
        ])
        
    def forward(self, x: torch.Tensor): 
        x = self.input(x)
        output = self.main(x)
        value = self.value(output)
        advantage = self.advantage(output)
        
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
    
    def load_weights(self, path: str): 
        self.load_state_dict(torch.load(path))
        
    def save_weights(self, path: str): 
        torch.save(self.state_dict(), path)
        
        