
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_linear = nn.Linear(in_features=64, out_features=1)
        self.fc_angular = nn.Linear(in_features=64, out_features=1)

        # self.log_std = nn.Parameter(torch.full((self.n_actions,),-2.5))
        self.log_std = nn.Parameter(torch.tensor([-4.0, -2.5]))
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean_linear = F.sigmoid(self.fc_linear(x))
        mean_angular = F.tanh(self.fc_angular(x))
        
        mean = torch.cat([mean_linear, mean_angular], dim=-1)        
        
        std = self.log_std.exp().expand_as(mean)
        
        return mean, std



class Critic(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
