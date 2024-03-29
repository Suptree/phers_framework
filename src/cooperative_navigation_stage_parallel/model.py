
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        # 小さいサイズ
        # self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        # self.fc2 = nn.Linear(in_features=64, out_features=64)
        # self.fc_linear = nn.Linear(in_features=64, out_features=1)
        # self.fc_angular = nn.Linear(in_features=64, out_features=1)
        ## ネットワークのアーキテクチャ情報を保存
        # self.architecture = {'layers': [n_states, 64, 64, n_actions]}
        
        # 論文サイズ
        self.fc1 = nn.Linear(in_features=self.n_states, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.f3 = nn.Linear(in_features=256, out_features=128)
        self.fc_linear = nn.Linear(in_features=128, out_features=1)
        self.fc_angular = nn.Linear(in_features=128, out_features=1)
        ## ネットワークのアーキテクチャ情報を保存
        self.architecture = {'layers': [n_states, 256, 256, 128, n_actions]}

        self.log_std = nn.Parameter(torch.full((self.n_actions,),-1.5))
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.f3(x))
        
        # mean_linear = F.tanh(F.relu((self.fc_linear(x))))
        mean_linear = F.tanh(self.fc_linear(x))
        mean_angular = F.tanh(self.fc_angular(x))
        
        mean = torch.cat([mean_linear, mean_angular], dim=-1)        
        
        std = self.log_std.exp().expand_as(mean)
        
        return mean, std

    def get_architecture(self):
        architecture = []
        for name, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                architecture.append({
                    'layer_type': 'Linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
            elif isinstance(layer, nn.Parameter):
                architecture.append({
                    'layer_type': 'Parameter',
                    'size': layer.size()
                })
        return architecture

class Critic(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states

        # 小さいサイズ
        # self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        # self.fc2 = nn.Linear(in_features=64, out_features=64)
        # self.fc3 = nn.Linear(in_features=64, out_features=1)

        # 論文サイズ
        self.fc1 = nn.Linear(in_features=self.n_states, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def get_architecture(self):
        architecture = []
        for name, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                architecture.append({
                    'layer_type': 'Linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })
        return architecture
