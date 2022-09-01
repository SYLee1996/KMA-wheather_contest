import torch
import torch.nn as nn

class DNN_block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(DNN_block,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        
        self.dim = config['input_dim']
        self.depth = config['depth']
        self.hidden_dim = config['hidden_dim']
        self.drop_out = config['drop_out']
        self.blocks = nn.ModuleList([])
        
        for _ in range(self.depth):
            self.blocks.append(
                DNN_block(self.dim, self.hidden_dim, self.drop_out), 
            )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.dim)
        self.batch_norm = nn.BatchNorm1d(self.dim)
        self.fc = nn.Linear(self.dim, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x) 
        x = self.batch_norm(x) 
        # x = self.layer_norm(x) 
        x_out1 = self.relu(self.fc(x))
        
        return x_out1