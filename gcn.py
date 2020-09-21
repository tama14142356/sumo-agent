import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import gym
import numpy as np
import random

args = {
    'mode': 'cui'
}


class Net(torch.nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        dataset = env.observation
        num_action = env.action_space[0][0].n + env.action_space[0][1].shape[0]
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, num_action)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        f = F.softmax(x, dim=1)
        return f


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('gym_sumo:sumo-v0', **args)
obs = env.reset()
model = Net(env).to(device)
action = []
for i in range(100):
    tmp = 0
    tmp2 = np.array([random.uniform(0.0, 1.0)], dtype=np.float32)
    action.append((tmp, tmp2))
obs, reward, done, _ = env.step(action)
dataset = obs
data = dataset.to(device)
test = model.forward(data)
print(test)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# model.train()  # モデルを訓練モードにする。
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
