import torch.nn as nn
import torch
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, recommender_attention, classifer_attention, dim_action):
        super(Critic, self).__init__()

        self.recommender_attention = recommender_attention
        self.classifer_attention = classifer_attention
        self.FC1 = nn.Linear(dim_action * 3 + 1, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, M, N, act_r, act_c):
        O = self.recommender_attention(obs)
        state = self.classifer_attention(M, N, act_r)
        all_state = torch.cat((O, state, act_c), 1)
        result = F.relu(self.FC1(all_state))
        result = torch.tanh(self.FC2(result))
        return self.FC4(F.relu(self.FC3(result)))