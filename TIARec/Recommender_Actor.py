import torch
import torch.nn as nn
import torch.nn.functional as F


# recommender actor
class recommender_attention(nn.Module):
    def __init__(self, dim_action):
        super(recommender_attention, self).__init__()
        self.W_o = nn.Linear(dim_action, dim_action)
        self.query_embedding = nn.Linear(dim_action, 1)

    def forward(self, obs):
        state = torch.tanh(self.W_o(obs))
        att = torch.softmax(self.query_embedding(state), 1)
        O = ((state * att)).mean(1)
        return O


class Actor1(nn.Module):
    def __init__(self, recommender_attention, dim_action):
        super(Actor1, self).__init__()

        self.recommender_attention = recommender_attention
        self.FC1 = nn.Linear(dim_action, 32)
        self.FC1.weight.data.normal_(0, 0.1)
        self.FC2 = nn.Linear(32, 128)
        self.FC2.weight.data.normal_(0, 0.1)
        self.FC3 = nn.Linear(128, dim_action)
        self.FC3.weight.data.normal_(0, 0.1)

    # action output between -2 and 2
    def forward(self, obs):
        O = self.recommender_attention(obs)
        result = torch.tanh(self.FC1(O))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        return result