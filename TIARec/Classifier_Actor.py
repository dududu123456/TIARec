import torch
import torch.nn.functional as F
import torch.nn as nn

class classifier_attention(nn.Module):
    def __init__(self, dim_action):
        super(classifier_attention, self).__init__()

        self.W_m = nn.Linear(dim_action, dim_action)
        self.W_n = nn.Linear(dim_action, dim_action)

    def forward(self, M, N, act_r):
        act_r = act_r.unsqueeze(2)
        M = F.relu(self.W_m(M))
        N = F.relu(self.W_n(N))
        att_m = torch.softmax(torch.bmm(M, act_r), 1)
        att_n = torch.softmax(torch.bmm(N, act_r), 1)
        M = ((M * att_m)).mean(1)
        N = ((N * att_n)).mean(1)
        state = torch.cat((M, N), 1)
        return state


class Actor2(nn.Module):
    def __init__(self, classifer_attention, dim_action):
        super(Actor2, self).__init__()
        self.classifer_attention = classifer_attention
        self.FC1 = nn.Linear(dim_action + dim_action, 128)
        self.FC2 = nn.Linear(128, 32)
        self.FC3 = nn.Linear(32, 1)

    def forward(self, M, N, act_r):
        state = self.classifer_attention(M, N, act_r)
        result = F.relu(self.FC1(state))
        result = torch.tanh(self.FC2(result))
        result = torch.sigmoid(self.FC3(result))

        return result