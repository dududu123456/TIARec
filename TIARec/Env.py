from ReplayMemory import ReplayMemory
import random
import datetime
import torch
from copy import deepcopy
import torch.nn as nn
from collections import namedtuple
from Classifier_Actor import *
from Recommender_Actor import *
from Critic import *
from torch.optim import Adam
from ReplayMemory import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class env(nn.Module):

    def __init__(self, config):
        super(env, self).__init__()
        self.n_actions = config.action_dim
        self.batch_size = config.batch_size
        self.tau = config.tau
        self.recommender_attention = recommender_attention(self.n_actions)
        self.classifier_attention = classifier_attention(self.n_actions)
        self.actors = [Actor1(self.recommender_attention, self.n_actions),
                       Actor2(self.classifier_attention, self.n_actions)]
        self.critics = Critic(self.recommender_attention, self.classifier_attention, self.n_actions)
        if (config.load_model == True):
            self.load(config.load_model_directory)
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.critic_optimizer = Adam(self.critics.parameters(), lr=self.critic_lr)
        self.actor_optimizer = torch.optim.SGD([
            {'params': self.actors[0].parameters()},
            {'params': self.actors[1].parameters()}
        ], self.actor_lr)
        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            self.critics.cuda()

            for x in self.actors_target:
                x.cuda()
            self.critics_target.cuda()
            self.recommender_attention.cuda()
            self.classifier_attention.cuda()
        self.steps_done = 0

    def selectAction(self, O, M, N, ground_truth, negative_item, topk, model, rating):



        O = O.view(1, -1, self.n_actions)
        virtual_item = self.actors[0](O.detach().to(device)).cpu().detach()

        all_sim = []
        sim = torch.cosine_similarity(virtual_item, ground_truth, 1).data
        all_sim.append(sim)
        max_sim = sim
        R_r = 0.0
        A1_action = ground_truth
        for i in negative_item:
            tmp_item_embedding = model.wv[str(i)]
            tmp_item_embedding = torch.from_numpy(tmp_item_embedding).view(1, -1)
            tmp_sim = torch.cosine_similarity(virtual_item, tmp_item_embedding).data
            all_sim.append(tmp_sim)
            if max_sim < tmp_sim:
                max_sim = tmp_sim
                A1_action = tmp_item_embedding
        all_sim.sort(reverse=True)
        ground_truth_index = all_sim.index(sim) + 1
        if ground_truth_index <= topk:
            R_r = 1.0

        A2_action = self.actors[1](M.detach().view(1, -1, self.n_actions).to(device),
                                   N.detach().view(1, -1, self.n_actions).to(device),
                                   A1_action.detach().view(1, -1).to(device))
        A2_action = A2_action.cpu().detach()

        self.steps_done += 1
        return A1_action, A2_action, R_r, sim

    def resetState(self, user_train_data, item_num, action_dim):
        O = user_train_data[0:item_num].ItemID.values
        M = torch.zeros(item_num, action_dim)
        L = M
        return M, L, O

    def step(self, O, M, N, A1_action, A2_action):

        next_O = torch.cat((O[1:], A1_action), 0)


        next_M = M
        next_N = N
        pr = float(A2_action.view(-1).data)
        Q = self.critics(O.view(1, -1, self.n_actions).detach().to(device),
                         M.view(1, -1, self.n_actions).detach().to(device),
                         N.view(1, -1, self.n_actions).detach().to(device), A1_action.detach().view(1, -1),
                         A2_action.detach().view(1, -1)).cpu().detach()

        probability = np.random.choice(2, 1, p=[pr, 1 - pr])[0]
        if probability == 0:
            next_N = torch.cat((next_N[1:], A1_action), 0)
            R_c = torch.tensor([1.0]).view(1, -1) - torch.cosine_similarity(A1_action.view(1, -1),
                                              torch.mean(N, dim=0).view(1, -1), dim=1).view(1, -1)
        else:
            # 放到M集合中
            next_M = torch.cat((next_M[1:], A1_action), 0)
            R_c = torch.cosine_similarity(A1_action.view(1, -1),
                                          torch.mean(M, dim=0).view(1, -1), dim=1).view(1, -1)

        return next_O, next_M, next_N, R_c, Q

    def update_policy(self):

        FloatTensor = torch.FloatTensor

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        O_state_batch = torch.stack(batch.O_states).type(FloatTensor)
        O = O_state_batch.view(self.batch_size, -1, self.n_actions)
        next_O_state_batch = torch.stack(batch.next_O_states).type(FloatTensor)
        next_O = next_O_state_batch.view(self.batch_size, -1, self.n_actions)

        M_state_batch = torch.stack(batch.M_states).type(FloatTensor)
        M = M_state_batch.view(self.batch_size, -1, self.n_actions)

        N_state_batch = torch.stack(batch.N_states).type(FloatTensor)
        N = N_state_batch.view(self.batch_size, -1, self.n_actions)

        next_M_state_batch = torch.stack(batch.next_M_states).type(FloatTensor)
        next_M = next_M_state_batch.view(self.batch_size, -1, self.n_actions)

        next_N_state_batch = torch.stack(batch.next_N_states).type(FloatTensor)
        next_N = next_N_state_batch.view(self.batch_size, -1, self.n_actions)

        act_r_batch = torch.stack(batch.act_r).type(FloatTensor)
        act_r = act_r_batch.view(self.batch_size, -1)
        act_c_batch = torch.stack(batch.act_c).type(FloatTensor)
        act_c = act_c_batch.view(self.batch_size, -1)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        reward = reward_batch.view(self.batch_size, -1)

        done = torch.stack(batch.done).type(FloatTensor).view(self.batch_size, -1)

        # update critic
        self.critic_optimizer.zero_grad()

        # current_Q
        current_Q = self.critics(O.detach().to(device), M.detach().to(device),
                                 N.detach().to(device), act_r.detach().to(device),
                                 act_c.detach().to(device)).cpu()


        # get next_act_r
        next_act_r = self.actors_target[0](next_O.detach().to(device)).cpu()
        # get next_act_c
        next_act_c = self.actors_target[1](next_M.detach().to(device),
                                           next_N.detach().to(device),
                                           next_act_r.detach().to(device))

        # target_Q
        target_Q = self.critics_target(
            next_O.to(device), next_M.to(device), next_N.to(device),
            next_act_r.to(device), next_act_c.to(device)
        ).cpu()


        target_Q = target_Q.mul(done) * self.GAMMA + (reward[:, 0] + self.a * reward[:, 1]).view(
            self.batch_size, -1)


        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())

        loss_Q.backward()

        self.critic_optimizer.step()

        # update agent
        O_batch = torch.stack(batch.O_states).type(FloatTensor)
        recommender_O = O_batch.view(self.batch_size, -1, self.n_actions)

        self.actor_optimizer.zero_grad()

        A1_action = self.actors[0](recommender_O.to(device))
        A1_action = A1_action.cpu()
        A2_action = self.actors[1](M.detach().to(device), N.detach().to(device), A1_action.detach().to(device)).cpu()
        act_r_batch = torch.stack(batch.act_r).type(FloatTensor)
        act_r = act_r_batch.view(self.batch_size, -1)


        critic1_loss = -self.critics(recommender_O.detach().to(device), M.detach().to(device),
                                     N.detach().to(device), A1_action.to(device),
                                     A2_action.to(device)).cpu().mean()
        actor1_loss = nn.MSELoss()(A1_action, act_r) + critic1_loss

        actor1_loss.backward()
        self.actor_optimizer.step()

        if self.steps_done % 20 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                self.soft_update(self.actors_target[i], self.actors[i], self.tau)
            self.soft_update(self.critics_target, self.critics, self.tau)
        return loss_Q.data, actor1_loss.data, 0

    # top-K
    def getTargetItem(self, O):
        O = O.view(1, -1, self.n_actions)
        item = self.actors[0](O.detach().to(device))
        return item.cpu().detach()

    # get negative
    def getNegativeItem(self, nega_num, a, c):
        negative_item = list(filter(lambda x: x not in a, c))
        negative_item = np.random.choice(negative_item, nega_num, replace=False)
        return negative_item

    # def calculate MRR and HR
    def calculateMrrAndHR(self, unreal_item, real_item, nega_item_list, model, top_k):
        mrr = []
        hr = []
        tmp_list = []
        hit = []
        ndcg = []
        sim = torch.cosine_similarity(unreal_item, real_item, 1).data
        tmp_list.append(sim)
        count = 0
        for i in nega_item_list:
            tmp_item_embedding = model.wv[str(i)]
            tmp_item_embedding = torch.from_numpy(tmp_item_embedding).view(1, -1)
            tmp_sim = torch.cosine_similarity(unreal_item, tmp_item_embedding).data
            tmp_list.append(tmp_sim)
            if (tmp_sim >= sim):
                count = count + 1
        tmp_list.sort(reverse=True)
        m = tmp_list.index(sim) + 1
        # print("##########",sim,"count",count,"index",m,"max_sim:",tmp_list[0])
        for j in top_k:
            if m > j:
                mrr.append(0)
                hr.append(0)
                hit.append(0)
                ndcg.append(0)
            else:
                hr.append(1)
                mrr.append(1 / m)
                ndcg.append(1 / (math.log(m + 1, 2)))
                hit.append(1)
        mrr = np.array(mrr)
        hr = np.array(hr)
        hit = np.array(hit)
        ndcg = np.array(ndcg)
        return mrr, hr, hit, ndcg

    def save(self, directory):
        # directory = '../model/movielens/'
        # t是当前时间
        t = datetime.datetime.now().isoformat()
        directory = directory + t
        torch.save(self.actors[0].state_dict(), directory + 'actor1.ptorch')
        torch.save(self.actors[1].state_dict(), directory + 'actor2.ptorch')
        torch.save(self.critics.state_dict(), directory + 'critic.ptorch')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def soft_update(target, source, t):
        for target_param, source_param in zip(target.parameters(),
                                              source.parameters()):
            target_param.data.copy_(
                (1 - t) * target_param.data + t * source_param.data)

    def load(self, directory):
        self.actors[0].load_state_dict(torch.load(directory + 'actor1.ptorch'))
        self.actors[1].load_state_dict(torch.load(directory + 'actor2.ptorch'))
        self.critics.load_state_dict(torch.load(directory + 'critic.ptorch'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

