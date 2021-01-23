import argparse
import torch
import numpy as np
from gensim.models import word2vec
from tqdm import tqdm
import pandas as pd
import math
import time
from Env import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--action_dim', type=int, default=64, help='item_embedding dimension size')
parser.add_argument('--select_topk', type=int, default=100)
parser.add_argument('--train_root', type=str, default="data/movielens/train.csv")
parser.add_argument('--test_root', type=str, default="data/movielens/test.csv")
parser.add_argument('--word2vec', type=str, default="data/movielens/item.model")
parser.add_argument('--nega_num', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--percentage', type=float, default=0.2)
parser.add_argument('--top_k', type=list, default=[1, 5, 10, 20, 30, 50])
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--load_model_directory', type=str, default="", help="load model")
parser.add_argument('--train', type=bool, default=True, help="training model")
parser.add_argument('--test', type=bool, default=True, help="testing model")
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--save_model', type=bool, default=False, help="save model")
parser.add_argument('--save_model_directory', type=str, default="", help="save model")
parser.add_argument('--capacity', type=int, default=2000, help='the size of replay buffer')
parser.add_argument('--alpha', type=float, default=0.5, help='controls total rewards')
parser.add_argument('--actor_lr', type=float, default=0.0002, help='the learning rate of both actor networks')
parser.add_argument('--critic_lr', type=float, default=0.0005, help='the learning rate of critic network')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
parser.add_argument('--tau', type=float, default=0.01, help='the discount factor')
parser.add_argument('--item_list', type=int, default=20)

config = parser.parse_args()


def main():


    train_data = pd.read_csv(config.train_root, sep=config.sep)

    test_data = pd.read_csv(config.test_root, sep=config.sep)
    net = env(config)
    model = word2vec.Word2Vec.load(config.word2vec)

    all_item_list = train_data.drop_duplicates('ItemID').ItemID.values.tolist()


    user_gather = train_data.drop_duplicates('UserID').UserID.values
    total_crttic_loss = []
    total_actor_loss = []
    # train_number
    train_number = 0

    time_start = time.time()
    for epoch in range(config.epoch):

        all_sim = []

        if config.train == True:
            for i in tqdm(range(len(user_gather)), desc="sampling:"):
                user_id = user_gather[i]
                user_train_data = train_data[train_data.UserID == user_id]

                O = torch.zeros(config.item_list, config.action_dim)

                M = O
                N = O
                interactions_list = user_train_data.ItemID.values.tolist()

                negative_item = net.getNegativeItem(config.nega_num, interactions_list, all_item_list)

                for num in range(0, len(user_train_data) - 1):

                    done = 1
                    if num == len(user_train_data) - 1:
                        done = 0
                    # get real_action
                    one_user_train_data = user_train_data[num:num + 1]
                    ground_truth = one_user_train_data.ItemID.values[0]
                    ground_truth = model.wv[str(ground_truth)]
                    ground_truth = torch.from_numpy(ground_truth).view(1, -1)
                    rating = one_user_train_data.Action.values[0]
                    # select action
                    A1_action, A2_action, R_r, sim = net.selectAction(O, M, N, ground_truth, negative_item,
                                                                      config.select_topk,
                                                                      model, rating)

                    # execute action and get reward
                    next_O, next_M, next_N, R_c, Q = net.step(O, M, N, A1_action, A2_action)
                    all_sim.append(sim)
                    # store train_sample in relayBuffer
                    current_O = O.view(1, -1)
                    current_M = M.view(1, -1)
                    current_N = N.view(1, -1)
                    next_O_state = next_O.view(1, -1)
                    next_M_state = next_M.view(1, -1)
                    next_N_state = next_N.view(1, -1)
                    current_act_r = A1_action.view(1, -1)
                    current_act_c = A2_action.view(1, -1)
                    reward_actor1 = torch.tensor([R_r]).view(1, -1)
                    reward_actor2 = R_c
                    current_reward = torch.cat((reward_actor1, reward_actor2), 1).view(1, -1)
                    current_done = torch.tensor([done]).view(1,-1)
                    net.memory.push(current_O, current_M, current_N, current_act_r, current_act_c, next_O_state,
                                    next_M_state, next_N_state, current_reward, current_done)
                    # if the number of the train_sample > capacity then start train
                    train_number = train_number + 1
                    O = next_O
                    N = next_N
                    M = next_M

                    if train_number >= config.capacity:
                        # start train
                        critic_loss, actor_loss = net.update_policy()
                        total_crttic_loss.append(critic_loss)
                        total_actor_loss.append(actor_loss)


            time_final = time.time()
            ttt = (int)(time_final - time_start)

            print("sim:", np.array(all_sim).mean(), "###total time:", ttt, "s", "#####finished epochs:", epoch,  "####crttic_loss:",
                  np.array(total_crttic_loss).mean(),
                  "###actor_loss:", np.array(total_actor_loss).mean())
        if config.test == True:
            list_user_id = user_gather.tolist()
            MRR = np.zeros(len(config.top_k))
            HR = np.zeros(len(config.top_k))
            Hit = np.zeros(len(config.top_k))
            NDCG = np.zeros(len(config.top_k))
            for i in tqdm(list_user_id, desc="testing:"):

                user_MRR = np.zeros(len(config.top_k))
                user_HR = np.zeros(len(config.top_k))
                user_Hit = np.zeros(len(config.top_k))
                user_NDCG = np.zeros(len(config.top_k))
                user_test_data = test_data[test_data.UserID == i]

                test_train_data = train_data[train_data.UserID == i]
                item_list = test_train_data.ItemID.values.tolist()

                test_real_item = user_test_data[0: 1].ItemID.values[0]
                test_real_item = model.wv[str(test_real_item)]
                test_real_item = torch.from_numpy(test_real_item).view(1, -1)
                # tmp = test_real_item
                tmp = torch.zeros(config.item_list, config.action_dim)
                tmp = torch.cat((tmp[1:], test_real_item), 0)
                # get t time
                for t in range(1, len(user_test_data)):

                    test_real_item = user_test_data[t:t + 1].ItemID.values[0]
                    test_real_item = model.wv[str(test_real_item)]
                    test_real_item = torch.from_numpy(test_real_item).view(1, -1)

                    test_tmp = tmp
                    for iii in range(len(tmp)):
                        if random.random() < config.percentage:
                            item_id = random.choice(all_item_list)
                            test_item_embedding = model.wv[str(item_id)]
                            test_tmp[iii] = torch.from_numpy(test_item_embedding)

                    # get target item，网络输出，虚拟的商品
                    unreal_targetItem = net.getTargetItem(test_tmp)
                    # 得到下一个状态
                    tmp = torch.cat((tmp[1:], test_real_item), 0)
                    # get negative_sample
                    negative_item = all_item_list

                    # calculate the MRR and HR
                    mrr, hr, hit, ndcg = net.calculateMrrAndHR(unreal_targetItem, test_real_item, negative_item, model,
                                                               config.top_k)
                    user_MRR = user_MRR + mrr
                    user_HR = user_HR + hr
                    user_NDCG = user_NDCG + ndcg
                    user_Hit = [user_Hit[i] or hit[i] for i in range(0, len(hit))]

                user_MRR = user_MRR / (len(user_test_data))
                user_HR = user_HR / (len(user_test_data))
                user_NDCG = user_NDCG / (len(user_test_data))
                MRR = MRR + user_MRR
                HR = HR + user_HR
                Hit = Hit + user_Hit
                NDCG = NDCG + user_NDCG

            MRR = MRR / len(list_user_id)
            HR = HR / len(list_user_id)
            Hit = Hit / len(list_user_id)
            NDCG = NDCG / len(list_user_id)

            print('Hit@1:{}, '
                  'Hit@5:{}, '
                  'Hit@10:{}, '
                  'Hit@20:{}, '
                  'Hit@30:{}, '
                  'Hit@50:{}, '
                  'HR@1:{}, '
                  'HR@5:{}, '
                  'HR@10:{}, '
                  'HR@20:{}, '
                  'HR@30:{}, '
                  'HR@50:{}, '
                  'NDCG@1:{}, '
                  'NDCG@5:{}, '
                  'NDCG@10:{}, '
                  'NDCG@20:{}, '
                  'NDCG@30:{}, '
                  'NDCG@50:{}, '
                  'Mrr@1:{}, '
                  'Mrr@5:{}, '
                  'Mrr@10:{}, '
                  'Mrr@20:{}, '
                  'Mrr@30:{}, '
                  'Mrr@50:{}, '.format(Hit[0], Hit[1], Hit[2], Hit[3], Hit[4], Hit[5],
                                       HR[0], HR[1], HR[2], HR[3], HR[4], HR[5],
                                       NDCG[0], NDCG[1], NDCG[2], NDCG[3], NDCG[4], NDCG[5],
                                       MRR[0], MRR[1], MRR[2], MRR[3], MRR[4], MRR[5]))

if __name__ == '__main__':

    main()