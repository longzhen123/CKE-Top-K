import time
import torch as t
import torch.nn as nn
import numpy as np
from torch import optim
from src.load_base import load_data
from src.evaluate import get_hit, get_ndcg


class CKE(nn.Module):

    def __init__(self, n_entity, n_user, n_item, n_rels, dim):

        super(CKE, self).__init__()
        self.dim = dim
        user_emb_matrix = t.randn(n_user, dim)
        item_emb_matrix = t.randn(n_item, dim)
        ent_emb_matrix = t.randn(n_entity, dim)
        Mr_matrix = t.randn(n_rels, dim, dim)
        rel_emb_matrix = t.randn(n_rels, dim)

        nn.init.xavier_uniform_(user_emb_matrix)
        nn.init.xavier_uniform_(item_emb_matrix)
        nn.init.xavier_uniform_(ent_emb_matrix)
        nn.init.xavier_uniform_(Mr_matrix)
        nn.init.xavier_uniform_(rel_emb_matrix)

        self.user_emb_matrix = nn.Parameter(user_emb_matrix)
        self.item_emb_matrix = nn.Parameter(item_emb_matrix)
        self.ent_emb_matrix = nn.Parameter(ent_emb_matrix)
        self.Mr_matrix = nn.Parameter(Mr_matrix)
        self.rel_emb_matrix = nn.Parameter(rel_emb_matrix)

    def forward(self, data, name):
        if name == 'kg':
            # print(data)
            heads_id = [i[0] for i in data]
            relations_id = [i[1] for i in data]
            pos_tails_id = [i[2] for i in data]
            neg_tails_id = [i[3] for i in data]
            head_emb = self.ent_emb_matrix[heads_id].view(-1, 1, self.dim)
            rel_emb = self.rel_emb_matrix[relations_id].view(-1, 1, self.dim)
            pos_tail_emb = self.ent_emb_matrix[pos_tails_id].view(-1, 1, self.dim)
            neg_tail_emb = self.ent_emb_matrix[neg_tails_id].view(-1, 1, self.dim)
            Mr = self.Mr_matrix[relations_id]

            pos_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(pos_tail_emb, Mr)).norm(dim=[1, 2]) ** 2
            neg_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(neg_tail_emb, Mr)).norm(dim=[1, 2]) ** 2
            # print(t.log(t.sigmoid(pos_stru_scores - neg_stru_scores)))
            stru_loss = t.sigmoid(pos_stru_scores - neg_stru_scores)
            stru_loss = t.log(stru_loss).sum()
            return stru_loss
        else:

        # print(uvv)
            users_id = [i[0] for i in data]
            poss_id = [i[1] for i in data]
            negs_id = [i[2] for i in data]
            users_emb = self.user_emb_matrix[users_id]
            pos_items_emb = self.item_emb_matrix[poss_id] + self.ent_emb_matrix[poss_id]
            neg_items_emb = self.item_emb_matrix[negs_id] + self.ent_emb_matrix[negs_id]
            base_loss = t.sigmoid(t.mul(users_emb, pos_items_emb).sum(dim=1) - t.mul(users_emb, neg_items_emb).sum(dim=1))
            base_loss = t.log(base_loss).sum()

            return base_loss

    def get_predict(self, pairs):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        user_emb = self.user_emb_matrix[users]

        item_emb = self.item_emb_matrix[items] + self.ent_emb_matrix[items]
        score = (user_emb * item_emb).sum(dim=1)

        return score.cpu().detach().view(-1).numpy().tolist()


def eval_topk(model, records, topk):
    HR, NDCG = [], []

    model.eval()
    for user, items in records.items():
        pairs = [[user, item] for item in items]

        predict_list = model.get_predict(pairs)
        # print(predict)
        n = len(pairs)
        user_scores = {items[i]: predict_list[i] for i in range(n)}
        item_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: topk]

        HR.append(get_hit(items[-1], item_list))
        NDCG.append(get_ndcg(items[-1], item_list))

    model.train()
    # print(np.mean(HR), np.mean(NDCG))
    return np.mean(HR), np.mean(NDCG)


def get_uvvs(pairs):
    positive_dict = {}
    negative_dict = {}
    for pair in pairs:
        user = pair[0]
        item = pair[1]
        label = pair[2]
        if label == 1:
            if user not in positive_dict:
                positive_dict[user] = []

            positive_dict[user].append(item)
        else:
            if user not in negative_dict:
                negative_dict[user] = []

            negative_dict[user].append(item)
    data = []
    for user in positive_dict:
        size = len(positive_dict[user])
        # print(len(positive_dict[user]), len(negative_dict[user]))
        for i in range(size):
            pos_item = positive_dict[user][i]
            neg_item = negative_dict[user][i]
            data.append([user, pos_item, neg_item])

    return data


def get_hrtts(kg_dict):
    # print('get hrtts...')

    entities = list(kg_dict)

    hrtts = []
    for head in kg_dict:
        for r_t in kg_dict[head]:
            relation = r_t[0]
            positive_tail = r_t[1]

            while True:
                negative_tail = np.random.choice(entities, 1)[0]
                if [relation, negative_tail] not in kg_dict[head]:
                    hrtts.append([head, relation, positive_tail, negative_tail])
                    break

    return hrtts


def train(args):
    np.random.seed(123)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_records, test_records = data[4], data[5], data[6]
    kg_dict = data[7]
    hrtts = get_hrtts(kg_dict)
    model = CKE(n_entity, n_user, n_item, n_relation, args.dim)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    uvvs = get_uvvs(train_set)
    train_data = [hrtts, uvvs]
    print(args.dataset + '----------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('learning_rate: %1.0e' % args.learning_rate, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)

    eval_HR_list = []
    eval_NDCG_list = []
    test_HR_list = []
    test_NDCG_list = []

    for epoch in range(args.epochs):

        start = time.clock()
        size = len(train_data[0])
        start_index = 0
        loss_sum = 0
        np.random.shuffle(train_set)
        np.random.shuffle(hrtts)
        while start_index < size:
            if start_index + args.batch_size <= size:
                hrtts = train_data[0][start_index: start_index + args.batch_size]
            else:
                hrtts = train_data[0][start_index:]
            loss = -model(hrtts, 'kg')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        start_index = 0
        size = len(train_data[-1])
        while start_index < size:
            if start_index + args.batch_size <= size:
                uvvs = train_data[-1][start_index: start_index + args.batch_size]
            else:
                uvvs = train_data[-1][start_index:]
            loss = -model(uvvs, 'cf')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        eval_HR, eval_NDCG = eval_topk(model, eval_records, args.topk)
        test_HR, test_NDCG = eval_topk(model, test_records, args.topk)
        eval_HR_list.append(eval_HR)
        eval_NDCG_list.append(eval_NDCG)
        test_HR_list.append(test_HR)
        test_NDCG_list.append(test_NDCG)
        end = time.clock()

        print('epoch: %d \t eval: HR %.4f NDCG %.4f \t test: HR %.4f NDCG %.4f \t loss: %d, \t time: %d'
              % ((epoch + 1), eval_HR, eval_NDCG, test_HR, test_NDCG, loss_sum, (end-start)))

    n_epoch = eval_HR_list.index(max(eval_HR_list))
    print('epoch: %d \t eval: HR %.4f NDCG %.4f \t test: HR %.4f NDCG %.4f' % (n_epoch+1, eval_HR_list[n_epoch], eval_NDCG_list[n_epoch], test_HR_list[n_epoch], test_NDCG_list[n_epoch]))
    return eval_HR_list[n_epoch], eval_NDCG_list[n_epoch], test_HR_list[n_epoch], test_NDCG_list[n_epoch]


