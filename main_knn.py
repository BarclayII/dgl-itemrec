import dgl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse as ssp
import tqdm
import pickle
from model.model import FISM
from model.pinsage import PinSage
from model.ranking import ndcg
from model.movielens import MovieLens

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

n_epoch = 200
iters_per_epoch = 20000
batch_size = 256
feature_size = 16
n_layers = 2
n_traces = 10
trace_len = 3
n_neighbors = 3
n_negs = 4
weight_decay = 1e-2
margin = 1
data_pickle = 'ml-1m.pkl'
data_path = 'data/ml-1m'

if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)
else:
    data = MovieLens(data_path)
    with open(data_pickle, 'wb') as f:
        pickle.dump(data, f)

ratings = data.ratings
ratings_train = ratings[~(ratings['valid_mask'] | ratings['test_mask'])]
user_latest_item_indices = (
        ratings_train.groupby('user_id')['timestamp'].transform(pd.Series.max) ==
        ratings_train['timestamp'])
user_latest_item = ratings_train[user_latest_item_indices]
user_latest_item = dict(
        zip(user_latest_item['user_idx'].values, user_latest_item['movie_idx'].values))
users_train = ratings_train['user_idx'].values
movies_train = ratings_train['movie_idx'].values
users_valid = ratings[ratings['valid_mask']]['user_idx'].values
movies_valid = ratings[ratings['valid_mask']]['movie_idx'].values
users_test = ratings[ratings['test_mask']]['user_idx'].values
movies_test = ratings[ratings['test_mask']]['movie_idx'].values
train_size = len(users_train)
valid_size = len(users_valid)
test_size = len(users_test)

HG = dgl.heterograph({
    ('user', 'um', 'movie'): (ratings_train['user_idx'], ratings_train['movie_idx']),
    ('movie', 'mu', 'user'): (ratings_train['movie_idx'], ratings_train['user_idx'])})
HG.nodes['movie'].data.update(data.movie_data)
HG.to(device)

model = PinSage(
        HG, 'movie', 'mu', 'um', [feature_size] * n_layers, n_neighbors, n_traces,
        trace_len, True)
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)


def cooccurrence_iterator(users, movies, batch_size, n_negs):
    M_um = ssp.coo_matrix((np.ones(train_size), (users_train, movies_train)))
    M_mm = (M_um.T * M_um).tocoo()

    rows, cols, counts = M_mm.row, M_mm.col, M_mm.data
    rows = torch.LongTensor(rows)
    cols = torch.LongTensor(cols)
    prob = counts / counts.sum()

    while True:
        # torch.multinomial is A LOT slower than np.random.choice.  Ugh.
        indices = np.random.choice(len(counts), batch_size, replace=True, p=prob)
        indices = torch.LongTensor(indices)
        yield rows[indices].to(device), \
              cols[indices].to(device), \
              torch.randint(0, M_mm.shape[0], (batch_size, n_negs)).to(device)
generator = cooccurrence_iterator(users_train, movies_train, batch_size, n_negs)


def train():
    for _ in range(n_epoch):
        # train
        neg, count = 0, 0
        with tqdm.trange(iters_per_epoch) as t:
            for _ in t:
                I_q, I_i, I_neg = next(generator)

                pos_overlap = [
                        len(np.intersect1d(
                            HG.successors(i, etype='mu').cpu().numpy(),
                            HG.successors(j, etype='mu').cpu().numpy())) > 0
                        for i, j in zip(I_q.cpu().numpy(), I_i.cpu().numpy())]
                neg_overlap = [
                        len(np.intersect1d(
                            HG.successors(i, etype='mu').cpu().numpy(),
                            HG.successors(j, etype='mu').cpu().numpy())) > 0
                        for i, j in zip(I_q.cpu().numpy(), I_neg[:, 0].cpu().numpy())]
                neg += sum(neg_overlap)
                count += len(neg_overlap)
                assert all(pos_overlap)

                z_q = model(I_q)
                z_i = model(I_i)
                z_neg = model(I_neg.view(-1)).view(batch_size, n_negs, -1)

                score_pos = (z_q * z_i).sum(1)
                score_neg = (z_q.unsqueeze(1) * z_neg).sum(2)
                loss = (score_neg - score_pos.unsqueeze(1) + margin).clamp(min=0).mean()

                opt.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    d_param = param.grad
                    assert not torch.isnan(d_param).any().item()
                opt.step()

                t.set_postfix({'loss': '%.06f' % loss.item()})
            print('neg:', neg / count)

        with torch.no_grad():
            # evaluate - precompute item embeddings
            I_list = torch.arange(len(data.movies)).split(batch_size)
            z = torch.cat([model(I.to(device)) for I in I_list])
            hits_10s = []
            ndcg_10s = []

            # evaluate one user-item interaction at a time
            for u, i in zip(users_valid, movies_valid):
                I_q = user_latest_item[u]
                I = torch.cat([torch.LongTensor([i]), torch.LongTensor(data.neg_valid[u])])
                Z_q = z[I_q]
                Z = z[I]
                score = (Z_q[None, :] * Z).sum(1).cpu().numpy()
                rank = score.argsort()
                hits_10 = rank[0] < 10
                relevance = (rank == 0)
                ndcg_10 = ndcg(relevance, 10)

                hits_10s.append(hits_10)
                ndcg_10s.append(ndcg_10)

            print('HITS@10:', np.mean(hits_10s), 'NDCG@10:', np.mean(ndcg_10s))

train()
