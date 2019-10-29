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
import argparse
from model.pinsage import PinSage
from model.ranking import ndcg
from model.movielens import MovieLens

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--n-epoch', type=int, default=200)
parser.add_argument('--iters-per-epoch', type=int, default=20000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--n-traces', type=int, default=10)
parser.add_argument('--trace-len', type=int, default=3)
parser.add_argument('--n-neighbors', type=int, default=3)
parser.add_argument('--n-negs', type=int, default=4)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--margin', type=float, default=1.)
parser.add_argument('--max-c', type=float, default=10.)
parser.add_argument('--data-pickle', type=str, default='ml-1m.pkl')
parser.add_argument('--data-path', type=str, default='/efs/quagan/movielens/ml-1m')
parser.add_argument('--id-as-feature', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
args = parser.parse_args()
n_epoch = args.n_epoch
iters_per_epoch = args.iters_per_epoch
batch_size = args.batch_size
feature_size = args.feature_size
n_layers = args.n_layers
n_traces = args.n_traces
trace_len = args.trace_len
n_neighbors = args.n_neighbors
n_negs = args.n_negs
weight_decay = args.weight_decay
margin = args.margin
max_c = np.inf
data_pickle = args.data_pickle
data_path = args.data_path
id_as_feature = args.id_as_feature
lr = args.lr

# Load the cached dataset object, or parse the raw MovieLens data
if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)
else:
    data = MovieLens(data_path)
    with open(data_pickle, 'wb') as f:
        pickle.dump(data, f)

# Fetch the interaction and movie data as numpy arrays
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

# Build the bidirectional bipartite graph and put the movie features
HG = dgl.heterograph({
    ('user', 'um', 'movie'): (ratings_train['user_idx'], ratings_train['movie_idx']),
    ('movie', 'mu', 'user'): (ratings_train['movie_idx'], ratings_train['user_idx'])})
HG.nodes['movie'].data.update(data.movie_data)
HG.to(device)

# Model and optimizer
model = PinSage(
        HG, 'movie', 'mu', 'um', [feature_size] * n_layers, n_neighbors, n_traces,
        trace_len, True, id_as_feature)
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)


def cooccurrence_iterator(users, movies, batch_size, n_negs):
    # Compute item-item cooccurrence matrix.
    # The entries are the number of cooccurrences.
    M_um = ssp.coo_matrix((np.ones(train_size), (users_train, movies_train)))
    M_mm = (M_um.T * M_um).tocoo()

    rows, cols, counts = M_mm.row, M_mm.col, M_mm.data
    rows = torch.LongTensor(rows)
    cols = torch.LongTensor(cols)
    prob = counts / counts.sum()
    counts = torch.LongTensor(counts).float()

    while True:
        # torch.multinomial is A LOT slower than np.random.choice.  Ugh.
        #indices = np.random.choice(len(counts), batch_size, replace=True, p=prob)
        #indices = torch.LongTensor(indices)
        indices_all = torch.randperm(len(counts))
        for indices in indices_all.split(batch_size):
            yield rows[indices].to(device), \
                  cols[indices].to(device), \
                  torch.randint(0, M_mm.shape[0], (batch_size, n_negs)).to(device), \
                  counts[indices].to(device)
generator = cooccurrence_iterator(users_train, movies_train, batch_size, n_negs)


def train():
    for _ in range(n_epoch):
        # train
        neg, count = 0, 0
        sum_loss = 0
        with tqdm.trange(iters_per_epoch) as t:
            for it in t:
                I_q, I_i, I_neg, c = next(generator)

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
                z_neg = model(I_neg.view(-1)).view(z_q.shape[0], n_negs, -1)

                score_pos = (z_q * z_i).sum(1)
                score_neg = (z_q.unsqueeze(1) * z_neg).sum(2)
                c = c.clamp(max=max_c)
                loss = (score_neg - score_pos.unsqueeze(1) + margin).clamp(min=0)
                loss = (loss.mean(1) * c).sum() / c.sum()

                opt.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    d_param = param.grad
                    assert not torch.isnan(d_param).any().item()
                opt.step()

                sum_loss += loss.item()
                t.set_postfix({'loss': '%.06f' % loss.item(), 'avg': '%.06f' % (sum_loss / (it + 1))})
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
                rank = scipy.stats.rankdata(-score, 'min')
                hits_10 = rank[0] <= 10
                relevance = ((-score).argsort() == 0)
                ndcg_10 = ndcg(relevance, 10)

                hits_10s.append(hits_10)
                ndcg_10s.append(ndcg_10)

            print('HITS@10:', np.mean(hits_10s), 'NDCG@10:', np.mean(ndcg_10s))

train()
