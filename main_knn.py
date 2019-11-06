import dgl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse as ssp
import tqdm
import pickle
import argparse
from model.pinsage import PinSage
from model.ranking import evaluate
from model.movielens import MovieLens
from model.randomwalk_sampler import CooccurrenceDataset, CooccurrenceNodeFlowGenerator
from model.randomwalk_sampler import NodeDataset, NodeFlowGenerator, to_device

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
parser.add_argument('--max-c', type=float, default=np.inf)
parser.add_argument('--data-pickle', type=str, default='ml-1m.pkl')
parser.add_argument('--data-path', type=str, default='/efs/quagan/movielens/ml-1m')
parser.add_argument('--id-as-feature', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--num-workers', type=int, default=0)
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
max_c = args.max_c
data_pickle = args.data_pickle
data_path = args.data_path
id_as_feature = args.id_as_feature
lr = args.lr
num_workers = args.num_workers

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
        HG, 'movie', 'mu', 'um', feature_size, n_layers, n_neighbors, n_traces,
        trace_len, True, id_as_feature)
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)


def cycle_iterator(loader):
    while True:
        it = iter(loader)
        for elem in it:
            yield elem


def train():
    train_dataset = CooccurrenceDataset(users_train, movies_train)
    valid_dataset = NodeDataset(len(data.movies))
    train_collator = CooccurrenceNodeFlowGenerator(
            HG, 'um', 'mu', n_neighbors, n_traces, trace_len, model.n_layers, n_negs)
    valid_collator = NodeFlowGenerator(
            HG, 'um', 'mu', n_neighbors, n_traces, trace_len, model.n_layers, n_negs)
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collator)
    valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=valid_collator)
    train_iter = cycle_iterator(train_loader)

    for _ in range(n_epoch):
        # train
        sum_loss = 0
        with tqdm.trange(iters_per_epoch) as t:
            for it in t:
                item = next(train_iter)
                I_q, I_i, I_neg, nf_q, nf_i, nf_neg, c = to_device(item, device)

                z_q = model(I_q, nf_q)
                z_i = model(I_i, nf_i)
                z_neg = model(I_neg.view(-1), nf_neg).view(I_neg.shape[0], n_negs, -1)

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

        with torch.no_grad():
            # evaluate - precompute item embeddings
            z = []
            for item in valid_loader:
                I, nf_i = to_device(item, device)
                z.append(model(I, nf_i))
            z = torch.cat(z)
            hits_10s = []
            ndcg_10s = []
            baseline_hits_10s = []
            baseline_ndcg_10s = []
            baseline_score_all = data.movies['movie_count'].values

            # evaluate one user-item interaction at a time
            for u, i in zip(users_valid, movies_valid):
                I_q = user_latest_item[u]
                I_pos = np.array([i])
                I_neg = data.neg_valid[u]
                relevance = np.array([1])

                I = torch.cat([torch.LongTensor(I_pos), torch.LongTensor(I_neg)])
                Z_q = z[I_q]
                Z = z[I]
                score = (Z_q[None, :] * Z).sum(1).cpu().numpy()
                baseline_score = baseline_score_all[I.numpy()]

                hits_10, ndcg_10 = evaluate(score, 1, relevance)
                hits_10s.append(hits_10)
                ndcg_10s.append(ndcg_10)

                hits_10, ndcg_10 = evaluate(baseline_score, 1, relevance)
                baseline_hits_10s.append(hits_10)
                baseline_ndcg_10s.append(ndcg_10)

            print('HITS@10:', np.mean(hits_10s), 'NDCG@10:', np.mean(ndcg_10s),
                  'HITS@10 (Most popular):', np.mean(baseline_hits_10s),
                  'NDCG@10 (Most popular):', np.mean(baseline_ndcg_10s))

train()
