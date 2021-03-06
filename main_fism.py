import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import tqdm
import os
import pickle
import argparse
from model.model import FISM
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
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--n-traces', type=int, default=10)
parser.add_argument('--trace-len', type=int, default=3)
parser.add_argument('--n-neighbors', type=int, default=3)
parser.add_argument('--n-negs', type=int, default=4)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--data-pickle', type=str, default='ml-1m.pkl')
parser.add_argument('--data-path', type=str, default='/efs/quagan/movielens/ml-1m')
parser.add_argument('--id-as-feature', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
args = parser.parse_args()
n_epoch = args.n_epoch
batch_size = args.batch_size
feature_size = args.feature_size
n_layers = args.n_layers
n_traces = args.n_traces
trace_len = args.trace_len
n_neighbors = args.n_neighbors
n_negs = args.n_negs
weight_decay = args.weight_decay
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
pinsage_p = PinSage(
        HG, 'movie', 'mu', 'um', [feature_size] * n_layers, n_neighbors, n_traces,
        trace_len, True, id_as_feature)
pinsage_q = PinSage(
        HG, 'movie', 'mu', 'um', [feature_size] * n_layers, n_neighbors, n_traces,
        trace_len, True, id_as_feature)
model = FISM(HG, pinsage_p, pinsage_q).to(device)

opt = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)


def train():
    for _ in range(n_epoch):
        train_indices = torch.randperm(len(ratings_train))
        train_batches = train_indices.split(batch_size)

        with tqdm.tqdm(train_batches) as t:
            for train_batch_indices in t:
                U = torch.LongTensor(users_train[train_batch_indices])
                I = torch.LongTensor(movies_train[train_batch_indices])
                I_neg = [torch.LongTensor(np.random.choice(data.neg_train[u], n_negs))
                         for u in U]
                I_neg = torch.stack(I_neg, 0)

                U = U.to(device)
                I = I.to(device)
                I_neg = I_neg.to(device)

                r, r_neg = model(I, U, I_neg)
                r_neg = r_neg.view(-1)
                r_all = torch.cat([r, r_neg])
                y = torch.cat([torch.ones_like(r), torch.zeros_like(r_neg)])
                loss = F.binary_cross_entropy_with_logits(r_all, y)

                opt.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    d_param = param.grad
                    assert not torch.isnan(d_param).any().item()
                opt.step()

                t.set_postfix({'loss': '%.06f' % loss.item()})

        valid_indices = torch.arange(valid_size)
        valid_batches = valid_indices.split(batch_size)

        hits_10s = []
        ndcg_10s = []
        with torch.no_grad():
            with tqdm.tqdm(valid_batches) as t:
                for valid_batch_indices in t:
                    U = torch.LongTensor(users_valid[valid_batch_indices])
                    I = torch.LongTensor(movies_valid[valid_batch_indices])
                    I_neg = [torch.LongTensor(data.neg_valid[u]) for u in U.numpy()]
                    I_neg = torch.stack(I_neg, 0)

                    U = U.to(device)
                    I = I.to(device)
                    I_neg = I_neg.to(device)

                    r, r_neg = model(I, U, I_neg)
                    r_all = torch.cat([r[:, None], r_neg], 1).cpu().numpy()
                    ranks = np.array([scipy.stats.rankdata(-_r, 'min')[0] for _r in r_all])
                    relevance = ((-r_all).argsort(1) == 0)

                    hits_10 = ranks <= 10
                    hits_10s.append(hits_10)
                    ndcg_10 = np.array([ndcg(_r, 10) for _r in relevance])
                    ndcg_10s.append(ndcg_10)

                    t.set_postfix({
                        'HITS@10': '%.03f' % hits_10.mean(),
                        'NDCG@10': '%.03f' % ndcg_10.mean()})

        hits_10 = np.concatenate(hits_10s).mean()
        ndcg_10 = np.concatenate(ndcg_10s).mean()
        print('HITS@10: %.6f NDCG@10: %.6f' % (hits_10, ndcg_10))

train()
