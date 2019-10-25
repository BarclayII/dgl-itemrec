import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import tqdm
import os
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
batch_size = 32
feature_size = 16
n_layers = 2
n_traces = 10
trace_len = 3
n_neighbors = 3
n_negs = 4
weight_decay = 1e-2
data_pickle = 'ml-1m.pkl'
data_path = '/efs/quagan/movielens/ml-1m'

if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)
else:
    data = MovieLens(data_path)
    with open(data_pickle, 'wb') as f:
        pickle.dump(data, f)

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

HG = dgl.heterograph({
    ('user', 'um', 'movie'): (ratings_train['user_idx'], ratings_train['movie_idx']),
    ('movie', 'mu', 'user'): (ratings_train['movie_idx'], ratings_train['user_idx'])})
HG.nodes['movie'].data.update(data.movie_data)
HG.to(device)

pinsage_p = PinSage(
        HG, 'movie', 'mu', 'um', [feature_size] * n_layers, n_neighbors, n_traces,
        trace_len, True)
pinsage_q = PinSage(
        HG, 'movie', 'mu', 'um', [feature_size] * n_layers, n_neighbors, n_traces,
        trace_len, True)
model = FISM(HG, pinsage_p, pinsage_q).to(device)

opt = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)

@profile
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

        valid_indices = torch.arange(len(ratings_valid))
        valid_batches = valid_indices.split(batch_size)

        hits_10s = []
        ndcg_10s = []
        with torch.no_grad():
            with tqdm.tqdm(valid_batches) as t:
                for valid_batch_indices in t:
                    U = users_valid[valid_batch_indices]
                    I = movies_valid[valid_batch_indices]
                    I_neg = [torch.LongTensor(data.neg_valid[u]) for u in U.numpy()]
                    I_neg = torch.stack(I_neg, 0)

                    U = U.to(device)
                    I = I.to(device)
                    I_neg = I_neg.to(device)

                    r, r_neg = model(HG, U, I, I_neg)
                    r_all = torch.cat([r[:, None], r_neg], 1).cpu().numpy()
                    ranks = np.array([scipy.stats.rankdata(-_r, 'min')[0] for _r in r_all])
                    relevance = (r_all.argsort(1) == 0).float().numpy()

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
