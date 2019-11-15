import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse as ssp
import scipy.stats
import tqdm
import os
import pickle
import argparse
import sh
from sklearn.metrics.pairwise import cosine_similarity
from model.model import FISM
from model.pinsage import PinSage
from model.ranking import evaluate
from model.movielens2 import MovieLens
from model.bookcrossing import BookCrossing
from model.yelp import Yelp2018
from model.randomwalk_sampler import EdgeDataset, EdgeNodeFlowGenerator, \
        NodeDataset, NodeFlowGenerator, to_device

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--n-epoch', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--feature-size', type=int, default=16)
parser.add_argument('--n-layers', type=int, default=1)
parser.add_argument('--n-traces', type=int, default=10)
parser.add_argument('--trace-len', type=int, default=3)
parser.add_argument('--n-neighbors', type=int, default=3)
parser.add_argument('--n-negs', type=int, default=4)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='movielens')
parser.add_argument('--data-pickle', type=str, default='ml-1m.pkl')
parser.add_argument('--data-path', type=str, default='ml-1m.dataset')
parser.add_argument('--model-path', type=str, default='model.pt')
parser.add_argument('--id-as-feature', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--loss-fn', type=str, default='sqr')
parser.add_argument('--neg-by-freq', action='store_true')
parser.add_argument('--neg-freq-min', type=float, default=1)
parser.add_argument('--neg-freq-max', type=float, default=np.inf)
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
dataset = args.dataset
data_pickle = args.data_pickle
data_path = args.data_path
model_path = args.model_path
id_as_feature = args.id_as_feature
lr = args.lr
num_workers = args.num_workers
alpha = args.alpha
pretrain = args.pretrain
optim = args.optim
loss_fn = args.loss_fn
neg_by_freq = args.neg_by_freq
neg_freq_max = args.neg_freq_max
neg_freq_min = args.neg_freq_min

# Load the cached dataset object, or parse the raw MovieLens data
if os.path.exists(data_pickle):
    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)
else:
    if dataset == 'movielens':
        data = MovieLens(data_path)
    elif dataset == 'bx':
        data = BookCrossing(data_path)
    elif dataset == 'yelp':
        data = Yelp2018(data_path)
    with open(data_pickle, 'wb') as f:
        pickle.dump(data, f)

# Fetch the interaction and movie data as numpy arrays
user_latest_item = data.user_latest_item
users_train = data.users_train
movies_train = data.movies_train
users_valid = data.users_valid
movies_valid = data.movies_valid
users_test = data.users_test
movies_test = data.movies_test
train_size = len(users_train)
valid_size = len(users_valid)
test_size = len(users_test)

# Build the bidirectional bipartite graph and put the movie features
HG = dgl.heterograph({
    ('user', 'um', 'movie'): (users_train, movies_train),
    ('movie', 'mu', 'user'): (movies_train, users_train)})
HG.nodes['movie'].data.update(data.movie_data)
HG.to(device)

# Model and optimizer
pinsage_p = PinSage(
        HG, 'movie', 'mu', 'um', feature_size, n_layers, n_neighbors, n_traces,
        trace_len, True, id_as_feature)
pinsage_q = PinSage(
        HG, 'movie', 'mu', 'um', feature_size, n_layers, n_neighbors, n_traces,
        trace_len, True, id_as_feature)
model = FISM(HG, pinsage_p, pinsage_q, alpha).to(device)

opt = getattr(torch.optim, optim)(model.parameters(), weight_decay=weight_decay, lr=lr)

# pretrain with matrix factorization
if pretrain:
    import tempfile
    tmpfile_train_data = tempfile.NamedTemporaryFile('w+')
    tmpfile_valid_data = tempfile.NamedTemporaryFile('w+')
    tmpfile_train_model = tmpfile_train_data.name + '.model'
    tmpfile_train_item_model = tmpfile_train_data.name + '.item'

    for u, m in zip(users_train, movies_train):
        print(u, m, 1, file=tmpfile_train_data)
    for u, m in zip(users_valid, movies_valid):
        print(u, m, 1, file=tmpfile_valid_data)
    mf_train = sh.Command('libmf/mf-train')
    mf_train('-f', 10, '-p', tmpfile_valid_data.name,
             '-k', feature_size, '-t', 6,
             tmpfile_train_data.name, tmpfile_train_model)
    with open(tmpfile_train_model) as f, open(tmpfile_train_item_model, 'w') as f_item:
        for l in f:
            if l.startswith('q'):
                id_, not_nan, item_data = l[1:].split(' ', 2)
                assert not_nan == 'T'
                print(item_data, file=f_item)
    item_emb = np.loadtxt(tmpfile_train_item_model, dtype=np.float32)
    sh.rm(tmpfile_train_model, tmpfile_train_item_model)
    item_emb = torch.FloatTensor(item_emb)
    pinsage_p.h.data[:] = item_emb
    pinsage_q.h.data[:] = item_emb


def train():
    # count number of occurrences for each movie
    if neg_by_freq:
        um = ssp.coo_matrix((np.ones_like(users_train), (users_train, movies_train)))
        movie_count = torch.FloatTensor(um.sum(0).A.squeeze())
    else:
        movie_count = None

    train_dataset = EdgeDataset(
            users_train, movies_train, data.neg_train, n_negs,
            movie_count, neg_freq_max, neg_freq_min)
    node_dataset = NodeDataset(data.num_movies)
    train_collator = EdgeNodeFlowGenerator(
            HG, 'um', 'mu', n_neighbors, n_traces, trace_len, pinsage_p.n_layers, n_negs)
    node_collator = NodeFlowGenerator(
            HG, 'um', 'mu', n_neighbors, n_traces, trace_len, pinsage_p.n_layers, n_negs)
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collator)
    node_loader = DataLoader(
            node_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=node_collator)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    for _ in range(n_epoch):
        # train
        sum_loss = 0
        with tqdm.tqdm(train_loader) as t:
            for U, I, I_neg, I_U, N_U, nf_i, nf_u, nf_neg, I_in_I_U in t:
                U = U.to(device)
                I = I.to(device)
                I_neg = I_neg.to(device)

                r, r_neg = model(I, U, I_neg, I_U, N_U, nf_i, nf_u, nf_neg, I_in_I_U)

                if loss_fn == 'bce':
                    r_neg = r_neg.view(-1)
                    r_all = torch.cat([r, r_neg])
                    y = torch.cat([torch.ones_like(r), torch.zeros_like(r_neg)])
                    loss = F.binary_cross_entropy_with_logits(r_all, y)
                elif loss_fn == 'sqr':
                    diff = 1 - (r.unsqueeze(1) - r_neg)
                    loss = (diff * diff / 2).sum()

                opt.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    d_param = param.grad
                    assert not torch.isnan(d_param).any().item()
                opt.step()

                t.set_postfix({'loss': '%.06f' % loss.item()})

        torch.save(model.state_dict(), model_path)

        with torch.no_grad():
            # evaluate - precompute item embeddings
            z_p = []
            z_q = []
            for item in node_loader:
                I, nf_i = to_device(item, device)
                z_p.append(pinsage_p(I, nf_i))
                z_q.append(pinsage_q(I, nf_i))
            z_p = torch.cat(z_p)
            z_q = torch.cat(z_q)

            metrics_valid = []
            metrics_test = []
            metrics_test_all = []
            for U in tqdm.trange(data.num_users):
                _, I_U = HG.out_edges(U, form='uv', etype='um')
                n_valid = n_test = 1
                n_valid_neg = len(data.neg_valid[U])
                n_test_neg = len(data.neg_test[U])
                n_test_neg_all = len(data.neg_test_complete[U])

                I = torch.LongTensor(np.concatenate([
                    [data.movies_valid[U]],
                    [data.movies_test[U]],
                    data.neg_valid[U],
                    data.neg_test[U],
                    data.neg_test_complete[U]])).to(device)
                U = torch.tensor(U, dtype=torch.int64).to(device)

                p_ctx = z_p[I_U].mean(0)
                q = z_q[I]

                relevance = np.array([1])
                score = model.rating(U, I, p_ctx, q)
                score_valid, score_test, score_valid_neg, score_test_neg, score_test_neg_all = \
                        score.split([n_valid, n_test, n_valid_neg, n_test_neg, n_test_neg_all])

                metrics_valid.append(evaluate(
                        torch.cat([score_valid, score_valid_neg]).cpu().numpy(),
                        1, relevance))
                metrics_test.append(evaluate(
                        torch.cat([score_test, score_test_neg]).cpu().numpy(),
                        1, relevance))
                metrics_test_all.append(evaluate(
                        torch.cat([score_test, score_test_neg_all]).cpu().numpy(),
                        1, relevance))

        metrics_valid = np.mean(metrics_valid, 0)
        metrics_test = np.mean(metrics_test, 0)
        metrics_test_all = np.mean(metrics_test_all, 0)

        print('HITS@10:', metrics_valid[0],
              'NDCG@10:', metrics_valid[1],
              'HITS@10 (Test):', metrics_test[0],
              'NDCG@10 (Test):', metrics_test[1],
              'HITS@10 (Test all):', metrics_test_all[0],
              'NDCG@10 (Test all):', metrics_test_all[1])


        '''
        if user_latest_item is None:
            continue

        # find most similar item embedding
        z_q = z_q.cpu().numpy()
        m_dist = cosine_similarity(z_q)

        hits_10s = []
        ndcg_10s = []
        for u, i in zip(users_valid, movies_valid):
            I_q = user_latest_item[u]
            I_pos = np.array([i])
            I_neg = data.neg_valid[u]
            relevance = np.array([1])

            I = torch.cat([torch.LongTensor(I_pos), torch.LongTensor(I_neg)])
            score = m_dist[I_q][I]
            hits_10, ndcg_10 = evaluate(score, 1, relevance)
            hits_10s.append(hits_10)
            ndcg_10s.append(ndcg_10)

        hits_10_knn_valid = np.mean(hits_10s)
        ndcg_10_knn_valid = np.mean(ndcg_10s)

        hits_10s = []
        ndcg_10s = []
        hits_10s_all = []
        ndcg_10s_all = []
        for u, i in zip(users_test, movies_test):
            I_q = user_latest_item[u]
            I_pos = np.array([i])
            I_neg = data.neg_test[u]
            I_neg_all = data.neg_test_complete[u]
            relevance = np.array([1])

            I = torch.cat([torch.LongTensor(I_pos), torch.LongTensor(I_neg)])
            score = m_dist[I_q][I]
            hits_10, ndcg_10 = evaluate(score, 1, relevance)
            hits_10s.append(hits_10)
            ndcg_10s.append(ndcg_10)

        hits_10_knn_test = np.mean(hits_10s)
        ndcg_10_knn_test = np.mean(ndcg_10s)

        print('NN-HITS@10: %.6f NN-NDCG@10: %.6f' % (hits_10_knn_valid, ndcg_10_knn_valid),
              'Test NN-HITS@10: %.6f NN-NDCG@10: %.6f' % (hits_10_knn_test, ndcg_10_knn_test))
        '''


train()
