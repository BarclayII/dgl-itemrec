from dgl.contrib.sampling.randomwalk import pinsage_neighbor_sampling
from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.sparse as ssp


def generate_pinsage_nodeflow(hg, etypes, seeds, num_neighbors, num_traces, trace_length,
                              num_layers):
    nodeflow = []
    cur_nodeset = torch.unique(seeds)

    for i in reversed(range(num_layers)):
        nb_nodes, nb_weights = pinsage_neighbor_sampling(
                hg, etypes, cur_nodeset, num_neighbors, num_traces, trace_length)
        nb_weights = nb_weights.float()
        nb_weights /= nb_weights.sum(1, keepdim=True)
        nodeflow.insert(0, (cur_nodeset, nb_weights, nb_nodes))
        cur_nodeset = torch.cat([nb_nodes.view(-1), cur_nodeset]).unique()

    return nodeflow


class BaseNodeFlowGenerator(object):
    """
    Works as a PyTorch DataLoader collator that "collates" the seed nodes into
    a computation dependency flow.
    """
    def __init__(self, HG, umtype, mutype, num_neighbors, num_traces, trace_length,
                 num_layers, n_negs):
        self.HG = HG
        self.umtype = umtype
        self.mutype = mutype
        self.utype, _, self.mtype = HG.to_canonical_etype(umtype)
        self.users, self.movies = HG.all_edges(etype=umtype)
        self.num_users = HG.number_of_nodes(self.utype)
        self.num_movies = HG.number_of_nodes(self.mtype)
        self.num_neighbors = num_neighbors
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.num_layers = num_layers
        self.n_negs = n_negs

    def generate(self, seeds):
        return generate_pinsage_nodeflow(
                self.HG,
                [self.mutype, self.umtype],
                seeds,
                self.num_neighbors,
                self.num_traces,
                self.trace_length,
                self.num_layers)


class NodeDataset(Dataset):
    def __init__(self, num_movies):
        self.num_movies = num_movies

    def __len__(self):
        return self.num_movies

    def __getitem__(self, i):
        return i


class NodeFlowGenerator(BaseNodeFlowGenerator):
    def __call__(self, I):
        I = torch.LongTensor(I)
        nf_i = self.generate(I)
        return I, nf_i


class CooccurrenceDataset(Dataset):
    def __init__(self, users, movies, m_nn=None):
        train_size = len(users)
        M_um = ssp.coo_matrix((np.ones(train_size), (users, movies)))
        M_mm = (M_um.T * M_um).tocoo()

        rows, cols, counts = M_mm.row, M_mm.col, M_mm.data

        self.rows = rows
        self.cols = cols
        self.counts = counts
        if m_nn is not None:
            self.m_nn_cols = m_nn.flatten()
            self.m_nn_rows = np.arange(m_nn.shape[0])[:, None].repeat(m_nn.shape[1]).flatten()
        else:
            self.m_nn_cols = self.m_nn_rows = None

    def __getitem__(self, i):
        if self.m_nn_cols is None:
            return self.rows[i], self.cols[i], self.counts[i]
        else:
            return self.m_nn_rows[i], self.m_nn_cols[i], 1

    def __len__(self):
        return len(self.rows) if self.m_nn_cols is None else len(self.m_nn_cols)


class CooccurrenceNodeFlowGenerator(BaseNodeFlowGenerator):
    def __init__(self, *args, movie_freq=None, movie_freq_min=1, movie_freq_max=100, eta=0.5):
        super().__init__(*args)
        if movie_freq is not None:
            self.movie_freq = (
                    movie_freq.float()
                    .clamp(min=movie_freq_min, max=movie_freq_max))
            self.movie_freq = self.movie_freq ** eta
        else:
            self.movie_freq = None
        self.eta = eta

    def __call__(self, batch):
        I_q, I_i, c = zip(*batch)
        I_q = torch.LongTensor(I_q)
        I_i = torch.LongTensor(I_i)
        c = torch.FloatTensor(c)
        if self.movie_freq is None:
            I_neg = torch.randint(0, self.num_movies, (len(I_q), self.n_negs))
        else:
            I_neg = torch.multinomial(self.movie_freq, len(I_q) * self.n_negs, replacement=True)
            I_neg = I_neg.view(-1, self.n_negs)

        nf_q = self.generate(I_q)
        nf_i = self.generate(I_i)
        nf_neg = self.generate(I_neg.view(-1))

        return I_q, I_i, I_neg, nf_q, nf_i, nf_neg, c


class EdgeDataset(Dataset):
    def __init__(self, users, movies, negs, n_negs_to_sample, movie_freq=None,
                 movie_freq_min=1, movie_freq_max=100, eta=0.5):
        self.users = users
        self.movies = movies
        self.negs = negs
        self.n_negs_to_sample = n_negs_to_sample
        if movie_freq is not None:
            self.movie_freq = np.maximum(np.minimum(movie_freq, movie_freq_max), movie_freq_min)
            self.movie_freq = self.movie_freq ** eta
        else:
            self.movie_freq = None

    def __getitem__(self, i):
        u = self.users[i]
        if self.movie_freq is not None:
            freq = self.movie_freq[self.negs[u]]
            p = freq / freq.sum()
            negs = np.random.choice(self.negs[u], self.n_negs_to_sample, replace=True, p=p)
        else:
            negs = np.random.choice(self.negs[u], self.n_negs_to_sample)
        return u, self.movies[i], negs

    def __len__(self):
        return len(self.users)


class FeedbackDataset(Dataset):
    def __init__(self, users, movies, feedback, n_negs_to_sample, movie_freq=None,
                 movie_freq_min=1, movie_freq_max=100, eta=0.5):
        self.users = users
        self.movies = movies
        self.feedback = feedback
        self.n_negs_to_sample = n_negs_to_sample
        if movie_freq is not None:
            self.movie_freq = np.maximum(np.minimum(movie_freq, movie_freq_max), movie_freq_min)
            self.movie_freq = self.movie_freq ** eta
        else:
            self.movie_freq = None

    #@profile
    def __getitem__(self, i):
        u = self.users[i]
        n_movies = self.feedback.shape[1]
        if self.movie_freq is not None:
            freq = self.movie_freq
            p = freq / freq.sum()
            negs = np.random.choice(n_movies, self.n_negs_to_sample, replace=True, p=p)
        else:
            negs = np.random.choice(n_movies, self.n_negs_to_sample)

        m = self.movies[i]
        fb_pos = self.feedback[u, m]
        fb_neg = self.feedback[u, negs]
        return u, m, negs, fb_pos, fb_neg

    def __len__(self):
        return len(self.users)


class EdgeNodeFlowGenerator(BaseNodeFlowGenerator):
    def __call__(self, batch):
        U, I, I_neg = zip(*batch)
        U = torch.LongTensor(U)
        I = torch.LongTensor(I)
        I_neg = torch.LongTensor(np.stack(I_neg, 0))
        _, I_U = self.HG.out_edges(U, form='uv', etype=self.umtype)
        N_U = self.HG[self.umtype].out_degrees(U)

        nf_i = self.generate(I)
        nf_u = self.generate(I_U)
        nf_neg = self.generate(I_neg.view(-1))

        I_Us = I_U.split(N_U.numpy().tolist())
        I_in_I_U = [
                _I.numpy() in _I_U.numpy()
                for _I, _I_U in zip(I, I_Us)]
        I_in_I_U = torch.LongTensor(I_in_I_U)

        return U, I, I_neg, I_U, N_U, nf_i, nf_u, nf_neg, I_in_I_U


class FeedbackNodeFlowGenerator(BaseNodeFlowGenerator):
    #@profile
    def __call__(self, batch):
        U, I, I_neg, fb_pos, fb_neg = zip(*batch)
        U = torch.LongTensor(U)
        I = torch.LongTensor(I)
        I_neg = torch.LongTensor(np.stack(I_neg, 0))
        fb_pos = torch.FloatTensor(fb_pos)
        fb_neg = torch.FloatTensor(np.stack(fb_neg, 0))
        _, I_U = self.HG.out_edges(U, form='uv', etype=self.umtype)
        N_U = self.HG[self.umtype].out_degrees(U)

        nf_i = self.generate(I)
        nf_u = self.generate(I_U)
        nf_neg = self.generate(I_neg.view(-1))

        I_Us = I_U.split(N_U.numpy().tolist())
        I_in_I_U = [
                _I.numpy() in _I_U.numpy()
                for _I, _I_U in zip(I, I_Us)]
        I_in_I_U = torch.LongTensor(I_in_I_U)

        return U, I, I_neg, I_U, N_U, nf_i, nf_u, nf_neg, I_in_I_U, fb_pos, fb_neg


def to_device(x, device):
    if isinstance(x, list):
        return [to_device(y, device) for y in x]
    elif isinstance(x, tuple):
        return tuple(to_device(y, device) for y in x)
    else:
        return x.to(device)
