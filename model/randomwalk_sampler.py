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
    def __init__(self, users, movies):
        train_size = len(users)
        M_um = ssp.coo_matrix((np.ones(train_size), (users, movies)))
        M_mm = (M_um.T * M_um).tocoo()

        rows, cols, counts = M_mm.row, M_mm.col, M_mm.data

        self.rows = rows
        self.cols = cols
        self.counts = counts

    def __getitem__(self, i):
        return self.rows[i], self.cols[i], self.counts[i]

    def __len__(self):
        return len(self.rows)


class CooccurrenceNodeFlowGenerator(BaseNodeFlowGenerator):
    
    def __call__(self, batch):
        I_q, I_i, c = zip(*batch)
        I_q = torch.LongTensor(I_q)
        I_i = torch.LongTensor(I_i)
        c = torch.FloatTensor(c)
        I_neg = torch.randint(0, self.num_movies, (len(I_q), self.n_negs))

        nf_q = self.generate(I_q)
        nf_i = self.generate(I_i)
        nf_neg = self.generate(I_neg.view(-1))

        return I_q, I_i, I_neg, nf_q, nf_i, nf_neg, c


class EdgeDataset(Dataset):
    def __init__(self, users, movies, negs, n_negs_to_sample):
        self.users = users
        self.movies = movies
        self.negs = negs
        self.n_negs_to_sample = n_negs_to_sample

    def __getitem__(self, i):
        u = self.users[i]
        return u, self.movies[i], \
                np.random.choice(self.negs[u], self.n_negs_to_sample)

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

        return U, I, I_neg, I_U, N_U, nf_i, nf_u, nf_neg


def to_device(x, device):
    if isinstance(x, list):
        return [to_device(y, device) for y in x]
    elif isinstance(x, tuple):
        return tuple(to_device(y, device) for y in x)
    else:
        return x.to(device)
