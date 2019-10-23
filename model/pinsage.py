import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from . import randomwalk
from .utils import cuda

def create_embeddings(n_nodes, n_features):
    return nn.Parameter(torch.randn(n_nodes, n_features))

def mix_embeddings(h, ndata, emb, proj):
    '''Combine node-specific trainable embedding ``h`` with categorical inputs
    (projected by ``emb``) and numeric inputs (projected by ``proj``).
    '''
    e = []
    for key, value in ndata.items():
        if value.dtype == torch.int64:
            e.append(emb[key](value))
        elif value.dtype == torch.float32:
            e.append(proj[key](value))
    return h + torch.stack(e, 0).sum(0)

def get_embeddings(h, nodeset):
    return h[nodeset]

def put_embeddings(h, nodeset, new_embeddings):
    n_nodes = nodeset.shape[0]
    n_features = h.shape[1]
    return h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), new_embeddings)

def safediv(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b)
    return a / b

def init_weight(w, func_name, nonlinearity):
    getattr(nn.init, func_name)(w, gain=nn.init.calculate_gain(nonlinearity))

def init_bias(w):
    nn.init.constant_(w, 0)

class PinSageConv(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(PinSageConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)

        init_weight(self.Q.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.Q.bias)
        init_bias(self.W.bias)


    def forward(self, h, nodeset, nb_nodes, nb_weights):
        '''
        h: node embeddings (num_total_nodes, in_features), or a container
           of the node embeddings (for distributed computing)
        nodeset: node IDs in this minibatch (num_nodes,)
        nb_nodes: neighbor node IDs of each node in nodeset (num_nodes, num_neighbors)
        nb_weights: weight of each neighbor node (num_nodes, num_neighbors)
        return: new node embeddings (num_nodes, out_features)
        '''
        n_nodes, T = nb_nodes.shape

        h_nodeset = get_embeddings(h, nodeset)  # (n_nodes, in_features)
        h_neighbors = get_embeddings(h, nb_nodes.view(-1)).view(n_nodes, T, self.in_features)

        h_neighbors = F.leaky_relu(self.Q(h_neighbors))
        h_agg = safediv(
                (nb_weights[:, :, None] * h_neighbors).sum(1),
                nb_weights.sum(1, keepdim=True))

        h_concat = torch.cat([h_nodeset, h_agg], 1)
        h_new = F.leaky_relu(self.W(h_concat))
        h_new = safediv(h_new, h_new.norm(dim=1, keepdim=True))

        return h_new

class PinSage(nn.Module):
    '''
    Completes a multi-layer PinSage convolution
    HG: DGLHeteroGraph
    ntype: node type whose embeddings are computed (items)
    forward_etype: item-to-user edge type
    backward_etype: user-to-item edge type
    feature_sizes: the dimensionality of input/hidden/output features
    T: number of neighbors we pick for each node
    n_traces: number of random walk traces to generate for top-k neighborhood sampling
    trace_len: length of each random walk trace
    '''
    def __init__(self, HG, ntype, forward_etype, backward_etype,
                 feature_sizes, T, n_traces, trace_len,
                 use_feature=False):
        super(PinSage, self).__init__()

        self.HG = HG
        self.ntype = ntype
        self.forward_etype = forward_etype
        self.backward_etype = backward_etype
        self.T = T
        self.n_traces = n_traces
        self.trace_len = trace_len

        self.in_features = feature_sizes[0]
        self.out_features = feature_sizes[-1]
        self.n_layers = len(feature_sizes) - 1

        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(PinSageConv(
                feature_sizes[i], feature_sizes[i+1], feature_sizes[i+1]))

        self.h = create_embeddings(HG.number_of_nodes(ntype), self.in_features)
        self.use_feature = use_feature

        if use_feature:
            self.emb = nn.ModuleDict()
            self.proj = nn.ModuleDict()

            # functions that project input categorical/real-valued features
            for key, scheme in HG.node_attr_schemes(ntype).items():
                if scheme.dtype == torch.int64:
                    self.emb[key] = nn.Embedding(
                            HG.nodes[ntype].data[key].max().item() + 1,
                            self.in_features,
                            padding_idx=0)
                elif scheme.dtype == torch.float32:
                    self.proj[key] = nn.Sequential(
                            nn.Linear(scheme.shape[0], self.in_features),
                            nn.LeakyReLU(),
                            )

    def forward(self, nodeset):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        nodeset: node IDs in this minibatch (num_nodes,)
        return: new node embeddings (num_nodes, out_features)
        '''
        if self.use_feature:
            h = mix_embeddings(self.h, self.HG.nodes[self.ntype].data,
                    self.emb, self.proj)
        else:
            h = self.h

        nodeflow = randomwalk.random_walk_nodeflow(
                HG, nodeset, self.n_layers, self.n_traces, self.trace_len,
                self.forward_etype, self.backward_etype, self.T)

        for i, (nodeset, nb_weights, nb_nodes) in enumerate(nodeflow):
            new_embeddings = self.convs[i](h, nodeset, nb_nodes, nb_weights)
            h = put_embeddings(h, nodeset, new_embeddings)

        h_new = get_embeddings(h, nodeset)
        return h_new
