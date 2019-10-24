import torch
import dgl
from collections import Counter



def bipartite_random_walk_sampler(
        HG, nodeset, n_traces, trace_len, forward_etype, backward_etype):
    '''
    HG: DGLGraph
    nodeset: 1D CPU Tensor of node IDs
    restart_prob: float
    n_traces: int
    trace_len: int
    forward_etype, backward_etype: edge types of item-to-user and user-to-item
    '''
    metapath = [forward_etype, backward_etype] * trace_len
    traces = dgl.contrib.sampling.randomwalk.metapath_random_walk(
            HG, metapath, nodeset, n_traces)
    traces = torch.stack([torch.stack(t) for t in traces])

    return traces[:, :, 1::2]

# Note: this function is not friendly to giant graphs since we use a matrix
# with size (num_nodes_in_nodeset, num_nodes_in_graph).

def random_walk_distribution(
        HG, nodeset, n_traces, trace_len, forward_etype, backward_etype):
    n_nodes = nodeset.shape[0]
    item_ntype = HG.to_canonical_etype(forward_etype)[0]
    n_available_nodes = HG.number_of_nodes(item_ntype)
    traces = bipartite_random_walk_sampler(
            HG, nodeset, n_traces, trace_len, forward_etype, backward_etype)
    visited_counts = torch.zeros(n_nodes, n_available_nodes)
    traces = traces.view(n_nodes, -1)
    visited_counts.scatter_add_(1, traces, torch.ones_like(traces, dtype=torch.float32))
    return visited_counts



def random_walk_distribution_topt(
        HG, nodeset, n_traces, trace_len, forward_etype, backward_etype, top_T):
    '''
    returns the top T important neighbors of each node in nodeset, as well as
    the weights of the neighbors.
    '''
    visited_prob = random_walk_distribution(
            HG, nodeset, n_traces, trace_len, forward_etype, backward_etype)
    weights, nodes = visited_prob.topk(top_T, 1)
    weights = weights / weights.sum(1, keepdim=True)
    return weights, nodes



def random_walk_nodeflow(
        HG, nodeset, n_layers, n_traces, trace_len, forward_etype, backward_etype, top_T):
    '''
    returns a list of triplets (
        "active" node IDs whose embeddings are computed at the i-th layer (num_nodes,)
        weight of each neighboring node of each "active" node on the i-th layer (num_nodes, top_T)
        neighboring node IDs for each "active" node on the i-th layer (num_nodes, top_T)
    )
    '''
    dev = nodeset.device
    nodeset = nodeset.cpu()
    nodeflow = []
    cur_nodeset = nodeset
    for i in reversed(range(n_layers)):
        nb_weights, nb_nodes = random_walk_distribution_topt(
                HG, cur_nodeset, n_traces, trace_len, forward_etype, backward_etype, top_T)
        nodeflow.insert(0, (cur_nodeset.to(dev), nb_weights.to(dev), nb_nodes.to(dev)))
        cur_nodeset = torch.cat([nb_nodes.view(-1), cur_nodeset]).unique()

    return nodeflow
