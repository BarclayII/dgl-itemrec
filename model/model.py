import torch
import torch.nn as nn

class FISM(nn.Module):
    r"""
    PinSAGE + FISM for item-based recommender systems

    The formulation of FISM goes as

    .. math::

       r_{ui} = b_u + b_i + \left(n_u^+\right)^{-\alpha}
       \sum_{j \in R_u^+} p_j q_i^\top

    In FISM, both :math:`p_j` and :math:`q_i` are trainable parameters.  Here
    we replace them as outputs from two PinSAGE models ``P`` and
    ``Q``.
    """
    def __init__(self, HG, P, Q, alpha=0):
        super().__init__()

        self.P = P
        self.Q = Q
        self.HG = HG
        self.b_u = nn.Parameter(torch.zeros(HG.number_of_nodes('user')))
        self.b_i = nn.Parameter(torch.zeros(HG.number_of_nodes('movie')))
        self.alpha = alpha

    
    def forward(self, I, U, I_neg=None):
        '''
        I: 1D LongTensor
        U: 1D LongTensor
        I_neg: 2D LongTensor (batch_size, n_negs)
        '''
        batch_size = I.shape[0]
        device = I.device
        # all item embeddings needed for computation
        _, I_U = self.HG.out_edges(U, form='uv', etype='um')
        I_U = I_U.to(device)
        # number of interacted items
        N_U = self.HG['um'].out_degrees(U).to(device)
        U_idx = torch.arange(U.shape[0], device=device).repeat_interleave(N_U)

        q = self.Q(I)

        p = self.P(I_U)
        p_self = self.P(I)
        p_sum = torch.zeros_like(q)
        p_sum = p_sum.scatter_add(0, U_idx[:, None].expand_as(p), p)    # batch_size, n_dims
        p_ctx = p_sum - p_self
        pq = (p_ctx * q).sum(1) / (N_U.float() ** self.alpha)
        r = self.b_u[U] + self.b_i[I] + pq

        if I_neg is not None:
            n_negs = I_neg.shape[1]
            I_neg_flat = I_neg.view(-1)
            q_neg = self.Q(I_neg_flat)
            q_neg = q_neg.view(batch_size, n_negs, -1)  # batch_size, n_negs, n_dims
            pq_neg = (p_ctx.unsqueeze(1) * q_neg).sum(2) / (N_U.float().unsqueeze(1) ** self.alpha)
            r_neg = self.b_u[U].unsqueeze(1) + self.b_i[I_neg] + pq_neg
            return r, r_neg
        else:
            return r
