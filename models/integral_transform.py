import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.channel_mlp import LinearChannelMLP
from neuralop.layers.segment_csr import segment_csr

"""
this code assumes same mesh (same x), different "PDE" solutions. batching logic gives it away
extend this to different meshes/geometries, unless in pre-processing all geometries are mapped to reference manifold
"""

class IntegralTransform(nn.Module):

    def __init__(
        self,
        channel_mlp,
        transform_type="linear",
        weights: str = "attention",   # NEW
        use_torch_scatter=True,
    ):
        super().__init__()
        self.channel_mlp = channel_mlp
        self.transform_type = transform_type
        self.weights = weights
        self.use_torch_scatter = use_torch_scatter

        assert weights in ["attention", "mean"], \
            "weights must be 'attention' or 'mean'"
        
def forward(
    self,
    x,                    # [n, d_x]
    y,                    # [m, d_y]
    neighbors_index,      # [M]
    row_splits,           # [m+1]
    x_embed=None,         # [n, d_emb] optional
    y_embed=None,         # [m, d_emb] optional
    f_x=None,             # [bs, n, f_ch]
    phi_x=None,           # [bs, n, out_ch]
    Q=None,               # [m, out_ch]
    K_att=None,           # [bs, n, out_ch]
):
    """
    Hybrid MINO + AGNO integral transform supporting:
        K(x_emb, y_emb, f(x))
        α(x,y) attention weights
        CSR aggregation
    """

    splits = row_splits
    neighbors = neighbors_index

    # batch detection (see MINO)
    batched = f_x is not None and f_x.ndim == 3
    bs = f_x.shape[0] if batched else 1

    # coordinates (x*, y*) over which K(x*,y*, f(x))) is computed
    x_kernel = x_embed if x_embed is not None else x
    y_kernel = y_embed if y_embed is not None else y

    # --------------------------------------------------
    # build edge features (MINO-style CSR)
    # --------------------------------------------------

    # y part
    rep_y = y_kernel[neighbors]                  # [M, d_emb or d_y]
    # x part
    num_reps = splits[1:] - splits[:-1]
    rep_x = torch.repeat_interleave(x_kernel, num_reps, dim=0)
    agg_features = torch.cat([rep_y, rep_x], dim=-1)
    if batched:
        agg_features = agg_features.unsqueeze(0).expand(bs, -1, -1)

    # --------------------------------------------------
    # nonlinear kernel case: add f(x)
    # 1. "nonlinear": K(x*, y*, f(x)) * Φ(f(x))
    # 2. "nonlinear_kernelonly": K(x*, y*, f(x))
    # --------------------------------------------------
    if f_x is not None and self.transform_type in ["nonlinear", "nonlinear_kernelonly"]:
        f_edge = (f_x[:, neighbors, :] if batched else f_x[neighbors]) # [bs, M, f_ch] or [M, f_ch]
        agg_features = torch.cat([agg_features, f_edge], dim=-1)

    # evaluate K() mlp
    K_xy = self.channel_mlp(agg_features)

    # Φ(f(x))
    if phi_x is not None: # "nonlinear"
        assert self.transform_type == 'nonlinear' and phi_x is not None, "Error: transform_type 'nonlinear' requires phi_x to be provided."
        phi_edge = (phi_x[:, neighbors, :] if batched else phi_x[neighbors])
        K_xy = K_xy * phi_edge

    # weights α(x,y)
    if self.weights == "attention":

        assert Q is not None and K_att is not None, "Error: attention weights require Q and K_att."

        # query_index from CSR splits
        m = splits.shape[0] - 1
        row_lengths = splits[1:] - splits[:-1]
        query_index = torch.repeat_interleave(torch.arange(m, device=splits.device),row_lengths)

        if batched:
            Q_edge = Q[query_index]                 # [M, out_ch]
            K_edge = K_att[:, neighbors, :]         # [bs, M, out_ch]
            scores = (Q_edge.unsqueeze(0) * K_edge).sum(-1)
        else:
            Q_edge = Q[query_index]
            K_edge = K_att[neighbors]
            scores = (Q_edge * K_edge).sum(-1)

        scores = scores / (Q.shape[-1] ** 0.5)
        scores_exp = torch.exp(scores)

        if batched:
            denom = segment_csr(scores_exp, splits.repeat(bs, 1), reduce="sum",use_scatter=self.use_torch_scatter)
            alpha = scores_exp / denom[:, query_index]
        else:
            denom = segment_csr(scores_exp, splits, reduce="sum", use_scatter=self.use_torch_scatter)
            alpha = scores_exp / denom[query_index]

        K_xy = K_xy * alpha.unsqueeze(-1)

    # final agregation
    out = segment_csr(K_xy, splits.repeat(bs, 1) if batched else splits, 
                      reduce="sum" if self.weights == "attention" else "mean",
                      use_scatter=self.use_torch_scatter)

    return out