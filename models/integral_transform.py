"""
code taken from https://github.com/yzshi5/MINO.git
@article{shi2025mesh,
  title={Mesh-Informed Neural Operator: A Transformer Generative Approach},
  author={Shi, Yaozhong and Ross, Zachary E and Asimaki, Domniki and Azizzadenesheli, Kamyar},
  journal={arXiv preprint arXiv:2506.16656},
  year={2025}
}
"""

import torch
from torch import nn
import torch.nn.functional as F
import neuralop

from neuralop.layers.channel_mlp import LinearChannelMLP
from neuralop.layers.segment_csr import segment_csr
#from .channel_mlp import LinearChannelMLP
#from .segment_csr import segment_csr

class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(y)} k(x, y) dx  
        (b) \int_{A(y)} k(x, y) * f(x) dx
        (c) \int_{A(y)} k(x, y, f(x)) dx
        (d) \int_{A(y)} k(x, y, f(x)) * f(x) dx

    y : Points for which the output is defined

    x : Points for which the input is defined
    A(y) : A subset of all points x (depending on\
        each y) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)
    
    f : Input function to integrate against given\
        on the points x

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(y) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        MLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the 
        kernel integral.
    channel_mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class
    channel_mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        LinearChannelMLP class. Only used if channel_mlp_layers is
        given and channel_mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_torch_scatter : bool, default 'True'
        Whether to use torch_scatter's implementation of 
        segment_csr or our native PyTorch version. torch_scatter 
        should be installed by default, but there are known versioning
        issues on some linux builds of CPU-only PyTorch. Try setting
        to False if you experience an error from torch_scatter.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_torch_scatter=True,
    ):
        super().__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter

        if (
            self.transform_type != "linear_kernelonly"
            and self.transform_type != "linear"
            and self.transform_type != "nonlinear_kernelonly"
            and self.transform_type != "nonlinear"
        ):
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
            

    """"
    

    Assumes y=x if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def forward(self, x, neighbors, y=None, f_x=None, weights=None):
        """Compute a kernel integral transform

        Parameters
        ----------
        x : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(y) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        y : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            y = x.
        f_x : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points x. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are x_{j+1} - x_j. If None,
            1/|A(y)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points y.
            d4 is the output size of the kernel k.
        """

        #print('x:{}, y:{}, f_x:{}'.format(x.shape, y.shape, f_x.shape))
        if y is None:
            y = x

        rep_features = y[neighbors["neighbors_index"]]

        # batching only matters if f_x (latent embedding) values are provided
        batched = False
        # f_x has a batch dim IFF batched=True
        if f_x is not None:
            if f_x.ndim == 3:
                batched = True
                batch_size = f_x.shape[0]
                in_features = f_x[:, neighbors["neighbors_index"], :]
            elif f_x.ndim == 2:
                batched = False
                in_features = f_x[neighbors["neighbors_index"]]

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )

        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=-1)
        #print('agg_features before.shape:{}, in_features:{}'.format(agg_features.shape, in_features.shape))
        if f_x is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        #print('agg_features.shape:{}'.format(agg_features.shape))
        rep_features = self.channel_mlp(agg_features)

        #print('rep_features:{}, in_features:{}'.format(rep_features.shape, in_features.shape))
        #if f_x is not None and self.transform_type != "nonlinear_kernelonly":
            #rep_features = rep_features * in_features
            #rep_features = rep_features.mul_(in_features)

        #print('rep_fatures.shape:{}'.format(rep_features.shape))
        reduction = "mean"

        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(rep_features, splits, reduce=reduction, use_scatter=self.use_torch_scatter)

        #print('out_features.{}'.format(out_features.shape))
        return out_features