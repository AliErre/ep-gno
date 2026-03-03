# contains nn modules
from typing import List


from pyparsing import alphas
import torch
from torch import nn
import torch.nn.functional as F

import neuralop
from neuralop.layers.channel_mlp import LinearChannelMLP
from .integral_transform import IntegralTransform
from neuralop.layers.neighbor_search import NeighborSearch
#from neuralop.layers.embeddings import SinusoidalEmbedding
#from kappamodules.layers import ContinuousSincosEmbed
#from .conditioner_timestep import ContinuousSincosEmbed
from .embeddings import LiftedPointEmbedding
from torch_geometric.utils import softmax

class AGNOBlock(nn.Module):
    def __init__(self, f_channels:int, out_channels:int, coord_dim:int, radius:float,
                 transform_type:str='linear', pos_embed:bool=False, pos_embedding_channels:int=32,
                 geometry_embedding:bool=False, geometry_embedding_channels:int=32,
                 channel_mlp_layers:List[int]=[128, 256, 128], channel_mlp_non_linearity=F.gelu,
                 channel_mlp:nn.Module=None, use_open3d_neighbor_search:bool=True, use_torch_scatter:bool=True):
        """
        transform_type == nonlinear: K(x,y,f(x)) 
        """
        
        super().__init__()
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim
        self.radius = radius
        self.pos_embed = pos_embed
        assert channel_mlp is None or channel_mlp_layers is None, "Error: cannot specify both channel_mlp and channel_mlp_layers."

        # MLP (linear lifting)
        self.lift_ffn = nn.Linear(self.f_channels, self.out_channels) # latent dim = out
        # point embedding through MLP (linear lifting), see GAOT. p_emb in [bs, n_pts, out_dim]
        if pos_embedding_channels is not None: # MINO embeds pos and computes K(x_embed, y_embed, f)
            # in MINO use sinusoidal
            self.pos_embedding_channels = pos_embedding_channels
            self.point_embedding = LiftedPointEmbedding(in_dim = self.coord_dim, out_dim = pos_embedding_channels) # latent dim
        else: # default: compute K on original domain coordinates
            self.pos_embedding_channels = self.out_channels
            self.point_embedding = LiftedPointEmbedding(in_dim = self.coord_dim, out_dim = self.out_channels) # latent dim
        
        # learnable weight matrices for attention based quadrature kernel weights
        # target position y -> query
        self.W_q = nn.Linear(self.pos_embedding_channels, self.out_channels)
        # source position x -> key
        self.W_k = nn.Linear(self.pos_embedding_channels, self.out_channels)
        # create object for directed graph computation
        if use_open3d_neighbor_search: # the graph is instantiated for x and y even when K(x_embed, y_embed)
            assert self.coord_dim == 3, f"Error: open3d is only designed for 3d data, \
                GNO instantiated for dim={coord_dim}"
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d_neighbor_search)


        # MLP for kernel K
        kernel_in_ch = self.f_channels + 2*self.pos_embedding_channels if self.pos_embed else self.f_channels + 2*self.coord_dim
        # K(x_emb, y_emb, MLP(f(x))) or K(x, y, MLP(f(x))) depending on pos_embed
        if channel_mlp is not None: # personalized channel if transform_type is nonlinear
            assert channel_mlp.in_channels == kernel_in_ch, f"Error: expected ChannelMLP to take\
                  input with {kernel_in_ch} channels (feature channels={kernel_in_ch}),\
                      got {channel_mlp.in_channels}."
            assert channel_mlp.out_channels == self.out_channels, f"Error: expected ChannelMLP to have\
                 {self.out_channels=} but got {channel_mlp.out_channels=}."
            self.channel_mlp = channel_mlp # personalized and passed as argument

        elif channel_mlp_layers is not None:
            if channel_mlp_layers[0] != kernel_in_ch:
                channel_mlp_layers = [kernel_in_ch] + channel_mlp_layers
            if channel_mlp_layers[-1] != self.out_channels:
                channel_mlp_layers.append(self.out_channels)       
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        
        # integral transform module
        self.integral_transform = IntegralTransform(channel_mlp=self.channel_mlp, transform_type=transform_type, use_torch_scatter=use_torch_scatter)

    def forward(self, x, y, f_x):
        # x: [bs, n_x, coord_dim]
        # y: [n_y, coord_dim]
        # f_x: [bs, n_x, f_channels]

        if self.pos_embed:
            x = self.point_embedding(x) # [bs, n_x, pos_embedding_channels]
            y = self.point_embedding(y) # [n_y, pos_embedding_channels]

        Q = self.W_q(y) # [n_y, out_ch]
        K = self.W_k(x) # [bs, n_x, out_ch]
        neighbors_dict = self.neighbor_search(data=x, queries=y, radius=self.radius) # neighbors of y
        
        src_idx, trg_idx = neighbors_dict['edge_index']

        # sparse aplhas (only for neighbors)
        # Q[trg_idx]: [num_edges, out_ch], K[:, src_idx]: [bs, num_edges, out_ch]
        dot_prod = torch.sum(Q[trg_idx] * K[:, src_idx], dim=-1) 
        scores = dot_prod / (self.out_channels ** 0.5)
        alphas =  softmax(scores, trg_idx, dim=-1) # [num_edges]
        f_emb = self.lift_ffn(f_x) # [bs, n_x, out_channels]
        f_y = self.integral_transform(x=x, neighbors=neighbors_dict, y=y, f_x=f_emb, weights=alphas) # [bs, n_y, out_channels]

        return f_y