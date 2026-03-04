from functools import partial
import torch
from torch import nn
from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from gno_block import AGNOBlock

class SupernodePooling(nn.Module):
    def __init__(self, f_channels, hidden_channels, coord_dim, radius):
        super().__init__()
        self.message = AGNOBlock(f_channels=f_channels, out_channels=hidden_channels,
                                coord_dim=coord_dim, radius=radius, transform_type='nonlinear', pos_embed=False, pos_embedding_channels=hidden_channels,
                                channel_mlp_layers=[hidden_channels*2, hidden_channels])
    def forward(self, source, query, fun):
        out = self.message(x=source[0], y=query[0], f_x=fun) # [0] makes me think of batched data
        return out
        
class Encoder(nn.Module):
    def __init__(self, input_dim, ndim, radius, enc_dim, enc_depth,
                 enc_num_heads, cond_dim=None, init_weights='truncnormal', init_gate_zero=True):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.radius = radius
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.condition_dim = cond_dim
        self.init_weights = init_weights
        self.init_gate_zero = init_gate_zero

        self.nodepooling = SupernodePooling(f_channels=input_dim, hidden_channels=enc_dim, coord_dim=ndim, radius=radius)
        self.enc_proj = LinearProjection(enc_dim, enc_dim, init_weights=init_weights, optional=True)

        if cond_dim is None: #?
            block_ctor = partial(PerceiverBlock, kv_dim=enc_dim, init_weights=init_weights)
        else:
            block_ctor = partial(DitPerceiverBlock, kv_dim=enc_dim, cond_dim=cond_dim, init_weights=init_weights,
                                 init_gate_zero=init_gate_zero)
            
        self.first_block = block_ctor(dim=enc_dim, num_heads=enc_num_heads)
        self.rest_block = nn.ModuleList(block_ctor(dim=enc_dim, num_heads=enc_num_heads) for _ in range(enc_depth-1))

    def forward(self, fun, source, query, condition=None):
        assert len(fun) == len(source), "expected input_feat and input_pos to have same length"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # supernode pooling
        x = self.nodepooling(fun=fun, source=source, query=query)
   
        # project to encoder dimension
        x = self.enc_proj(x)                            #"batch_size seqlen dim 

        h = self.first_block(kv=x, q=x, **cond_kwargs)  # [B, L, enc_dim]

        for blk in self.rest_block:                      # N-1 times
            h = blk(q=h, kv=x, **cond_kwargs)
            
        return h       
        