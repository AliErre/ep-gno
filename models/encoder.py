from functools import partial
import torch
from torch import nn
from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from gno_block import AGNOBlock

class SupernodePooling(nn.Module):
    def __init__(self, f_channels, hidden_channels, coord_dim, radius, transform_type):
        super().__init__()
        self.message = AGNOBlock(f_channels=f_channels, out_channels=hidden_channels,
                                coord_dim=coord_dim, radius=radius, transform_type=transform_type, pos_embed=False, pos_embedding_channels=hidden_channels,
                                channel_mlp_layers=[hidden_channels*2, hidden_channels])
    def forward(self, source, query, fun):
        out = self.message(x=source[0], y=query[0], f_x=fun) # [0] makes me think of batched data
        return out
        
class Encoder(nn.Module):
    def __init__(self, input_dim, ndim, radius, transform_type, enc_dim, enc_depth,
                 enc_num_heads, cond_dim=None, cond_flow=True, init_weights='truncnormal', init_gate_zero=True):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.radius = radius
        self.transform_type = transform_type
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.condition_dim = cond_dim
        self.cond_flow = cond_flow
        self.init_weights = init_weights
        self.init_gate_zero = init_gate_zero

        if cond_flow:
            # 1D-CNN: (B, Channels, L) -> (B, cond_dim)
            self.ecg_extractor = nn.Sequential(
                nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, cond_dim, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool1d(1), 
                nn.Flatten())
            # concatenate ECG embedding (cond_dim) + time embedding (cond_dim) -> cond_dim
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2*cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim))

        # 1. GNO
        self.nodepooling = SupernodePooling(f_channels=input_dim, hidden_channels=enc_dim, coord_dim=ndim, radius=radius, transform_type=transform_type)
        # 2. linear projection of GNO output
        self.enc_proj = LinearProjection(enc_dim, enc_dim, init_weights=init_weights, optional=True) # h_0
        # 3. Multi head cross attention (perceiver style; K = h_0, V = h_0, Q = h_{j-1} for j=1,...,enc_depth)
        if cond_dim is None: #?
            block_ctor = partial(PerceiverBlock, kv_dim=enc_dim, init_weights=init_weights)
        else: # uses time conditioning for adaptive layer normalization
            block_ctor = partial(DitPerceiverBlock, kv_dim=enc_dim, cond_dim=cond_dim, init_weights=init_weights,
                                 init_gate_zero=init_gate_zero)
            
        self.first_block = block_ctor(dim=enc_dim, num_heads=enc_num_heads)
        self.rest_block = nn.ModuleList(block_ctor(dim=enc_dim, num_heads=enc_num_heads) for _ in range(enc_depth-1))

    def forward(self, fun, source, query, time_condition=None, condition=None):
        assert len(fun) == len(source), "expected function and sources to have same length"
        if time_condition is not None:
            assert time_condition.ndim == 2, "expected time embedding of shape (batch_size, cond_dim)"
        else:
            raise ValueError("time_condition is required in flow matching.")
        
        # unique embedding for time and condition
        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_emb = self.ecg_extractor(condition)
            # concatenate
            combined_cond = torch.cat([cond_emb, time_condition], dim=-1) # time_condition is already the sinusoidal embedding
            # final conditioning
            condition = self.fusion_mlp(combined_cond)
            cond_kwargs["cond"] = condition
        else:
            cond_kwargs["cond"] = time_condition
            
        # supernode pooling
        x = self.nodepooling(fun=fun, source=source, query=query)
   
        # project to encoder dimension
        x = self.enc_proj(x)                            #"batch_size seqlen dim 

        h = self.first_block(kv=x, q=x, **cond_kwargs)  # [B, L, enc_dim]

        for blk in self.rest_block:                      # N-1 times
            h = blk(q=h, kv=x, **cond_kwargs)
            
        return h       
        