from functools import partial

import einops
import torch
from kappamodules.layers import LinearProjection
from embeddings import ContinuousSincosEmbed
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from torch import nn
import math
class DecoderPerceiver(nn.Module):
    """
    Decodes the function back to physical space at source positions x.
    """
    def __init__(self, input_dim, output_dim, ndim, dim, depth, num_heads, unbatch_mode='dense_to_sparse_unpadded',
                 perc_dim=None, perc_num_heads=None, cond_dim=None, cond_flow = True, init_weights='truncnormal002',
                 init_gate_zero=False, in_out_dim_same=True, val_dim=1, use_pos_embed=False, **kwargs):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.unbatch_mode = unbatch_mode
        self.cond_dim = cond_dim
        self.cond_flow = cond_flow
        self.init_weights = init_weights
        self.use_pos_embed = use_pos_embed
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        if cond_dim is None:
            block_ctor = partial(PerceiverBlock, kv_dim=dim)
        else:
            block_ctor = partial(DitPerceiverBlock, kv_dim=dim, cond_dim=cond_dim, init_weights=init_weights,
                                 init_gate_zero=init_gate_zero)
        # --- CONDITIONING MODULES (Matching Encoder) ---
            if self.cond_flow:
                # 1D-CNN: (B, Channels, L) -> (B, cond_dim)
                self.ecg_extractor = nn.Sequential(
                    nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, cond_dim, kernel_size=3, stride=2, padding=1),
                    nn.AdaptiveAvgPool1d(1), 
                    nn.Flatten()
                )
                # Fusion: ECG (cond_dim) + time (cond_dim) -> final cond_dim
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(2 * cond_dim, cond_dim),
                    nn.SiLU(),
                    nn.Linear(cond_dim, cond_dim)
                )
        self.blocks = nn.ModuleList([block_ctor(dim=dim, num_heads=num_heads, init_weights=init_weights) for _ in range(depth)])

        if self.use_pos_embed:
            self.pos_embed = ContinuousSincosEmbed(dim=self.perc_dim, ndim=ndim) # positional embedding of source coordinates
            pos_input_dim=self.perc_dim
        else:
            self.pos_embed = nn.Identity()
            pos_input_dim = ndim # raw coordinates dimension
        
        if in_out_dim_same:
            val_dim = output_dim
        
        self.val_embed = nn.Sequential(LinearProjection(val_dim, self.perc_dim, init_weights=init_weights), nn.GELU(), 
                                       LinearProjection(self.perc_dim, self.perc_dim, init_weoghts=init_weights))
        
        self.query_proj = nn.Sequential(LinearProjection(pos_input_dim + self.perc_dim, perc_dim*2, init_weights=init_weights), nn.GELU(),
                                        LinearProjection(perc_dim*2, perc_dim, init_weights=init_weights))
        
        # projection
        self.pred = nn.Sequential(nn.LayerNorm(perc_dim, eps=1e-6), LinearProjection(perc_dim, output_dim, init_weights=init_weights))
    

    def forward(self, x, output_val, output_pos, time_condition=None, condition=None):
        # output_val = f_t(x), x = output_pos
        if time_condition is not None:
            assert time_condition.ndim == 2, 'expected shape (batch_size, cond_dim)'
        if condition is not None:
            # Process ECG and fuse with time
            cond_emb = self.ecg_extractor(condition)
            combined_cond = torch.cat([cond_emb, time_condition], dim=-1)
            final_cond = self.fusion_mlp(combined_cond)
            cond_kwargs["cond"] = final_cond
        else:
            # Use time only
            cond_kwargs["cond"] = time_condition
        cond_kwargs = {}
        # complete to match the same conditioning interface as the encoder (time_condition and condition)
        query_pos = self.pos_embed(output_pos)
        query_val = self.val_embed(output_val) # MLP(f_t)
        query = self.query_proj(torch.cat([query_pos, query_val], dim=-1))

        # apply the cross attention DiT
        # cros attention + self-attention
        ## placeholder, placeholder, placeholder
        block_id = 0
        for block in self.blocks:
            if block_id == 0:
                x = block(q=query, kv=x, **cond_kwargs)
            else:
                x = block(q=x, kv=x, **cond_kwargs) #self attention
            block_id = block_id+1

        x = self.pred(x)

        if self.unbatch_mode == 'dense_to_sparse_unpadded':
            x = einops.rearrange(
                x,
                'batch_size seqlen dim -> batch_size dim seqlen',
            )
        elif self.unbatch_mode == 'image':
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                'batch_size (height width) dim -> batch_size dim height width',
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x  
