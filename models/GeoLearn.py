import torch
import torch.nn as nn
from encoder import Encoder
from decoder import DecoderPerceiver
import einops

class GeoLearn(nn.Module):
    def __init__(self, encoder:Encoder, decoder:DecoderPerceiver, conditioner=None):
        """
        encoder: AGNO + multi-head cross attention blocks (Perceiver style)
        decoder: Perceiver decoder
        conditioner: module for embedding of timestep
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.conditioner = conditioner

    def forward(self, f_x, source_pos, query_pos, time_condition=None, condition=None):
        """
        f_x: field values at source positions [f_dim, N_x], f_dim = codomain cardinality
        source_pos: x coordinates in physical space [dim, N_x]
        query_pos: y coordinates in uniform grid [dim, N_y]
        """
        x_dim = source_pos.shape[1]
        n_chan = f_x.shape[1]
        batch_size = len(f_x)

        if self.conditioner is not None:
            if time_condition.dim() == 0 or time_condition.numel() == 1:
                time_condition = torch.ones(batch_size, device = time_condition.device) * time_condition
            time_emb = self.conditioner(time_condition) # [batch_size, cond_dim]
        else:
            raise ValueError("Conditioner is required for time conditioning in GeoLearn.")

        output_pos = source_pos.permute(0, 2, 1)
        output_val = f_x.permute(0, 2, 1)
        input_pos = einops.rearrange(source_pos, "batch_size dim seq_len -> batch_size seq_len dim ",
                                     batch_size = batch_size,
                                     dim = x_dim)
        input_val = einops.rearrange(f_x, "batch_size dim seq_len -> batch_size seq_len dim",
                                      batch_size = batch_size,
                                      dim = n_chan) 
        query_pos = einops.rearrange(query_pos, "batch_size dim seq_len -> batch_size seq_len dim ",
                                     batch_size = batch_size,
                                     dim = x_dim)       
        # encode
        # input_val = x_t
        enc_out = self.encoder(fun = input_val, source = input_pos, query = query_pos, time_condition = time_emb, condition = condition)
        # decode
        dec_out = self.decoder(x = enc_out, output_val = output_val, output_pos = output_pos, time_condition = time_condition, condition = condition)
        return dec_out