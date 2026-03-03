import torch
import torch.nn as nn

class LiftedPointEmbedding(nn.Module):
    """
    A learnable point-based positional embedding for 3D coordinates.
    Replaces ContinuousSincosEmbed by using a point-wise MLP to map
    raw 2D/3D coordinates to a high-dimensional feature space.
    """
    def __init__(self, out_dim, in_dim, hidden_dim=None):
        super().__init__()
        self.out_dim = out_dim            # target output dimension (e.g., 256)
        self.in_dim = in_dim      # input coordinate dimension (e.g., 3 for x, y, z)
        
        # wider hidden layer
        if hidden_dim is None:
            hidden_dim = self.out_dim * 2
            
        # point-wise MLP: [in_dim, n_pts] -> [out_dim, n_pts]
        self.mlp = nn.Sequential(nn.Linear(self.in_dim, hidden_dim), nn.GELU(),
                                 nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.out_dim))

    def forward(self, coords):
        # coords: [batch_size, n_pts, ndim] or [n_pts, ndim] if unbatched
        assert coords.shape[-1] == self.in_dim, f"Expected coord dim {self.in_dim}, got {coords.shape[-1]}"
        
        p_emb = self.mlp(coords)
        return p_emb # [batch_size, num_points, out_dim]

  
def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000, scale : int = 128):
    # The grid is assumed to be [-1 ,1] or [0, 1] (small)
    if dim % 2 == 0:
        padding = None
    else:
        padding = torch.zeros(*grid.shape, 1)
        dim -= 1
    # generate frequencies for sin/cos (e.g. dim=8 -> omega = [1.0, 0.1, 0.01, 0.001])
    omega = 1.0 / max_wavelength ** (torch.arange(0, dim, 2, dtype=torch.double) / dim).to(
        grid.device
    )
    # create grid of frequencies with timesteps
    # Example seqlen=5 dim=8
    # [0, 0.0, 0.00, 0.000]
    # [1, 0.1, 0.01, 0.001]
    # [2, 0.2, 0.02, 0.002]
    # [3, 0.3, 0.03, 0.003]
    # [4, 0.4, 0.04, 0.004]
    # Note: supports cases where grid is more than 1d
    out = scale *grid.unsqueeze(-1) @ omega.unsqueeze(0)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concat([emb_sin, emb_cos], dim=-1).float()
    if padding is None:
        return emb
    else:
        return torch.concat([emb, padding], dim=-1)


class ConditionerTimestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        cond_dim = dim * 4
        self.dim = dim
        self.cond_dim = cond_dim
        """
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=num_timesteps, dim=dim),
        )
        """
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, timestep):
        # checks + preprocess
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten().double()
        # embed
        # embed = self.mlp(self.timestep_embed[timestep])
        embed = self.mlp(get_sincos_1d_from_grid(timestep, dim=self.dim))
        return embed