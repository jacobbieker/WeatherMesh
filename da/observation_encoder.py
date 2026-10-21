"""AI DA encoding phase.

Observations of a particular type (shape (N, Ch)) are encoded into latent
tokens and injected into a HEALPix background latent of shape
(B, Nvert, D, C) via cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._meshes import HEALPixMesh


def assign_to_vertices(locations_latlon: torch.Tensor, mesh: HEALPixMesh) -> torch.Tensor:
    """Assign each observation to its nearest HEALPix vertex.

    Args:
        locations_latlon: (B, N, 2) lat/lon in degrees.
        mesh: HEALPix mesh with (lats, lons) of shape (Nvert,).

    Returns:
        vertex indices of shape (B, N), values in [0, Nvert).
    """
    lat = torch.deg2rad(locations_latlon[..., 0])  # (B, N)
    lon = torch.deg2rad(locations_latlon[..., 1])
    mlat = torch.deg2rad(torch.as_tensor(mesh.lats, dtype=locations_latlon.dtype, device=locations_latlon.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, Nvert)
    mlon = torch.deg2rad(torch.as_tensor(mesh.lons, dtype=locations_latlon.dtype, device=locations_latlon.device)).unsqueeze(0).unsqueeze(0)
    dlon = lon.unsqueeze(-1) - mlon  # (B, N, Nvert)
    dist = (
        torch.sin((lat.unsqueeze(-1) - mlat) / 2) ** 2
        + torch.cos(lat.unsqueeze(-1)) * torch.cos(mlat) * torch.sin(dlon / 2) ** 2
    )
    return dist.argmin(dim=-1)  # (B, N)


class ObservationEncoder(nn.Module):
    """Encode one observation type into the HEALPix latent space via cross-attention."""

    def __init__(self, n_obs_channels: int, latent_size: int, num_heads: int = 8):
        super().__init__()
        self.obs_proj = nn.Linear(n_obs_channels, latent_size)
        self.cross_attn = nn.MultiheadAttention(latent_size, num_heads, batch_first=True)
        self.out_proj = nn.Linear(latent_size, latent_size)

    def forward(
        self,
        obs: torch.Tensor,
        obs_locations: torch.Tensor,
        background_latent: torch.Tensor,
        mesh: HEALPixMesh,
    ) -> torch.Tensor:
        # obs: (B, N, Ch); obs_locations: (B, N, 2)
        # background_latent: (B, Nvert, D, C)
        B, Nvert, D, C = background_latent.shape
        N = obs.shape[1]
        obs_tokens = self.obs_proj(obs)  # (B, N, C)

        # Assign observations to vertices; weight each vertex's update by its obs
        vertex_idx = assign_to_vertices(obs_locations, mesh)  # (B, N)
        # Scatter observation tokens onto their vertices (sum, then normalise by count)
        vertex_tokens = torch.zeros(B, Nvert, C, device=obs.device, dtype=obs.dtype)
        count = torch.zeros(B, Nvert, 1, device=obs.device, dtype=obs.dtype)
        vertex_tokens.scatter_add_(1, vertex_idx.unsqueeze(-1).expand(-1, -1, C), obs_tokens)
        count.scatter_add_(1, vertex_idx.unsqueeze(-1), torch.ones(B, N, 1, device=obs.device, dtype=obs.dtype))
        vertex_tokens = vertex_tokens / count.clamp(min=1.0)  # (B, Nvert, C)

        # Cross-attention: each latent vertex (across depth) queries the observation tokens
        q = background_latent.reshape(B, Nvert * D, C)  # (B, Nvert*D, C)
        attn_out, _ = self.cross_attn(q, obs_tokens, obs_tokens)  # (B, Nvert*D, C)
        attn_out = self.out_proj(attn_out)
        # Blend in the vertex-scattered observation signal
        vertex_signal = vertex_tokens.unsqueeze(2).expand(-1, -1, D, -1).reshape(B, Nvert * D, C)
        attn_out = attn_out + vertex_signal
        return attn_out.reshape(B, Nvert, D, C)
