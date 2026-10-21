"""WindBorne AI Data Assimilation (AI DA) - top-level system.

Based on https://windbornesystems.com/blog/windborne-ai-da

Architecture-only: encodes observations into a HEALPix latent space and runs a
three-stage assimilation to produce an analysis increment that initialises a
forecast model (e.g. WeatherMesh-4).  No weights are distributed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from da.observation_encoder import ObservationEncoder, assign_to_vertices
from da.assimilation import AssimilationPhase
from models._meshes import HEALPixMesh


class WindBorneAIDA(nn.Module):
    def __init__(
        self,
        mesh: HEALPixMesh,
        n_levels: int,
        obs_channels_per_type: dict[str, int],
        n_background_vars: int,
        latent_size: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.mesh = mesh
        self.n_levels = n_levels
        self.latent_size = latent_size

        # Background encoder: (B, Nvert, D, n_vars) -> (B, Nvert, D, C)
        self.bg_in = nn.Linear(n_background_vars, latent_size)
        self.bg_out = nn.Linear(latent_size, n_background_vars)

        # One encoder per observation type
        self.encoders = nn.ModuleDict(
            {name: ObservationEncoder(ch, latent_size, num_heads) for name, ch in obs_channels_per_type.items()}
        )

        self.assimilation = AssimilationPhase(latent_size, num_heads=num_heads)

    def encode_background(self, background: torch.Tensor) -> torch.Tensor:
        # background: (B, Nvert, D, n_vars) -> (B, Nvert, D, C)
        B, Nvert, D, nv = background.shape
        x = background.reshape(B * Nvert * D, nv)
        x = self.bg_in(x)
        return x.reshape(B, Nvert, D, self.latent_size)

    def decode_background(self, latent: torch.Tensor, n_background_vars: int) -> torch.Tensor:
        # latent: (B, Nvert, D, C) -> (B, Nvert, D, n_vars)
        B, Nvert, D, _ = latent.shape
        x = latent.reshape(B * Nvert * D, self.latent_size)
        x = self.bg_out(x)
        return x.reshape(B, Nvert, D, n_background_vars)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        obs_locations: dict[str, torch.Tensor],
        background: torch.Tensor,
    ) -> torch.Tensor:
        # observations: {obs_type: (B, N, Ch)}
        # obs_locations: {obs_type: (B, N, 2) lat/lon}
        # background: (B, Nvert, D, n_vars)
        bg_latent = self.encode_background(background)  # (B, Nvert, D, C)

        increments = []
        for name, enc in self.encoders.items():
            obs = observations[name]  # (B, N, Ch)
            locs = obs_locations[name]  # (B, N, 2)
            inc = enc(obs, locs, bg_latent, self.mesh)  # (B, Nvert, D, C)
            increments.append(inc)

        analysis_increment = self.assimilation(bg_latent, increments)  # (B, Nvert, D, C)
        analysis_latent = bg_latent + analysis_increment
        analysis = self.decode_background(analysis_latent, background.shape[-1])
        return analysis
