"""AI DA assimilation phase.

Three transformer stages combine observation-type increments with the
background latent, all of shape (B, Nvert, D, C):

  1. each observation type talks to the background directly (one-way);
  2. the background is updated using all observations (where DA happens),
     forced to occur at a specific location;
  3. observations at different locations communicate with one another.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AssimilationPhase(nn.Module):
    def __init__(self, latent_size: int, num_heads: int = 8, depth: int = 2):
        super().__init__()
        self.latent_size = latent_size
        # Stage 1: one-way observation -> background (per encoder layer)
        self.stage1 = nn.ModuleList(
            [nn.TransformerEncoderLayer(latent_size, num_heads, batch_first=True) for _ in range(depth)]
        )
        # Stage 2: background updated using all observations (the DA step)
        self.stage2 = nn.TransformerEncoderLayer(latent_size, num_heads, batch_first=True)
        # Stage 3: inter-observation communication across locations
        self.stage3 = nn.TransformerEncoderLayer(latent_size, num_heads, batch_first=True)
        self.output_proj = nn.Linear(latent_size, latent_size)

    def forward(self, background: torch.Tensor, increments: list[torch.Tensor]) -> torch.Tensor:
        # background, increments: (B, Nvert, D, C)
        B, Nvert, D, C = background.shape
        x = background.reshape(B, Nvert * D, C)

        # Stage 1: inject each observation increment, let background self-organise
        for inc in increments:
            x = x + inc.reshape(B, Nvert * D, C)
            x = self.stage1[0](x)

        # Stage 2: the DA step -- background updated using all observations
        x = self.stage2(x)

        # Stage 3: observations at different locations communicate
        x = self.stage3(x)

        x = self.output_proj(x)
        return x.reshape(B, Nvert, D, C)
