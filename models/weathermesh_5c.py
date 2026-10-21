"""WeatherMesh-5c (WM-5c) -- Continuous Forecasting.

Based on https://windbornesystems.com/blog/weathermesh-5c-release-notes

Key documented innovations, reconstructed here:
  * Continuous Forecasting: first continuously updated global weather model,
    running every 20 minutes (18 updates between ECMWF 6-hourly cycles).
  * Independent assimilation: powered by WindBorne's proprietary AI DA rather
    than ECMWF/NOAA analyses published every 6 hours.
  * Lightweight: ~1/3 the size of the previous generation, same forecasting
    skill; a 24-member ensemble runs in parallel on 8 GPUs in ~16 minutes.
  * Improved precipitation (calibration + extremes) and 10 m / 100 m wind
    speed parameters.

Architecture-only: no weights.  ``get_WeatherMesh5c`` builds a smaller
HEALPix forecast model; ``WM5cContinuousForecaster`` wraps it with the
20-minute cycle / "flavour" logic described in the blog.
"""

from __future__ import annotations

import torch

from models._blocks import ForecastConfig, ForecastModel, ResConvDecoder, ResConvEncoder
from models._meshes import HEALPixMesh

PRESSURE_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]
SFC_VARS = ["165_10u", "166_10v", "167_2t", "151_msl", "10uv_100m"]  # + 100m wind
LEVELS = [
    10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550,
    600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000,
]


def get_WeatherMesh5c(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 384,  # ~1/3 of WM-4's 1024
    encdec_tr_depth: int = 1,
    num_heads: int = 12,
    window_size: int = 7,
):
    """Build the lightweight WeatherMesh-5c architecture (~1/3 the size of WM-4)."""
    state_mesh = HEALPixMesh(depth=depth_state)
    latent_mesh = HEALPixMesh(depth=depth_latent)

    n_pr_vars = len(PRESSURE_VARS)
    n_sfc_vars = len(SFC_VARS)
    n_levels = len(LEVELS)
    H, W = state_mesh.ring_shape

    config = ForecastConfig(
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
        affine=True,
    )

    encoder = ResConvEncoder(config, n_pr_vars, n_sfc_vars, n_levels)
    decoder = ResConvDecoder(config, n_pr_vars, n_sfc_vars, n_levels)

    model = ForecastModel(config, encoder, decoder, n_levels, H, W)
    model.state_mesh = state_mesh
    model.latent_mesh = latent_mesh
    model.levels = LEVELS
    model.pressure_vars = PRESSURE_VARS
    model.sfc_vars = SFC_VARS
    model.n_pr_vars = n_pr_vars
    model.n_sfc_vars = n_sfc_vars
    return model


class WM5cContinuousForecaster:
    """Wrapper modelling WM-5c's continuous 20-minute forecasting cycle.

    The blog describes "flavours": each 6-hourly synoptic cycle (0z/6z/12z/18z)
    produces a fresh run every hour (Flavour 0 at synoptic time, ... Flavour 5
    one hour before the next cycle), with skill improving as more recent
    observations are assimilated.  WM-5c runs every 20 minutes with a
    24-member ensemble.
    """

    def __init__(self, model, n_members: int = 24, cycle_minutes: int = 20):
        self.model = model
        self.n_members = n_members
        self.cycle_minutes = cycle_minutes
        # Updates per 6-hourly ECMWF cycle: 6h / 20min = 18
        self.updates_per_cycle = 6 * 60 // cycle_minutes

    def flavour_schedule(self, synoptic_hour: int):
        """Return the flavour run times (hours after synoptic time) for a cycle.

        Flavour 0 at synoptic time, ready within 30-40 min; Flavour 5 is the
        last run before the next ECMWF cycle becomes available.
        """
        return list(range(0, 6, 1))[: self.updates_per_cycle]

    def forecast_ensemble(self, x):
        """Run the model as an ``n_members`` ensemble (perturbed inputs).

        Architecture-only: returns the stacked ensemble output without
        applying real perturbations (no weights to perturb).
        """
        outs = [self.model(x) for _ in range(self.n_members)]
        return torch.stack(outs, dim=0)  # (n_members, B, H, W, n_vars)
