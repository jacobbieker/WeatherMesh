"""WeatherMesh-5c (WM-5c) architecture.

Based on https://windbornesystems.com/blog/weathermesh-5c-release-notes

Architecture-only: no weights are distributed.  This is a lightweight HEALPix
forecast model (~1/3 the size of WM-4); ``WM5cContinuousForecaster`` wraps it
with the 20-minute cycle / "flavour" logic described in the blog.
"""

from __future__ import annotations

from models._blocks import BaseEnsemble, build_forecast_model
from models.constants import WM5C_LEVELS, WM5C_PRESSURE_VARS, WM5C_SFC_VARS


def get_WeatherMesh5c(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 384,  # ~1/3 of WM-4's 1024
    encdec_tr_depth: int = 1,
    num_heads: int = 12,
    window_size: int = 7,
):
    """Build the lightweight WeatherMesh-5c architecture (~1/3 the size of WM-4)."""
    return build_forecast_model(
        WM5C_PRESSURE_VARS,
        WM5C_SFC_VARS,
        WM5C_LEVELS,
        depth_state=depth_state,
        depth_latent=depth_latent,
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
    )


class WM5cContinuousForecaster(BaseEnsemble):
    """Wrapper modelling WM-5c's continuous 20-minute forecasting cycle.

    The blog describes "flavours": each 6-hourly synoptic cycle produces a
    fresh run every hour, with skill improving as more recent observations are
    assimilated.  WM-5c runs every 20 minutes with a 24-member ensemble.
    """

    def __init__(self, model, n_members: int = 24, cycle_minutes: int = 20):
        super().__init__(model, n_members)
        self.cycle_minutes = cycle_minutes
        # Updates per 6-hourly ECMWF cycle: 6h / 20min = 18
        self.updates_per_cycle = 6 * 60 // cycle_minutes

    def flavour_schedule(self, synoptic_hour: int):
        """Return the flavour run times (hours after synoptic time) for a cycle.

        Flavour 0 at synoptic time; Flavour 5 is the last run before the next
        ECMWF cycle becomes available.
        """
        return list(range(0, 6, 1))[: self.updates_per_cycle]
