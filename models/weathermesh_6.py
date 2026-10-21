"""WeatherMesh-6 (WM-6) architecture.

Based on https://windbornesystems.com/blog/introducing-wm-6

Architecture-only: no weights are distributed.  This is a HEALPix forecast
model with WM-6's expanded variable catalog; ``WM6Ensemble`` runs it as a
latent-space ensemble and computes calibrated statistics.
"""

from __future__ import annotations

import torch

from models._blocks import BaseEnsemble, build_forecast_model
from models.constants import (
    WM6_LEVELS,
    WM6_PRESSURE_VARS,
    WM6_SFC_VARS,
)


def _build_wm6(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
    n_levels: int | None = None,
):
    levels = WM6_LEVELS if n_levels is None else WM6_LEVELS[:n_levels]
    return build_forecast_model(
        WM6_PRESSURE_VARS,
        WM6_SFC_VARS,
        levels,
        depth_state=depth_state,
        depth_latent=depth_latent,
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
    )


def get_WeatherMesh6(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
):
    """Build the WeatherMesh-6 Global architecture.

    Use :class:`WM6Ensemble` for the latent-space ensemble.
    """
    return _build_wm6(
        depth_state=depth_state,
        depth_latent=depth_latent,
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
    )


def get_WeatherMesh6_3km(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
):
    """Build the WeatherMesh-6 3 km regional architecture.

    Same model family as WM-6 Global, configured for 3 km resolution.
    """
    return _build_wm6(
        depth_state=depth_state,
        depth_latent=depth_latent,
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
    )


class WM6Ensemble(BaseEnsemble):
    """Latent-space ensemble wrapper for WM-6 with calibrated statistics."""

    def __init__(self, model, n_members: int = 128):
        super().__init__(model, n_members)

    def ensemble_statistics(self, ensemble_output, precip_var_index=None):
        """Compute calibrated ensemble statistics.

        Args:
            ensemble_output: (n_members, B, H, W, n_vars).
            precip_var_index: index of the precipitation variable (optional).

        Returns:
            dict with mean, std, percentiles (10/50/90), and, if
            ``precip_var_index`` is given, exceedance probability of
            precipitation > 0.25 mm/3h.
        """
        mean = ensemble_output.mean(dim=0)
        std = ensemble_output.std(dim=0)
        pct = torch.quantile(
            ensemble_output,
            torch.tensor([0.1, 0.5, 0.9], device=ensemble_output.device),
            dim=0,
        )  # (3, B, H, W, n_vars)
        stats = {"mean": mean, "std": std, "percentiles": pct}
        if precip_var_index is not None:
            precip = ensemble_output[..., precip_var_index]
            stats["precip_exceed_0.25mm"] = (precip > 0.25).float().mean(dim=0)
        return stats
