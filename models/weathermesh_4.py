"""WeatherMesh-4 (WM-4) architecture.

Based on https://windbornesystems.com/blog/weathermesh-4-technical-blog

Architecture-only: no weights are distributed.  This is a faithful
reconstruction of the publicly described design (HEALPix mesh replacing the
latlon grid; depth-8 state, depth-5 latent), not the exact internal
implementation.
"""

from __future__ import annotations

from models._blocks import build_forecast_model
from models.constants import WM4_LEVELS, WM4_PRESSURE_VARS, WM4_SFC_VARS


def get_WeatherMesh4(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
):
    """Build the WeatherMesh-4 architecture.

    The state (input/output) lives on a HEALPix mesh of ``depth_state``;
    the latent space lives on ``depth_latent``.  The encoder's 8x spatial
    downsampling maps depth-8 -> depth-5 when ``depth_state=8,
    depth_latent=5``.

    Returns a ForecastModel with mesh and variable metadata attached.
    """
    return build_forecast_model(
        WM4_PRESSURE_VARS,
        WM4_SFC_VARS,
        WM4_LEVELS,
        depth_state=depth_state,
        depth_latent=depth_latent,
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
    )
