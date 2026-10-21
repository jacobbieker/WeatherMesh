"""WeatherMesh-4 (WM-4).

Based on https://windbornesystems.com/blog/weathermesh-4-technical-blog

Key documented innovations, reconstructed here:
  * HEALPix mesh replaces the latitude-longitude grid.
    - Depth 8  -> nside=256,  Npix=786,432  (~0.23 deg)  weather state grid
    - Depth 5  -> nside=32,   Npix=12,288   (~1.8 deg)   weather latent space
  * Hierarchical / equal-area / isoLatitude; each pixel divides into 4 at the
    next depth, so the encoder naturally maps the depth-8 state grid onto the
    depth-5 latent grid (8x spatial downsampling).
  * Deterministic forecasts are interpreted as ensemble means (blur is a
    feature, not a bug); evaluated against ERA-5.

Architecture-only: no weights.  Returns a :class:`models._blocks.ForecastModel`
with the HEALPix mesh and variable metadata attached.
"""

from __future__ import annotations

from models._blocks import ForecastConfig, ForecastModel, ResConvDecoder, ResConvEncoder
from models._meshes import HEALPixMesh

# ECMWF GRIB parameter IDs (https://codes.ecmwf.int/grib/param-db/)
PRESSURE_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]  # z, t, u, v, q
SFC_VARS = ["165_10u", "166_10v", "167_2t", "151_msl"]  # 10u, 10v, 2t, msl

# Medium pressure level set (28 levels), consistent with WM-3/WM-4 training.
LEVELS = [
    10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550,
    600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000,
]


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

    Returns a ForecastModel with ``.state_mesh``, ``.latent_mesh``,
    ``.levels``, ``.pressure_vars``, ``.sfc_vars`` attached.
    """
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
