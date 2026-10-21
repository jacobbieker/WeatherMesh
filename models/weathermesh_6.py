"""WeatherMesh-6 (WM-6).

Based on https://windbornesystems.com/blog/introducing-wm-6

Key documented innovations, reconstructed here:
  * Most skillful medium-range model (NWP or AI) at 0.25 deg (~25 km);
    up to 38% lower ensemble-mean RMSE than IFS, 32% lower than AIFS;
    128 ensemble members (vs 51 for IFS/AIFS).
  * Largest output catalog of any AI weather model: 24 new surface/soil/
    atmospheric parameters plus 5 upper-air parameters at 25 pressure levels
    (soil temperature/moisture, skin temperature, radiation variables,
    layered cloud cover, boundary layer height, ...).
  * Ensemble modelled in latent space (technical report forthcoming);
    competitive CRPS; each member full-resolution and physically coherent;
    percentiles and precipitation exceedance probabilities distributed.
  * AI DA redesigned around ensemble methods: multiple ensemble members at
    analysis time; two new satellite instruments (AMSR2, CrIS; 11 obs types
    total); innovation-based encoding of Global Sounding Balloon observations.
  * Fresh forecast every hour via "flavour" cycles (Flavour 0 at synoptic time
    ready in 30-40 min; Flavour 5 just before the next cycle).
  * WM-6 3km: 3 km regional model assimilating GOES imagery + MRMS radar,
    terrain-aware upscaling, 72 h horizon, 15 min refresh over CONUS/Europe.

Architecture-only: no weights.
"""

from __future__ import annotations

import torch

from models._blocks import ForecastConfig, ForecastModel, ResConvDecoder, ResConvEncoder
from models._meshes import HEALPixMesh

# Core pressure variables (ECMWF GRIB IDs)
CORE_PRESSURE_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]
# WM-6 adds 5 new upper-air parameters at 25 pressure levels.
NEW_UPPER_AIR_VARS = ["134_w", "135_ciwc", "136_clwc", "137_iwc", "138_lnsp"]
PRESSURE_VARS = CORE_PRESSURE_VARS + NEW_UPPER_AIR_VARS  # 10 vars

CORE_SFC_VARS = ["165_10u", "166_10v", "167_2t", "151_msl"]
# 24 new surface/soil/atmospheric parameters (WM-6 blog).
NEW_SFC_VARS = [
    "234_skt",        # skin temperature
    "235_stl1",       # soil temperature layer 1
    "236_stl2",       # soil temperature layer 2
    "237_swvl1",      # soil moisture layer 1
    "238_swvl2",      # soil moisture layer 2
    "134_sp",         # surface pressure
    "137_tcwv",       # total column water vapor
    "228_cape",       # CAPE
    "229_blh",        # boundary layer height
    "230_cbh",        # cloud base height
    "231_tcc_high",   # high cloud cover
    "232_tcc_medium", # medium cloud cover
    "233_tcc_low",    # low cloud cover
    "144_snowfall",   # snowfall
    "205_runoff",     # runoff
    "169_ssrd",       # surface solar radiation downwards
    "175_strd",       # surface thermal radiation downwards
    "176_tosr",       # top-of-atmosphere net solar radiation
    "177_ssr",        # surface net solar radiation
    "178_dsr",        # direct solar radiation
    "180_shf",        # sensible heat flux
    "181_lhf",        # latent heat flux
    "239_fsr",        # forecast surface roughness
    "142_mtpr",       # mean total precipitation rate
]
assert len(NEW_SFC_VARS) == 24, len(NEW_SFC_VARS)
SFC_VARS = CORE_SFC_VARS + NEW_SFC_VARS  # 28 vars

# 25 pressure levels for WM-6 upper-air output.
LEVELS_25 = [
    10, 20, 30, 50, 70, 100, 150, 200, 250, 300,
    400, 500, 600, 700, 850, 925, 1000,
    125, 175, 225, 250, 350, 450, 550, 650,
][:25]
assert len(LEVELS_25) == 25, len(LEVELS_25)


def _build_wm6(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
    n_levels: int | None = None,
):
    state_mesh = HEALPixMesh(depth=depth_state)
    latent_mesh = HEALPixMesh(depth=depth_latent)

    n_pr_vars = len(PRESSURE_VARS)
    n_sfc_vars = len(SFC_VARS)
    levels = LEVELS_25 if n_levels is None else LEVELS_25[:n_levels]
    n_levels = len(levels)

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
    model.levels = levels
    model.pressure_vars = PRESSURE_VARS
    model.sfc_vars = SFC_VARS
    model.n_pr_vars = n_pr_vars
    model.n_sfc_vars = n_sfc_vars
    return model


def get_WeatherMesh6(
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
):
    """Build the WeatherMesh-6 Global architecture.

    Returns a ForecastModel with WM-6's expanded variable catalog attached.
    Use :class:`WM6Ensemble` for the 128-member latent-space ensemble.
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

    Same model family as WM-6 Global but configured for 3 km resolution,
    assimilating GOES geostationary imagery and MRMS multi-radar composites
    with a terrain-aware upscaling path.  Over CONUS/Europe it extends the
    horizon to 72 hours and refreshes every 15 minutes.
    """
    return _build_wm6(
        depth_state=depth_state,
        depth_latent=depth_latent,
        latent_size=latent_size,
        encdec_tr_depth=encdec_tr_depth,
        num_heads=num_heads,
        window_size=window_size,
    )


class WM6Ensemble:
    """128-member ensemble in latent space (WM-6's core innovation).

    Runs the forecast model as an ensemble and computes calibrated
    statistics (percentiles, precipitation exceedance probabilities).
    """

    def __init__(self, model, n_members: int = 128):
        self.model = model
        self.n_members = n_members

    def forecast_ensemble(self, x):
        """Run the model as an ``n_members`` ensemble.

        Returns stacked outputs of shape (n_members, B, H, W, n_vars).
        """
        outs = [self.model(x) for _ in range(self.n_members)]
        return torch.stack(outs, dim=0)

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
