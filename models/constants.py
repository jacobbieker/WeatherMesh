"""Shared constants for the WeatherMesh model architectures.

Central location for variable catalogs and pressure levels used by the
WeatherMesh-4 / 5c / 6 reconstructions.  All are architecture-only: no weights
are distributed.
"""

from __future__ import annotations

# ECMWF GRIB parameter IDs (https://codes.ecmwf.int/grib/param-db/)
CORE_PRESSURE_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]  # z, t, u, v, q
CORE_SFC_VARS = ["165_10u", "166_10v", "167_2t", "151_msl"]  # 10u, 10v, 2t, msl

# Medium pressure level set (28 levels), consistent with WM-3/WM-4 training.
LEVELS_MEDIUM = [
    10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550,
    600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000,
]

# WeatherMesh-6 additions
NEW_UPPER_AIR_VARS = ["134_w", "135_ciwc", "136_clwc", "137_iwc", "138_lnsp"]
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

# 25 pressure levels for WM-6 upper-air output.
LEVELS_25 = [
    10, 20, 30, 50, 70, 100, 150, 200, 250, 300,
    400, 500, 600, 700, 850, 925, 1000,
    125, 175, 225, 250, 350, 450, 550, 650,
]
assert len(LEVELS_25) == 25, len(LEVELS_25)

# Per-model variable catalogs
WM4_PRESSURE_VARS = list(CORE_PRESSURE_VARS)
WM4_SFC_VARS = list(CORE_SFC_VARS)
WM4_LEVELS = list(LEVELS_MEDIUM)

WM5C_PRESSURE_VARS = list(CORE_PRESSURE_VARS)
WM5C_SFC_VARS = list(CORE_SFC_VARS) + ["10uv_100m"]  # + 100m wind
WM5C_LEVELS = list(LEVELS_MEDIUM)

WM6_PRESSURE_VARS = list(CORE_PRESSURE_VARS) + list(NEW_UPPER_AIR_VARS)
WM6_SFC_VARS = list(CORE_SFC_VARS) + list(NEW_SFC_VARS)
WM6_LEVELS = list(LEVELS_25)
