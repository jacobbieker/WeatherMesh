# WeatherMesh

Architecture definitions for WindBorne Systems' WeatherMesh family, starting
from the open-source WeatherMesh-3 (this repo's original model) and extending
to WeatherMesh-4, WeatherMesh-5c, WeatherMesh-6, and WindBorne AI DA, as
described in WindBorne's technical blogs.

## Models

| Model | Blog | Core idea |
|-------|------|-----------|
| WeatherMesh-3 | [arXiv:2503.22235](https://arxiv.org/abs/2503.22235) | LatLon-grid forecast model (this repo's original `model.py`) |
| WeatherMesh-4 | [weathermesh-4-technical-blog](https://windbornesystems.com/blog/weathermesh-4-technical-blog) | HEALPix mesh replaces latlon grid; depth-8 state (~0.23°), depth-5 latent (~1.8°) |
| AI DA | [windborne-ai-da](https://windbornesystems.com/blog/windborne-ai-da) | Encoding + 3-stage assimilation; obs → HEALPix latent via cross-attention |
| WeatherMesh-5c | [weathermesh-5c-release-notes](https://windbornesystems.com/blog/weathermesh-5c-release-notes) | Continuous forecasting (every 20 min), ~1/3 size, 24-member ensemble, AI DA init |
| WeatherMesh-6 | [introducing-wm-6](https://windbornesystems.com/blog/introducing-wm-6) | 128-member latent ensemble, 24 new sfc + 5 upper-air vars @ 25 levels, ensemble AI DA |

All new models are **architecture-only**: no weights or training data are
distributed, and the architectures are faithful reconstructions of the publicly
described designs (full architectural details and weights are not public).

## Installation with pixi

The recommended way to create the environment is with [pixi](https://pixi.sh):

```bash
pixi install
```

This installs PyTorch 2.13 (CPU and CUDA variants), NumPy, NATTEN 0.21.7
(libnatten via PyPI, CPU and CUDA variants), and `matepoint` as defined in
`pixi.toml`.  To run the WeatherMesh-3 architecture demo:

```bash
pixi run python model.py
```

A legacy `./install.sh` (pip-based) is kept as a fallback.

## Usage

```python
from models import get_WeatherMesh4, get_WeatherMesh5c, get_WeatherMesh6
from da import WindBorneAIDA

# WeatherMesh-4: HEALPix state (depth 8) -> latent (depth 5)
wm4 = get_WeatherMesh4()
print(wm4)

# WeatherMesh-5c: lightweight continuous-forecasting model
wm5c = get_WeatherMesh5c()

# WeatherMesh-6: 128-member latent-space ensemble
wm6 = get_WeatherMesh6()
ensemble = WM6Ensemble(wm6, n_members=128)

# WindBorne AI DA: observation encoding + 3-stage assimilation
aida = WindBorneAIDA(mesh, n_levels, obs_channels_per_type, n_background_vars)
analysis = aida(observations, obs_locations, background)
```

## Repository layout

```
model.py                  # WeatherMesh-3 architecture demo (latlon grid)
model_latlon/             # WeatherMesh-3 latlon encoder/decoder/transformer
meshes.py                 # WeatherMesh-3 LatLonGrid
utils.py                  # levels, variable constants, normalization
constants/                # normalization.json, variable_weights.json, nan_mask, additional_variables
models/                   # NEW: WM-4 / WM-5c / WM-6 + HEALPix mesh + NATTEN blocks
  weathermesh_4.py        # WeatherMesh-4 (HEALPix mesh)
  weathermesh_5c.py       # WeatherMesh-5c (continuous forecasting)
  weathermesh_6.py        # WeatherMesh-6 (128-member latent ensemble)
  _meshes.py              # HEALPix equal-area isoLatitude mesh
  _blocks.py              # ResConv encoder/decoder + NATTEN windowed attention
da/                       # NEW: WindBorne AI DA (encoding + assimilation)
  aida.py
  observation_encoder.py
  assimilation.py
pixi.toml                 # pixi environment (torch 2.13, natten 0.21.7)
install.sh                # legacy pip install
requirements/             # legacy pip requirements
```

## Notes

* PyTorch 2.13 (CPU and CUDA variants) and NATTEN 0.21.7 (libnatten via PyPI,
  CPU and CUDA variants) are used exclusively for attention; there is no
  pure-torch attention fallback.  The new `models/` and `da/` packages use
  NATTEN and run in the pixi environment.
* The original WeatherMesh-3 code (`model.py`, `model_latlon/`) was written
  against natten 0.17.x and is CUDA-specific; it has been updated to the
  natten 0.21 API but still targets CUDA for training.  ``matepoint`` is
  pulled from test.pypi.
