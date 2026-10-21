"""WeatherMesh model implementations.

Flat directory of model definitions reconstructed from the WindBorne technical
blogs:

  * WeatherMesh-4  -- https://windbornesystems.com/blog/weathermesh-4-technical-blog
  * WeatherMesh-5c -- https://windbornesystems.com/blog/weathermesh-5c-release-notes
  * WeatherMesh-6  -- https://windbornesystems.com/blog/introducing-wm-6
  * AI DA          -- https://windbornesystems.com/blog/windborne-ai-da

All are architecture-only: no weights are distributed, and the architectures
are faithful reconstructions of the publicly described designs rather than the
exact internal WindBorne implementations (full architectural details and
weights are not public).
"""

from models.weathermesh_4 import get_WeatherMesh4
from models.weathermesh_5c import get_WeatherMesh5c, WM5cContinuousForecaster
from models.weathermesh_6 import get_WeatherMesh6, get_WeatherMesh6_3km, WM6Ensemble

__all__ = [
    "get_WeatherMesh4",
    "get_WeatherMesh5c",
    "WM5cContinuousForecaster",
    "get_WeatherMesh6",
    "get_WeatherMesh6_3km",
    "WM6Ensemble",
]
