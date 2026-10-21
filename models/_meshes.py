"""HEALPix-style equal-area, isoLatitude hierarchical spherical mesh.

Implements the core WeatherMesh-4 innovation documented in the WindBorne
WeatherMesh-4 technical blog: replacing the latitude-longitude grid with a
HEALPix (Hierarchical Equal Area isoLatitude Pixelation) mesh.

Properties (matching the blog):
  * Hierarchical: resolution controlled by `depth`; each pixel divides into 4
    at the next depth level (pixel count scales by 4x per depth).
  * Equal Area: every pixel represents the same area on the sphere.
  * IsoLatitude: nodes lie on constant-latitude rings.

The mesh is laid out as ``n_rings x pixels_per_ring`` where
``n_rings = 4 * nside`` and ``pixels_per_ring = 3 * nside`` with
``nside = 2 ** depth``.  This gives ``Npix = 12 * nside**2`` pixels and a
2D ring layout that admits efficient reshape/linear "convolution"-like
operations, as described in the blog.

Depth 8  -> nside=256,  Npix=786,432  (~0.23 deg, weather state)
Depth 5  -> nside=32,   Npix=12,288   (~1.8 deg, weather latent space)
"""

from __future__ import annotations

import numpy as np


class HEALPixMesh:
    """Equal-area isoLatitude spherical mesh (HEALPix-style ring ordering)."""

    def __init__(self, depth: int = 5, resolution_deg: float | None = None):
        self.depth = int(depth)
        self.nside = 2 ** self.depth
        self.n_pix = 12 * self.nside ** 2
        self.n_rings = 4 * self.nside
        self.pixels_per_ring = 3 * self.nside

        # Equal-area ring boundaries in z = cos(theta), evenly spaced from 1 to -1
        z_edges = np.linspace(1.0, -1.0, self.n_rings + 1)
        z_center = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Ring index for each pixel, repeated pixels_per_ring times
        ring_idx = np.repeat(np.arange(self.n_rings), self.pixels_per_ring)
        # Longitude position within each ring, tiled
        lon_pos = np.tile(np.arange(self.pixels_per_ring), self.n_rings)

        # Latitude from z = cos(theta)
        theta = np.arccos(np.clip(z_center[ring_idx], -1.0, 1.0))
        lats = 90.0 - np.degrees(theta)

        # Longitude evenly spaced within each ring
        lons = (360.0 * lon_pos / self.pixels_per_ring) % 360.0
        lons[lons >= 180.0] -= 360.0

        self.lats = lats.astype(np.float32)
        self.lons = lons.astype(np.float32)
        self.z = z_center[ring_idx].astype(np.float32)

        # 2D ring-grid shape (n_rings, pixels_per_ring) for reshape-based conv
        self.ring_shape = (self.n_rings, self.pixels_per_ring)

        # 8-direction neighbor index map (used by attention / conv)
        self.neighbors = self._build_neighbors()

        # String id
        self.string_id = f"healpix>d{self.depth}n{self.nside}p{self.n_pix}"

    def _build_neighbors(self) -> np.ndarray:
        """8 nearest neighbors per pixel on the ring grid.

        Neighbors are the same-ring left/right plus three pixels in the ring
        above and three in the ring below (aligned by longitude index).
        Boundaries are handled by clamping to the valid ring range.
        """
        n_rings, ppr = self.ring_shape
        neigh = np.full((self.n_pix, 8), -1, dtype=np.int64)

        ring_idx = np.arange(self.n_pix) // ppr
        pos_idx = np.arange(self.n_pix) % ppr

        # same ring: left / right (circular longitude)
        neigh[:, 0] = ring_idx * ppr + (pos_idx - 1) % ppr   # W
        neigh[:, 1] = ring_idx * ppr + (pos_idx + 1) % ppr   # E

        # ring above (lower latitude index -> higher latitude)
        up_ring = np.clip(ring_idx - 1, 0, n_rings - 1)
        neigh[:, 2] = up_ring * ppr + (pos_idx - 1) % ppr  # NW
        neigh[:, 3] = up_ring * ppr + pos_idx              # N
        neigh[:, 4] = up_ring * ppr + (pos_idx + 1) % ppr  # NE

        # ring below
        dn_ring = np.clip(ring_idx + 1, 0, n_rings - 1)
        neigh[:, 5] = dn_ring * ppr + (pos_idx - 1) % ppr  # SW
        neigh[:, 6] = dn_ring * ppr + pos_idx              # S
        neigh[:, 7] = dn_ring * ppr + (pos_idx + 1) % ppr  # SE

        return neigh

    def shape(self, n_vars: int) -> tuple:
        """Return the per-sample tensor shape (n_pix, n_vars)."""
        return (self.n_pix, n_vars)

    def to_2d(self, x):
        """Reshape a (B, n_pix, C) tensor into (B, n_rings, pixels_per_ring, C)."""
        B, N, C = x.shape
        assert N == self.n_pix, f"Expected {self.n_pix} pixels, got {N}"
        return x.reshape(B, self.n_rings, self.pixels_per_ring, C)

    def from_2d(self, x):
        """Reshape (B, n_rings, pixels_per_ring, C) back to (B, n_pix, C)."""
        return x.reshape(x.shape[0], self.n_pix, x.shape[-1])

    def summary_str(self) -> str:
        return (
            f"HEALPix depth={self.depth}, nside={self.nside}, "
            f"n_pix={self.n_pix}, rings={self.n_rings}x{pixels_per_ring := self.pixels_per_ring}"
        )
