"""Shared neural building blocks for the new WeatherMesh models.

Uses NATTEN (libnatten via PyPI, CPU and CUDA variants) exclusively for
windowed/neighbourhood attention; no pure-torch attention fallback.

Mirrors the WM-3 design (ResConv encoder/decoder + sliding-window
transformer) but operates on a HEALPix ring-grid and uses NATTEN.

Encoder design:
  * pressure vars: Conv3d over (Cpr, D, H, W) -> (C, D', H', W'), then depth
    flattened into channels and projected to latent_size -> (B, H', W', C)
  * surface vars: 2D conv over (H, W) -> (C, H', W') -> projected to latent_size
  * latent = pr_latent + sfc_latent, processed by the 2D transformer
Decoder design:
  * latent -> Linear -> pressure volume (ConvTranspose3d) + surface (2D conv)
  * depth is padded to a multiple of 4 in the encoder and trimmed in the
    decoder, so any ``n_levels`` (e.g. WM-6's 25) works.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import natten
from natten.modules import NeighborhoodAttention2D as _NattenNA2D

from models._meshes import HEALPixMesh

natten.use_kv_parallelism_in_fused_na(True)
natten.set_memory_usage_preference("unrestricted")


def _pad2d(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Pad (B, C, H, W): reflect on latitude (H), circular on longitude (W)."""
    x = F.pad(x, (0, 0, pad_h, pad_h), mode="reflect")
    x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
    return x


class EarthResBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int | None = None,
        kernel_size: int = 3,
        affine: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        intermediate_channels = intermediate_channels or out_channels
        k = kernel_size // 2
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=affine)
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=kernel_size, padding=0)
        self.norm2 = nn.InstanceNorm2d(intermediate_channels, affine=affine)
        self.conv2 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = _pad2d(h, self._k, self._k)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = _pad2d(h, self._k, self._k)
        h = self.conv2(h)
        return h + self.nin_shortcut(x)


class EarthConvDown2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self._k = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _pad2d(x, self._k, self._k)
        return self.conv(h)


class EarthConvUp2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False
        )
        self._pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hs, Ws = x.shape
        H = Hs * self.stride - (Hs % 2)
        W = Ws * self.stride
        h = self.conv(x)
        if self._pad > 0:
            h = torch.cat([h, h[..., : self._pad]], dim=-1)
        h = h[..., self._pad : self._pad + W]
        h = h[:, :, self._pad : self._pad + H, :]
        return h


class EarthConvEncoder2d(nn.Module):
    def __init__(self, in_channels, conv_dims=None, affine=True):
        super().__init__()
        conv_dims = conv_dims or [64, 192, 576, 1024]
        self.down_layers = nn.ModuleList()
        for i in range(len(conv_dims) - 1):
            in_dim = conv_dims[i] if i != 0 else in_channels
            mid_dim = conv_dims[i]
            next_dim = conv_dims[i + 1]
            layer = nn.Sequential(
                EarthResBlock2d(in_dim, mid_dim, affine=affine),
                EarthResBlock2d(mid_dim, mid_dim, affine=affine),
                EarthConvDown2d(mid_dim, next_dim),
            )
            self.down_layers.append(layer)
        self.latent_dim = conv_dims[-1]
        # 3 down layers of stride 2 => 8x spatial downsampling
        self.downsample = 2 ** (len(conv_dims) - 1)

    def forward(self, x):
        for layer in self.down_layers:
            x = layer(x)
        return x


class EarthConvDecoder2d(nn.Module):
    def __init__(self, out_channels, conv_dims=None, affine=True):
        super().__init__()
        conv_dims = conv_dims or [1024, 576, 192, 64]
        self.up_layers = nn.ModuleList()
        for i in range(len(conv_dims) - 1):
            in_dim = conv_dims[i]
            out_dim = conv_dims[i + 1]
            layer = nn.Sequential(
                EarthResBlock2d(in_dim, in_dim, affine=affine),
                EarthResBlock2d(in_dim, in_dim, affine=affine),
                EarthConvUp2d(in_dim, out_dim),
            )
            self.up_layers.append(layer)
        self.final = nn.Sequential(
            EarthResBlock2d(conv_dims[-1], conv_dims[-1], affine=affine),
            EarthResBlock2d(conv_dims[-1], out_channels, affine=affine),
        )

    def forward(self, x):
        for layer in self.up_layers:
            x = layer(x)
        return self.final(x)


class WindowedAttention2d(nn.Module):
    """NATTEN neighbourhood attention over (B, H, W, C).

    Wraps ``natten.modules.NeighborhoodAttention2D`` so the new models use
    NATTEN exclusively (CPU and CUDA variants via libnatten).  The effective
    kernel is clamped to the input spatial size so any mesh depth works.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self._attns = nn.ModuleDict()

    def _get_attn(self, kernel_size: int) -> _NattenNA2D:
        key = str(kernel_size)
        if key not in self._attns:
            self._attns[key] = _NattenNA2D(
                embed_dim=self.dim,
                num_heads=self.num_heads,
                kernel_size=kernel_size,
                qkv_bias=True,
                qk_scale=self.head_dim ** -0.5,
                proj_drop=0.0,
            )
        return self._attns[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NATTEN expects (B, H, W, C); returns (B, H, W, C).
        B, H, W, C = x.shape
        k = min(self.window_size, H, W)
        return self._get_attn(k)(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention2d(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SlideLayers(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, window_size) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class ForecastConfig:
    """Configuration shared by the new WeatherMesh forecast models."""

    def __init__(
        self,
        latent_size: int = 1024,
        encdec_tr_depth: int = 2,
        num_heads: int = 32,
        window_size: int = 7,
        affine: bool = True,
        processor_dts: list[int] | None = None,
    ):
        self.latent_size = latent_size
        self.encdec_tr_depth = encdec_tr_depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.affine = affine
        self.processor_dts = processor_dts or [6]


class ResConvEncoder(nn.Module):
    """Encodes pressure + surface inputs on a HEALPix ring-grid into a latent map.

    Input x: (B, H, W, n_vars) where n_vars = n_levels * n_pr_vars + n_sfc_vars.
    Output latent: (B, H', W', latent_size) with H'=H/8, W'=W/8.
    """

    def __init__(self, config: ForecastConfig, n_pr_vars: int, n_sfc_vars: int, n_levels: int):
        super().__init__()
        self.config = config
        self.n_pr_vars = n_pr_vars
        self.n_sfc_vars = n_sfc_vars
        self.n_levels = n_levels
        self._pr_depth_pad = ((n_levels + 3) // 4) * 4

        self.pr_encoder = nn.Conv3d(
            in_channels=n_pr_vars,
            out_channels=config.latent_size,
            kernel_size=(4, 8, 8),
            stride=(4, 8, 8),
        )
        self.pr_to_latent = nn.Linear(config.latent_size * (self._pr_depth_pad // 4), config.latent_size)

        self.sfc_encoder = EarthConvEncoder2d(
            n_sfc_vars,
            conv_dims=[96, 192, 512, config.latent_size],
            affine=config.affine,
        )
        self.sfc_to_latent = nn.Linear(config.latent_size, config.latent_size)

        self.tr = SlideLayers(
            dim=config.latent_size,
            depth=config.encdec_tr_depth,
            num_heads=config.num_heads,
            window_size=config.window_size,
        )

    def forward(self, x: torch.Tensor):
        B, H, W, _ = x.shape
        Cpr = self.n_pr_vars
        D = self.n_levels
        Dpad = self._pr_depth_pad
        Dp = Dpad // 4

        # Pressure: (B, Cpr, D, H, W) -> Conv3d -> (B, C, D', H', W')
        x_pr = x[..., : D * Cpr].reshape(B, H, W, Cpr, D).permute(0, 3, 4, 1, 2)
        if Dpad > D:
            x_pr = F.pad(x_pr, (0, 0, 0, 0, 0, Dpad - D))
        x_pr = self.pr_encoder(x_pr)  # (B, C, Dp, H', W')
        _, _, _, Hp, Wp = x_pr.shape
        # Flatten depth into channels, project to latent
        x_pr = x_pr.permute(0, 3, 4, 2, 1).reshape(B, Hp, Wp, Dp * self.config.latent_size)
        x_pr = self.pr_to_latent(x_pr)  # (B, H', W', latent)

        # Surface: (B, Csfc, H, W) -> conv -> (B, C, H', W') -> project
        x_sfc = x[..., D * Cpr :].permute(0, 3, 1, 2)
        x_sfc = self.sfc_encoder(x_sfc)
        x_sfc = x_sfc.permute(0, 2, 3, 1)
        x_sfc = self.sfc_to_latent(x_sfc)  # (B, H', W', latent)

        x = x_pr + x_sfc
        x = self.tr(x)  # (B, H', W', latent)
        return x, (Hp, Wp, Dp)


class ResConvDecoder(nn.Module):
    """Decodes a latent map back to pressure + surface outputs."""

    def __init__(self, config: ForecastConfig, n_pr_vars: int, n_sfc_vars: int, n_levels: int):
        super().__init__()
        self.config = config
        self.n_pr_vars = n_pr_vars
        self.n_sfc_vars = n_sfc_vars
        self.n_levels = n_levels
        self._pr_depth_pad = ((n_levels + 3) // 4) * 4
        self._Dp = self._pr_depth_pad // 4

        self.tr = SlideLayers(
            dim=config.latent_size,
            depth=config.encdec_tr_depth,
            num_heads=config.num_heads,
            window_size=config.window_size,
        )
        self.latent_to_pr = nn.Linear(config.latent_size, self._Dp * config.latent_size)
        self.latent_to_sfc = nn.Linear(config.latent_size, config.latent_size)
        self.sfc_decoder = EarthConvDecoder2d(
            n_sfc_vars,
            conv_dims=[config.latent_size, 512, 192, 96],
            affine=config.affine,
        )
        self.pr_decoder = nn.ConvTranspose3d(
            out_channels=n_pr_vars,
            in_channels=config.latent_size,
            kernel_size=(4, 8, 8),
            stride=(4, 8, 8),
        )

    def forward(self, x: torch.Tensor, shape_info, n_levels_out: int, H: int, W: int):
        # x: (B, H', W', latent)
        B, Hp, Wp, C = x.shape
        _, _, Dp = shape_info
        x = self.tr(x)

        # Pressure: latent -> (B, H', W', Dp*C) -> (B, C, Dp, H', W') -> ConvTranspose3d
        x_pr = self.latent_to_pr(x)  # (B, H', W', Dp*C)
        x_pr = x_pr.reshape(B, Hp, Wp, Dp, C).permute(0, 4, 3, 1, 2)  # (B, C, Dp, H', W')
        pr = self.pr_decoder(x_pr)  # (B, Cpr, Dpad, H, W)
        pr = pr[..., :n_levels_out, :, :]  # trim padded depth
        pr = pr.permute(0, 2, 3, 4, 1).reshape(B, n_levels_out, H, W, self.n_pr_vars)
        pr = pr.reshape(B, H, W, n_levels_out * self.n_pr_vars)

        # Surface: latent -> (B, H', W', C) -> 2D conv -> (B, Csfc, H, W)
        x_sfc = self.latent_to_sfc(x)  # (B, H', W', C)
        x_sfc = x_sfc.permute(0, 3, 1, 2)
        sfc = self.sfc_decoder(x_sfc)  # (B, Csfc, H, W)
        sfc = sfc.permute(0, 2, 3, 1)  # (B, H, W, Csfc)

        return torch.cat([pr, sfc], dim=-1)


class ForecastModel(nn.Module):
    """Encoder-processor-decoder forecast model on HEALPix (or latlon) meshes.

    Stores the output grid/shape so ``forward(x)`` is self-contained.
    """

    def __init__(
        self,
        config: ForecastConfig,
        encoder: ResConvEncoder,
        decoder: ResConvDecoder,
        n_levels: int,
        H: int,
        W: int,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.n_levels = n_levels
        self.H = H
        self.W = W

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def process(self, latent, shape_info):
        # latent: (B, H', W', C); identity processor in this rebuild.
        # Subclasses can override to add iterative / multi-step processing.
        return latent

    def decode(self, latent, shape_info):
        return self.decoder(latent, shape_info, self.n_levels, self.H, self.W)

    def forward(self, x: torch.Tensor):
        latent, shape_info = self.encode(x)
        latent = self.process(latent, shape_info)
        out = self.decode(latent, shape_info)
        return out


def build_forecast_model(
    pressure_vars: list[str],
    sfc_vars: list[str],
    levels: list[int],
    depth_state: int = 8,
    depth_latent: int = 5,
    latent_size: int = 1024,
    encdec_tr_depth: int = 2,
    num_heads: int = 32,
    window_size: int = 7,
) -> ForecastModel:
    """Build a HEALPix forecast model from variable catalogs.

    Shared constructor used by get_WeatherMesh4 / 5c / 6.  Returns a
    ForecastModel with state_mesh, latent_mesh, levels, pressure_vars,
    sfc_vars, n_pr_vars and n_sfc_vars attached.
    """
    state_mesh = HEALPixMesh(depth=depth_state)
    latent_mesh = HEALPixMesh(depth=depth_latent)

    n_pr_vars = len(pressure_vars)
    n_sfc_vars = len(sfc_vars)
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
    model.pressure_vars = pressure_vars
    model.sfc_vars = sfc_vars
    model.n_pr_vars = n_pr_vars
    model.n_sfc_vars = n_sfc_vars
    return model


class BaseEnsemble:
    """Base wrapper that runs a forecast model as an ``n_members`` ensemble."""

    def __init__(self, model, n_members: int):
        self.model = model
        self.n_members = n_members

    def forecast_ensemble(self, x):
        """Run the model as an ``n_members`` ensemble.

        Returns stacked outputs of shape (n_members, B, H, W, n_vars).
        Architecture-only: no perturbations are applied (no weights).
        """
        outs = [self.model(x) for _ in range(self.n_members)]
        return torch.stack(outs, dim=0)

