import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from model_latlon.primatives2d import call_checkpointed
from typing import Optional, Tuple
import warnings

import natten
from natten.modules import NeighborhoodAttention3D as _NattenNA3D

# Initialize NATTEN and error logging
os.environ['NATTEN_LOG_LEVEL'] = 'error'
warnings.simplefilter(action='ignore', category=FutureWarning)
natten.use_kv_parallelism_in_fused_na(True)
natten.set_memory_usage_preference("unrestricted")


def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    (_, f, h, w, dim), device, dtype = patches

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
        indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


def add_posemb(x):
    B, D, H, W, C = x.shape
    pe = posemb_sincos_3d(((0, D, H, W, C), x.device, x.dtype)).unsqueeze(0).reshape(B,D,H,W,C)
    x = x + pe
    return x


class CustomNeighborhoodAttention3D(nn.Module):
    """
    Neighborhood Attention 3D Module backed by NATTEN 0.21 (libnatten).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int | Tuple[int, int, int],
        dilation: int | Tuple[int, int, int] = 1,
        is_causal: bool | Tuple[bool, bool, bool] = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        embedding_module = None
    ):
        assert dilation == 1
        assert not is_causal
        if embedding_module is not None and embedding_module != "nothing":
            raise NotImplementedError("embedding_module is not supported by NATTEN 0.21")
        if any(is_causal) if isinstance(is_causal, tuple) else is_causal:
            pass
        if rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_causal = is_causal

        self.attn = _NattenNA3D(
            embed_dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            qkv_bias=qkv_bias,
            qk_scale=self.scale,
            proj_drop=proj_drop,
        )
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"NeighborhoodAttention3D expected a rank-5 input tensor; got {x.dim()=}."
            )
        x = self.attn(x)
        x = self.attn_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
        )
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class Natten3DTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=None, 
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, embedding_module=None):
        super().__init__()
        assert window_size is not None

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)        
        self.attn = CustomNeighborhoodAttention3D(
            dim,
            num_heads,
            window_size,
            rel_pos_bias=True,
            embedding_module=embedding_module
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        B, D, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        wpad = self.window_size[2]//2
        x[:, :, :, :wpad] = x[:, :, :, -wpad-wpad:-wpad].clone()
        x[:, :, :, -wpad:] = x[:, :, :, wpad:wpad+wpad].clone()

        return x
    
class SlideLayers3D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, embedding_module=None, checkpoint_type="matepoint"):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.checkpoint_type = checkpoint_type

        mlist = []
        for _ in range(depth):
            tb_block = Natten3DTransformerBlock(
                dim,
                num_heads,
                window_size=window_size,
                embedding_module=embedding_module
            )
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)

    def forward(self, x):

        for _, blk in enumerate(self.blocks):
            x = call_checkpointed(blk, x, checkpoint_type=self.checkpoint_type)

        return x
