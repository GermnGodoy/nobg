import math
from numbers import Integral, Real

from torch import nn
import torch
from torch.nn import functional as F


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}")


def _stage_values(name, values, count):
    """Accept one value shared by all stages or exactly one value per stage."""
    if isinstance(values, Real):
        return (values,) * count
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a number or a sequence of {count} values") from exc
    if len(values) != count:
        raise ValueError(f"{name} must contain {count} values. Got {len(values)}")
    return values


class DropPath(nn.Module):
    """Drop a residual branch per sample during training, preserving its expectation."""

    def __init__(self, p=0.0):
        super().__init__()
        if not isinstance(p, Real) or not 0 <= p < 1:
            raise ValueError("DropPath probability must satisfy 0 <= p < 1")
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

# -------------------------------------
# Blocks

class PatchEmbedding(nn.Module):
    """Project overlapping patches to tokens normalized across channels."""

    def __init__(self, in_channels=3, embed_dim=32, kernel_size=7, stride=4, padding=3):
        super().__init__()

        if kernel_size <= stride:
            raise ValueError("kernel_size must be greater than stride for overlapping patches")

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.projection(x)  # [B, embed_dim, H_out, W_out]
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # [B, H_out*W_out, embed_dim]
        x = self.norm(x)
        return x, height, width


class SelfAttention(nn.Module):
    """Self Attention block optimized following the ideas of SegFormer paper."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio = 1):
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("num_heads", num_heads)
        self.num_heads = num_heads

        if dim % num_heads != 0:
            raise ValueError(f"num_heads must be a positive integer that divides dim. Got num_heads={num_heads}, dim={dim}")

        _positive_int("sr_ratio", sr_ratio)

        self.sr_ratio = sr_ratio
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        if sr_ratio > 1: # Implement optimization
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        _positive_int("H", H)
        _positive_int("W", W)

        if N != H * W:
            raise ValueError(f"Input sequence length N={N} does not match expected H*W={H*W}. Please ensure that the input tensor has the correct shape.")

        # Compute Q
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        reduced = x

        if self.sr_ratio > 1:
            if H < self.sr_ratio or W < self.sr_ratio:
                raise ValueError(
                    f"Feature map {H}x{W} is smaller than sr_ratio={self.sr_ratio}"
                )
            # Reshape x to [B, C, H, W] for spatial reduction
            reduced = reduced.transpose(1, 2).reshape(B, C, H, W)
            reduced = self.sr(reduced)  # [B, C, floor(H/r), floor(W/r)]
            reduced = reduced.flatten(2).transpose(1, 2)  # [B, N', C]
            reduced = self.norm(reduced)

        M = reduced.shape[1]  # New sequence length after spatial reduction
        kv = self.kv(reduced).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    """Mix Feed Forward Network block optimized following the ideas of SegFormer paper."""

    def __init__(self, dim, ffn_expansion_factor=4, drop=0., kernel_size=3):
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("kernel_size", kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions")
        if (not isinstance(ffn_expansion_factor, Real)
                or not math.isfinite(ffn_expansion_factor)
                or ffn_expansion_factor <= 0):
            raise ValueError("ffn_expansion_factor must be finite and positive")
        hidden_features = int(dim * ffn_expansion_factor)
        _positive_int("hidden_features", hidden_features)

        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (kernel_size - 1) // 2  # In order to not change the spatial dimensions

        self.fc1 = nn.Linear(dim, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, _ = x.shape

        _positive_int("H", H)
        _positive_int("W", W)

        if N != H * W:
            raise ValueError("N must equal H * W")

        x = self.fc1(x)  # [B, N, hidden_features]

        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)  # [B, hidden_features, H, W]
        x = x.flatten(2).transpose(1, 2)

        x = self.drop(self.act(x))
        x = self.drop(self.fc2(x))
        return x  # [B, N, dim]


class TransformerBlock(nn.Module):
    """Transformer block optimized following the ideas of SegFormer paper."""

    def __init__(self, dim, num_heads, ffn_expansion_factor=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., sr_ratio=1, kernel_size=3):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFFN(dim, ffn_expansion_factor=ffn_expansion_factor, drop=drop, kernel_size=kernel_size)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


class MiTEncoder(nn.Module):
    """Four-stage Mix Transformer; defaults follow MiT-B0 in supplemental Table 4."""

    def __init__(self, in_channels=3, embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8), ffn_expansion_factors=(8, 8, 4, 4), qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., sr_ratios=(8, 4, 2, 1), kernel_sizes=(3, 3, 3, 3), depths=(2, 2, 2, 2)):
        super().__init__()

        _positive_int("in_channels", in_channels)
        self.in_channels = in_channels
        self.embed_dims = tuple(embed_dims)
        self.num_stages = len(self.embed_dims)
        if self.num_stages != 4:
            raise ValueError("MiTEncoder requires exactly four embed_dims")
        for dim in self.embed_dims:
            _positive_int("embed_dim", dim)
        self.num_heads = _stage_values("num_heads", num_heads, self.num_stages)
        self.ffn_expansion_factors = _stage_values("ffn_expansion_factors", ffn_expansion_factors, self.num_stages)
        self.sr_ratios = _stage_values("sr_ratios", sr_ratios, self.num_stages)
        self.kernel_sizes = _stage_values("kernel_sizes", kernel_sizes, self.num_stages)
        self.depths = _stage_values("depths", depths, self.num_stages)
        for depth in self.depths:
            _positive_int("depth", depth)
        # Validate even if the first block's scheduled rate is zero.
        DropPath(drop_path_rate)
        path_rates = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        block_index = 0

        self.patch_embed_stages = nn.ModuleList()
        self.encoder_stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for dim in self.embed_dims])

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=self.embed_dims[i], kernel_size=7, stride=4, padding=3)
            else:
                patch_embed = PatchEmbedding(in_channels=self.embed_dims[i-1], embed_dim=self.embed_dims[i], kernel_size=3, stride=2, padding=1)

            self.patch_embed_stages.append(patch_embed)

            encoder_stage = nn.ModuleList([
                TransformerBlock(
                    dim=self.embed_dims[i],
                    num_heads=self.num_heads[i],
                    ffn_expansion_factor=self.ffn_expansion_factors[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=path_rates[block_index + j],
                    sr_ratio=self.sr_ratios[i],
                    kernel_size=self.kernel_sizes[i]
                ) for j in range(self.depths[i])
            ])
            self.encoder_stages.append(encoder_stage)
            block_index += self.depths[i]

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected images [B, {self.in_channels}, H, W]")
        features = []
        
        for i in range(self.num_stages):
            x, H, W = self.patch_embed_stages[i](x)

            for block in self.encoder_stages[i]:
                x = block(x, H, W)
            
            x = self.stage_norms[i](x)

            x = x.transpose(1, 2).reshape(x.size(0), -1, H, W).contiguous()

            features.append(x)
        return features


class SegFormerDecoder(nn.Module):
    """All-MLP decoder following Eq. (4); output uses the first stage's resolution.

    Each scale is projected C_i -> decoder_dim, resized and concatenated.
    A 1x1 convolution implements a shared linear map at every pixel.
    """

    def __init__(self, in_channels=(32, 64, 160, 256), decoder_dim=256,
                 num_classes=2, dropout=0.0):
        super().__init__()
        self.in_channels = tuple(in_channels)
        if len(self.in_channels) != 4:
            raise ValueError("SegFormerDecoder requires four feature maps")
        _positive_int("decoder_dim", decoder_dim)
        _positive_int("num_classes", num_classes)
        for channels in self.in_channels:
            _positive_int("in_channels", channels)
        self.projections = nn.ModuleList([
            nn.Linear(channels, decoder_dim) for channels in self.in_channels
        ])
        self.fuse = nn.Conv2d(4 * decoder_dim, decoder_dim, kernel_size=1)
        self.drop = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Expected four encoder feature maps")
        if features[0].ndim != 4:
            raise ValueError("Feature maps must have shape [B, C, H, W]")
        batch = features[0].shape[0]
        target_size = features[0].shape[-2:]
        projected = []
        for i, (feature, projection) in enumerate(zip(features, self.projections)):
            if (feature.ndim != 4 or feature.shape[0] != batch
                    or feature.shape[1] != self.in_channels[i]):
                raise ValueError(
                    f"Feature {i} must have shape [B={batch}, C={self.in_channels[i]}, H, W]"
                )
            B, _, H, W = feature.shape
            tokens = projection(feature.flatten(2).transpose(1, 2))
            feature = tokens.transpose(1, 2).reshape(B, -1, H, W)
            if (H, W) != target_size:
                feature = F.interpolate(feature, size=target_size,
                                        mode="bilinear", align_corners=False)
            projected.append(feature)
        fused = self.fuse(torch.cat(projected, dim=1))  # [B, decoder_dim, H1, W1]
        return self.classifier(self.drop(fused))  # [B, num_classes, H1, W1]


class SegFormer(nn.Module):
    """From-scratch encoder and decoder returning raw segmentation logits.

    By default logits are resized to the image size. Pass upsample_logits=False
    to obtain the native decoder resolution. Apply softmax only for probabilities,
    not before CrossEntropyLoss.
    """

    def __init__(self, num_classes=2, decoder_dim=256, decoder_dropout=0.0,
                 upsample_logits=True, **encoder_kwargs):
        super().__init__()
        self.encoder = MiTEncoder(**encoder_kwargs)
        self.decoder = SegFormerDecoder(
            in_channels=self.encoder.embed_dims, decoder_dim=decoder_dim,
            num_classes=num_classes, dropout=decoder_dropout,
        )
        self.upsample_logits = upsample_logits

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        if self.upsample_logits:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits
