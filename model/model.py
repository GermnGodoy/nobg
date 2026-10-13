from pyexpat import features

from torch import nn
import torch

# -------------------------------------
# To Do
#  - Implement DropPath in TransformerBlock


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
        self.num_heads = num_heads

        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must be a positive integer that divides dim. Got num_heads={num_heads}, dim={dim}")

        if sr_ratio < 1 or not isinstance(sr_ratio, int):
            raise ValueError(f"sr_ratio must be a positive integer greater than or equal to 1. Got sr_ratio={sr_ratio}")

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

        if N != H * W:
            raise ValueError(f"Input sequence length N={N} does not match expected H*W={H*W}. Please ensure that the input tensor has the correct shape.")

        # Compute Q
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        reduced = x

        if self.sr_ratio > 1:
            # Reshape x to [B, C, H, W] for spatial reduction
            reduced = reduced.transpose(1, 2).reshape(B, C, H, W)
            reduced = self.sr(reduced)                   # [B, C, H/sr_ratio, W/sr_ratio]
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

    def __init__(self, dim, ffn_expansion_factor=4, drop=0., sr_ratio=1, kernel_size=3): 
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        if sr_ratio < 1 or not isinstance(sr_ratio, int):
            raise ValueError(f"sr_ratio must be a positive integer greater than or equal to 1. Got sr_ratio={sr_ratio}")

        self.sr_ratio = sr_ratio
        self.norm = nn.LayerNorm(hidden_features, eps=1e-6)

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
        self.ffn = MixFFN(dim, ffn_expansion_factor=ffn_expansion_factor, drop=drop, sr_ratio=sr_ratio, kernel_size=kernel_size)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MiTEncoder(nn.Module):
    """MiT complete block (in the paper it is called Transformer Block of the Encoder)"""

    def __init__(self, in_channels=3, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], ffn_expansion_factors=[8, 8, 4, 4], qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., sr_ratios=[8, 4, 2, 1], kernel_sizes=[3, 3, 3, 3]):
        super().__init__()

        self.num_stages = len(embed_dims)

        self.num_heads = num_heads if num_heads is not None and len(num_heads) == self.num_stages else [num_heads] * self.num_stages
        
        self.ffn_expansion_factors = ffn_expansion_factors if ffn_expansion_factors is not None and len(ffn_expansion_factors) == self.num_stages else [ffn_expansion_factors] * self.num_stages

        self.sr_ratios = sr_ratios if sr_ratios is not None and len(sr_ratios) == self.num_stages else [sr_ratios] * self.num_stages

        self.kernel_sizes = kernel_sizes if kernel_sizes is not None and len(kernel_sizes) == self.num_stages else [kernel_sizes] * self.num_stages

        self.patch_embed_stages = nn.ModuleList()
        self.encoder_stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for dim in embed_dims])

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dims[i], kernel_size=7, stride=4, padding=3)
            else:
                patch_embed = PatchEmbedding(in_channels=embed_dims[i-1], embed_dim=embed_dims[i], kernel_size=3, stride=2, padding=1)

            self.patch_embed_stages.append(patch_embed)

            encoder_stage = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    ffn_expansion_factor=ffn_expansion_factors[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    sr_ratio=sr_ratios[i],
                    kernel_size=kernel_sizes[i]
                ) for _ in range(2)  
            ])
            self.encoder_stages.append(encoder_stage)

    def forward(self, x):
        features = []
        
        for i in range(self.num_stages):
            x, H, W = self.patch_embed_stages[i](x)

            for block in self.encoder_stages[i]:
                x = block(x, H, W)
            
            x = self.stage_norms[i](x)

            x = x.transpose(1, 2).reshape(x.size(0), -1, H, W)

            features.append(x)
        return features