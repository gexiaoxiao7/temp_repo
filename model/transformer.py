from torch import nn, einsum
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from model.utils import LayerNorm
from model.customize_visiontransformer import TSTransformer


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FSATransformerEncoder(nn.Module):
    """Factorized Self-Attention Transformer Encoder"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                 ]))

    def forward(self, x):
        x = x.unsqueeze(2)
        b = x.shape[0]
        x = torch.flatten(x, start_dim=0, end_dim=1)  # extract spatial tokens from x

        for sp_attn, temp_attn, ff in self.layers:
            t = sp_attn(x)
            sp_attn_x = t + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_attn_x = sp_attn_x.chunk(b, dim=0)
            sp_attn_x = [temp[None] for temp in sp_attn_x]
            sp_attn_x = torch.cat(sp_attn_x, dim=0).transpose(1, 2)
            sp_attn_x = torch.flatten(sp_attn_x, start_dim=0, end_dim=1)

            t = temp_attn(sp_attn_x)
            temp_attn_x = t + sp_attn_x  # Temporal attention


            x = ff(temp_attn_x) + temp_attn_x  # MLP

            # Again reshape tensor for spatial attention
            x = x.chunk(b, dim=0)
            x = [temp[None] for temp in x]
            x = torch.cat(x, dim=0).transpose(1, 2)
            x = torch.flatten(x, start_dim=0, end_dim=1)

        # Reshape vector to [b, nt*nh*nw, dim]
        x = x.chunk(b, dim=0)
        x = [temp[None] for temp in x]
        x = torch.cat(x, dim=0)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        return x

class TemporalVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 T=8, temporal_modeling_type=None, use_checkpoint=False, num_experts=0, expert_insert_layers=[],
                 record_routing=False, routing_type='patch-level'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.temporal_modeling_type = temporal_modeling_type
        self.T = T
        self.use_checkpoint = use_checkpoint
        self.record_routing = record_routing
        self.routing_type = routing_type

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = TSTransformer(width, layers, heads, use_checkpoint=self.use_checkpoint, T=self.T,
                                         temporal_modeling_type=self.temporal_modeling_type, num_experts=num_experts,
                                         expert_insert_layers=expert_insert_layers, record_routing=record_routing,
                                         routing_type=routing_type)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x, [maskf, mask] = x
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width] # (bxt) l c
        B, l, c = x.shape
        b = B // self.T

        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        if self.record_routing:
            x, routing_state = self.transformer(x)
        else:
            x = self.transformer(x)

        feature = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(feature[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        if self.record_routing:
            return x
        else:
            return x