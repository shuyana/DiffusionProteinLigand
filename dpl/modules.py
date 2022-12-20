import math
from typing import Callable, Optional, Tuple

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn

from .features import ALLOWABLE_ATOM_FEATURES, ALLOWABLE_BOND_FEATURES


class AtomEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(allowable_list), embed_dim)
                for allowable_list in ALLOWABLE_ATOM_FEATURES.values()
            ]
        )
        self.num_features = len(self.embeddings)
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats_embed = 0.0
        for i in range(self.num_features):
            feats_embed += self.scale * self.embeddings[i](feats[..., i])
        return feats_embed  # type: ignore[return-value]


class BondEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(allowable_list), embed_dim)
                for allowable_list in ALLOWABLE_BOND_FEATURES.values()
            ]
        )
        self.num_features = len(self.embeddings)
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats_embed = 0.0
        for i in range(self.num_features):
            feats_embed += self.scale * self.embeddings[i](feats[..., i])
        return feats_embed  # type: ignore[return-value]


class RadialBasisProjection(nn.Module):
    def __init__(self, embed_dim: int, min_val: float = 0.0, max_val: float = 2.0):
        super().__init__()
        self.scale = (embed_dim - 1) / (max_val - min_val)
        self.center = nn.Parameter(
            torch.linspace(min_val, max_val, embed_dim), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.scale * torch.square(x.unsqueeze(-1) - self.center))


class SinusoidalProjection(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even: {embed_dim}.")
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
            torch.logspace(-4.0, 0.0, self.embed_dim // 2), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wx = self.weight * x.unsqueeze(-1)
        return torch.cat([torch.sin(wx), torch.cos(wx)], dim=-1)


def variance_scaling_init_(
    weight: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal",
) -> None:
    fan_out, fan_in = weight.shape
    if mode == "fan_in":
        scale = scale / max(1.0, fan_in)
    elif mode == "fan_out":
        scale = scale / max(1.0, fan_out)
    elif mode == "fan_avg":
        scale = scale / max(1.0, (fan_in + fan_out) / 2.0)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if distribution == "truncated_normal":
        std = math.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(weight, 0.0, std)
    elif distribution == "normal":
        std = math.sqrt(scale)
        nn.init.normal_(weight, 0.0, std)
    elif distribution == "uniform":
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(weight, -limit, limit)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                variance_scaling_init_(self.weight, 1.0, "fan_in", "truncated_normal")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "relu":
                variance_scaling_init_(self.weight, 2.0, "fan_in", "truncated_normal")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "glorot":
                variance_scaling_init_(self.weight, 1.0, "fan_avg", "uniform")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "normal":
                variance_scaling_init_(self.weight, 1.0, "fan_in", "normal")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "gating":
                nn.init.zeros_(self.weight)
                if bias:
                    nn.init.ones_(self.bias)
            elif init == "final":
                nn.init.zeros_(self.weight)
                if bias:
                    nn.init.zeros_(self.bias)
            else:
                raise ValueError(f"Invalid init: {init}")


class Attention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.inf = 2.0 ** 15
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.q_proj = Linear(embed_dim, num_heads * head_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, num_heads * head_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, num_heads * head_dim, bias=False, init="glorot")
        self.gate_proj = Linear(embed_dim, num_heads * head_dim, init="gating")
        self.out_proj = Linear(num_heads * head_dim, embed_dim, init="final")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        query = einops.rearrange(
            self.q_proj(x),
            "... i (h c) -> ... h i c",
            h=self.num_heads,
            c=self.head_dim,
        )
        key = einops.rearrange(
            self.k_proj(x),
            "... j (h c) -> ... h j c",
            h=self.num_heads,
            c=self.head_dim,
        )
        value = einops.rearrange(
            self.v_proj(x),
            "... j (h c) -> ... h j c",
            h=self.num_heads,
            c=self.head_dim,
        )
        gate = einops.rearrange(
            torch.sigmoid(self.gate_proj(x)),
            "... i (h c) -> ... h i c",
            h=self.num_heads,
            c=self.head_dim,
        )
        logits = torch.einsum("...ic,...jc->...ij", self.scale * query, key)
        if attn_bias is not None:
            logits += attn_bias
        attn_mask = einops.rearrange(mask, "... j -> ... 1 1 j")
        logits = logits.masked_fill(attn_mask < 0.5, -self.inf)
        attn = torch.softmax(logits, dim=-1)
        out = gate * torch.einsum("...ij,...jc -> ...ic", attn, value)
        out = einops.rearrange(out, "... h i c -> ... i (h c)")
        out = self.out_proj(out)
        return out


class TriangleAttention(nn.Module):
    def __init__(self, pair_dim: int, head_dim: int, num_heads: int, mode: str):
        super().__init__()
        if mode not in ("starting", "ending"):
            raise ValueError(f"Invalid mode: {mode}")
        self.attn = Attention(pair_dim, head_dim, num_heads)
        self.mode = mode

    def forward(self, pair: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        if self.mode == "ending":
            pair = einops.rearrange(pair, "... i j d -> ... j i d")
            mask_2d = einops.rearrange(mask_2d, "... i j -> ... j i")
        out = self.attn(pair, mask_2d)
        if self.mode == "ending":
            out = einops.rearrange(out, "... j i d -> ... i j d")
        return out


class TriangleMultiplication(nn.Module):
    def __init__(self, pair_dim: int, mode: str):
        super().__init__()
        if mode == "outgoing":
            self.equation = "...ikd,...jkd->...ijd"
        elif mode == "incoming":
            self.equation = "...kid,...kjd->...ijd"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.norm = nn.LayerNorm(pair_dim, elementwise_affine=False)
        self.ab_proj = Linear(pair_dim, pair_dim * 2, init="default")
        self.ab_gate = Linear(pair_dim, pair_dim * 2, init="gating")
        self.ab_norm = nn.LayerNorm(pair_dim, elementwise_affine=False)
        self.out_proj = Linear(pair_dim, pair_dim, init="final")
        self.out_gate = Linear(pair_dim, pair_dim, init="gating")

    def forward(self, pair: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        pair = self.norm(pair)
        a, b = torch.chunk(
            mask_2d.unsqueeze(-1)
            * torch.sigmoid(self.ab_gate(pair))
            * self.ab_proj(pair),
            2,
            dim=-1,
        )
        out = torch.sigmoid(self.out_gate(pair)) * self.out_proj(
            self.ab_norm(torch.einsum(self.equation, a, b))
        )
        return out


class OuterLinear(nn.Module):
    def __init__(self, single_dim: int, pair_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(single_dim, elementwise_affine=False)
        self.linear = Linear(single_dim * 2, pair_dim, init="final")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_i = einops.rearrange(x, "... i d -> ... i 1 d")
        x_j = einops.rearrange(x, "... j d -> ... 1 j d")
        return self.linear(torch.cat([x_i * x_j, x_i - x_j], dim=-1))


class FoldingBlock(nn.Module):
    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        head_dim: int,
        num_heads: int,
        transition_factor: int,
    ):
        super().__init__()
        self.attn_bias = nn.Sequential(
            nn.LayerNorm(pair_dim, elementwise_affine=False),
            Linear(pair_dim, num_heads, init="normal"),
            Rearrange("... i j h -> ... h i j"),
        )
        self.single_attn = Attention(single_dim, head_dim, num_heads)
        self.single_fc = nn.Sequential(
            nn.LayerNorm(single_dim, elementwise_affine=False),
            Linear(single_dim, single_dim * transition_factor, init="relu"),
            nn.ReLU(),
            Linear(single_dim * transition_factor, single_dim, init="final"),
        )
        self.outer_linear = OuterLinear(single_dim, pair_dim)
        self.pair_mul_outgoing = TriangleMultiplication(pair_dim, "outgoing")
        self.pair_mul_incoming = TriangleMultiplication(pair_dim, "incoming")
        self.pair_attn_starting = TriangleAttention(
            pair_dim, head_dim, num_heads, "starting"
        )
        self.pair_attn_ending = TriangleAttention(
            pair_dim, head_dim, num_heads, "ending"
        )
        self.pair_fc = nn.Sequential(
            nn.LayerNorm(pair_dim, elementwise_affine=False),
            Linear(pair_dim, pair_dim * transition_factor, init="relu"),
            nn.ReLU(),
            Linear(pair_dim * transition_factor, pair_dim, init="final"),
        )

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        single = single + self.single_attn(single, mask, attn_bias=self.attn_bias(pair))
        single = single + self.single_fc(single)
        pair = pair + self.outer_linear(single)
        pair = pair + self.pair_mul_outgoing(pair, mask_2d)
        pair = pair + self.pair_mul_incoming(pair, mask_2d)
        pair = pair + self.pair_attn_starting(pair, mask_2d)
        pair = pair + self.pair_attn_ending(pair, mask_2d)
        pair = pair + self.pair_fc(pair)
        return single, pair
