# transformer1d.py ----------------------------------------------------------
# Block-wise cross-attention backbone with MoE tail
# --------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


# ───────────────────────────────── Time embedding ─────────────────────────
class TimeEmb(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : (B,)  integer diffusion step
        return (B, dim)
        """
        half = self.lin.in_features // 2
        freq = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device) / (half - 1))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)        # (B, half)
        emb  = torch.cat([emb.sin(), emb.cos()], -1)             # (B, dim)

        return self.lin(emb)


# ───────────────────────────── Block cross-attention ──────────────────────
class BlockCrossAttention(nn.Module):
    """
    One query (noisy frame) attends to *all* approximation blocks.
    If the attention weight for a block is w_b, every token inside that
    block is weighted by the same w_b.

    q : (B, H, d)
    k : (B, H, S, L, d)
    v : (B, H, S, L, d)
    """
    def __init__(self, d_model: int, n_heads: int, drop: float=0.5, normalize_entropy=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.nh = n_heads
        self.dh = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.scale = self.dh ** -0.5

        self.dropout = nn.Dropout(p=drop)

        self.normalize_entropy = normalize_entropy

    # ------------------------------------------------------------------
    def forward(self, q, k, v):
        B, H, S, L, d = k.shape                # T = H * S
        nh, dh = self.nh, self.dh

        # ── project and reshape ───────────────────────────────────────
        attn_list = []
        for level in range(L):
            _q = self.q_proj(q).view(B, H, nh, dh)                         # (B, H, nh, dh)

            _k = (
                self.k_proj(k[..., level, :])                             # (B, H, S, d)
                .view(B, H, S, nh, dh)                                    # (B, H, S, nh, dh)
                .transpose(2, 3)                                          # (B, H, nh, S, dh)
            )
            _v = (
                self.v_proj(v[..., level, :])
                .view(B, H, S, nh, dh)
                .transpose(2, 3)                                          # (B, H, nh, S, dh)
            )

            # ── aggregate inside each block ──────────────────────────────
            k_blk = _k.mean(dim=-2)                                        # (B, H, nh, dh)
            v_blk = _v.sum(dim=-2)                                         # (B, H, nh, dh)

            # ── block-level attention ────────────────────────────────────
            attn = (_q.transpose(1, 2) @ k_blk.transpose(1, 2).transpose(2, 3)) * self.scale         # (B, nh, H, H)
            attn = F.softmax(attn, dim=-1)  # (B, nh, H, H)
            attn = self.dropout(attn)

            attn_list.append(attn)

        # ---- stack for vectorised ops: (L, B, nh, H, H)
        attn = torch.stack(attn_list, dim=0)

        # per‑token entropy: e_{k,b,nh,i} = −Σ_j A_{ij}^{(k)} log A_{ij}^{(k)}
        entropy = -(attn * (attn + 1e-8).log()).sum(-1)     # (L, B, nh, H)
        entropy = entropy.mean(2)  # average over heads → (L, B, N)

        if self.normalize_entropy:
            entropy = entropy / torch.log(torch.tensor(float(H)))

        # ---- softmax over K branches, per token i: w_{ik} ∝ exp(-h_{ik})
        weights = F.softmax((-entropy).permute(1, 2, 0), dim=-1)  # (B, H, L)

        # ---- fuse rows:  Σ_k w_{ik} A_{ij}^{(k)}
        w_br = weights.permute(2, 0, 1).unsqueeze(2).unsqueeze(-1)  # broadcast weights to (L, B, 1, H, 1)
        fused_attn = (w_br * attn).sum(0)  # (B, nh, H, H)

        fused_attn = self.dropout(fused_attn)

        # ── weighted sum of *block* value vectors ────────────────────
        ctx  = fused_attn @ v_blk.transpose(1, 2)                     # (B, H, nh, dh)
        ctx  = ctx.reshape(B, H, nh * dh)                             # (B, H, d)

        return self.o_proj(ctx)


# ─────────────────────────── Transformer encoder block ─────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, drop: float=0.5):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = BlockCrossAttention(d_model, n_heads, drop=drop)
        self.attn_drop = nn.Dropout(drop)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.fnn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

        self.ffn_drop = nn.Dropout(drop)

    def forward(self, h, k, v):
        """
        Args:
            h: (B, H, d)
            k: (B, H, S, L, d)
            v: (B, H, S, L, d)

        Returns: (B, H, d)
        """

        h = h + self.attn_drop(self.attn(self.attn_norm(h), k, v))  # (B, H, d)

        h = h + self.ffn_drop(self.fnn(self.ffn_norm(h)))

        return h


# ──────────────────────── Transformer ───────────────────────
class StochMultiScaleTransformer(nn.Module):
    """
    Multiscale Cross-attention transformer.

    Parameters
    ----------
    S         : int   samples per action (≈33)
    H         : int   horizon
    act_dim   : int   4
    force_dim : int   6
    d_model   : int   latent width for the transformer
    n_layers  : int   transformer depth
    n_heads   : int   attention heads
    drop      : float dropout rate
    """
    def __init__(self,
                 S: int = 33,
                 H: int = 6,
                 act_dim: int = 4,
                 force_dim: int = 6,
                 d_model: int = 128,
                 n_layers: int = 4,
                 n_heads: int  = 4,
                 drop: float = 0.5
                 ):
        super().__init__()
        self.S = S
        self.H = H
        self.act_dim   = act_dim
        self.force_dim = force_dim
        self.d_model   = d_model

        # embeddings
        self.q_proj = nn.Linear(act_dim, d_model)
        self.kv_proj_f = nn.Linear(force_dim, d_model)
        self.kv_proj_t = nn.Linear(force_dim, d_model)
        self.kv_proj = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model),
                                     nn.LayerNorm(2 * d_model),
                                     nn.GELU(),
                                     nn.Linear(2 * d_model, d_model))

        self.temb = TimeEmb(d_model)
        self.t_mlp = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model))

        # transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=4, drop=drop) for _ in range(n_layers)
        ])

        # map back to act_dim
        self.pre_out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, act_dim)

        self.fc_mu = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.fc_logvar = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()      # standard deviation
        eps = torch.randn_like(std)     # ε ~ N(0,1)

        return mu + eps * std  # z = μ + σ * ε

    # ------------------------------------------------------------------
    def forward(self, noisy, t, apx_f, apx_t):
        """
        noisy    : (B, H, act_dim)
        t        : (B,)
        apx_f, apx_t  : (B, S·H, L, force_dim)   -- one scale
        """
        S, H = self.S, self.H

        # -------- reshape approximation into (B, H, S, L, force_dim) -----
        apx_f_blk = torch.split(apx_f, S, dim=1)
        apx_f_blk = torch.stack(apx_f_blk, dim=1)   # (B, H, S, L, force_dim)

        apx_t_blk = torch.split(apx_t, S, dim=1)
        apx_t_blk = torch.stack(apx_t_blk, dim=1)  # (B, H, S, L, force_dim)

        # -------- embed queries / keys / values -----------------------
        q = self.q_proj(noisy)                            # (B, H, d)
        f_k = self.kv_proj_f(apx_f_blk)                   # (B, H, S, L, d)
        v_k = self.kv_proj_t(apx_t_blk)                   # (B, H, S, L, d)
        k = torch.cat([f_k, v_k], dim=-1)         # (B, H, S, L, 2 * d)
        k = self.kv_proj(k)                               # (B, H, S, L, d)

        mu, logvar = self.fc_mu(k), self.fc_logvar(k)

        if self.training:
            k = self.reparameterize(mu, logvar)
        else:
            k = mu

        v = k                                             # share proj

        # add time embedding to queries once (broadcast)
        t_emb = self.t_mlp(self.temb(t)).unsqueeze(1)     # (B, 1, d)
        h = q + t_emb

        # -------- transformer encoder --------------------------------
        for blk in self.blocks:
            h = blk(h, k, v)                              # (B, H, d)

        # map to action-dim
        h = self.pre_out_norm(h)
        out = self.out_proj(h)                            # (B, H, act_dim)

        # KL divergence term
        mu = mu.view(-1, self.d_model)          # (-1, D)
        logvar = logvar.view(-1, self.d_model)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return out, kld
