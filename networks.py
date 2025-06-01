import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ViT with RoPE ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device=None):
        t = torch.arange(seq_len, device=self.inv_freq.device if device is None else device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    @staticmethod
    def apply_rotary(x, rope_emb):
        head_dim = x.shape[-1]
        half_dim = head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:2 * half_dim]
        sin = rope_emb[:, :half_dim].unsqueeze(0).unsqueeze(2)
        cos = rope_emb[:, half_dim:2 * half_dim].unsqueeze(0).unsqueeze(2)
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        x_rot = torch.cat([x_rot1, x_rot2], dim=-1)
        if head_dim > 2 * half_dim:
            x_rot = torch.cat([x_rot, x[..., 2 * half_dim:]], dim=-1)
        return x_rot

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.heads = heads
        self.head_dim = dim // heads
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, N, D = x.shape
        x_norm = self.norm1(x)
        q = k = v = x_norm
        q = q.view(B, N, self.heads, self.head_dim)
        k = k.view(B, N, self.heads, self.head_dim)
        rope_emb = self.rope(N, device=x.device)
        q = RotaryEmbedding.apply_rotary(q, rope_emb)
        k = RotaryEmbedding.apply_rotary(k, rope_emb)
        q = q.reshape(B, N, D)
        k = k.reshape(B, N, D)
        attn_out, _ = self.attn(q, k, v)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, view_size, patch_size=8, in_chans=4, embed_dim=128, depth=6, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_proj = nn.ModuleList([
            nn.Linear(in_chans * patch_size * patch_size, embed_dim)
            for _ in range(num_levels)
        ])
        self.transformer = nn.ModuleList([
            nn.Sequential(*[
                TransformerBlock(embed_dim, heads=8, mlp_ratio=8)
                for _ in range(depth)
            ]) for _ in range(num_levels)
        ])
        self.fc_out = nn.Linear(embed_dim * num_levels, embed_dim * num_levels)
        self.view_size = view_size

    def forward(self, x):
        feats = []
        for i in range(self.num_levels):
            xi = x[:, i]
            B, C, H, W = xi.shape
            patches = xi.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(B, -1, C * self.patch_size * self.patch_size)
            tokens = self.patch_proj[i](patches)
            tokens = self.transformer[i](tokens)
            pooled = tokens.mean(dim=1)
            feats.append(pooled)
        out = torch.cat(feats, dim=-1)
        out = self.fc_out(out)
        return out

# --- Simple ConvNet ---
class ConvEncoder(nn.Module):
    def __init__(self, view_size, in_chans=4, num_levels=3, out_dim=384):
        super().__init__()
        self.num_levels = num_levels
        self.view_size = view_size
        self.out_dim = out_dim
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ) for _ in range(num_levels)
        ])
        self.fc = nn.Linear(128 * num_levels, out_dim)

    def forward(self, x):
        feats = []
        for i in range(self.num_levels):
            xi = x[:, i]
            feat = self.convs[i](xi)
            feat = feat.view(xi.size(0), -1)
            feats.append(feat)
        out = torch.cat(feats, dim=-1)
        out = self.fc(out)
        return out
