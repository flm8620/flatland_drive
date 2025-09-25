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
    def __init__(self, view_size, patch_size=8, in_chans=2, embed_dim=128, depth=6, num_levels=3, state_dim=2):
        super().__init__()
        self.num_levels = num_levels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        
        # Patch projection for each pyramid level (now only 2 channels: cost + target)
        self.patch_proj = nn.ModuleList([
            nn.Linear(in_chans * patch_size * patch_size, embed_dim)
            for _ in range(num_levels)
        ])
        
        # Transformer blocks for each level
        self.transformer = nn.ModuleList([
            nn.Sequential(*[
                TransformerBlock(embed_dim, heads=8, mlp_ratio=8)
                for _ in range(depth)
            ]) for _ in range(num_levels)
        ])
        
        # State encoder for velocity
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(),
        )
        
        # Final output layer
        combined_dim = embed_dim * num_levels + embed_dim  # image features + state features
        self.fc_out = nn.Linear(combined_dim, combined_dim)
        self.view_size = view_size

    def forward(self, obs_dict):
        # Extract image and state from dict
        image_obs = obs_dict['image']  # (batch_size, num_levels, 2, H, W) uint8
        state_obs = obs_dict['state']  # (batch_size, state_dim) float32
        
        # Convert image to float and normalize
        image_obs = image_obs.float() / 255.0  # uint8 -> float32 [0,1]
        
        # Process each pyramid level
        image_feats = []
        for i in range(self.num_levels):
            xi = image_obs[:, i]  # (batch_size, 2, H, W)
            B, C, H, W = xi.shape
            
            # Create patches
            patches = xi.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(B, -1, C * self.patch_size * self.patch_size)
            
            # Project to embeddings
            tokens = self.patch_proj[i](patches)
            
            # Apply transformer
            tokens = self.transformer[i](tokens)
            
            # Global average pooling
            pooled = tokens.mean(dim=1)
            image_feats.append(pooled)
        
        # Concatenate image features from all levels
        image_features = torch.cat(image_feats, dim=-1)  # (batch_size, embed_dim * num_levels)
        
        # Process state features
        state_features = self.state_encoder(state_obs)  # (batch_size, embed_dim)
        
        # Combine image and state features
        combined_features = torch.cat([image_features, state_features], dim=-1)
        
        # Final output
        out = self.fc_out(combined_features)
        return out

# --- Simple ConvNet ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ConvEncoder(nn.Module):
    def __init__(self, view_size, in_chans=2, num_levels=3, out_dim=384, state_dim=2):
        super().__init__()
        self.num_levels = num_levels
        self.view_size = view_size
        self.out_dim = out_dim
        self.state_dim = state_dim
        
        # Image encoders for each pyramid level (now only 2 channels: cost + target)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                ResidualBlock(32, 64, stride=2),
                ResidualBlock(64, 128, stride=2),
                ResidualBlock(128, 256, stride=2),
                ResidualBlock(256, 256, stride=2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
            ) for _ in range(num_levels)
        ])
        
        # State encoder for velocity
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final MLP to combine image features and state features
        combined_dim = 256 * num_levels + 64  # image features + state features
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, obs_dict):
        # Extract image and state from dict
        image_obs = obs_dict['image']  # (batch_size, num_levels, 2, H, W) uint8
        state_obs = obs_dict['state']  # (batch_size, state_dim) float32
        
        # Convert image to float and normalize
        image_obs = image_obs.float() / 255.0  # uint8 -> float32 [0,1]
        
        # Process each pyramid level
        image_feats = []
        for i in range(self.num_levels):
            xi = image_obs[:, i]  # (batch_size, 2, H, W)
            feat = self.encoders[i](xi)  # (batch_size, 256)
            image_feats.append(feat)
        
        # Concatenate image features from all levels
        image_features = torch.cat(image_feats, dim=-1)  # (batch_size, 256 * num_levels)
        
        # Process state features
        state_features = self.state_encoder(state_obs)  # (batch_size, 64)
        
        # Combine image and state features
        combined_features = torch.cat([image_features, state_features], dim=-1)
        
        # Final MLP
        out = self.mlp(combined_features)
        return out
