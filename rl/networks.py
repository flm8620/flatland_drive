import torch
import torch.nn as nn
import torch.nn.functional as F
# Import proper ViT with RoPE implementation
from utils.vit_rope import ViTEncoder

# --- ViT with RoPE (Fixed Implementation) ---
class ViTEncoderWrapper(nn.Module):
    """Wrapper around the proper ViT implementation for backward compatibility"""
    
    def __init__(self, view_size, patch_size=8, in_chans=2, embed_dim=128, depth=6, num_levels=3, state_dim=2):
        super().__init__()
        # Use the proper ViT implementation
        self.vit_encoder = ViTEncoder(
            view_size=view_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            heads=8,
            mlp_ratio=4,
            dropout=0.1,
            num_levels=num_levels,
            state_dim=state_dim,
            use_2d_rope=True,  # Use 2D RoPE for spatial understanding
            spatial_size=view_size // patch_size
        )
    
    def forward(self, obs_dict):
        return self.vit_encoder(obs_dict)

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
