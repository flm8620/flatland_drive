#!/usr/bin/env python3
"""
Vision Transformer with Rotary Position Embedding (RoPE)
Proper implementation that can be reused across different projects.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RoPE2D(nn.Module):
    """
    2D Rotary Positional Embedding (RoPE) for images.
    Splits the last dimension head_dim into two halves:
      - First half used for height direction rotary encoding
      - Second half used for width direction rotary encoding
    Each direction applies standard RoPE with paired rotations (even/odd pairing).

    Supports two input formats:
      1) x.shape == (B, n_heads, H, W, D)
      2) x.shape == (B, L, n_heads, D) with seq_shape=(H, W)

    Usage (note q/k must use same phase):
        rope = RoPE2D(head_dim, height, width)
        q, k = rope(q, k)                           # for 5D input
        q, k = rope(q, k, seq_shape=(H, W))         # for 4D (flattened) input

    Args:
      head_dim : Channel dimension D for each attention head, must be divisible by 4
      height   : Height of the 2D grid
      width    : Width of the 2D grid
      base     : RoPE base, default 10000.0
    """
    def __init__(self, head_dim: int, height: int, width: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 (2D RoPE needs H/W halves with paired rotation)."
        self.head_dim = head_dim
        self.height = height
        self.width = width
        self.base = float(base)

        # Precompute cos/sin tables at initialization
        self._build_cos_sin_tables()

    def _build_cos_sin_tables(self):
        """Build cos/sin tables at initialization time"""
        D = self.head_dim
        H, W = self.height, self.width
        
        Dh = D // 2                  # Dimensions for height direction
        Dw = D // 2                  # Dimensions for width direction
        Dh_pairs = Dh // 2           # Number of pairs (even/odd pairing)
        Dw_pairs = Dw // 2

        # Standard RoPE frequencies: base^{-2i/d}
        inv_freq_h = 1.0 / (self.base ** (torch.arange(0, Dh, 2, dtype=torch.float32) / Dh))
        inv_freq_w = 1.0 / (self.base ** (torch.arange(0, Dw, 2, dtype=torch.float32) / Dw))

        pos_h = torch.arange(H, dtype=torch.float32)[:, None]     # (H, 1)
        pos_w = torch.arange(W, dtype=torch.float32)[:, None]     # (W, 1)

        ang_h = pos_h * inv_freq_h[None, :]                       # (H, Dh_pairs)
        ang_w = pos_w * inv_freq_w[None, :]                       # (W, Dw_pairs)

        cos_h, sin_h = torch.cos(ang_h), torch.sin(ang_h)         # (H, Dh_pairs)
        cos_w, sin_w = torch.cos(ang_w), torch.sin(ang_w)         # (W, Dw_pairs)

        # Register as buffers so they move with the model to GPU/CPU
        self.register_buffer('cos_h', cos_h)  # (H, Dh_pairs)
        self.register_buffer('sin_h', sin_h)  # (H, Dh_pairs)
        self.register_buffer('cos_w', cos_w)  # (W, Dw_pairs)
        self.register_buffer('sin_w', sin_w)  # (W, Dw_pairs)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply paired rotation to tensor x's last dimension.
        x:   (..., D_half) where D_half is half-dimension for one direction, must be even
        cos: (..., D_half/2)
        sin: (..., D_half/2)
        Returns: (..., D_half)
        """
        # Split last dimension into pairs, shape (..., pairs, 2)
        x = x.view(*x.shape[:-1], -1, 2)
        x_even = x[..., 0]        # (..., pairs)
        x_odd  = x[..., 1]        # (..., pairs)
        # Apply rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., pairs, 2)
        return x_rot.view(*x_rot.shape[:-2], -1)              # (..., D_half)

    def _apply_2d_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D RoPE to flattened input tensor.
        Args:
            x: (B*n_heads, L, D) where L = H * W
        Returns:
            x with 2D RoPE applied, same shape (B*n_heads, L, D)
        """
        BH, L, D = x.shape
        H, W = self.height, self.width
        
        assert L == H * W, f"Sequence length {L} must equal H*W={H*W}"
        assert D == self.head_dim, f"Head dimension {D} must match expected {self.head_dim}"
        
        # Reshape to spatial format: (B*n_heads, H, W, D)
        x_spatial = x.view(BH, H, W, D)
        
        # Split into height and width halves
        Dh = D // 2
        x_h = x_spatial[..., :Dh]  # (B*n_heads, H, W, Dh)
        x_w = x_spatial[..., Dh:]  # (B*n_heads, H, W, Dw)
        
        # Apply paired rotations
        x_h_rot = self._rotate_pairs(x_h, self.cos_h[None, :, None, :], self.sin_h[None, :, None, :])
        x_w_rot = self._rotate_pairs(x_w, self.cos_w[None, None, :, :], self.sin_w[None, None, :, :])
        
        # Concatenate and reshape back to flat format
        x_rot = torch.cat([x_h_rot, x_w_rot], dim=-1)  # (B*n_heads, H, W, D)
        return x_rot.view(BH, L, D)  # (B*n_heads, L, D)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE to (q, k) with consistent phase.
        Input shape: (B*n_heads, L, D) where L = H * W
        
        Returns:
          (q_rot, k_rot) with same shape as input
        """
        assert q.shape == k.shape, "q and k must have identical shapes."
        assert q.dim() == 3, f"Expected 3D tensor (B*n_heads, L, D), got {q.dim()}D tensor"
        
        BH, L, D = q.shape
        assert D == self.head_dim, f"Last dimension {D} doesn't match expected head_dim {self.head_dim}."
        assert D % 4 == 0, "head_dim must be divisible by 4."
        assert L == self.height * self.width, f"Sequence length {L} must equal H*W={self.height * self.width}"

        # Apply 2D RoPE to both q and k
        q_rot = self._apply_2d_flat(q)
        k_rot = self._apply_2d_flat(k)

        return q_rot, k_rot


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding for sequence data"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Create frequency bands
        freqs = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        freqs = 1.0 / (10000 ** freqs)
        self.register_buffer('freqs', freqs)
    
    def forward(self, seq_len, device=None):
        """Generate rotary embeddings for sequence length"""
        if device is None:
            device = self.freqs.device
            
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = t[:, None] * self.freqs[None, :]  # (seq_len, dim//2)
        
        # Create sin/cos embeddings
        emb = torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # (seq_len, dim)
        return emb
    
    def apply_rotary_emb(self, x, pos_emb):
        """
        Apply 1D rotary embedding
        Args:
            x: (..., seq_len, dim)
            pos_emb: (seq_len, dim) 
        """
        half_dim = self.dim // 2
        
        # Split into pairs for rotation
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:half_dim*2]
        
        # Extract cos/sin
        cos = pos_emb[..., :half_dim]
        sin = pos_emb[..., half_dim:half_dim*2]
        
        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Concatenate rotated pairs
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
        
        # Handle odd dimensions
        if self.dim > half_dim * 2:
            x_rot = torch.cat([x_rot, x[..., half_dim*2:]], dim=-1)
            
        return x_rot


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention with RoPE support"""
    
    def __init__(self, dim, heads=8, dropout=0.1, rope_2d=None, rope_1d=None):
        super().__init__()
        assert dim % heads == 0
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE embeddings
        self.rope_2d = rope_2d
        self.rope_1d = rope_1d
        
        # Initialize parameters following transformer best practices
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following transformer best practices"""
        # Initialize QKV projection with Xavier uniform
        nn.init.xavier_uniform_(self.qkv.weight)
        
        # Initialize output projection with Xavier uniform and scale by sqrt(2 * num_layers)
        # Note: We don't know num_layers here, so we use a reasonable default scale
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (batch_size, seq_len, dim) - Always batch-first format
            attn_mask: (seq_len, seq_len) attention mask
        """
        B, L, D = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if available
        if self.rope_2d is not None:
            # Apply 2D RoPE efficiently by vectorizing over batch and heads
            # q, k shape: (B, heads, L, head_dim)

            # Reshape to combine batch and heads: (B*heads, L, head_dim)
            B, heads, L, head_dim = q.shape
            q_flat = q.reshape(B * heads, L, head_dim)
            k_flat = k.reshape(B * heads, L, head_dim)
            
            # Use dimensions from RoPE2D initialization - no need to calculate seq_shape
            # RoPE2D already knows the expected grid dimensions
            q_rope, k_rope = self.rope_2d(q_flat, k_flat)  # (B*heads, N, head_dim)

            # Reshape back to separate batch and heads: (B, heads, L, head_dim)
            q = q_rope.reshape(B, heads, L, head_dim)
            k = k_rope.reshape(B, heads, L, head_dim)

        elif self.rope_1d is not None:
            # Apply 1D RoPE to q and k
            pos_emb = self.rope_1d(L, device=x.device)  # (L, head_dim)

            q_reshaped = q.transpose(1, 2).reshape(B * L, self.heads * self.head_dim)
            k_reshaped = k.transpose(1, 2).reshape(B * L, self.heads * self.head_dim)

            # Expand pos_emb for all heads
            pos_emb_expanded = pos_emb.repeat(1, self.heads)  # (L, D)
            pos_emb_batch = pos_emb_expanded[None, :, :].expand(B, -1, -1).reshape(B * L, -1)

            q_reshaped = self.rope_1d.apply_rotary_emb(q_reshaped[:, None, :], pos_emb_batch[:, None, :]).squeeze(1)
            k_reshaped = self.rope_1d.apply_rotary_emb(k_reshaped[:, None, :], pos_emb_batch[:, None, :]).squeeze(1)

            q = q_reshaped.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
            k = k_reshaped.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, L, L)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        
        # Final projection
        out = self.proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with RoPE support"""
    
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.1, rope_2d=None, rope_1d=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout, rope_2d, rope_1d)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize MLP parameters following transformer best practices"""
        # Initialize MLP layers
        nn.init.xavier_uniform_(self.mlp[0].weight)  # First linear layer
        nn.init.constant_(self.mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.mlp[3].weight)  # Second linear layer  
        nn.init.constant_(self.mlp[3].bias, 0.0)
        
        # LayerNorm parameters are already properly initialized by PyTorch
        # (weight=1.0, bias=0.0)
        
    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), attn_mask)
        
        # MLP with residual  
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTSpatialEncoder(nn.Module):
    """Vision Transformer encoder that preserves spatial tokens (for ACT)"""
    
    def __init__(self, 
                 view_size=64,
                 patch_size=8, 
                 in_channels=2, 
                 embed_dim=256,
                 depth=6, 
                 heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 num_levels=3,
                 use_2d_rope=True):
        super().__init__()
        self.view_size = view_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # Calculate number of patches per side
        self.patches_per_side = view_size // patch_size  # Should be 8 for 64x64 with patch_size=8
        self.num_patches = self.patches_per_side ** 2  # Should be 64
        
        # Patch embedding for each level using Conv2d (much cleaner than unfold)
        self.patch_embed = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_levels)
        ])
        
        # RoPE embeddings  
        self.use_2d_rope = use_2d_rope
        if use_2d_rope:
            head_dim = embed_dim // heads  # Each attention head dimension
            # Calculate grid dimensions from patch configuration
            grid_height = grid_width = self.patches_per_side  # e.g., 8x8 = 64 patches
            self.rope_2d = RoPE2D(head_dim, grid_height, grid_width)
            self.rope_1d = None
        else:
            self.rope_2d = None 
            head_dim = embed_dim // heads
            self.rope_1d = RoPE1D(head_dim)
        
        # Transformer blocks for each level
        self.transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_dim, heads, mlp_ratio, dropout, self.rope_2d, self.rope_1d)
                for _ in range(depth)
            ]) for _ in range(num_levels)
        ])
        
        # Level embeddings to distinguish tokens from different zoom levels
        self.level_embeddings = nn.Parameter(torch.randn(num_levels, embed_dim) * 0.02)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following ViT best practices"""
        # Initialize patch embedding conv layers
        for patch_embed in self.patch_embed:
            # Conv2d initialization (similar to linear layer)
            nn.init.xavier_uniform_(patch_embed.weight)
            if patch_embed.bias is not None:
                nn.init.constant_(patch_embed.bias, 0.0)
        
        # Level embeddings are already initialized with small random values
        # No additional initialization needed
        
    def forward(self, images):
        """
        Args:
            images: (batch_size, num_levels, in_channels, H, W)
        Returns:
            spatial_tokens: (batch_size, num_levels * num_patches, embed_dim) - spatial tokens from all levels
        """
        # Convert to float if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        batch_size = images.size(0)
        
        # Process each level and collect spatial tokens
        all_level_tokens = []
        for level in range(self.num_levels):
            level_img = images[:, level]  # (B, C, H, W)
            
            # Patchify using conv2d: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
            patches = self.patch_embed[level](level_img)  # (B, embed_dim, patches_per_side, patches_per_side)
            
            # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
            tokens = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
            
            # Add level-specific embeddings to distinguish different zoom levels
            tokens = tokens + self.level_embeddings[level][None, None, :]  # Broadcasting: (1, 1, embed_dim) -> (B, num_patches, embed_dim)
            
            # Apply transformer blocks
            for transformer_block in self.transformers[level]:
                tokens = transformer_block(tokens)
            
            # Collect tokens from this level (B, num_patches, embed_dim)
            all_level_tokens.append(tokens)
        
        # Combine multi-scale tokens: concatenate along sequence (token) dimension
        # (B, num_levels * num_patches, embed_dim)
        spatial_tokens = torch.cat(all_level_tokens, dim=1)
        
        return spatial_tokens


class ViTEncoder(nn.Module):
    """Vision Transformer encoder with global pooling (for non-ACT use cases)"""
    
    def __init__(self, 
                 view_size=64,
                 patch_size=8, 
                 in_channels=2, 
                 embed_dim=256,
                 depth=6, 
                 heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 num_levels=3,
                 state_dim=2,
                 use_2d_rope=True):
        super().__init__()
        self.view_size = view_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.state_dim = state_dim
        
        # Use the spatial encoder
        self.spatial_encoder = ViTSpatialEncoder(
            view_size=view_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_levels=num_levels,
            use_2d_rope=use_2d_rope
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # Output projection
        combined_dim = embed_dim * 2  # image features + state features
        self.output_proj = nn.Linear(combined_dim, combined_dim)
    
    def forward(self, obs_dict):
        """
        Args:
            obs_dict: Dictionary with keys:
                - 'image': (batch_size, num_levels, in_channels, H, W)
                - 'state': (batch_size, state_dim)
        Returns:
            combined_features: (batch_size, combined_dim)
        """
        images = obs_dict['image']  # (B, num_levels, C, H, W) 
        states = obs_dict['state']  # (B, state_dim)
        
        # Get spatial tokens from all levels
        spatial_tokens = self.spatial_encoder(images)  # (B, num_levels * num_patches, embed_dim)
        
        # Global average pooling to get image feature
        image_features = spatial_tokens.mean(dim=1)  # (B, embed_dim)
        
        # Process state
        state_features = self.state_encoder(states)  # (B, embed_dim)
        
        # Combine features
        combined = torch.cat([image_features, state_features], dim=-1)  # (B, combined_dim)
        
        # Final projection
        output = self.output_proj(combined)
        return output



