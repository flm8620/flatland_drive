#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers) for Flatland Driving
Adapted for the maze navigation task with multi-scale observations
"""

import os
import h5py
import logging
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Import timer utilities
from timer import get_timer, set_timing_enabled, reset_timer, print_timing_report

# Import proper ViT with RoPE
from vit_rope import ViTSpatialEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reparametrize(mu, logvar):
    """Reparameterization trick for VAE"""
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def analyze_and_filter_segments(h5_file_path: str, 
                               chunk_size: int,
                               stride: int,
                               max_no_op_ratio: float,
                               min_meaningful_actions: int,
                               verbose: bool = True):
    """
    Cut episodes into segments (chunks) and filter segments based on action distribution.
    
    Args:
        h5_file_path: Path to the H5 file containing demonstrations
        chunk_size: Size of each segment/chunk
        stride: Stride between segment starts
        max_no_op_ratio: Maximum allowed ratio of no-op actions in a segment
        min_meaningful_actions: Minimum number of meaningful (non-no-op) actions per segment
        verbose: Whether to print filtering statistics
    
    Returns:
        dict: Contains episode indices and their valid segment starts
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        total_episodes = h5_file.attrs['total_episodes']
        episodes = h5_file['episodes'][:total_episodes]
        
        if verbose:
            print(f"=== ANALYZING AND FILTERING SEGMENTS ===")
            print(f"Total episodes: {total_episodes}")
            print(f"Chunk size: {chunk_size}, Stride: {stride}")
            print(f"Filtering criteria:")
            print(f"  - Max no-op ratio per segment: {max_no_op_ratio:.2f}")
            print(f"  - Min meaningful actions per segment: {min_meaningful_actions}")
        
        # Process each episode and extract valid segments
        episode_segment_data = {}
        total_segments_found = 0
        total_segments_kept = 0
        total_no_ops_before = 0
        total_no_ops_after = 0
        total_actions_before = 0
        total_actions_after = 0
        
        for ep_idx in range(total_episodes):
            episode = episodes[ep_idx]
            start_idx = int(episode['start_idx'])
            length = int(episode['length'])
            
            # Skip initial no-op actions at the beginning of each episode
            skip_count = 0
            for i in range(length):
                action_idx = start_idx + i
                action = h5_file['actions'][action_idx]
                if action == 0:  # no-op action
                    skip_count += 1
                else:
                    break  # Found first non-no-op action
            
            episode_start = start_idx + skip_count
            remaining_length = length - skip_count
            
            # Generate all possible segments for this episode
            valid_segment_starts = []
            
            for i in range(0, max(0, remaining_length - chunk_size + 1), stride):
                segment_start = episode_start + i
                
                # Analyze the action distribution in this segment
                segment_actions = h5_file['actions'][segment_start:segment_start + chunk_size]
                no_op_count = np.sum(segment_actions == 0)
                meaningful_count = chunk_size - no_op_count
                no_op_ratio = no_op_count / chunk_size
                
                total_segments_found += 1
                total_no_ops_before += no_op_count
                total_actions_before += chunk_size
                
                # Apply filtering criteria
                if no_op_ratio <= max_no_op_ratio and meaningful_count >= min_meaningful_actions:
                    valid_segment_starts.append(int(segment_start))
                    total_segments_kept += 1
                    total_no_ops_after += no_op_count
                    total_actions_after += chunk_size
            
            if len(valid_segment_starts) > 0:
                episode_segment_data[int(ep_idx)] = valid_segment_starts
        
        if verbose:
            print(f"\n=== SEGMENT FILTERING RESULTS ===")
            print(f"Episodes with valid segments: {len(episode_segment_data)} / {total_episodes}")
            print(f"Total segments found: {total_segments_found}")
            print(f"Segments kept: {total_segments_kept} ({total_segments_kept/total_segments_found*100:.1f}%)")
            print(f"Segments rejected: {total_segments_found - total_segments_kept} ({(total_segments_found - total_segments_kept)/total_segments_found*100:.1f}%)")
            
            # Calculate no-op ratio improvement
            no_op_ratio_before = total_no_ops_before / total_actions_before if total_actions_before > 0 else 0
            no_op_ratio_after = total_no_ops_after / total_actions_after if total_actions_after > 0 else 0
            
            print(f"No-op ratio before filtering: {no_op_ratio_before:.3f} ({no_op_ratio_before*100:.1f}%)")
            print(f"No-op ratio after filtering: {no_op_ratio_after:.3f} ({no_op_ratio_after*100:.1f}%)")
            print(f"No-op ratio improvement: {(no_op_ratio_before - no_op_ratio_after)*100:.1f} percentage points")
            print(f"Total training segments after filtering: {total_segments_kept}")
        
        return episode_segment_data


def create_episode_split(h5_file_path: str, 
                        chunk_size: int,
                        train_split: float, 
                        random_seed: int,
                        max_no_op_ratio: float, 
                        min_meaningful_actions: int,
                        stride: int):
    """
    Create explicit train/val episode splits for reproducible dataset creation.
    Episodes are processed to extract valid segments, then segments are split into train/val.
    
    Args:
        h5_file_path: Path to the H5 file containing demonstrations
        chunk_size: Size of each segment/chunk
        train_split: Fraction of episodes to use for training
        random_seed: Random seed for reproducible splits
        max_no_op_ratio: Maximum allowed ratio of no-op actions in a segment
        min_meaningful_actions: Minimum number of meaningful actions per segment
        stride: Stride between segment starts
    
    Returns:
        tuple: (train_episode_segment_data, val_episode_segment_data)
        Each is a dict mapping episode_idx -> list of valid segment starts
    """
    # First, filter segments to reduce no-op bias
    all_episode_segment_data = analyze_and_filter_segments(
        h5_file_path, 
        chunk_size=chunk_size,
        stride=stride,
        max_no_op_ratio=max_no_op_ratio,
        min_meaningful_actions=min_meaningful_actions,
        verbose=True
    )
    
    if len(all_episode_segment_data) == 0:
        raise ValueError("No episodes have valid segments. Consider relaxing the filtering constraints.")
    
    # Get episodes that have valid segments
    episodes_with_valid_segments = list(all_episode_segment_data.keys())
    
    # Create reproducible episode indices from episodes with valid segments
    np.random.seed(random_seed)
    np.random.shuffle(episodes_with_valid_segments)
    
    # Split based on train_split
    split_idx = int(len(episodes_with_valid_segments) * train_split)
    train_episode_indices = episodes_with_valid_segments[:split_idx]
    val_episode_indices = episodes_with_valid_segments[split_idx:]
    
    # Sort indices to maintain some ordering
    train_episode_indices = sorted(train_episode_indices)
    val_episode_indices = sorted(val_episode_indices)
    
    # Split the episode segment data into train/val
    train_episode_segment_data = {ep_idx: all_episode_segment_data[ep_idx] for ep_idx in train_episode_indices}
    val_episode_segment_data = {ep_idx: all_episode_segment_data[ep_idx] for ep_idx in val_episode_indices}
    
    print(f"\n=== FINAL EPISODE SPLIT ===")
    print(f"Episodes with valid segments: {len(episodes_with_valid_segments)}")
    print(f"Train: {len(train_episode_indices)} episodes, Val: {len(val_episode_indices)} episodes")
    print(f"Train episodes: {train_episode_indices[:10]}{'...' if len(train_episode_indices) > 10 else ''}")
    print(f"Val episodes: {val_episode_indices[:10]}{'...' if len(val_episode_indices) > 10 else ''}")
    
    return train_episode_segment_data, val_episode_segment_data


class FlatlandDataset(Dataset):
    """Dataset for loading Flatland human demonstration data from H5 files."""
    
    def __init__(self, h5_file_path: str, chunk_size: int = 32, 
                 episode_segment_data: dict = None,
                 mode: str = 'train'):
        """
        Args:
            h5_file_path: Path to the H5 file containing demonstrations
            chunk_size: Number of actions to predict per observation
            episode_segment_data: Dictionary mapping episode indices to valid segment starts
            mode: 'train' or 'val' (used for logging only)
        """
        if episode_segment_data is None:
            raise ValueError("episode_segment_data must be provided. Use create_episode_split() to generate it.")
        
        self.h5_file_path = h5_file_path
        self.chunk_size = chunk_size
        self.mode = mode
        self.episode_segment_data = episode_segment_data
        
        # Extract episode indices from the segment data
        self.episode_indices = list(episode_segment_data.keys())
        
        # Open HDF5 file and keep it open for the dataset lifetime
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        # Get metadata
        total_episodes = self.h5_file.attrs['total_episodes']
        self.view_size = self.h5_file.attrs['view_size']
        self.num_levels = self.h5_file.attrs['num_levels']
        
        # Validate episode indices
        if len(self.episode_indices) > 0 and np.max(self.episode_indices) >= total_episodes:
            raise ValueError(f"Episode index {np.max(self.episode_indices)} >= total episodes {total_episodes}")
        
        # Load episode metadata for selected episodes only
        episodes_array = self.h5_file['episodes']
        assert len(episodes_array) == total_episodes, \
            f"Episodes array length {len(episodes_array)} does not match total_episodes {total_episodes}"
        
        if len(self.episode_indices) > 0:
            self.episodes = episodes_array[self.episode_indices]
        else:
            self.episodes = []
        
        print(f"{mode.capitalize()} split: {len(self.episodes)} episodes (indices: {len(self.episode_indices)})")
        
        # Build list of valid starting positions from pre-filtered segments
        self.valid_starts = []
        
        print(f'Dataset mode: {mode}')
        for ep_idx in self.episode_indices:
            segment_starts = self.episode_segment_data[ep_idx]
            self.valid_starts.extend(segment_starts)
            print(f'Episode {ep_idx}: added {len(segment_starts)} pre-filtered segments')
        
        print(f"Total valid segment starts: {len(self.valid_starts)}")
        
        # Print final statistics by sampling a few segments
        if len(self.valid_starts) > 0:
            # Sample some segments to verify no-op ratio
            sample_size = min(100, len(self.valid_starts))
            sample_indices = np.random.choice(len(self.valid_starts), sample_size, replace=False)
            total_sampled_actions = 0
            total_sampled_no_ops = 0
            
            for sample_idx in sample_indices:
                segment_start = self.valid_starts[sample_idx]
                segment_actions = self.h5_file['actions'][segment_start:segment_start + chunk_size]
                total_sampled_actions += len(segment_actions)
                total_sampled_no_ops += np.sum(segment_actions == 0)
            
            final_no_op_ratio = total_sampled_no_ops / total_sampled_actions if total_sampled_actions > 0 else 0
            print(f"Final estimated no-op ratio in {mode} data: {final_no_op_ratio:.3f} ({final_no_op_ratio*100:.1f}%)")
        else:
            print("WARNING: No valid segments found!")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        """
        Returns:
            observation: TensorDict with 'image' and 'state' keys
            actions: (chunk_size,) - future actions to predict
            mask: (chunk_size,) - padding mask (all True for now)
        """
        with get_timer("dataset_getitem"):
            start_idx = self.valid_starts[idx]
            
            with get_timer("h5_file_read"):
                # Load observations at start position - NEW FORMAT
                # Images: (num_levels, 2, view_size, view_size) uint8
                obs_images = self.h5_file['obs_images'][start_idx]  
                # State: (2,) float32 - velocity information
                obs_states = self.h5_file['obs_states'][start_idx]
                
                # Load next chunk_size actions
                actions = self.h5_file['actions'][start_idx:start_idx + self.chunk_size]
            
            with get_timer("tensor_conversion"):
                # Convert to tensors
                obs_images_tensor = torch.from_numpy(obs_images).float() / 255.0  # Convert uint8 to float [0,1]
                obs_states_tensor = torch.from_numpy(obs_states).float()
                actions_tensor = torch.from_numpy(actions).long()
                
                # Create observation dictionary (plain dict, no batch dimension for single sample)
                obs_dict = {
                    'image': obs_images_tensor,  # (num_levels, 2, view_size, view_size)
                    'state': obs_states_tensor   # (2,) velocity
                }
                
                # Create mask (no padding for now)
                mask = torch.ones(self.chunk_size, dtype=torch.bool)
            
            return {
                'observations': obs_dict,
                'actions': actions_tensor,
                'mask': mask,
            }
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoidal positional encoding"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """Simplified transformer encoder layer for ACT"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Simplified transformer decoder layer for ACT"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos=None):
        # Self attention on target sequence (no causal mask needed for ACT)
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention with memory
        query = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.multihead_attn(query=query, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class FlatlandACT(nn.Module):
    """ACT model adapted for Flatland navigation task with spatial features and optional CVAE"""
    
    def __init__(self, 
                 num_levels: int = 3,
                 view_size: int = 64,
                 action_dim: int = 9,
                 chunk_size: int = 32,
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 spatial_size: int = 8,
                 use_cvae: bool = True,
                 latent_dim: int = 32):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.spatial_size = spatial_size
        self.num_levels = num_levels
        self.num_spatial_tokens = spatial_size * spatial_size  # 64 tokens for 8x8
        self.use_cvae = use_cvae
        self.latent_dim = latent_dim
        
        # ViT encoder for spatial features (image only)
        self.vit_spatial_encoder = ViTSpatialEncoder(
            view_size=view_size,
            patch_size=view_size // spatial_size,  # Ensure we get spatial_size x spatial_size patches
            in_channels=2,
            embed_dim=hidden_dim,
            depth=4,
            heads=8,
            mlp_ratio=4,
            dropout=dropout,
            num_levels=num_levels,
            use_2d_rope=True
        )
        
        # State encoder for velocity information  
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 64),  # Input: (2,) velocity
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder (for observations)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.num_encoder_layers = num_encoder_layers
        
        # Transformer decoder (for action sequence generation)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.num_decoder_layers = num_decoder_layers
        
        # Action queries (learnable embeddings for each position in action sequence)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))
        
        # Positional encodings for action sequence (1D)
        self.register_buffer(
            'action_pos_encoding', 
            get_sinusoid_encoding_table(chunk_size, hidden_dim)
        )
        
        # CVAE components (optional)
        if self.use_cvae:
            # VAE encoder for learning latent action style from GT actions during training
            self.cvae_encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
                for _ in range(2)  # Smaller encoder for CVAE
            ])
            
            # CLS token for CVAE encoder
            self.cls_embed = nn.Parameter(torch.randn(1, hidden_dim))
            
            # Action embedding for CVAE encoder
            self.action_embed = nn.Linear(action_dim, hidden_dim)
            
            # State embedding for CVAE encoder (to condition on current state)
            self.cvae_state_embed = nn.Linear(2, hidden_dim)  # velocity state
            
            # Positional encoding for CVAE encoder (CLS + state + action_sequence)
            self.register_buffer(
                'cvae_pos_encoding',
                get_sinusoid_encoding_table(1 + 1 + chunk_size, hidden_dim)  # [CLS, state, actions]
            )
            
            # Latent projection (outputs mu and logvar)
            self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
            
            # Latent to hidden projection for decoder conditioning
            self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, observations, actions=None, mask=None):
        """
        Args:
            observations: Dictionary with 'image' and 'state' keys
            actions: (batch_size, chunk_size) - ground truth actions for training (required for CVAE)
            mask: (batch_size, chunk_size) - padding mask
            
        Returns:
            logits: (batch_size, chunk_size, action_dim)
            latent_info: [mu, logvar] if using CVAE and training, else [None, None]
            
        For training with CVAE, actions should be provided.
        For inference, actions should be None.
        """
        # Get batch size from the observations dictionary
        batch_size = observations['image'].size(0)
        device = observations['image'].device
        is_training = actions is not None
        
        # Initialize latent variables
        mu, logvar = None, None
        
        # CVAE Encoder: Extract latent style from ground truth actions (training only)
        if self.use_cvae:
            with get_timer("cvae_encoding"):
                if is_training:
                    # Convert actions to one-hot and embed
                    actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()  # (B, chunk_size, action_dim)
                    action_embeddings = self.action_embed(actions_onehot)  # (B, chunk_size, hidden_dim)
                    
                    # Embed current state
                    state_embedding = self.cvae_state_embed(observations['state']).unsqueeze(1)  # (B, 1, hidden_dim)
                    
                    # CLS token
                    cls_token = self.cls_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
                    
                    # Concatenate: [CLS, state, action_sequence]
                    cvae_input = torch.cat([cls_token, state_embedding, action_embeddings], dim=1)  # (B, 1+1+chunk_size, hidden_dim)
                    
                    # Add positional encoding
                    pos_encoding = self.cvae_pos_encoding.expand(batch_size, -1, -1)  # (B, seq_len, hidden_dim)
                    cvae_input = cvae_input + pos_encoding
                    
                    # Pass through CVAE encoder
                    cvae_output = cvae_input
                    for layer in self.cvae_encoder_layers:
                        cvae_output = layer(cvae_output)
                    
                    # Extract CLS token output for latent projection
                    cls_output = cvae_output[:, 0]  # (B, hidden_dim)
                    
                    # Project to latent space (mu and logvar)
                    latent_info = self.latent_proj(cls_output)  # (B, latent_dim * 2)
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    
                    # Reparameterization trick
                    latent_sample = reparametrize(mu, logvar)  # (B, latent_dim)
                else:
                    # During inference, sample from prior (zero mean, unit variance)
                    latent_sample = torch.zeros(batch_size, self.latent_dim, device=device)
                
                # Project latent to hidden dimension for decoder conditioning
                latent_conditioning = self.latent_out_proj(latent_sample)  # (B, hidden_dim)
        else:
            latent_conditioning = None
        
        # Encode observations to spatial tokens and state features
        with get_timer("observation_encoding"):
            # Encode images to spatial tokens
            with get_timer("vit_spatial_encoding"):
                spatial_tokens = self.vit_spatial_encoder(observations['image'])  # (B, num_levels * num_patches, hidden_dim)
                
                # Verify we get the expected number of spatial tokens (from all levels)
                expected_tokens = self.num_levels * self.spatial_size * self.spatial_size
                assert spatial_tokens.size(1) == expected_tokens, \
                    f"Expected {expected_tokens} spatial tokens, got {spatial_tokens.size(1)}"
            
            # Encode state information
            with get_timer("state_encoding"):
                state_features = self.state_encoder(observations['state'])  # (batch_size, hidden_dim)
        
        # Combine spatial and state tokens for memory
        with get_timer("memory_preparation"):
            # Add state as an additional token
            state_token = state_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            memory_tokens = [spatial_tokens, state_token]
            
            # Add latent conditioning as an additional memory token if using CVAE 
            if self.use_cvae:
                latent_token = latent_conditioning.unsqueeze(1)  # (batch_size, 1, hidden_dim)
                memory_tokens.append(latent_token)
            
            memory = torch.cat(memory_tokens, dim=1)  # (B, num_levels*spatial_size^2 + 1 + [latent], hidden_dim)
            
            # Pass through encoder layers
            for encoder_layer in self.encoder_layers:
                memory = encoder_layer(memory)  # (batch_size, memory_seq_len, hidden_dim)
        
        # Prepare action queries
        with get_timer("query_preparation"):
            # Expand action queries for batch: (batch_size, chunk_size, hidden_dim)
            tgt = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Expand positional encoding for batch: (batch_size, chunk_size, hidden_dim)
            pos_enc = self.action_pos_encoding.expand(batch_size, -1, -1)
        
        # Pass through decoder layers
        with get_timer("decoder_forward"):
            output = tgt
            for decoder_layer in self.decoder_layers:
                output = decoder_layer(
                    tgt=output,
                    memory=memory,
                    query_pos=pos_enc
                )  # (batch_size, chunk_size, hidden_dim)
        
        # Generate action logits (no transpose needed with batch_first=True)
        with get_timer("output_projection"):
            # Generate action logits
            logits = self.action_head(output)  # (batch_size, chunk_size, action_dim)
        
        return logits, [mu, logvar]
    
    def generate(self, observations, temperature=1.0, top_k=None, top_p=None):
        """
        Generate actions for inference with various sampling strategies
        
        Args:
            observations: Dictionary with 'image' and 'state' keys
            temperature: Temperature for softmax sampling (1.0 = normal, >1.0 = more random, <1.0 = more confident)
            top_k: If set, only sample from top-k most likely actions
            top_p: If set, use nucleus sampling (sample from top actions with cumulative prob <= top_p)
        
        Returns:
            actions: (batch_size, chunk_size) - sampled actions
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(observations)  # (batch_size, chunk_size, action_dim)
            
            batch_size, chunk_size, action_dim = logits.shape
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (batch_size, chunk_size, action_dim)
            
            # Apply top-k filtering if specified
            if top_k is not None and top_k < action_dim:
                # Get top-k indices and values
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                
                # Create mask for non-top-k actions
                probs_filtered = torch.zeros_like(probs)
                probs_filtered.scatter_(-1, top_k_indices, top_k_probs)
                probs = probs_filtered
            
            # Apply nucleus (top-p) sampling if specified
            if top_p is not None and top_p < 1.0:
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create mask for actions to keep (cumulative prob <= top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least the first action
                sorted_indices_to_remove[:, :, 0] = False
                
                # Convert back to original indices and set filtered probs to 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
            
            # Renormalize probabilities after filtering
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Sample from the probability distribution
            # Flatten for multinomial sampling
            probs_flat = probs.view(-1, action_dim)  # (batch_size * chunk_size, action_dim)
            
            # Sample actions
            actions_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # (batch_size * chunk_size,)
            
            # Reshape back
            actions = actions_flat.view(batch_size, chunk_size)  # (batch_size, chunk_size)
            
        return actions
    
    def generate_deterministic(self, observations):
        """
        Generate actions using deterministic argmax (original behavior)
        
        Args:
            observations: Dictionary with 'image' and 'state' keys  
            
        Returns:
            actions: (batch_size, chunk_size) - predicted actions
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(observations)
            actions = torch.argmax(logits, dim=-1)  # (batch_size, chunk_size)
        return actions


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard examples and down-weights easy examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ACTTrainer:
    """Trainer for the Flatland ACT model"""
    
    def __init__(self, cfg: DictConfig):
        reset_timer()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up run directories - following same pattern as PPO training
        run_name = cfg.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = to_absolute_path(cfg.logging.output_dir)
        self.run_dir = os.path.join(output_dir, run_name)
        
        # Create subdirectories
        self.log_dir = os.path.join(self.run_dir, "log")
        self.save_dir = os.path.join(self.run_dir, "checkpoints")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"Run directory: {self.run_dir}")
        logger.info(f"Tensorboard logs: {self.log_dir}")
        logger.info(f"Model checkpoints: {self.save_dir}")
        
        # Enable timing if requested
        if hasattr(cfg, 'enable_timer') and cfg.enable_timer:
            set_timing_enabled(True)
        
        # Create explicit episode splits for reproducible train/val separation
        with get_timer("episode_split_creation"):
            # Get filtering parameters from config (with defaults)
            max_no_op_ratio = cfg.data.max_no_op_ratio
            min_meaningful_actions = cfg.data.min_meaningful_actions
            stride = cfg.data.stride

            train_episode_segment_data, val_episode_segment_data = create_episode_split(
                h5_file_path=cfg.data.h5_file_path,
                chunk_size=cfg.model.chunk_size,
                train_split=cfg.data.train_split,
                random_seed=getattr(cfg.data, 'split_seed', 42),  # Allow configurable seed
                max_no_op_ratio=max_no_op_ratio,
                min_meaningful_actions=min_meaningful_actions,
                stride=stride
            )
        
        # Create datasets with episode segment data
        with get_timer("dataset_creation"):
            self.train_dataset = FlatlandDataset(
                h5_file_path=cfg.data.h5_file_path,
                chunk_size=cfg.model.chunk_size,
                episode_segment_data=train_episode_segment_data,
                mode='train',
            )
            
            self.val_dataset = FlatlandDataset(
                h5_file_path=cfg.data.h5_file_path,
                chunk_size=cfg.model.chunk_size,
                episode_segment_data=val_episode_segment_data,
                mode='val',
            )
        
        # Create data loaders
        with get_timer("dataloader_creation"):
            # Configure timing for multiprocessing
            if cfg.training.num_workers > 0:
                # With multiprocessing, we need to configure timers for worker processes
                # The main process keeps timers enabled, workers will disable them
                worker_init_fn = self._worker_init_fn
            else:
                # Single-threaded: keep all timers enabled
                worker_init_fn = None
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn
            )
        
        # Create model
        with get_timer("model_creation"):
            self.model = FlatlandACT(
                num_levels=self.train_dataset.num_levels,
                view_size=self.train_dataset.view_size,
                action_dim=cfg.model.action_dim,
                chunk_size=cfg.model.chunk_size,
                hidden_dim=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                num_encoder_layers=cfg.model.num_encoder_layers,
                num_decoder_layers=cfg.model.num_decoder_layers,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                spatial_size=cfg.model.spatial_size,
                use_cvae=cfg.model.use_cvae,
                latent_dim=cfg.model.latent_dim
            ).to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        
        # Calculate warmup steps (typically 5-10% of total training steps)
        warmup_epochs = max(1, int(cfg.training.epochs * 0.1))  # 10% of total epochs
        total_steps = len(self.train_loader) * cfg.training.epochs
        warmup_steps = len(self.train_loader) * warmup_epochs
        
        # Create warmup + cosine annealing scheduler
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1.0 - 0.01) + 0.01
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss function - use Focal Loss to handle class imbalance
        self.criterion = FocalLoss(
            alpha=cfg.training.focal_alpha,
            gamma=cfg.training.focal_gamma,
            reduction='none'  # Don't reduce yet, we'll handle masking
        )
        
        # KL divergence weight for CVAE loss
        self.kl_weight = cfg.training.kl_weight
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Global step counter for consistent TensorBoard logging
        self.global_step = 0
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def _worker_init_fn(self, worker_id):
        """Initialize worker processes for DataLoader multiprocessing.
        
        This function disables timers in worker processes to avoid CUDA issues,
        while keeping them enabled in the main process.
        """
        # Disable timers in worker processes to avoid CUDA synchronization issues
        set_timing_enabled(False)
    
    def compute_loss_and_accuracy(self, logits, actions, mask, latent_info):
        """
        Compute loss and accuracy for both training and validation.
        
        Args:
            logits: (batch_size, chunk_size, action_dim) - predicted action logits
            actions: (batch_size, chunk_size) - ground truth actions
            mask: (batch_size, chunk_size) - padding mask
            latent_info: [mu, logvar] from CVAE encoder
            
        Returns:
            dict: Contains 'total_loss', 'action_loss', 'kl_loss', 'accuracy'
        """
        mu, logvar = latent_info
        
        # Action prediction loss
        action_loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),  # (B * chunk_size, action_dim)
            actions.reshape(-1)  # (B * chunk_size,)
        )
        action_loss = action_loss.reshape(actions.shape)  # (B, chunk_size)
        
        # Apply mask and average
        valid_action_loss = (action_loss * mask.float()).sum() / mask.sum()
        
        # KL divergence loss (only if using CVAE)
        total_loss = valid_action_loss
        kl_loss = torch.tensor(0.0, device=self.device)
        
        if self.model.use_cvae:
            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl_loss.mean()  # Average over batch
            total_loss = valid_action_loss + self.kl_weight * kl_loss
        
        # Calculate accuracy
        pred_actions = torch.argmax(logits, dim=-1)
        correct = (pred_actions == actions) * mask
        accuracy = correct.sum().float() / mask.sum()
        
        return {
            'total_loss': total_loss,
            'action_loss': valid_action_loss,
            'kl_loss': kl_loss,
            'accuracy': accuracy
        }
    
    def log_prediction_examples(self, logits, actions, mask, phase, num_samples=3):
        """
        Log prediction vs ground truth examples to TensorBoard
        
        Args:
            logits: (batch_size, chunk_size, action_dim) - predicted action logits
            actions: (batch_size, chunk_size) - ground truth actions  
            mask: (batch_size, chunk_size) - padding mask
            phase: 'train' or 'val'
            epoch: current epoch number
            batch_idx: current batch index (for train) 
            num_samples: number of samples to log
        """
        batch_size = actions.size(0)
        chunk_size = actions.size(1)
        
        # Get predicted actions
        pred_actions = torch.argmax(logits, dim=-1)  # (batch_size, chunk_size)
        
        # Get prediction probabilities for analysis
        pred_probs = F.softmax(logits, dim=-1)  # (batch_size, chunk_size, action_dim)
        
        # Sample random examples from the batch
        sample_indices = torch.randperm(batch_size)[:min(num_samples, batch_size)]
        
        action_names = ["No-op", "Up", "Up-Right", "Right", "Down-Right", 
                       "Down", "Down-Left", "Left", "Up-Left"]
        
        # Log individual examples
        for i, sample_idx in enumerate(sample_indices):
            sample_idx = sample_idx.item()
            
            gt_actions = actions[sample_idx].cpu().numpy()  # (chunk_size,)
            predicted_actions = pred_actions[sample_idx].cpu().numpy()  # (chunk_size,)
            sample_mask = mask[sample_idx].cpu().numpy()  # (chunk_size,)
            sample_probs = pred_probs[sample_idx].detach().cpu().numpy()  # (chunk_size, action_dim)
            
            # Create text summary
            text_lines = [f"=== Sample {i+1} ==="]
            text_lines.append("Step | GT Action    | Pred Action   | Confidence | Correct")
            text_lines.append("-" * 65)
            
            for step in range(chunk_size):
                if not sample_mask[step]:
                    continue
                    
                gt_action = int(gt_actions[step])
                pred_action = int(predicted_actions[step])
                confidence = sample_probs[step, pred_action]
                is_correct = "✓" if gt_action == pred_action else "✗"
                
                gt_name = action_names[gt_action] if gt_action < len(action_names) else f"Action{gt_action}"
                pred_name = action_names[pred_action] if pred_action < len(action_names) else f"Action{pred_action}"
                
                text_lines.append(f"{step:4d} | {gt_name:<12} | {pred_name:<12} | {confidence:.3f}     | {is_correct}")
            
            # Calculate accuracy for this sample
            valid_steps = sample_mask.sum()
            correct_steps = ((gt_actions == predicted_actions) * sample_mask).sum()
            sample_accuracy = correct_steps / max(valid_steps, 1)
            
            text_lines.append(f"\nSample Accuracy: {sample_accuracy:.3f} ({correct_steps}/{valid_steps})")
            
            # Log to tensorboard with HTML formatting for monospace display
            global_step = self.global_step if phase == 'train' else self.global_step
            html_content = f'<pre style="font-family: monospace; font-size: 12px;">' + '\n'.join(text_lines) + '</pre>'
            self.writer.add_text(
                f'{phase}/prediction_examples/sample_{i+1}', 
                html_content, 
                global_step
            )
        
        # Log action distribution statistics
        all_gt_actions = actions[mask].detach().cpu().numpy()
        all_pred_actions = pred_actions[mask].detach().cpu().numpy()
        
        # Compute action histograms
        gt_hist = np.bincount(all_gt_actions, minlength=9)
        pred_hist = np.bincount(all_pred_actions, minlength=9)
        
        # Normalize to probabilities
        gt_dist = gt_hist / gt_hist.sum()
        pred_dist = pred_hist / pred_hist.sum()
        
        # Log action distributions as bar charts
        global_step = self.global_step
        
        # Create distribution comparison text
        dist_lines = ["Action | GT Prob | Pred Prob | Difference"]
        dist_lines.append("-" * 45)
        for action_idx in range(9):
            action_name = action_names[action_idx]
            gt_prob = gt_dist[action_idx]
            pred_prob = pred_dist[action_idx]
            diff = pred_prob - gt_prob
            dist_lines.append(f"{action_name:<8} | {gt_prob:.3f}   | {pred_prob:.3f}    | {diff:+.3f}")
        
        # Log action distribution with HTML formatting for monospace display
        html_dist_content = f'<pre style="font-family: monospace; font-size: 12px;">' + '\n'.join(dist_lines) + '</pre>'
        self.writer.add_text(
            f'{phase}/action_distribution',
            html_dist_content,
            global_step
        )
        
        # Log action distributions as histograms instead of individual scalars
        # Create weighted samples for histogram visualization
        gt_samples = np.repeat(np.arange(9), (gt_hist * 1000).astype(int))  # Scale up for better visualization
        pred_samples = np.repeat(np.arange(9), (pred_hist * 1000).astype(int))
        
        if len(gt_samples) > 0:  # Only log if we have samples
            self.writer.add_histogram(f'{phase}/action_dist/ground_truth', gt_samples, global_step)
        if len(pred_samples) > 0:
            self.writer.add_histogram(f'{phase}/action_dist/predictions', pred_samples, global_step)
    
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss_accum = 0
        total_acc = 0
        num_batches = 0
    
        dataloader_iter = iter(self.train_loader)
        pbar = tqdm(range(len(self.train_loader)), desc=f"Epoch {epoch+1} Training")
        
        for batch_idx in pbar:
            
            with get_timer("fetch_data"):
                batch = next(dataloader_iter)

            with get_timer("data_to_device"):
                # Transfer dictionary of observations to device
                observations = batch['observations']
                observations = {
                    'image': observations['image'].to(self.device),
                    'state': observations['state'].to(self.device)
                }
                actions = batch['actions'].to(self.device)  # (B, chunk_size)
                mask = batch['mask'].to(self.device)  # (B, chunk_size)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            with get_timer("forward_pass"):
                logits, latent_info = self.model(observations, actions, mask)  # (B, chunk_size, action_dim)
            
            # Compute losses and accuracy
            with get_timer("loss_computation"):
                loss_dict = self.compute_loss_and_accuracy(logits, actions, mask, latent_info)
                total_loss = loss_dict['total_loss']
                valid_action_loss = loss_dict['action_loss']
                kl_loss = loss_dict['kl_loss']
                accuracy = loss_dict['accuracy']
            
            # Backward pass
            with get_timer("backward_pass"):
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate after each step
            
            total_loss_accum += total_loss.item()
            total_acc += accuracy.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{accuracy.item():.3f}",
                'kl': f"{kl_loss.item():.4f}" if self.model.use_cvae else "0.0000"
            })
            
            # Log to tensorboard
            with get_timer("tensorboard_logging"):
                self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/action_loss', valid_action_loss.item(), self.global_step)
                if self.model.use_cvae:
                    self.writer.add_scalar('train/kl_loss', kl_loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', accuracy.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Log prediction examples only on the first batch to avoid too much logging
                if batch_idx == 0:
                    self.log_prediction_examples(logits, actions, mask, 'train', num_samples=1)
                
                # Increment global step after each training batch
                self.global_step += 1

            print_timing_report()
        avg_loss = total_loss_accum / num_batches
        avg_acc = total_acc / num_batches
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.3f}")
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with get_timer(f"val_epoch_{epoch}"):
            with torch.no_grad():
                dataloader_iter = iter(self.val_loader)
                pbar = tqdm(range(len(self.val_loader)), desc=f"Epoch {epoch+1} Validation")
                
                for batch_idx in pbar:
                    with get_timer("fetch_data"):
                        batch = next(dataloader_iter)
                    
                    with get_timer("val_data_transfer"):
                        # Transfer dictionary of observations to device
                        observations = batch['observations']
                        observations = {
                            'image': observations['image'].to(self.device),
                            'state': observations['state'].to(self.device)
                        }
                        actions = batch['actions'].to(self.device)
                        mask = batch['mask'].to(self.device)
                    
                    # Forward pass
                    with get_timer("val_forward_pass"):
                        logits, latent_info = self.model(observations, actions, mask)
                        mu, logvar = latent_info
                    
                    # Compute loss and accuracy
                    with get_timer("val_loss_computation"):
                        loss_dict = self.compute_loss_and_accuracy(logits, actions, mask, latent_info)
                        valid_loss = loss_dict['total_loss']
                        accuracy = loss_dict['accuracy']
                    
                    total_loss += valid_loss.item()
                    total_acc += accuracy.item()
                    num_batches += 1
                    
                    # Log prediction examples on the first batch of each validation
                    if batch_idx == 0:
                        self.log_prediction_examples(logits, actions, mask, 'val', num_samples=3)
                    
                    pbar.set_postfix({
                        'loss': f"{valid_loss.item():.4f}",
                        'acc': f"{accuracy.item():.3f}"
                    })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        if epoch == -1:
            logger.info(f"Initial Validation - Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.3f}")
        else:
            logger.info(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.3f}")
        
        # Log to tensorboard (only for actual training epochs, not initial validation)
        if epoch >= 0:
            self.writer.add_scalar('val/loss', avg_loss, self.global_step)
            self.writer.add_scalar('val/accuracy', avg_acc, self.global_step)
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch+1}")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.cfg.logging.save_interval == 0:
            periodic_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
    
    def train(self):
        logger.info("Starting training...")
        
        # Run initial validation to get baseline score from random network
        logger.info("Running initial validation to get baseline score...")
        initial_val_loss, initial_val_acc = self.validate(epoch=-1)  # Use epoch=-1 to indicate initial validation
        logger.info(f"Initial random network - Val Loss: {initial_val_loss:.4f}, Val Acc: {initial_val_acc:.3f}")
        
        # Log initial validation to tensorboard at epoch -1 for reference
        self.writer.add_scalar('val/loss_initial', initial_val_loss, 0)
        self.writer.add_scalar('val/accuracy_initial', initial_val_acc, 0)
        
        best_val_loss = initial_val_loss  # Initialize with baseline score
        
        for epoch in range(self.cfg.training.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log epoch summary
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
        
        logger.info("Training completed!")
        self.writer.close()


@hydra.main(version_base=None, config_path=".", config_name="act_config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    
    if not os.path.exists(cfg.data.h5_file_path):
        print(f"ERROR: Data file not found at {cfg.data.h5_file_path}")
        print("Please make sure you have recorded human demonstration data first.")
        print("Run: python human_player.py")
        return
    
    # Create trainer and start training
    trainer = ACTTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
