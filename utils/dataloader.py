#!/usr/bin/env python3
"""
Shared dataloader for Flatland imitation learning tasks.
Supports both ACT (chunked actions) and ConvNet (single actions) models.

Key features:
1. Configurable history window frames (for future temporal modeling)
2. Configurable action chunk size (1 for ConvNet, 32+ for ACT)
3. Flexible no-op filtering with percentage-based exclusion
4. Reproducible train/val splits at episode level
"""

import os
import h5py
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset

# Import timer utilities
from utils.timer import get_timer

# Setup logging
logger = logging.getLogger(__name__)


def analyze_and_filter_segments(h5_file_path: str, 
                               history_frames: int,
                               action_chunk_size: int,
                               stride: int,
                               max_no_op_ratio: float,
                               no_op_filter_percentage: float = 1.0,
                               verbose: bool = True) -> Dict[int, List[int]]:
    """
    Analyze episodes and extract valid segments with flexible no-op filtering.
    
    Args:
        h5_file_path: Path to the H5 file containing demonstrations
        history_frames: Number of historical observation frames needed
        action_chunk_size: Size of each action chunk/segment  
        stride: Stride between segment starts
        max_no_op_ratio: Maximum allowed ratio of no-op actions in a segment
        no_op_filter_percentage: Percentage of segments that must meet no-op filtering criteria (0.0-1.0)
                               1.0 = all segments must pass filtering (strict)
                               0.5 = 50% of segments can violate filtering criteria  
                               0.0 = no filtering applied (keep all segments)
        verbose: Whether to print filtering statistics
    
    Returns:
        dict: Maps episode index -> list of valid segment starts
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        total_episodes = h5_file.attrs['total_episodes']
        episodes = h5_file['episodes'][:total_episodes]
        
        if verbose:
            print(f"\n=== SEGMENT ANALYSIS ===")
            print(f"History frames: {history_frames}, Action chunk size: {action_chunk_size}, Stride: {stride}")
            print(f"Max no-op ratio: {max_no_op_ratio}")
            print(f"No-op filter percentage: {no_op_filter_percentage} ({'strict' if no_op_filter_percentage == 1.0 else 'flexible'})")
        
        # Process each episode and extract all possible segments
        episode_all_segments = {}  # ep_idx -> [(start_idx, no_op_ratio, meaningful_count)]
        total_segments_found = 0
        total_no_ops_before = 0
        total_actions_before = 0
        
        for ep_idx in range(total_episodes):
            episode = episodes[ep_idx]
            start_idx = int(episode['start_idx'])
            length = int(episode['length'])
            
            # Skip initial no-op actions at the beginning of each episode
            skip_count = 0
            for i in range(length):
                action_idx = start_idx + i
                if action_idx < len(h5_file['actions']):
                    action = h5_file['actions'][action_idx]
                    if action == 0:  # no-op action
                        skip_count += 1
                    else:
                        break  # Found first non-no-op action
                else:
                    break
            
            episode_start = start_idx + skip_count
            remaining_length = length - skip_count
            
            # We need at least history_frames + action_chunk_size timesteps
            min_required_length = history_frames + action_chunk_size
            if remaining_length < min_required_length:
                continue  # Skip episodes that are too short
            
            # Generate all possible segments for this episode  
            episode_segments = []
            max_start_offset = remaining_length - min_required_length + 1
            
            for i in range(0, max_start_offset, stride):
                segment_start = episode_start + i
                
                # The segment needs history_frames obs + action_chunk_size actions
                # For observations: we take the frame at (segment_start + history_frames - 1)
                # For actions: we take actions from segment_start + history_frames - 1 to segment_start + history_frames - 1 + action_chunk_size
                action_start = segment_start + history_frames - 1
                action_end = action_start + action_chunk_size
                
                # Make sure we don't exceed the episode boundaries
                if action_end <= start_idx + length:
                    # Analyze the action distribution in this segment
                    segment_actions = h5_file['actions'][action_start:action_end]
                    no_op_count = np.sum(segment_actions == 0).item()
                    meaningful_count = action_chunk_size - no_op_count
                    no_op_ratio = no_op_count / action_chunk_size
                    
                    episode_segments.append((int(segment_start), no_op_ratio, meaningful_count))
                    total_segments_found += 1
                    total_no_ops_before += no_op_count
                    total_actions_before += action_chunk_size
            
            if len(episode_segments) > 0:
                episode_all_segments[int(ep_idx)] = episode_segments
        
        # Apply flexible filtering
        episode_segment_data = {}
        total_segments_kept = 0
        total_no_ops_after = 0
        total_actions_after = 0
        
        for ep_idx, segments in episode_all_segments.items():
            # Separate segments into those that pass and fail the filtering criteria
            passing_segments = []
            failing_segments = []
            
            for segment_start, no_op_ratio, meaningful_count in segments:
                if no_op_ratio <= max_no_op_ratio:
                    passing_segments.append(segment_start)
                else:
                    failing_segments.append(segment_start)
            
            # Determine how many segments to keep based on no_op_filter_percentage
            total_segments_in_episode = len(segments)
            if total_segments_in_episode == 0:
                continue
                
            if no_op_filter_percentage >= 1.0:
                # Strict filtering: only keep passing segments
                valid_segments = passing_segments
            elif no_op_filter_percentage <= 0.0:
                # No filtering: keep all segments
                valid_segments = [start for start, _, _ in segments]
            else:
                # Flexible filtering: keep all passing segments + some failing segments
                num_required_passing = int(np.ceil(total_segments_in_episode * no_op_filter_percentage))
                num_passing = len(passing_segments)
                
                if num_passing >= num_required_passing:
                    # We have enough passing segments, sample some failing ones too
                    num_failing_to_add = total_segments_in_episode - num_passing
                    if num_failing_to_add > 0 and len(failing_segments) > 0:
                        # Randomly sample failing segments
                        num_failing_to_add = min(num_failing_to_add, len(failing_segments))
                        np.random.seed(ep_idx)  # Deterministic sampling per episode
                        sampled_failing = np.random.choice(failing_segments, num_failing_to_add, replace=False).tolist()
                        valid_segments = passing_segments + sampled_failing
                    else:
                        valid_segments = passing_segments
                else:
                    # Not enough passing segments, keep all and pad with failing segments if needed
                    num_failing_needed = num_required_passing - num_passing
                    if num_failing_needed > 0 and len(failing_segments) > 0:
                        num_failing_needed = min(num_failing_needed, len(failing_segments))
                        np.random.seed(ep_idx)
                        sampled_failing = np.random.choice(failing_segments, num_failing_needed, replace=False).tolist()
                        valid_segments = passing_segments + sampled_failing
                    else:
                        valid_segments = passing_segments
            
            if len(valid_segments) > 0:
                episode_segment_data[ep_idx] = sorted(valid_segments)
                
                # Calculate statistics for kept segments
                for segment_start in valid_segments:
                    # Find the segment info 
                    for start, no_op_ratio, meaningful_count in segments:
                        if start == segment_start:
                            no_op_count = action_chunk_size - meaningful_count
                            total_segments_kept += 1
                            total_no_ops_after += no_op_count
                            total_actions_after += action_chunk_size
                            break
        
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
            improvement = (no_op_ratio_before - no_op_ratio_after) * 100
            print(f"No-op ratio improvement: {improvement:.1f} percentage points")
            print(f"Total training segments after filtering: {total_segments_kept}")
        
        return episode_segment_data


def create_episode_split(h5_file_path: str,
                        history_frames: int,
                        action_chunk_size: int,
                        train_split: float, 
                        random_seed: int,
                        max_no_op_ratio: float, 
                        stride: int,
                        no_op_filter_percentage: float = 1.0) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Create explicit train/val episode splits for reproducible dataset creation.
    Episodes are processed to extract valid segments, then segments are split into train/val.
    
    Args:
        h5_file_path: Path to the H5 file containing demonstrations
        history_frames: Number of historical observation frames needed
        action_chunk_size: Size of each action chunk/segment
        train_split: Fraction of episodes to use for training
        random_seed: Random seed for reproducible splits
        max_no_op_ratio: Maximum allowed ratio of no-op actions in a segment
        stride: Stride between segment starts
        no_op_filter_percentage: Percentage of segments that must meet filtering criteria
    
    Returns:
        tuple: (train_episode_segment_data, val_episode_segment_data)
        Each is a dict mapping episode_idx -> list of valid segment starts
    """
    # First, filter segments with flexible no-op filtering
    all_episode_segment_data = analyze_and_filter_segments(
        h5_file_path, 
        history_frames=history_frames,
        action_chunk_size=action_chunk_size,
        stride=stride,
        max_no_op_ratio=max_no_op_ratio,
        no_op_filter_percentage=no_op_filter_percentage,
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
    """
    Shared dataset for loading Flatland human demonstration data from H5 files.
    
    Supports both ACT (chunked actions) and ConvNet (single actions) models through
    configurable parameters:
    
    - history_frames: Number of observation frames in the history window (currently 1)
    - action_chunk_size: Number of future actions to predict (1 for ConvNet, 32+ for ACT)
    - episode_segment_data: Pre-filtered valid segments per episode
    """
    
    def __init__(self, 
                 h5_file_path: str, 
                 history_frames: int = 1,
                 action_chunk_size: int = 1, 
                 episode_segment_data: Optional[Dict[int, List[int]]] = None,
                 mode: str = 'train'):
        """
        Args:
            h5_file_path: Path to the H5 file containing demonstrations
            history_frames: Number of historical observation frames (for future temporal modeling)
            action_chunk_size: Number of actions to predict per observation
            episode_segment_data: Dictionary mapping episode indices to valid segment starts
            mode: 'train' or 'val' (used for logging only)
        """
        if episode_segment_data is None:
            raise ValueError("episode_segment_data must be provided. Use create_episode_split() to generate it.")
        
        self.h5_file_path = h5_file_path
        self.history_frames = history_frames
        self.action_chunk_size = action_chunk_size
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
        print(f"Dataset config - History frames: {history_frames}, Action chunk size: {action_chunk_size}")
        
        # Build list of valid starting positions from pre-filtered segments
        self.valid_starts = []
        
        for ep_idx in self.episode_indices:
            segment_starts = self.episode_segment_data[ep_idx]
            self.valid_starts.extend(segment_starts)
            if len(segment_starts) <= 10:
                print(f'Episode {ep_idx}: added {len(segment_starts)} segments: {segment_starts}')
            else:
                print(f'Episode {ep_idx}: added {len(segment_starts)} segments: {segment_starts[:5]}...{segment_starts[-5:]}')
        
        print(f"Total valid segment starts: {len(self.valid_starts)}")
        
        # Print final statistics by sampling a few segments
        if len(self.valid_starts) > 0:
            # Sample some segments to verify action distribution
            sample_size = min(100, len(self.valid_starts))
            sample_indices = np.random.choice(len(self.valid_starts), sample_size, replace=False)
            total_sampled_actions = 0
            total_sampled_no_ops = 0
            
            for sample_idx in sample_indices:
                segment_start = self.valid_starts[sample_idx]
                # Calculate action range for this segment
                action_start = segment_start + history_frames - 1 
                action_end = action_start + action_chunk_size
                segment_actions = self.h5_file['actions'][action_start:action_end]
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
            observations: Dict or List[Dict] with 'image' and 'state' keys
                         - If history_frames=1: Dict with current observation
                         - If history_frames>1: List of Dicts for temporal sequence
            actions: (action_chunk_size,) - future actions to predict  
            mask: (action_chunk_size,) - padding mask (all True for now)
        """
        with get_timer("dataset_getitem"):
            segment_start = self.valid_starts[idx]
            
            with get_timer("h5_file_read"):
                # Load observation history
                # For history_frames=1, we take observation at (segment_start + history_frames - 1) = segment_start
                # For history_frames>1, we take observations from segment_start to segment_start + history_frames - 1
                obs_history = []
                
                for h in range(self.history_frames):
                    obs_idx = segment_start + h
                    
                    # Load observations - check which format is used in the H5 file
                    try:
                        # Try new format first: 'obs_images' and 'obs_states' 
                        obs_images = self.h5_file['obs_images'][obs_idx]  # (num_levels, 2, view_size, view_size)
                        obs_states = self.h5_file['obs_states'][obs_idx]  # (2,)
                    except KeyError:
                        # Fall back to old format: 'observations/images' and 'observations/states'
                        obs_images = self.h5_file['observations/images'][obs_idx]
                        obs_states = self.h5_file['observations/states'][obs_idx]
                    
                    obs_history.append((obs_images, obs_states))
                
                # Load future actions  
                action_start = segment_start + self.history_frames - 1
                action_end = action_start + self.action_chunk_size
                actions = self.h5_file['actions'][action_start:action_end]
            
            with get_timer("tensor_conversion"):
                # Convert observations to tensors
                if self.history_frames == 1:
                    # Single observation case (most common)
                    obs_images, obs_states = obs_history[0]
                    obs_images_tensor = torch.from_numpy(obs_images).float() / 255.0  
                    obs_states_tensor = torch.from_numpy(obs_states).float()
                    
                    observations = {
                        'image': obs_images_tensor,  # (num_levels, 2, view_size, view_size)
                        'state': obs_states_tensor   # (2,) velocity
                    }
                else:
                    # Multiple observation case (for temporal modeling)
                    observations = []
                    for obs_images, obs_states in obs_history:
                        obs_images_tensor = torch.from_numpy(obs_images).float() / 255.0
                        obs_states_tensor = torch.from_numpy(obs_states).float()
                        
                        observations.append({
                            'image': obs_images_tensor,
                            'state': obs_states_tensor
                        })
                
                # Convert actions to tensor
                actions_tensor = torch.from_numpy(actions).long()
                
                # Create mask (no padding for now)
                mask = torch.ones(self.action_chunk_size, dtype=torch.bool)
            
            return {
                'observations': observations,
                'actions': actions_tensor,
                'mask': mask,
            }
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def create_flatland_dataloader(h5_file_path: str,
                              history_frames: int = 1,
                              action_chunk_size: int = 1,
                              train_split: float = 0.8,
                              random_seed: int = 42,
                              max_no_op_ratio: float = 0.8,
                              stride: int = 1,
                              no_op_filter_percentage: float = 1.0,
                              batch_size: int = 32,
                              num_workers: int = 0,
                              **dataloader_kwargs) -> Tuple[Dataset, Dataset]:
    """
    Convenience function to create train and validation datasets with proper filtering.
    
    Args:
        h5_file_path: Path to the H5 file containing demonstrations
        history_frames: Number of historical observation frames
        action_chunk_size: Number of future actions to predict
        train_split: Fraction of episodes to use for training  
        random_seed: Random seed for reproducible splits
        max_no_op_ratio: Maximum allowed ratio of no-op actions in a segment
        stride: Stride between segment starts
        no_op_filter_percentage: Percentage of segments that must meet filtering criteria
        batch_size: Batch size (not used here, returned for convenience)
        num_workers: Number of data loading workers (not used here, returned for convenience)
        **dataloader_kwargs: Additional arguments passed to DataLoader (returned for convenience)
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create episode splits with filtering
    train_episode_segment_data, val_episode_segment_data = create_episode_split(
        h5_file_path=h5_file_path,
        history_frames=history_frames,
        action_chunk_size=action_chunk_size,
        train_split=train_split,
        random_seed=random_seed,
        max_no_op_ratio=max_no_op_ratio,
        stride=stride,
        no_op_filter_percentage=no_op_filter_percentage
    )
    
    # Create datasets
    train_dataset = FlatlandDataset(
        h5_file_path=h5_file_path,
        history_frames=history_frames,
        action_chunk_size=action_chunk_size,
        episode_segment_data=train_episode_segment_data,
        mode='train'
    )
    
    val_dataset = FlatlandDataset(
        h5_file_path=h5_file_path,
        history_frames=history_frames,
        action_chunk_size=action_chunk_size,
        episode_segment_data=val_episode_segment_data,
        mode='val'
    )
    
    return train_dataset, val_dataset
