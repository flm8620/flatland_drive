"""
Efficient data loader for HDF5 human demonstration data.
Provides high-speed random access for imitation learning.
"""

import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class HumanDemonstrationDataset(Dataset):
    """
    PyTorch Dataset for loading human demonstration data from HDF5 files.
    Optimized for random access patterns needed in imitation learning.
    """
    
    def __init__(self, hdf5_path: str, sequence_length: int = 1, 
                 successful_only: bool = False, device: str = 'cpu'):
        """
        Initialize the dataset.
        
        Args:
            hdf5_path: Path to the HDF5 file
            sequence_length: Length of sequences to return (1 for single steps)
            successful_only: If True, only include successful episodes
            device: Device to load tensors on
        """
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length
        self.device = device
        
        # Open HDF5 file
        self.h5file = h5py.File(hdf5_path, 'r')
        
        # Load metadata
        self.config = json.loads(self.h5file.attrs['config'])
        self.view_size = self.h5file.attrs['view_size']
        self.num_levels = self.h5file.attrs['num_levels']
        self.total_episodes = self.h5file.attrs.get('total_episodes', len(self.h5file['episodes']))
        self.total_steps = self.h5file.attrs.get('total_steps', len(self.h5file['observations']))
        
        print(f"Loaded HDF5 dataset: {self.total_episodes} episodes, {self.total_steps} steps")
        
        # Build valid step indices
        self.valid_indices = []
        self._build_valid_indices(successful_only)
        
        print(f"Valid samples: {len(self.valid_indices)}")
    
    def _build_valid_indices(self, successful_only: bool):
        """Build list of valid step indices for sampling"""
        episodes = self.h5file['episodes'][:self.total_episodes]
        
        for episode in episodes:
            if successful_only and not episode['success']:
                continue
            
            start_idx = episode['start_idx']
            length = episode['length']
            
            # Add valid starting positions for sequences
            for i in range(length - self.sequence_length + 1):
                self.valid_indices.append(start_idx + i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a sample (observation, action) pair or sequence"""
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Load data
        observations = torch.from_numpy(
            self.h5file['observations'][start_idx:end_idx]
        ).to(self.device)
        
        actions = torch.from_numpy(
            self.h5file['actions'][start_idx:end_idx]
        ).to(self.device)
        
        rewards = torch.from_numpy(
            self.h5file['rewards'][start_idx:end_idx]
        ).to(self.device)
        
        if self.sequence_length == 1:
            # Return single step
            return {
                'observation': observations[0],
                'action': actions[0],
                'reward': rewards[0]
            }
        else:
            # Return sequence
            return {
                'observations': observations,
                'actions': actions,
                'rewards': rewards
            }
    
    def get_episode_data(self, episode_idx: int):
        """Get complete episode data"""
        if episode_idx >= self.total_episodes:
            raise IndexError(f"Episode {episode_idx} not found")
        
        episode = self.h5file['episodes'][episode_idx]
        start_idx = episode['start_idx']
        length = episode['length']
        end_idx = start_idx + length
        
        return {
            'observations': torch.from_numpy(self.h5file['observations'][start_idx:end_idx]),
            'actions': torch.from_numpy(self.h5file['actions'][start_idx:end_idx]),
            'rewards': torch.from_numpy(self.h5file['rewards'][start_idx:end_idx]),
            'episode_reward': float(episode['reward']),
            'success': bool(episode['success']),
            'timestamp': episode['timestamp'].decode('utf-8')
        }
    
    def get_statistics(self):
        """Get dataset statistics"""
        episodes = self.h5file['episodes'][:self.total_episodes]
        
        total_reward = np.sum([ep['reward'] for ep in episodes])
        success_count = np.sum([ep['success'] for ep in episodes])
        episode_lengths = [ep['length'] for ep in episodes]
        
        actions = self.h5file['actions'][:self.total_steps]
        action_counts = np.bincount(actions, minlength=9)
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'success_rate': success_count / self.total_episodes,
            'avg_episode_reward': total_reward / self.total_episodes,
            'avg_episode_length': np.mean(episode_lengths),
            'episode_length_std': np.std(episode_lengths),
            'action_distribution': action_counts / np.sum(action_counts)
        }
    
    def close(self):
        """Close the HDF5 file"""
        if hasattr(self, 'h5file'):
            self.h5file.close()
    
    def __del__(self):
        self.close()


class HumanDataAnalyzer:
    """Utility class for analyzing human demonstration data"""
    
    def __init__(self, hdf5_path: str):
        self.dataset = HumanDemonstrationDataset(hdf5_path)
        self.action_names = [
            "No Action", "Up", "Up-Right", "Right", "Down-Right",
            "Down", "Down-Left", "Left", "Up-Left"
        ]
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.dataset.get_statistics()
        
        print("=== Dataset Statistics ===")
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average episode reward: {stats['avg_episode_reward']:.2f}")
        print(f"Average episode length: {stats['avg_episode_length']:.1f} ± {stats['episode_length_std']:.1f}")
        print()
        
        print("Action distribution:")
        for i, (action_name, prob) in enumerate(zip(self.action_names, stats['action_distribution'])):
            print(f"  {i}: {action_name:<12} {prob:.3f} ({prob*100:.1f}%)")
    
    def plot_learning_curves(self, window_size: int = 50):
        """Plot learning curves from human data"""
        episode_rewards = []
        success_rates = []
        
        for i in range(self.dataset.total_episodes):
            episode_data = self.dataset.get_episode_data(i)
            episode_rewards.append(episode_data['episode_reward'])
            success_rates.append(1.0 if episode_data['success'] else 0.0)
        
        # Compute moving averages
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        rewards_ma = moving_average(episode_rewards, window_size)
        success_ma = moving_average(success_rates, window_size)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Reward plot
        ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
        ax1.plot(range(window_size-1, len(episode_rewards)), rewards_ma, 
                label=f'Moving Average ({window_size})', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Human Player Reward Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rate plot
        ax2.plot(success_rates, alpha=0.3, label='Success (0/1)')
        ax2.plot(range(window_size-1, len(success_rates)), success_ma, 
                label=f'Success Rate MA ({window_size})', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Human Player Success Rate Over Time')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                         successful_only: bool = False, sequence_length: int = 1):
        """Create a PyTorch DataLoader for training"""
        dataset = HumanDemonstrationDataset(
            self.dataset.hdf5_path, 
            sequence_length=sequence_length,
            successful_only=successful_only
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # HDF5 doesn't work well with multiprocessing
            pin_memory=True
        )


def merge_hdf5_files(input_files: List[str], output_file: str):
    """Merge multiple HDF5 human demonstration files into one"""
    if not input_files:
        print("No input files provided")
        return
    
    print(f"Merging {len(input_files)} HDF5 files into {output_file}")
    
    # Open first file to get structure and metadata
    first_file = h5py.File(input_files[0], 'r')
    view_size = first_file.attrs['view_size']
    num_levels = first_file.attrs['num_levels']
    obs_shape = (num_levels, 4, view_size, view_size)
    
    # Count total size needed
    total_steps = 0
    total_episodes = 0
    
    for filepath in input_files:
        with h5py.File(filepath, 'r') as f:
            total_steps += f.attrs.get('total_steps', len(f['observations']))
            total_episodes += f.attrs.get('total_episodes', len(f['episodes']))
    
    print(f"Total steps: {total_steps}, Total episodes: {total_episodes}")
    
    # Create output file
    with h5py.File(output_file, 'w') as out_f:
        # Create datasets
        obs_ds = out_f.create_dataset('observations', 
                                     shape=(total_steps, *obs_shape),
                                     dtype=np.float32, chunks=True, compression='gzip')
        act_ds = out_f.create_dataset('actions', 
                                     shape=(total_steps,),
                                     dtype=np.int32, chunks=True, compression='gzip')
        rew_ds = out_f.create_dataset('rewards', 
                                     shape=(total_steps,),
                                     dtype=np.float32, chunks=True, compression='gzip')
        
        episode_dtype = first_file['episodes'].dtype
        ep_ds = out_f.create_dataset('episodes',
                                    shape=(total_episodes,),
                                    dtype=episode_dtype, chunks=True)
        
        # Copy data
        step_offset = 0
        episode_offset = 0
        
        for filepath in input_files:
            print(f"Processing {filepath}...")
            with h5py.File(filepath, 'r') as in_f:
                file_steps = in_f.attrs.get('total_steps', len(in_f['observations']))
                file_episodes = in_f.attrs.get('total_episodes', len(in_f['episodes']))
                
                # Copy step data
                obs_ds[step_offset:step_offset+file_steps] = in_f['observations'][:file_steps]
                act_ds[step_offset:step_offset+file_steps] = in_f['actions'][:file_steps]
                rew_ds[step_offset:step_offset+file_steps] = in_f['rewards'][:file_steps]
                
                # Copy and adjust episode data
                episodes = in_f['episodes'][:file_episodes].copy()
                # Adjust start indices
                episodes['start_idx'] += step_offset
                ep_ds[episode_offset:episode_offset+file_episodes] = episodes
                
                step_offset += file_steps
                episode_offset += file_episodes
        
        # Set attributes
        out_f.attrs['view_size'] = view_size
        out_f.attrs['num_levels'] = num_levels
        out_f.attrs['total_steps'] = total_steps
        out_f.attrs['total_episodes'] = total_episodes
        out_f.attrs['merged_from'] = [os.path.basename(f) for f in input_files]
        
        # Use config from first file
        out_f.attrs['config'] = first_file.attrs['config']
    
    first_file.close()
    print(f"Successfully merged into {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hdf5_data_loader.py <hdf5_file>")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    if not os.path.exists(hdf5_file):
        print(f"File not found: {hdf5_file}")
        sys.exit(1)
    
    # Analyze the data
    analyzer = HumanDataAnalyzer(hdf5_file)
    analyzer.print_statistics()
    
    # Example: Create a dataloader
    print("\n=== Creating DataLoader Example ===")
    dataloader = analyzer.create_dataloader(batch_size=32, successful_only=True)
    
    # Test loading a batch
    for batch in dataloader:
        print(f"Batch shape - Observations: {batch['observation'].shape}, Actions: {batch['action'].shape}")
        break
    
    # Plot learning curves if requested
    if len(sys.argv) > 2 and sys.argv[2] == '--plot':
        analyzer.plot_learning_curves()
