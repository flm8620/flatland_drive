"""
Filter and combine human play H5 files.

This script can:
1. Filter episodes based on success and minimum length
2. Combine multiple H5 files into a single file
3. Preserve the original data structure and metadata

Usage:
    python filter_human_data.py input1.h5 [input2.h5 ...] --output output.h5 [--min-length 30] [--success-only]
"""

import argparse
import h5py
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional


def load_episodes_info(h5_file: h5py.File) -> List[dict]:
    """Load episode information from H5 file."""
    episodes_dataset = h5_file['episodes']
    episodes_info = []
    
    for i in range(len(episodes_dataset)):
        episode = episodes_dataset[i]
        episodes_info.append({
            'index': i,
            'start_idx': episode['start_idx'],
            'length': episode['length'],
            'reward': episode['reward'],
            'success': bool(episode['success']),
            'timestamp': episode['timestamp'].decode('utf-8') if isinstance(episode['timestamp'], bytes) else episode['timestamp']
        })
    
    return episodes_info


def filter_episodes(episodes_info: List[dict], min_length: int = 30, success_only: bool = False) -> List[dict]:
    """Filter episodes based on criteria."""
    filtered = []
    
    for episode in episodes_info:
        # Filter by length
        if episode['length'] < min_length:
            continue
        
        # Filter by success if requested
        if success_only and not episode['success']:
            continue
        
        filtered.append(episode)
    
    return filtered


def copy_filtered_data(input_files: List[str], output_file: str, min_length: int = 30, success_only: bool = False):
    """Copy filtered data from input files to output file."""
    
    print(f"Processing {len(input_files)} input file(s)...")
    print(f"Filters: min_length={min_length}, success_only={success_only}")
    
    # Collect all episodes and data info
    all_episodes = []
    total_original_episodes = 0
    total_original_steps = 0
    
    # First pass: collect episode information from all files
    file_episodes = []
    file_handles = []
    
    try:
        for i, input_file in enumerate(input_files):
            print(f"\nAnalyzing file {i+1}/{len(input_files)}: {input_file}")
            
            h5f = h5py.File(input_file, 'r')
            file_handles.append(h5f)
            
            episodes_info = load_episodes_info(h5f)
            original_count = len(episodes_info)
            original_steps = sum(ep['length'] for ep in episodes_info)
            
            print(f"  Original episodes: {original_count}, steps: {original_steps}")
            
            # Filter episodes
            filtered_episodes = filter_episodes(episodes_info, min_length, success_only)
            filtered_count = len(filtered_episodes)
            filtered_steps = sum(ep['length'] for ep in filtered_episodes)
            
            print(f"  Filtered episodes: {filtered_count}, steps: {filtered_steps}")
            print(f"  Kept: {filtered_count}/{original_count} episodes ({100*filtered_count/original_count:.1f}%)")
            
            if filtered_count == original_count:
                print(f"    ⚠️  No episodes were filtered out from this file!")
            else:
                dropped = original_count - filtered_count
                print(f"    ✅ Dropped {dropped} episodes that didn't meet criteria")
            
            # Add file reference to each episode
            for episode in filtered_episodes:
                episode['file_index'] = i
            
            file_episodes.append(filtered_episodes)
            all_episodes.extend(filtered_episodes)
            
            total_original_episodes += original_count
            total_original_steps += original_steps
        
        # Summary
        total_filtered_episodes = len(all_episodes)
        total_filtered_steps = sum(ep['length'] for ep in all_episodes)
        
        print(f"\n=== Summary ===")
        print(f"Total original: {total_original_episodes} episodes, {total_original_steps} steps")
        print(f"Total filtered: {total_filtered_episodes} episodes, {total_filtered_steps} steps")
        print(f"Overall kept: {total_filtered_episodes}/{total_original_episodes} episodes ({100*total_filtered_episodes/total_original_episodes:.1f}%)")
        
        if total_filtered_episodes == total_original_episodes:
            print("⚠️  WARNING: No episodes were filtered out! Output file will be same size as input.")
            print("   Check your filter criteria (--min-length, --success-only)")
        
        if total_filtered_episodes == 0:
            print("No episodes passed the filters. Output file will not be created.")
            return
        
        # Get metadata from first file
        first_file = file_handles[0]
        view_size = first_file.attrs.get('view_size', 64)
        num_levels = first_file.attrs.get('num_levels', 3)
        
        # Verify all files have compatible shapes
        image_shape = (num_levels, 2, view_size, view_size)
        state_shape = (2,)
        
        for i, h5f in enumerate(file_handles):
            if h5f['obs_images'].shape[1:] != image_shape:
                raise ValueError(f"File {i} has incompatible image shape: {h5f['obs_images'].shape[1:]} vs {image_shape}")
            if h5f['obs_states'].shape[1:] != state_shape:
                raise ValueError(f"File {i} has incompatible state shape: {h5f['obs_states'].shape[1:]} vs {state_shape}")
            if h5f['agent_positions'].shape[1:] != (2,):
                raise ValueError(f"File {i} has incompatible agent_positions shape: {h5f['agent_positions'].shape[1:]} vs (2,)")
        
        print(f"\nCreating output file: {output_file}")
        
        # Create output file
        with h5py.File(output_file, 'w') as out_file:
            # Create datasets
            chunk_size = 1
            
            # Image observations
            obs_images_dataset = out_file.create_dataset(
                'obs_images',
                shape=(total_filtered_steps, *image_shape),
                dtype=np.uint8,
                chunks=(chunk_size, *image_shape),
                compression='gzip',
                compression_opts=1
            )
            
            # State observations
            obs_states_dataset = out_file.create_dataset(
                'obs_states',
                shape=(total_filtered_steps, *state_shape),
                dtype=np.float32,
                chunks=(chunk_size, *state_shape)
            )
            
            # Actions
            actions_dataset = out_file.create_dataset(
                'actions',
                shape=(total_filtered_steps,),
                dtype=np.int32,
                chunks=(chunk_size,)
            )
            
            # Rewards
            rewards_dataset = out_file.create_dataset(
                'rewards',
                shape=(total_filtered_steps,),
                dtype=np.float32,
                chunks=(chunk_size,)
            )
            
            # Agent positions
            agent_positions_dataset = out_file.create_dataset(
                'agent_positions',
                shape=(total_filtered_steps, 2),
                dtype=np.float32,
                chunks=(chunk_size, 2)
            )
            
            # Episodes metadata (updated with world state data)
            episodes_chunk_size = min(100, total_filtered_episodes)  # Chunk size can't exceed dataset size
            episodes_dataset = out_file.create_dataset(
                'episodes',
                shape=(total_filtered_episodes,),
                dtype=[
                    ('start_idx', 'i4'),      # Starting index in the main arrays
                    ('length', 'i4'),         # Episode length
                    ('reward', 'f4'),         # Total episode reward
                    ('success', 'b1'),        # Success flag
                    ('timestamp', 'S20'),     # Timestamp string
                    ('world_map', 'f4', (512, 512)),  # World cost map for this episode
                    ('start_pos', 'i4', (2,)),        # Start position [x, y]
                    ('target_pos', 'i4', (2,))        # Target position [x, y]
                ],
                chunks=(episodes_chunk_size,) if episodes_chunk_size > 0 else None
            )
            
            # Copy data
            current_step_idx = 0
            current_episode_idx = 0
            
            print("Copying data...")
            
            for file_idx, episodes in enumerate(file_episodes):
                if not episodes:
                    continue
                
                print(f"  Processing file {file_idx + 1}/{len(file_episodes)}: {len(episodes)} episodes")
                h5f = file_handles[file_idx]
                
                for ep_idx, episode in enumerate(episodes):
                    # Get data range from original file
                    start_idx = episode['start_idx']
                    end_idx = start_idx + episode['length']
                    
                    # Copy step data
                    new_end_idx = current_step_idx + episode['length']
                    
                    obs_images_dataset[current_step_idx:new_end_idx] = h5f['obs_images'][start_idx:end_idx]
                    obs_states_dataset[current_step_idx:new_end_idx] = h5f['obs_states'][start_idx:end_idx]
                    actions_dataset[current_step_idx:new_end_idx] = h5f['actions'][start_idx:end_idx]
                    rewards_dataset[current_step_idx:new_end_idx] = h5f['rewards'][start_idx:end_idx]
                    agent_positions_dataset[current_step_idx:new_end_idx] = h5f['agent_positions'][start_idx:end_idx]
                    
                    # Get original episode record to extract world state data
                    original_episode = h5f['episodes'][episode['index']]
                    
                    # Create episode record with world state data
                    episode_record = (
                        current_step_idx,  # new start_idx
                        episode['length'],
                        episode['reward'],
                        episode['success'],
                        episode['timestamp'].encode('utf-8') if isinstance(episode['timestamp'], str) else episode['timestamp'],
                        original_episode['world_map'],  # Copy world map from original episode
                        original_episode['start_pos'],  # Copy start position from original episode  
                        original_episode['target_pos']  # Copy target position from original episode
                    )
                    
                    episodes_dataset[current_episode_idx] = episode_record
                    
                    current_step_idx = new_end_idx
                    current_episode_idx += 1
                    
                    if (ep_idx + 1) % 50 == 0 or (ep_idx + 1) == len(episodes):
                        print(f"    Copied {ep_idx + 1}/{len(episodes)} episodes from file {file_idx + 1}")
            
            # Set metadata
            out_file.attrs['view_size'] = view_size
            out_file.attrs['num_levels'] = num_levels
            out_file.attrs['total_episodes'] = total_filtered_episodes
            out_file.attrs['total_steps'] = current_step_idx  # Use actual steps written
            out_file.attrs['created_time'] = time.strftime("%Y%m%d_%H%M%S")
            out_file.attrs['filter_min_length'] = min_length
            out_file.attrs['filter_success_only'] = success_only
            out_file.attrs['source_files'] = json.dumps(input_files)
            
            # Trim datasets to actual size used
            if current_step_idx < total_filtered_steps:
                print(f"Trimming datasets from {total_filtered_steps} to {current_step_idx} steps")
                obs_images_dataset.resize((current_step_idx, *image_shape))
                obs_states_dataset.resize((current_step_idx, *state_shape))
                actions_dataset.resize((current_step_idx,))
                rewards_dataset.resize((current_step_idx,))
                agent_positions_dataset.resize((current_step_idx, 2))
            
            if current_episode_idx < total_filtered_episodes:
                print(f"Trimming episodes from {total_filtered_episodes} to {current_episode_idx}")
                episodes_dataset.resize((current_episode_idx,))
            
            # Copy config from first file if available
            if 'config' in first_file.attrs:
                out_file.attrs['config'] = first_file.attrs['config']
        
        print(f"\nOutput file created successfully: {output_file}")
        print(f"Final dataset: {current_episode_idx} episodes, {current_step_idx} steps")
    
    finally:
        # Close all file handles
        for h5f in file_handles:
            h5f.close()


def print_file_info(filename: str):
    """Print information about an H5 file."""
    try:
        with h5py.File(filename, 'r') as h5f:
            episodes_info = load_episodes_info(h5f)
            
            total_episodes = len(episodes_info)
            total_steps = sum(ep['length'] for ep in episodes_info)
            success_count = sum(1 for ep in episodes_info if ep['success'])
            
            lengths = [ep['length'] for ep in episodes_info]
            min_length = min(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0
            avg_length = np.mean(lengths) if lengths else 0
            
            print(f"\nFile: {filename}")
            print(f"  Episodes: {total_episodes}")
            print(f"  Steps: {total_steps}")
            print(f"  Success rate: {success_count}/{total_episodes} ({100*success_count/total_episodes:.1f}%)")
            print(f"  Episode length: min={min_length}, max={max_length}, avg={avg_length:.1f}")
            
            # Show distribution of episode lengths
            short_episodes = sum(1 for length in lengths if length < 30)
            if short_episodes > 0:
                print(f"  Short episodes (< 30 steps): {short_episodes} ({100*short_episodes/total_episodes:.1f}%)")
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter and combine human play H5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter a single file, removing failed episodes and episodes < 30 steps
  python filter_human_data.py input.h5 --output filtered.h5
  
  # Combine multiple files, keeping only successful episodes
  python filter_human_data.py file1.h5 file2.h5 file3.h5 --output combined.h5 --success-only
  
  # Set custom minimum episode length
  python filter_human_data.py input.h5 --output filtered.h5 --min-length 50
  
  # Just show info about files without filtering
  python filter_human_data.py input1.h5 input2.h5 --info-only
        """
    )
    
    parser.add_argument('input_files', nargs='+', help='Input H5 files to process')
    parser.add_argument('--output', '-o', help='Output H5 file (required unless --info-only)')
    parser.add_argument('--min-length', type=int, default=30, help='Minimum episode length (default: 30)')
    parser.add_argument('--success-only', action='store_true', help='Keep only successful episodes')
    parser.add_argument('--info-only', action='store_true', help='Only show file information, do not create output')
    
    args = parser.parse_args()
    
    # Validate input files
    for input_file in args.input_files:
        if not Path(input_file).exists():
            print(f"Error: Input file does not exist: {input_file}")
            return 1
    
    if args.info_only:
        print("=== File Information ===")
        for input_file in args.input_files:
            print_file_info(input_file)
        return 0
    
    if not args.output:
        print("Error: --output is required unless using --info-only")
        return 1
    
    # Check if output file already exists
    if Path(args.output).exists():
        response = input(f"Output file {args.output} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    try:
        copy_filtered_data(
            args.input_files,
            args.output,
            min_length=args.min_length,
            success_only=args.success_only
        )
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
