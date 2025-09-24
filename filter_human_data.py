#!/usr/bin/env python3
"""
Filter human data to remove bad episodes.
Removes failure episodes and episodes that are too short.
"""

import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm


def filter_human_data(input_file, output_file, min_steps=20, keep_failures=False):
    """
    Filter human data episodes based on quality criteria.
    
    Args:
        input_file (str): Path to input HDF5 file
        output_file (str): Path to output HDF5 file
        min_steps (int): Minimum episode length to keep
        keep_failures (bool): Whether to keep failure episodes
    """
    
    print(f"Filtering human data from {input_file}")
    print(f"Criteria: min_steps={min_steps}, keep_failures={keep_failures}")
    
    with h5py.File(input_file, 'r') as input_h5:
        # Read episode metadata
        episodes = input_h5['episodes'][:]
        total_episodes = len(episodes)
        
        print(f"Total episodes in input: {total_episodes}")
        
        # Filter episodes based on criteria
        good_episode_indices = []
        for i, episode in enumerate(episodes):
            episode_length = episode['length']
            is_success = episode['success']
            
            # Check length criterion
            if episode_length < min_steps:
                continue
            
            # Check success criterion
            if not keep_failures and not is_success:
                continue
            
            good_episode_indices.append(i)
        
        print(f"Episodes passing filter: {len(good_episode_indices)} / {total_episodes}")
        
        if len(good_episode_indices) == 0:
            print("No episodes pass the filter criteria!")
            return
        
        # Calculate total steps in filtered episodes
        good_episodes = episodes[good_episode_indices]
        total_filtered_steps = sum(episode['length'] for episode in good_episodes)
        
        print(f"Total steps in filtered episodes: {total_filtered_steps}")
        
        # Create output file
        with h5py.File(output_file, 'w') as output_h5:
            # Copy metadata attributes
            for attr_name, attr_value in input_h5.attrs.items():
                output_h5.attrs[attr_name] = attr_value
            
            # Update metadata for filtered data
            output_h5.attrs['total_episodes'] = len(good_episodes)
            output_h5.attrs['total_steps'] = total_filtered_steps
            output_h5.attrs['filtered_from'] = input_file.encode('utf-8')
            output_h5.attrs['filter_criteria'] = f"min_steps={min_steps}, keep_failures={keep_failures}".encode('utf-8')
            
            # Get shapes from input data
            obs_shape = input_h5['observations'].shape[1:]  # Remove first dimension
            
            # Create output datasets
            chunk_size = min(1000, total_filtered_steps)
            
            obs_dataset = output_h5.create_dataset(
                'observations',
                shape=(total_filtered_steps, *obs_shape),
                dtype=input_h5['observations'].dtype,
                chunks=(chunk_size, *obs_shape),
                compression='gzip',
                compression_opts=1
            )
            
            actions_dataset = output_h5.create_dataset(
                'actions',
                shape=(total_filtered_steps,),
                dtype=input_h5['actions'].dtype,
                chunks=(chunk_size,),
                compression='gzip',
                compression_opts=1
            )
            
            rewards_dataset = output_h5.create_dataset(
                'rewards',
                shape=(total_filtered_steps,),
                dtype=input_h5['rewards'].dtype,
                chunks=(chunk_size,),
                compression='gzip',
                compression_opts=1
            )
            
            episodes_dataset = output_h5.create_dataset(
                'episodes',
                shape=(len(good_episodes),),
                dtype=input_h5['episodes'].dtype,
                chunks=(min(100, len(good_episodes)),)
            )
            
            # Copy filtered data
            output_step_idx = 0
            
            print("Copying filtered episodes...")
            for output_episode_idx, input_episode_idx in enumerate(tqdm(good_episode_indices)):
                episode = episodes[input_episode_idx]
                start_idx = episode['start_idx']
                length = episode['length']
                end_idx = start_idx + length
                
                # Copy step data
                obs_dataset[output_step_idx:output_step_idx + length] = \
                    input_h5['observations'][start_idx:end_idx]
                actions_dataset[output_step_idx:output_step_idx + length] = \
                    input_h5['actions'][start_idx:end_idx]
                rewards_dataset[output_step_idx:output_step_idx + length] = \
                    input_h5['rewards'][start_idx:end_idx]
                
                # Update episode metadata with new start index
                new_episode = np.array([
                    (
                        output_step_idx,  # New start index
                        length,
                        episode['reward'],
                        episode['success'],
                        episode['timestamp']
                    )
                ], dtype=episodes_dataset.dtype)
                
                episodes_dataset[output_episode_idx] = new_episode[0]
                
                output_step_idx += length
    
    print(f"Filtered data saved to {output_file}")
    
    # Print summary
    print("\n=== Filtering Summary ===")
    with h5py.File(output_file, 'r') as output_h5:
        episodes = output_h5['episodes'][:]
        success_count = sum(episode['success'] for episode in episodes)
        failure_count = len(episodes) - success_count
        
        print(f"Output episodes: {len(episodes)}")
        print(f"  Success: {success_count}")
        print(f"  Failure: {failure_count}")
        print(f"Output steps: {len(output_h5['observations'])}")
        
        # Episode length statistics
        lengths = [episode['length'] for episode in episodes]
        print(f"Episode length stats:")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Min: {np.min(lengths)}")
        print(f"  Max: {np.max(lengths)}")
        print(f"  Std: {np.std(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Filter human demonstration data')
    parser.add_argument('input_file', help='Input HDF5 file path')
    parser.add_argument('output_file', help='Output HDF5 file path')
    parser.add_argument('--min-steps', type=int, default=20, 
                       help='Minimum episode length to keep (default: 20)')
    parser.add_argument('--keep-failures', action='store_true',
                       help='Keep failure episodes (default: remove them)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    # Check if output file exists and confirm overwrite
    if os.path.exists(args.output_file):
        response = input(f"Output file {args.output_file} exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        filter_human_data(args.input_file, args.output_file, 
                         args.min_steps, args.keep_failures)
        print("Filtering completed successfully!")
        
    except Exception as e:
        print(f"Error during filtering: {e}")
        # Clean up incomplete output file
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
            print("Cleaned up incomplete output file")


if __name__ == '__main__':
    main()
