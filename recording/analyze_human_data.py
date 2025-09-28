"""
Clean analysis script for human-recorded HDF5 data.

Provides two main functionalities:
1. Summary statistics (episodes, steps, rewards, outcomes)
2. Episode video generation (three observation levels side by side)
"""

import h5py
import numpy as np
import cv2
import argparse
from pathlib import Path


def print_h5_structure(filepath, max_shape_display=5):
    """
    Print the structure of an HDF5 file including groups, datasets, and their shapes.
    
    Args:
        filepath: Path to HDF5 file
        max_shape_display: Maximum number of shape dimensions to display
    """
    def print_item(name, obj, indent=0):
        """Recursively print HDF5 items"""
        spaces = "  " * indent
        if isinstance(obj, h5py.Group):
            print(f"{spaces}📁 {name}/ (Group)")
            # Print group attributes
            if obj.attrs:
                for attr_name, attr_val in obj.attrs.items():
                    print(f"{spaces}    @{attr_name}: {attr_val}")
        elif isinstance(obj, h5py.Dataset):
            shape_str = str(obj.shape)
            if len(obj.shape) > max_shape_display:
                shape_str = f"({', '.join(map(str, obj.shape[:max_shape_display]))}, ...)"
            dtype_str = str(obj.dtype)
            size_mb = obj.nbytes / (1024 * 1024)
            print(f"{spaces}📄 {name} → shape{shape_str}, dtype={dtype_str}, size={size_mb:.2f}MB")
            
            # Print dataset attributes
            if obj.attrs:
                for attr_name, attr_val in obj.attrs.items():
                    print(f"{spaces}    @{attr_name}: {attr_val}")
    
    print(f"\n=== HDF5 FILE STRUCTURE ===")
    print(f"File: {filepath}")
    
    with h5py.File(filepath, 'r') as h5file:
        # Print file-level attributes
        if h5file.attrs:
            print("📋 File Attributes:")
            for attr_name, attr_val in h5file.attrs.items():
                print(f"  @{attr_name}: {attr_val}")
            print()
        
        # Print structure
        print("📂 Structure:")
        h5file.visititems(print_item)
        
        # File size
        import os
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\n💾 Total file size: {file_size_mb:.2f}MB")
    print()


def analyze_summary(filepath):
    """
    Analyze HDF5 file and provide episode summary.
    
    Returns:
        dict: Summary with episode count, step counts, discounted rewards, and outcomes
    """
    with h5py.File(filepath, 'r') as h5file:
        # Get metadata
        total_episodes = h5file.attrs.get('total_episodes', len(h5file['episodes']))
        episodes = h5file['episodes'][:total_episodes]
        
        # Extract episode information
        episode_info = []
        for i, episode in enumerate(episodes):
            info = {
                'episode': i,
                'steps': int(episode['length']),
                'discounted_reward': float(episode['reward']),
                'success': bool(episode['success']),
                'timestamp': episode['timestamp'].decode('utf-8')
            }
            episode_info.append(info)
        
        summary = {
            'total_episodes': total_episodes,
            'episodes': episode_info
        }
        
        return summary


def print_summary(summary):
    """Print formatted summary to console"""
    print(f"\n=== HUMAN DATA SUMMARY ===")
    print(f"Total Episodes: {summary['total_episodes']}")
    print(f"\nEpisode Details:")
    print(f"{'Ep':>3} | {'Steps':>5} | {'Reward':>8} | {'Outcome':>8} | Timestamp")
    print("-" * 60)
    
    success_count = 0
    total_steps = 0
    total_reward = 0.0
    
    for ep in summary['episodes']:
        outcome = "SUCCESS" if ep['success'] else "FAIL"
        if ep['success']:
            success_count += 1
        total_steps += ep['steps']
        total_reward += ep['discounted_reward']
        
        print(f"{ep['episode']:>3} | {ep['steps']:>5} | {ep['discounted_reward']:>8.2f} | {outcome:>8} | {ep['timestamp']}")
    
    print("-" * 60)
    print(f"Success Rate: {success_count}/{summary['total_episodes']} ({100*success_count/summary['total_episodes']:.1f}%)")
    print(f"Total Steps: {total_steps}")
    print(f"Average Steps: {total_steps/summary['total_episodes']:.1f}")
    print(f"Average Reward: {total_reward/summary['total_episodes']:.2f}")


def obs_level_to_image(obs_level, canvas_size=200):
    """Convert observation level to BGR image for visualization"""
    # obs_level shape: (2, 64, 64) - uint8
    # Channel 0: cost map (0=drivable/black, 255=wall/white)
    # Channel 1: target map (0=nothing, 128=agent center, 255=goal)
    cost = obs_level[0].astype(np.float32)  # (64, 64)
    target_map = obs_level[1].astype(np.float32)  # (64, 64)
    
    # Create base grayscale image from cost map
    # cost: 0 (drivable) -> black, 255 (wall) -> white
    cost_norm = cost / 255.0  # 0..1
    img = np.stack([cost_norm, cost_norm, cost_norm], axis=-1)  # (64, 64, 3)
    
    # Overlay target markers with transparency
    # Agent center (value 128): green overlay
    agent_mask = target_map == 128
    if np.any(agent_mask):
        img[agent_mask, 0] = 0.0   # Blue = 0
        img[agent_mask, 1] = 1.0   # Green = 1  
        img[agent_mask, 2] = 0.0   # Red = 0
    
    # Goal (value 255): magenta overlay
    goal_mask = target_map == 255
    if np.any(goal_mask):
        img[goal_mask, 0] = 1.0   # Blue = 1
        img[goal_mask, 1] = 0.0   # Green = 0
        img[goal_mask, 2] = 1.0   # Red = 1
    
    # Convert to 0-255 range
    img = (img * 255).astype(np.uint8)
    
    # Resize to canvas size
    img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)
    
    return img


def visualize_world_trajectory(filepath, episode_idx, output_path):
    """
    Create a visualization of the full world map with agent trajectory.
    
    Args:
        filepath: Path to HDF5 file
        episode_idx: Episode index to visualize
        output_path: Output image file path
    """
    with h5py.File(filepath, 'r') as h5file:
        total_episodes = h5file.attrs.get('total_episodes', len(h5file['episodes']))
        
        if episode_idx >= total_episodes:
            print(f"Error: Episode {episode_idx} not found. Available episodes: 0-{total_episodes-1}")
            return False
        
        # Get episode info
        episode = h5file['episodes'][episode_idx]
        start_idx = episode['start_idx']
        length = episode['length']
        
        # Load trajectory data
        agent_positions = h5file['agent_positions'][start_idx:start_idx + length]  # Shape: (length, 2)
        
        # Get episode-level world state data
        world_map = h5file['world_maps'][episode_idx]  # Shape: (512, 512)
        start_pos = episode['start_pos']  # Shape: (2,)
        target_pos = episode['target_pos']  # Shape: (2,)
        
        # Create visualization
        # Convert world map to RGB (cost map: 0=drivable/black, 1=wall/white)
        world_rgb = np.stack([world_map, world_map, world_map], axis=-1)  # (512, 512, 3)
        world_rgb = (world_rgb * 255).astype(np.uint8)
        
        # Draw trajectory as colored line
        if length > 1:
            trajectory_points = agent_positions.astype(np.int32)
            for i in range(len(trajectory_points) - 1):
                pt1 = tuple(trajectory_points[i])
                pt2 = tuple(trajectory_points[i + 1])
                # Draw trajectory in blue
                cv2.line(world_rgb, pt1, pt2, (255, 0, 0), 2)  # Blue line, thickness 2
        
        # Draw start position as green circle
        start_pt = tuple(start_pos.astype(np.int32))
        cv2.circle(world_rgb, start_pt, 8, (0, 255, 0), -1)  # Green filled circle
        cv2.circle(world_rgb, start_pt, 8, (0, 0, 0), 2)    # Black border
        
        # Draw target position as red circle
        target_pt = tuple(target_pos.astype(np.int32))
        cv2.circle(world_rgb, target_pt, 8, (0, 0, 255), -1)  # Red filled circle
        cv2.circle(world_rgb, target_pt, 8, (0, 0, 0), 2)    # Black border
        
        # Draw agent final position as yellow circle
        if length > 0:
            final_pos = tuple(agent_positions[-1].astype(np.int32))
            cv2.circle(world_rgb, final_pos, 6, (0, 255, 255), -1)  # Yellow filled circle
            cv2.circle(world_rgb, final_pos, 6, (0, 0, 0), 2)      # Black border
        
        # Add legend text
        legend_y = 30
        cv2.putText(world_rgb, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(world_rgb, "Green: Start", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(world_rgb, "Red: Target", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(world_rgb, "Blue: Trajectory", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(world_rgb, "Yellow: Final", (10, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add episode info
        info_text = f"Episode {episode_idx}: {length} steps, Reward: {episode['reward']:.2f}, Success: {episode['success']}"
        cv2.putText(world_rgb, info_text, (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save image
        cv2.imwrite(output_path, world_rgb)
        print(f"World trajectory visualization saved to: {output_path}")
        return True


def episode_to_video(filepath, episode_idx, output_path, fps=5):
    """
    Convert episode to video showing RGB images from the HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        episode_idx: Episode index to convert
        output_path: Output video file path
        fps: Video framerate
    """
    with h5py.File(filepath, 'r') as h5file:
        total_episodes = h5file.attrs.get('total_episodes', len(h5file['episodes']))
        
        if episode_idx >= total_episodes:
            print(f"Error: Episode {episode_idx} not found. Available episodes: 0-{total_episodes-1}")
            return False
        
        # Get episode info
        episode = h5file['episodes'][episode_idx]
        start_idx = episode['start_idx']
        length = episode['length']
        
        # Load episode data - using correct dataset names from file structure
        obs_images = h5file['obs_images'][start_idx:start_idx + length]  # Shape: (length, 3, 2, 64, 64)
        obs_states = h5file['obs_states'][start_idx:start_idx + length]  # Shape: (length, 2)
        actions = h5file['actions'][start_idx:start_idx + length]
        rewards = h5file['rewards'][start_idx:start_idx + length]
        agent_positions = h5file['agent_positions'][start_idx:start_idx + length]  # Shape: (length, 2)
        
        # Get episode-level world state data
        world_map = h5file['world_maps'][episode_idx]  # Shape: (512, 512)
        start_pos = episode['start_pos']  # Shape: (2,)
        target_pos = episode['target_pos']  # Shape: (2,)
        
        print(f"Converting Episode {episode_idx}:")
        print(f"  Steps: {length}")
        print(f"  Total Reward: {episode['reward']:.2f}")
        print(f"  Success: {episode['success']}")
        print(f"  Obs Images Shape: {obs_images.shape}")
        print(f"  Obs States Shape: {obs_states.shape}")
        print(f"  Agent Positions Shape: {agent_positions.shape}")
        print(f"  World Map Shape: {world_map.shape}")
        print(f"  Start Position: [{start_pos[0]}, {start_pos[1]}]")
        print(f"  Target Position: [{target_pos[0]}, {target_pos[1]}]")
        
        # Action names for display
        action_names = [
            "No Action", "Up", "Up-Right", "Right", "Down-Right",
            "Down", "Down-Left", "Left", "Up-Left"
        ]
        
        # Create video writer with fallback codecs for better compatibility
        canvas_size = 200
        
        # obs_images has shape (length, 3, 2, 64, 64)
        # 3 pyramid levels, 2 channels (cost map, target map), 64x64 resolution
        num_levels = obs_images.shape[1]  # Should be 3 pyramid levels
        frame_width = canvas_size * num_levels  # Three levels side by side
        frame_height = canvas_size + 80  # Extra space for text
        
        # Try different codecs in order of preference
        codecs = [
            # ('H264', 'mp4'),  # H.264 codec, best compatibility
            # ('XVID', 'avi'),  # Xvid codec, good fallback
            ('MJPG', 'avi'),  # Motion JPEG, widely supported
            ('mp4v', 'mp4'),  # Original codec as last resort
        ]
        
        video_writer = None
        final_output_path = output_path
        
        for codec_name, extension in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                # Update output path with correct extension
                base_name = output_path.rsplit('.', 1)[0]  # Remove original extension
                test_output_path = f"{base_name}.{extension}"
                
                video_writer = cv2.VideoWriter(test_output_path, fourcc, fps, (frame_width, frame_height))
                
                if video_writer.isOpened():
                    final_output_path = test_output_path
                    print(f"Using codec: {codec_name} -> {final_output_path}")
                    break
                else:
                    video_writer.release()
                    video_writer = None
            except Exception as e:
                print(f"Failed to initialize {codec_name} codec: {e}")
                continue
        
        if not video_writer or not video_writer.isOpened():
            print(f"Error: Could not initialize any video codec")
            return False
        
        # Process each step
        for step in range(length):
            obs_img = obs_images[step]  # Shape: (3, 2, 64, 64)
            obs_state = obs_states[step]  # Shape: (2,) - velocity
            action = actions[step]
            reward = rewards[step]
            
            # Create images for each pyramid level
            level_images = []
            for level in range(num_levels):
                # Extract observation level: (2, 64, 64) - cost map and target map
                obs_level = obs_img[level]  # Shape: (2, 64, 64)
                display_img = obs_level_to_image(obs_level, canvas_size)
                
                # Add level label
                cv2.putText(display_img, f"Level {level}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2)
                
                level_images.append(display_img)
            
            # Concatenate levels horizontally
            frame_top = np.concatenate(level_images, axis=1)
            
            # Create bottom section with step info
            info_section = np.zeros((80, frame_width, 3), dtype=np.uint8)
            
            # Add step information
            action_name = action_names[action] if action < len(action_names) else f"Action {action}"
            agent_pos = agent_positions[step]
            text_info1 = f"Step: {step+1}/{length}  Action: {action_name}  Reward: {reward:.2f}"
            text_info2 = f"State: [{obs_state[0]:.2f}, {obs_state[1]:.2f}]  Agent: [{agent_pos[0]:.1f}, {agent_pos[1]:.1f}]"
            text_info3 = f"Start: [{start_pos[0]}, {start_pos[1]}]  Target: [{target_pos[0]}, {target_pos[1]}]"
            
            cv2.putText(info_section, text_info1, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_section, text_info2, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_section, text_info3, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine top and bottom sections
            frame = np.concatenate([frame_top, info_section], axis=0)
            
            # Write frame to video
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to: {final_output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Analyze human trajectory data')
    parser.add_argument('data_file', help='Path to human data HDF5 file (.h5)')
    parser.add_argument('--detail', type=int, metavar='N',
                       help='Detail analysis for episode N: creates both video and world trajectory')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video framerate (default: 30)')

    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.data_file).exists():
        print(f"Error: File {args.data_file} not found")
        return
    
    # Check file extension
    if not args.data_file.endswith('.h5'):
        print(f"Error: Expected .h5 file, got {args.data_file}")
        return
    
    try:
        if args.detail is not None:
            # Detail mode: create both video and world trajectory visualization
            episode_num = args.detail
            print(f"\n=== DETAIL ANALYSIS FOR EPISODE {episode_num} ===")
            
            # Create video
            video_output = f'episode_{episode_num}_video.mp4'
            print(f"\n1. Creating episode video...")
            video_success = episode_to_video(args.data_file, episode_num, video_output, args.fps)
            
            # Create world trajectory visualization
            trajectory_output = f'episode_{episode_num}_trajectory.png'
            print(f"\n2. Creating world trajectory visualization...")
            trajectory_success = visualize_world_trajectory(args.data_file, episode_num, trajectory_output)
            
            if video_success and trajectory_success:
                print(f"\n✅ Detail analysis complete for episode {episode_num}:")
                print(f"   📹 Video: {video_output}")
                print(f"   🗺️  World map: {trajectory_output}")
            else:
                print(f"\n❌ Detail analysis failed for episode {episode_num}")
                
        else:
            # Show file structure first
            print_h5_structure(args.data_file)
            
            # Then show summary
            summary = analyze_summary(args.data_file)
            print_summary(summary)
            
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
