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
    # obs_level shape: (4, view_size, view_size)
    # Channel 0: cost map, Channel 3: target map
    cost = obs_level[0]
    target_map = obs_level[3]
    
    # Normalize cost: -1 (drivable) -> 0 (black), 1+ (wall) -> 255 (white)
    cost_norm = np.clip((cost + 1) / 2, 0, 1)
    img = np.stack([cost_norm, cost_norm, cost_norm], axis=-1)
    
    # Overlay target as magenta
    target_mask = target_map > 0.5
    img[target_mask, 0] = 1.0  # Blue in BGR
    img[target_mask, 1] = 0.0  # Green
    img[target_mask, 2] = 1.0  # Red in BGR
    
    # Convert to 0-255 range
    img = (img * 255).astype(np.uint8)
    
    # Resize
    img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)
    
    # Draw agent center (green circle)
    center = canvas_size // 2
    cv2.circle(img, (center, center), 3, (0, 255, 0), 2)
    
    return img


def episode_to_video(filepath, episode_idx, output_path, fps=5):
    """
    Convert episode to video with three observation levels side by side.
    
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
        
        # Load episode data
        observations = h5file['observations'][start_idx:start_idx + length]
        actions = h5file['actions'][start_idx:start_idx + length]
        rewards = h5file['rewards'][start_idx:start_idx + length]
        
        print(f"Converting Episode {episode_idx}:")
        print(f"  Steps: {length}")
        print(f"  Total Reward: {episode['reward']:.2f}")
        print(f"  Success: {episode['success']}")
        
        # Action names for display
        action_names = [
            "No Action", "Up", "Up-Right", "Right", "Down-Right",
            "Down", "Down-Left", "Left", "Up-Left"
        ]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        canvas_size = 200
        frame_width = canvas_size * 3  # Three levels side by side
        frame_height = canvas_size + 50  # Extra space for text
        
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return False
        
        # Process each step
        for step in range(length):
            obs = observations[step]  # Shape: (num_levels, 4, view_size, view_size)
            action = actions[step]
            reward = rewards[step]
            
            # Create images for each level
            level_images = []
            for level in range(min(3, obs.shape[0])):  # Up to 3 levels
                img = obs_level_to_image(obs[level], canvas_size)
                
                # Add level label
                cv2.putText(img, f"Level {level}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                level_images.append(img)
            
            # Pad with black images if less than 3 levels
            while len(level_images) < 3:
                black_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
                cv2.putText(black_img, "N/A", (canvas_size//2-20, canvas_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                level_images.append(black_img)
            
            # Concatenate levels horizontally
            frame_top = np.concatenate(level_images, axis=1)
            
            # Create bottom section with step info
            info_section = np.zeros((50, frame_width, 3), dtype=np.uint8)
            
            # Add step information
            action_name = action_names[action] if action < len(action_names) else f"Action {action}"
            text_info = f"Step: {step}/{length-1}  Action: {action_name}  Reward: {reward:.2f}"
            
            cv2.putText(info_section, text_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine top and bottom sections
            frame = np.concatenate([frame_top, info_section], axis=0)
            
            # Write frame to video
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Analyze human trajectory data')
    parser.add_argument('data_file', help='Path to human data HDF5 file (.h5)')
    parser.add_argument('--episode-video', type=int, metavar='N', 
                       help='Convert episode N to video')
    parser.add_argument('--output', '-o', default='episode_video.mp4',
                       help='Output video filename (default: episode_video.mp4)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Video framerate (default: 5)')
    
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
        if args.episode_video is not None:
            # Convert episode to video
            success = episode_to_video(args.data_file, args.episode_video, 
                                     args.output, args.fps)
            if not success:
                return
        else:
            # Show summary
            summary = analyze_summary(args.data_file)
            print_summary(summary)
            
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
