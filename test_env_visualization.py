#!/usr/bin/env python3
"""
Test script to create an environment and visualize the world map, start/target positions, 
and geodesic distance map as images.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects
from env import ParallelDrivingEnv
import cv2  # Still needed for the get_human_frame_from_obs function


def create_visualization(env, env_idx=0, output_dir="test_viz"):
    """
    Create visualizations for the given environment index.
    
    Args:
        env: ParallelDrivingEnv instance
        env_idx: Environment index to visualize (default: 0)
        output_dir: Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data from the environment (convert tensors to numpy)
    cost_map = env.cost_maps[env_idx].cpu().numpy()  # (H, W)
    geodesic_map = env.geodesic_dist_maps[env_idx].cpu().numpy()  # (H, W)
    start_pos = env.starts[env_idx].cpu().numpy()  # (2,) - x, y
    target_pos = env.targets[env_idx].cpu().numpy()  # (2,) - x, y
    
    print(f"Environment {env_idx} info:")
    print(f"  Map shape: {cost_map.shape}")
    print(f"  Start position: ({start_pos[0]}, {start_pos[1]})")
    print(f"  Target position: ({target_pos[0]}, {target_pos[1]})")
    print(f"  Geodesic distance range: {geodesic_map.min():.2f} to {geodesic_map.max():.2f}")
    
    # 1. Save cost map (walls vs drivable areas)
    plt.figure(figsize=(10, 10))
    plt.imshow(cost_map, cmap='gray', origin='lower')
    plt.title('Cost Map (Black=Drivable, White=Walls)')
    plt.colorbar(label='Cost (0=drivable, 1=wall)')
    
    # Mark start and target positions
    plt.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', 
                label=f'Start ({start_pos[0]}, {start_pos[1]})', edgecolors='black', linewidth=2)
    plt.scatter(target_pos[0], target_pos[1], c='red', s=100, marker='s', 
                label=f'Target ({target_pos[0]}, {target_pos[1]})', edgecolors='black', linewidth=2)
    plt.legend()
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cost_map_env_{env_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Save geodesic distance map
    plt.figure(figsize=(10, 10))
    
    # Create a custom colormap where infinite values are black
    geodesic_vis = geodesic_map.copy()
    max_finite_dist = np.max(geodesic_vis[np.isfinite(geodesic_vis)])
    geodesic_vis[~np.isfinite(geodesic_vis)] = -1  # Set inf values to -1 for visualization
    
    # Use a colormap that makes -1 values black
    cmap = plt.cm.viridis.copy()
    cmap.set_under('black')
    
    im = plt.imshow(geodesic_vis, cmap=cmap, origin='lower', vmin=0, vmax=max_finite_dist)
    plt.title('Geodesic Distance Map (Black=Unreachable)')
    plt.colorbar(im, label='Distance from target')
    
    # Mark start and target positions
    plt.scatter(start_pos[0], start_pos[1], c='lime', s=100, marker='o', 
                label=f'Start (dist={geodesic_map[start_pos[1], start_pos[0]]:.1f})', 
                edgecolors='black', linewidth=2)
    plt.scatter(target_pos[0], target_pos[1], c='red', s=100, marker='s', 
                label=f'Target (dist=0)', edgecolors='white', linewidth=2)
    plt.legend()
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'geodesic_map_env_{env_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Save combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Cost map
    axes[0].imshow(cost_map, cmap='gray', origin='lower')
    axes[0].scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', 
                   label='Start', edgecolors='black', linewidth=2)
    axes[0].scatter(target_pos[0], target_pos[1], c='red', s=100, marker='s', 
                   label='Target', edgecolors='black', linewidth=2)
    axes[0].set_title('Cost Map')
    axes[0].legend()
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    
    # Geodesic distance map
    im = axes[1].imshow(geodesic_vis, cmap=cmap, origin='lower', vmin=0, vmax=max_finite_dist)
    axes[1].scatter(start_pos[0], start_pos[1], c='lime', s=100, marker='o', 
                   label='Start', edgecolors='black', linewidth=2)
    axes[1].scatter(target_pos[0], target_pos[1], c='red', s=100, marker='s', 
                   label='Target', edgecolors='white', linewidth=2)
    axes[1].set_title('Geodesic Distance Map')
    axes[1].legend()
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    
    # Add colorbar to the right
    plt.colorbar(im, ax=axes[1], label='Distance from target')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'combined_view_env_{env_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Save raw data as numpy arrays
    np.save(os.path.join(output_dir, f'cost_map_env_{env_idx}.npy'), cost_map)
    np.save(os.path.join(output_dir, f'geodesic_map_env_{env_idx}.npy'), geodesic_map)
    np.save(os.path.join(output_dir, f'start_pos_env_{env_idx}.npy'), start_pos)
    np.save(os.path.join(output_dir, f'target_pos_env_{env_idx}.npy'), target_pos)
    
    # 5. Create local crop around target with distance numbers
    crop_size = 8  # Size of the local crop
    target_x, target_y = int(target_pos[0]), int(target_pos[1])
    
    # Define crop boundaries (centered on target)
    half_crop = crop_size // 2
    x_min = max(0, target_x - half_crop)
    x_max = min(cost_map.shape[1], target_x + half_crop)
    y_min = max(0, target_y - half_crop)
    y_max = min(cost_map.shape[0], target_y + half_crop)
    
    # Extract crops
    cost_crop = cost_map[y_min:y_max, x_min:x_max]
    geodesic_crop = geodesic_map[y_min:y_max, x_min:x_max]
    
    # Create annotated geodesic crop with distance numbers
    create_annotated_geodesic_crop(geodesic_crop, cost_crop, 
                                 start_pos[0] - x_min, start_pos[1] - y_min,
                                 target_pos[0] - x_min, target_pos[1] - y_min,
                                 output_dir, env_idx)
    
    # Create simple distance grid for debugging
    create_simple_distance_grid(geodesic_crop, cost_crop, output_dir, env_idx)
    
    print(f"Visualizations saved to {output_dir}/")
    
    return cost_map, geodesic_map, start_pos, target_pos


def create_annotated_geodesic_crop(geodesic_crop, cost_crop, agent_rel_x, agent_rel_y, 
                                 target_rel_x, target_rel_y, output_dir, env_idx):
    """
    Create a single, clear visualization of the geodesic map crop with distance numbers.
    """
    crop_h, crop_w = geodesic_crop.shape
    
    # Create a single plot
    plt.figure(figsize=(10, 10))
    
    # Prepare data
    max_finite_dist = np.max(geodesic_crop[np.isfinite(geodesic_crop)]) if np.any(np.isfinite(geodesic_crop)) else 1
    
    # Create a masked array to hide walls
    masked_geodesic = np.ma.masked_where(cost_crop > 0, geodesic_crop)
    
    # Display the geodesic distance map
    im = plt.imshow(masked_geodesic, cmap='viridis', origin='lower', 
                   vmin=0, vmax=max_finite_dist, interpolation='nearest')
    
    # Add grid lines
    plt.xticks(np.arange(crop_w))
    plt.yticks(np.arange(crop_h))
    plt.grid(True, alpha=0.4, color='white', linewidth=1)
    
    # Add distance numbers and wall markers
    for y in range(crop_h):
        for x in range(crop_w):
            if cost_crop[y, x] > 0:  # Wall
                plt.text(x, y, 'WALL', ha='center', va='center', 
                        fontsize=10, color='red', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            elif np.isfinite(geodesic_crop[y, x]):
                # Distance number with good contrast
                dist_val = geodesic_crop[y, x]
                plt.text(x, y, f'{dist_val:.0f}', ha='center', va='center', 
                        fontsize=12, color='white', weight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
            else:
                # Unreachable
                plt.text(x, y, '∞', ha='center', va='center', 
                        fontsize=14, color='red', weight='bold')
    
    # Mark agent and target
    if 0 <= agent_rel_x < crop_w and 0 <= agent_rel_y < crop_h:
        plt.scatter(agent_rel_x, agent_rel_y, c='lime', s=400, marker='o', 
                   label='Agent', edgecolors='black', linewidth=3, zorder=10)
    
    if 0 <= target_rel_x < crop_w and 0 <= target_rel_y < crop_h:
        plt.scatter(target_rel_x, target_rel_y, c='red', s=400, marker='s', 
                   label='Target', edgecolors='white', linewidth=3, zorder=10)
    
    plt.title(f'Geodesic Distance Crop (8x8) - Environment {env_idx}\nCentered on Target', 
              fontsize=14, pad=20)
    plt.xlabel('X (relative to crop)')
    plt.ylabel('Y (relative to crop)')
    plt.colorbar(im, label='Distance to Target', shrink=0.8)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'geodesic_crop_annotated_env_{env_idx}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    """
    Create a single, clear visualization of the geodesic map crop with distance numbers.
    """
    crop_h, crop_w = geodesic_crop.shape
    
    # Create a single plot
    plt.figure(figsize=(10, 10))
    
    # Prepare data
    max_finite_dist = np.max(geodesic_crop[np.isfinite(geodesic_crop)]) if np.any(np.isfinite(geodesic_crop)) else 1
    
    # Create a masked array to hide walls
    masked_geodesic = np.ma.masked_where(cost_crop > 0, geodesic_crop)
    
    # Display the geodesic distance map
    im = plt.imshow(masked_geodesic, cmap='viridis', origin='lower', 
                   vmin=0, vmax=max_finite_dist, interpolation='nearest')
    
    # Add grid lines
    plt.xticks(np.arange(crop_w))
    plt.yticks(np.arange(crop_h))
    plt.grid(True, alpha=0.4, color='white', linewidth=1)
    
    # Add distance numbers and wall markers
    for y in range(crop_h):
        for x in range(crop_w):
            if cost_crop[y, x] > 0:  # Wall
                plt.text(x, y, 'WALL', ha='center', va='center', 
                        fontsize=10, color='red', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            elif np.isfinite(geodesic_crop[y, x]):
                # Distance number with good contrast
                dist_val = geodesic_crop[y, x]
                plt.text(x, y, f'{dist_val:.0f}', ha='center', va='center', 
                        fontsize=12, color='white', weight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
            else:
                # Unreachable
                plt.text(x, y, '∞', ha='center', va='center', 
                        fontsize=14, color='red', weight='bold')
    
    # Mark agent and target
    if 0 <= agent_rel_x < crop_w and 0 <= agent_rel_y < crop_h:
        plt.scatter(agent_rel_x, agent_rel_y, c='lime', s=400, marker='o', 
                   label='Agent', edgecolors='black', linewidth=3, zorder=10)
    
    if 0 <= target_rel_x < crop_w and 0 <= target_rel_y < crop_h:
        plt.scatter(target_rel_x, target_rel_y, c='red', s=400, marker='s', 
                   label='Target', edgecolors='white', linewidth=3, zorder=10)
    
    plt.title(f'Geodesic Distance Crop (8x8) - Environment {env_idx}\nCentered on Target', 
              fontsize=14, pad=20)
    plt.xlabel('X (relative to crop)')
    plt.ylabel('Y (relative to crop)')
    plt.colorbar(im, label='Distance to Target', shrink=0.8)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'geodesic_crop_annotated_env_{env_idx}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def create_simple_distance_grid(geodesic_crop, cost_crop, output_dir, env_idx):
    """Create a simple text-based distance grid for debugging."""
    crop_h, crop_w = geodesic_crop.shape
    
    print(f"\nEnvironment {env_idx} - Local Distance Grid:")
    print("=" * (crop_w * 6 + 1))
    
    grid_text = []
    for y in range(crop_h):
        row_text = "|"
        for x in range(crop_w):
            if cost_crop[y, x] > 0:  # Wall
                cell_text = " WALL"
            elif np.isfinite(geodesic_crop[y, x]):
                cell_text = f"{geodesic_crop[y, x]:5.0f}"
            else:
                cell_text = "  INF"
            row_text += cell_text + "|"
        grid_text.append(row_text)
        print(row_text)
    
    print("=" * (crop_w * 6 + 1))
    
    # Save to text file
    with open(os.path.join(output_dir, f'distance_grid_env_{env_idx}.txt'), 'w') as f:
        f.write(f"Environment {env_idx} - Local Distance Grid:\n")
        f.write("=" * (crop_w * 6 + 1) + "\n")
        for row in grid_text:
            f.write(row + "\n")
        f.write("=" * (crop_w * 6 + 1) + "\n")


def main():
    """Main function to create environment and generate visualizations."""
    
    print("Creating ParallelDrivingEnv...")
    
    # Environment parameters (based on env_config.yaml)
    env_params = {
        'num_envs': 3,  # Create 3 environments for testing
        'view_size': 64,
        'dt': 0.1,
        'a_max': 2.0,
        'v_max': 5.0,
        'w_col': 1.0,
        'R_goal': 100.0,
        'disk_radius': 2.0,
        'render_dir': None,  # No video rendering
        'video_fps': 30,
        'w_dist': 5.0,
        'w_accel': 0.1,
        'max_steps': 2000,
        'hitwall_cost': -50.0,
        'pyramid_levels': [0, 2, 4],
        'num_levels': 3,
        'min_start_goal_dist': 50,
        'max_start_goal_dist': 200,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {env_params['device']}")
    
    # Create environment
    env = ParallelDrivingEnv(**env_params)
    
    # Reset to generate initial maps
    print("Resetting environment to generate maps...")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs['image'].shape}, {obs['state'].shape}")
    print(f"Info keys: {info.keys()}")
    
    # Create visualizations for each environment
    output_dir = "test_env_viz"
    for env_idx in range(env_params['num_envs']):
        print(f"\nGenerating visualization for environment {env_idx}...")
        create_visualization(env, env_idx, output_dir)
    
    print(f"\nAll visualizations complete! Check the '{output_dir}' directory.")
    
    # Also test the observation visualization function
    print("\nTesting observation visualization...")
    obs_single = {
        'image': obs['image'][0],  # First environment
        'state': obs['state'][0]
    }
    
    # Convert to numpy for visualization
    frame = get_human_frame_from_obs(obs_single, env.vel[0].cpu().numpy(), env.v_max)
    cv2.imwrite(os.path.join(output_dir, 'observation_visualization.png'), frame)
    print(f"Observation visualization saved to {output_dir}/observation_visualization.png")


def get_human_frame_from_obs(obs_dict, vel, v_max, action=None, info=None, a_max=None):
    """
    Generate visualization from observation dict (similar to get_human_frame in env.py).
    """
    from env import obs_level_to_image, ParallelDrivingEnv
    
    image_obs = obs_dict['image']  # (num_levels, 2, view_size, view_size)
    state_obs = obs_dict['state']  # (2,) - normalized velocity
    
    vis_levels = []
    canvas_size = 256
    num_levels = image_obs.shape[0]
    
    for i in range(num_levels):
        img = obs_level_to_image(image_obs[i], canvas_size=canvas_size)
        # Draw agent (center)
        cx, cy = canvas_size // 2, canvas_size // 2
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), 2)
        # Draw velocity vector
        v_norm = np.linalg.norm(vel)
        if v_norm > 1e-3:
            v_dir = vel / (v_norm + 1e-6)
            v_len = int((v_norm / v_max) * (canvas_size // 4))
            tip = (int(cx + v_dir[0] * v_len), int(cy + v_dir[1] * v_len))
            cv2.arrowedLine(img, (cx, cy), tip, (255, 255, 0), 2, tipLength=0.3)
        vis_levels.append(img)
    
    frame = np.concatenate(vis_levels, axis=1)
    
    # Overlay text info
    if info is None:
        info = {}
    info_lines = [
        f"Velocity: ({vel[0]:.2f}, {vel[1]:.2f})",
        f"Speed: {np.linalg.norm(vel):.2f} / {v_max:.2f}",
        f"State (norm): ({state_obs[0]:.2f}, {state_obs[1]:.2f})",
    ]
    y0 = 30
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    main()
