"""
Human data recording system for the flatland driving environment.
Allows playing the game with keyboard controls and recording trajectories.
"""

import os
import sys
import time
import json
import numpy as np
import pygame
import torch
import cv2
import h5py
from collections import deque
from omegaconf import DictConfig, OmegaConf
import hydra

from env import ParallelDrivingEnv, get_human_frame


class HumanPlayer:
    """
    Human player interface using pygame for the flatland driving environment.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize pygame
        pygame.init()
        
        # Display settings
        self.view_size = cfg.env.view_size
        self.num_levels = cfg.env.num_levels
        self.level_display_size = 256  # Size of each level display
        self.info_panel_width = 300
        self.window_width = self.level_display_size * self.num_levels + self.info_panel_width
        self.window_height = self.level_display_size + 100  # Extra space for controls
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Flatland Human Player")
        
        # Font for text rendering
        self.font = pygame.font.Font(None, 18)
        self.small_font = pygame.font.Font(None, 14)


        min_start_goal_dist = 10.0
        max_start_goal_dist = 200.0
        
        # Create single environment (num_envs=1)
        self.env = ParallelDrivingEnv(
            num_envs=1,
            view_size=cfg.env.view_size,
            dt=cfg.env.dt,
            a_max=cfg.env.a_max,
            v_max=cfg.env.v_max,
            w_col=cfg.env.w_col,
            R_goal=cfg.env.R_goal,
            disk_radius=cfg.env.disk_radius,
            render_dir=None,  # No video recording during play
            video_fps=cfg.env.video_fps,
            w_dist=cfg.env.w_dist,
            w_accel=cfg.env.w_accel,
            max_steps=cfg.train.max_steps,
            hitwall_cost=cfg.env.hitwall_cost,
            pyramid_levels=cfg.env.pyramid_levels,
            num_levels=cfg.env.num_levels,
            min_start_goal_dist=min_start_goal_dist,
            max_start_goal_dist=max_start_goal_dist,
            device=self.device
        )
        
        # Action mapping (same as in the environment)
        self.action_names = [
            "No Action",      # 0: (0, 0)
            "Up",            # 1: (0, 1)
            "Up-Right",      # 2: (1/√2, 1/√2)
            "Right",         # 3: (1, 0)
            "Down-Right",    # 4: (1/√2, -1/√2)
            "Down",          # 5: (0, -1)
            "Down-Left",     # 6: (-1/√2, -1/√2)
            "Left",          # 7: (-1, 0)
            "Up-Left",       # 8: (-1/√2, 1/√2)
        ]
        
        # Key to action mapping
        self.key_to_action = {
            pygame.K_x: 1,      # Up
            pygame.K_c: 2,      # Up-Right
            pygame.K_d: 3,      # Right
            pygame.K_e: 4,      # Down-Right
            pygame.K_w: 5,      # Down
            pygame.K_q: 6,      # Down-Left
            pygame.K_a: 7,      # Left
            pygame.K_z: 8,      # Up-Left
        }
        
        # Game state
        self.current_action = 0
        self.obs = None
        self.info = None
        self.done = False
        self.episode_reward = 0.0
        self.step_count = 0
        
        # Recording data
        self.recording = True
        self.current_episode_data = []
        self.episode_count = 0
        self.session_start_time = time.strftime("%Y%m%d_%H%M%S")
        
        # Setup recording directory and HDF5 file
        self.record_dir = "human_data"
        os.makedirs(self.record_dir, exist_ok=True)
        
        # Always use the same filename and append to it
        self.hdf5_filename = os.path.join(self.record_dir, "human_data.h5")
        self.hdf5_file = None
        self.total_steps_recorded = 0
        self._init_or_append_hdf5_dataset()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.BORDER_COLOR = (200, 200, 200)
        
        print("Human Player initialized!")
        print("Controls:")
        print("  W/A/S/D/Q/E/Z/C - Movement")
        print("  R - Start/Stop Recording")
        print("  SPACE - Reset Environment")
        print("  ESC - Quit")
    
    def obs_level_to_pygame_surface(self, obs_level):
        """Convert observation level to pygame surface"""
        # Convert tensor to numpy
        if hasattr(obs_level, 'cpu'):
            obs_level = obs_level.cpu().numpy()
        
        cost = obs_level[0]
        vel_x = obs_level[1]
        vel_y = obs_level[2]
        target_map = obs_level[3]
        
        # Create RGB image
        # Cost channel: -1 (drivable) = black, 1+ (wall) = white
        cost_norm = np.clip((cost + 1) / 2, 0, 1)  # -1..1 -> 0..1
        img = np.stack([cost_norm, cost_norm, cost_norm], axis=-1)
        
        # Overlay target as magenta
        target_mask = target_map > 0.5
        img[target_mask, 0] = 1.0  # Red
        img[target_mask, 1] = 0.0  # Green
        img[target_mask, 2] = 1.0  # Blue
        
        # Convert to 0-255 range
        img = (img * 255).astype(np.uint8)
        
        # Resize to display size
        img = cv2.resize(img, (self.level_display_size, self.level_display_size), 
                        interpolation=cv2.INTER_NEAREST)
        
        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        return surface
    
    def draw_level_info(self, surface, level_idx, obs_level):
        """Draw information overlay on a level surface"""
        if hasattr(obs_level, 'cpu'):
            obs_level = obs_level.cpu().numpy()
        
        vel_x = obs_level[1, self.view_size//2, self.view_size//2]
        vel_y = obs_level[2, self.view_size//2, self.view_size//2]
        
        # Draw level number
        text = self.small_font.render(f"Level {level_idx}", True, self.WHITE)
        surface.blit(text, (5, 5))
        
        # Draw velocity info
        vel_text = f"Vel: ({vel_x:.2f}, {vel_y:.2f})"
        text = self.small_font.render(vel_text, True, self.WHITE)
        surface.blit(text, (5, 25))
        
        # Draw agent center marker
        cx, cy = self.level_display_size // 2, self.level_display_size // 2
        pygame.draw.circle(surface, self.GREEN, (cx, cy), 4, 2)
        
        # Draw velocity vector
        vel = np.array([vel_x, vel_y]) * self.cfg.env.v_max
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 1e-3:
            vel_dir = vel / vel_norm
            vel_len = int(vel_norm / self.cfg.env.v_max * 30)  # Scale for visualization
            tip = (int(cx + vel_dir[0] * vel_len), int(cy + vel_dir[1] * vel_len))
            pygame.draw.line(surface, self.YELLOW, (cx, cy), tip, 2)
            # Arrow tip
            pygame.draw.circle(surface, self.YELLOW, tip, 3)
    
    def draw_info_panel(self):
        """Draw the information panel"""
        panel_x = self.level_display_size * self.num_levels
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.window_height)
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)
        
        y_offset = 15
        line_height = 18
        
        def draw_text(text, color=self.BLACK):
            nonlocal y_offset
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (panel_x + 10, y_offset))
            y_offset += line_height
        
        # Game info
        draw_text("=== Game Info ===", self.BLACK)
        draw_text(f"Step: {self.step_count}")
        draw_text(f"Reward: {self.episode_reward:.2f}")
        draw_text(f"Episodes: {self.episode_count}")
        
        if self.info is not None:
            draw_text(f"Progress: {self.info['r_progress'][0].item():.3f}")
            draw_text(f"Goal: {self.info['r_goal'][0].item():.3f}")
            draw_text(f"Hitwall: {self.info['hitwall'][0].item():.3f}")
            
            vel = self.info['vel'][0]
            draw_text(f"Vel: ({vel[0].item():.2f}, {vel[1].item():.2f})")
        
        y_offset += 8
        
        # Current action
        draw_text("=== Current Action ===", self.BLACK)
        action_name = self.action_names[self.current_action]
        draw_text(f"{self.current_action}: {action_name}", self.BLUE)
        
        y_offset += 8
        
        # Recording status
        status_color = self.RED if self.recording else self.BLACK
        draw_text("=== Recording ===", self.BLACK)
        status_text = "RECORDING" if self.recording else "NOT RECORDING"
        draw_text(status_text, status_color)
        saved_episodes = self.get_episode_count_in_session()
        draw_text(f"Episodes saved: {saved_episodes}")
        
        y_offset += 8
        
        # Controls
        draw_text("=== Controls ===", self.BLACK)
        controls = [
            "W/A/S/D/Q/E/Z/C: Move",
            "R: Toggle Recording",
            "SPACE: Reset",
            "ESC: Quit"
        ]
        for control in controls:
            surface = self.small_font.render(control, True, self.BLACK)
            self.screen.blit(surface, (panel_x + 10, y_offset))
            y_offset += 16
        
        # Status
        if self.done:
            y_offset += 15
            status_text = "EPISODE FINISHED!"
            surface = self.font.render(status_text, True, self.RED)
            self.screen.blit(surface, (panel_x + 10, y_offset))
    
    def get_pressed_action(self, keys):
        """Get action from currently pressed keys"""
        for key, action in self.key_to_action.items():
            if keys[key]:
                return action
        return 0  # No action
    
    def reset_environment(self):
        """Reset the environment for a new episode"""
        self.obs, self.info = self.env.reset()
        self.done = False
        self.episode_reward = 0.0
        self.step_count = 0
        self.current_action = 0
        print(f"Environment reset. Starting episode {self.episode_count + 1}")
    
    def _init_or_append_hdf5_dataset(self):
        """Initialize HDF5 dataset for efficient storage or append to existing file"""
        
        # Check if file already exists
        if os.path.exists(self.hdf5_filename):
            print(f"Found existing HDF5 file: {self.hdf5_filename}")
            self._append_to_existing_hdf5()
        else:
            print(f"Creating new HDF5 file: {self.hdf5_filename}")
            self._create_new_hdf5()
    
    def _append_to_existing_hdf5(self):
        """Append to existing HDF5 file"""
        self.hdf5_file = h5py.File(self.hdf5_filename, 'a')  # Open in append mode
        
        # Verify file structure is compatible
        required_datasets = ['observations', 'actions', 'rewards', 'episodes']
        for dataset_name in required_datasets:
            if dataset_name not in self.hdf5_file:
                raise ValueError(f"Existing HDF5 file missing required dataset: {dataset_name}")
        
        # Check compatibility of observation shape
        obs_shape = (self.num_levels, 4, self.view_size, self.view_size)
        existing_obs_shape = self.hdf5_file['observations'].shape[1:]
        if existing_obs_shape != obs_shape:
            raise ValueError(f"Observation shape mismatch: existing {existing_obs_shape} vs required {obs_shape}")
        
        # Get existing datasets
        self.obs_dataset = self.hdf5_file['observations']
        self.actions_dataset = self.hdf5_file['actions']
        self.rewards_dataset = self.hdf5_file['rewards']
        self.episodes_dataset = self.hdf5_file['episodes']
        
        # Get current counts from existing data
        self.total_steps_recorded = self.hdf5_file.attrs.get('total_steps', len(self.obs_dataset))
        self.episode_count = self.hdf5_file.attrs.get('total_episodes', len(self.episodes_dataset))
        
        # Trim datasets to actual size (in case they were pre-allocated larger)
        if self.total_steps_recorded > 0:
            current_step_size = self.obs_dataset.shape[0]
            if self.total_steps_recorded < current_step_size:
                # Dataset is larger than actual data, but we keep it for efficiency
                pass
        
        print(f"Appending to existing file - Current: {self.episode_count} episodes, {self.total_steps_recorded} steps")
    
    def _create_new_hdf5(self):
        """Create new HDF5 file"""
        # Estimate initial size (can be resized later)
        initial_episodes = 1000
        initial_steps = 100000  # Estimate steps per episode * episodes
        obs_shape = (self.num_levels, 4, self.view_size, self.view_size)
        
        self.hdf5_file = h5py.File(self.hdf5_filename, 'w')
        
        # Create datasets with chunking for efficient random access
        chunk_size = 1000  # Number of steps per chunk
        
        # Observations: [total_steps, num_levels, 4, view_size, view_size]
        self.obs_dataset = self.hdf5_file.create_dataset(
            'observations', 
            shape=(initial_steps, *obs_shape),
            maxshape=(None, *obs_shape),  # Unlimited in first dimension
            dtype=np.float32,
            chunks=(chunk_size, *obs_shape),
            compression='gzip',
            compression_opts=1  # Light compression for speed
        )
        
        # Actions: [total_steps]
        self.actions_dataset = self.hdf5_file.create_dataset(
            'actions',
            shape=(initial_steps,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=1
        )
        
        # Rewards: [total_steps]
        self.rewards_dataset = self.hdf5_file.create_dataset(
            'rewards',
            shape=(initial_steps,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=1
        )
        
        # Episode metadata: [num_episodes]
        self.episodes_dataset = self.hdf5_file.create_dataset(
            'episodes',
            shape=(initial_episodes,),
            maxshape=(None,),
            dtype=[
                ('start_idx', 'i4'),      # Starting index in the main arrays
                ('length', 'i4'),         # Episode length
                ('reward', 'f4'),         # Total episode reward
                ('success', 'b1'),        # Success flag
                ('timestamp', 'S20')      # Timestamp string
            ],
            chunks=(100,)
        )
        
        # Store metadata
        self.hdf5_file.attrs['session_start_time'] = self.session_start_time
        self.hdf5_file.attrs['view_size'] = self.view_size
        self.hdf5_file.attrs['num_levels'] = self.num_levels
        self.hdf5_file.attrs['total_episodes'] = 0
        self.hdf5_file.attrs['total_steps'] = 0
        
        # Save config as JSON string
        config_str = json.dumps(OmegaConf.to_container(self.cfg, resolve=True))
        self.hdf5_file.attrs['config'] = config_str
        
        # Initialize counters
        self.total_steps_recorded = 0
        self.episode_count = 0
    
    def _resize_datasets_if_needed(self, required_steps, required_episodes):
        """Resize datasets if more space is needed"""
        if self.hdf5_file is None:
            return
        
        # Check and resize step datasets
        current_steps = self.obs_dataset.shape[0]
        if required_steps > current_steps:
            new_size = max(required_steps, int(current_steps * 1.5))
            self.obs_dataset.resize((new_size, *self.obs_dataset.shape[1:]))
            self.actions_dataset.resize((new_size,))
            self.rewards_dataset.resize((new_size,))
        
        # Check and resize episode dataset
        current_episodes = self.episodes_dataset.shape[0]
        if required_episodes > current_episodes:
            new_size = max(required_episodes, int(current_episodes * 1.5))
            self.episodes_dataset.resize((new_size,))

    def save_episode_data(self, episode_data):
        """Save a single episode data immediately to HDF5 file"""
        if not episode_data or self.hdf5_file is None:
            return
        
        observations = episode_data['observations']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        episode_length = len(observations)
        
        # Convert observations to numpy array
        obs_array = torch.stack(observations).cpu().numpy()  # [episode_length, num_levels, 4, view_size, view_size]
        actions_array = np.array(actions, dtype=np.int32)
        rewards_array = np.array(rewards, dtype=np.float32)
        
        # Resize datasets if needed
        required_steps = self.total_steps_recorded + episode_length
        required_episodes = self.episode_count + 1
        self._resize_datasets_if_needed(required_steps, required_episodes)
        
        # Write step data
        start_idx = self.total_steps_recorded
        end_idx = start_idx + episode_length
        
        self.obs_dataset[start_idx:end_idx] = obs_array
        self.actions_dataset[start_idx:end_idx] = actions_array
        self.rewards_dataset[start_idx:end_idx] = rewards_array
        
        # Write episode metadata
        episode_record = np.array([
            (
                start_idx,
                episode_length,
                episode_data['episode_reward'],
                episode_data.get('success', False),
                time.strftime("%Y%m%d_%H%M%S").encode('utf-8')
            )
        ], dtype=self.episodes_dataset.dtype)
        
        self.episodes_dataset[self.episode_count] = episode_record[0]
        
        # Update counters
        self.total_steps_recorded += episode_length
        
        # Flush to disk periodically for safety
        if self.episode_count % 10 == 0:
            self.hdf5_file.flush()
        
        print(f"Saved episode {self.episode_count} with {episode_length} steps (total steps: {self.total_steps_recorded})")
    
    def get_episode_count_in_session(self):
        """Get the count of episodes saved in this session"""
        return self.episode_count
    
    def close_hdf5_file(self):
        """Safely close the HDF5 file"""
        if self.hdf5_file is not None:
            # Update final attributes
            self.hdf5_file.attrs['total_episodes'] = self.episode_count
            self.hdf5_file.attrs['total_steps'] = self.total_steps_recorded
            self.hdf5_file.attrs['session_end_time'] = time.strftime("%Y%m%d_%H%M%S")
            
            # Trim datasets to actual size
            if self.total_steps_recorded > 0:
                self.obs_dataset.resize((self.total_steps_recorded, *self.obs_dataset.shape[1:]))
                self.actions_dataset.resize((self.total_steps_recorded,))
                self.rewards_dataset.resize((self.total_steps_recorded,))
            
            if self.episode_count > 0:
                self.episodes_dataset.resize((self.episode_count,))
            
            self.hdf5_file.close()
            print(f"Closed HDF5 file with {self.episode_count} episodes and {self.total_steps_recorded} steps")
            self.hdf5_file = None
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Initialize environment
        self.reset_environment()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Finish current episode if recording
                        if self.recording and self.current_episode_data:
                            episode_data = {
                                'observations': [step['obs'] for step in self.current_episode_data],
                                'actions': [step['action'] for step in self.current_episode_data],
                                'rewards': [step['reward'] for step in self.current_episode_data],
                                'episode_reward': sum(step['reward'] for step in self.current_episode_data),
                                'episode_length': len(self.current_episode_data)
                            }
                            self.episode_count += 1
                            self.save_episode_data(episode_data)
                            self.current_episode_data = []
                        
                        self.reset_environment()
                    elif event.key == pygame.K_r:
                        self.recording = not self.recording
                        if self.recording:
                            print("Started recording")
                        else:
                            print("Stopped recording")
                            # Don't save partial episodes - just discard current data
                            if self.current_episode_data:
                                print(f"Discarded partial episode with {len(self.current_episode_data)} steps")
                                self.current_episode_data = []
            
            # Get current action from keyboard
            keys = pygame.key.get_pressed()
            self.current_action = self.get_pressed_action(keys)
            
            # Step environment if not done
            if not self.done:
                action_tensor = torch.tensor([self.current_action], device=self.device)
                
                # Record data if recording
                if self.recording:
                    self.current_episode_data.append({
                        'obs': self.obs[0].cpu().clone(),  # Store observation
                        'action': self.current_action,
                        'reward': 0.0  # Will be updated after step
                    })
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action_tensor)
                
                # Update game state
                self.obs = next_obs
                self.info = info
                self.done = terminated[0].item() or truncated[0].item()
                episode_reward_delta = reward[0].item()
                self.episode_reward += episode_reward_delta
                self.step_count += 1
                
                # Update reward in recorded data
                if self.recording and self.current_episode_data:
                    self.current_episode_data[-1]['reward'] = episode_reward_delta
                
                # Handle episode end
                if self.done:
                    success = info['success'][0].item()
                    if success:
                        print(f"SUCCESS! Episode finished! Reward: {self.episode_reward:.2f}, Steps: {self.step_count}")
                    else:
                        print(f"FAILED! Episode finished! Reward: {self.episode_reward:.2f}, Steps: {self.step_count}")
                    
                    if self.recording and self.current_episode_data:
                        episode_data = {
                            'observations': [step['obs'] for step in self.current_episode_data],
                            'actions': [step['action'] for step in self.current_episode_data],
                            'rewards': [step['reward'] for step in self.current_episode_data],
                            'episode_reward': self.episode_reward,
                            'episode_length': len(self.current_episode_data),
                            'success': success
                        }
                        self.episode_count += 1
                        self.save_episode_data(episode_data)
                        self.current_episode_data = []
                    
                    # Auto-restart after a brief pause
                    pygame.time.wait(1000)  # Wait 1 second
                    self.reset_environment()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw observation levels
            if self.obs is not None:
                for level_idx in range(self.num_levels):
                    obs_level = self.obs[0, level_idx]  # [0] for first (and only) env
                    surface = self.obs_level_to_pygame_surface(obs_level)
                    self.draw_level_info(surface, level_idx, obs_level)
                    
                    x_pos = level_idx * self.level_display_size
                    
                    # Draw border around the image
                    border_rect = pygame.Rect(x_pos - 2, -2, self.level_display_size + 4, self.level_display_size + 4)
                    pygame.draw.rect(self.screen, self.BORDER_COLOR, border_rect, 2)
                    
                    self.screen.blit(surface, (x_pos, 0))
            
            # Draw info panel
            self.draw_info_panel()
            
            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        
        # Close HDF5 file before quitting
        self.close_hdf5_file()
        pygame.quit()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function to run human player"""
    print("Starting Human Player...")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    
    player = HumanPlayer(cfg)
    try:
        player.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Don't save partial episodes - only complete ones should be recorded
        if hasattr(player, 'current_episode_data') and player.current_episode_data:
            print(f"Discarded partial episode with {len(player.current_episode_data)} steps on exit")
        
        # Ensure HDF5 file is properly closed
        if hasattr(player, 'close_hdf5_file'):
            player.close_hdf5_file()
        
        print("Human player session ended.")


if __name__ == '__main__':
    main()
