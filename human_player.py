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


        min_start_goal_dist = 100.0
        max_start_goal_dist = 400.0
        
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
            max_steps=10000,
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
        
        # Key to action mapping - WASD with combinations (corrected for inverted Y axis)
        self.key_to_action = {
            # Single keys - corrected for pygame screen coordinates vs environment coordinates
            (pygame.K_w,): 5,           # Up on screen (W moves up visually, maps to action 5)
            (pygame.K_d,): 3,           # Right (D) 
            (pygame.K_s,): 1,           # Down on screen (S moves down visually, maps to action 1)
            (pygame.K_a,): 7,           # Left (A)
            # Key combinations for diagonal movement - corrected
            (pygame.K_w, pygame.K_d): 4,  # Up-Right on screen (W+D: up visually + right)
            (pygame.K_s, pygame.K_d): 2,  # Down-Right on screen (S+D: down visually + right)
            (pygame.K_s, pygame.K_a): 8,  # Down-Left on screen (S+A: down visually + left)
            (pygame.K_w, pygame.K_a): 6,  # Up-Left on screen (W+A: up visually + left)
        }
        
        # Recording data
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
        print("  WASD - Movement (combine for diagonals, e.g. W+A for up-left)")
        print("  R - Skip current episode and start new one")
        print("  ESC - Quit")
        print()
        print("Data being recorded:")
        print("  - Observations (image + state)")
        print("  - Actions and rewards") 
        print("  - Full 512x512 world maps")
        print("  - Agent positions in world coordinates")
        print("  - Start and target positions")
        print("  -> Scenes can be fully recreated from this data!")
    
    def obs_level_to_pygame_surface(self, obs_level):
        """Convert observation level to pygame surface"""
        # Convert tensor to numpy
        if hasattr(obs_level, 'cpu'):
            obs_level = obs_level.cpu().numpy()
        
        # New format: only 2 channels (cost + target)
        cost = obs_level[0]  # uint8 values: 0 (drivable) or 255 (wall)
        target_map = obs_level[1]  # uint8 values: 0 (no target), 128 (agent center), 255 (goal)
        
        # Create RGB image
        # Cost channel: 0 (drivable) = black, 255 (wall) = white
        cost_norm = cost.astype(np.float32) / 255.0  # 0..255 -> 0..1
        img = np.stack([cost_norm, cost_norm, cost_norm], axis=-1)
        
        # Overlay target as magenta (goal) and green (agent center)
        goal_mask = target_map == 255
        agent_mask = target_map == 128
        
        img[goal_mask, 0] = 1.0  # Red for goal
        img[goal_mask, 1] = 0.0  # Green
        img[goal_mask, 2] = 1.0  # Blue for goal -> magenta
        
        img[agent_mask, 0] = 0.0  # Red for agent center
        img[agent_mask, 1] = 1.0  # Green for agent center -> green
        img[agent_mask, 2] = 0.0  # Blue
        
        # Convert to 0-255 range
        img = (img * 255).astype(np.uint8)
        
        # Resize to display size
        img = cv2.resize(img, (self.level_display_size, self.level_display_size), 
                        interpolation=cv2.INTER_NEAREST)
        
        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        return surface
    
    def draw_level_info(self, surface, level_idx, state_obs):
        """Draw information overlay on a level surface"""
        if hasattr(state_obs, 'cpu'):
            state_obs = state_obs.cpu().numpy()
        
        vel_x = state_obs[0]  # Normalized velocity x
        vel_y = state_obs[1]  # Normalized velocity y
        
        # Draw level number
        text = self.small_font.render(f"Level {level_idx}", True, self.WHITE)
        surface.blit(text, (5, 5))
        
        # Draw velocity info (denormalized)
        vel_actual_x = vel_x * self.cfg.env.v_max
        vel_actual_y = vel_y * self.cfg.env.v_max
        vel_text = f"Vel: ({vel_actual_x:.2f}, {vel_actual_y:.2f})"
        text = self.small_font.render(vel_text, True, self.WHITE)
        surface.blit(text, (5, 25))
        
        # Draw agent center marker
        cx, cy = self.level_display_size // 2, self.level_display_size // 2
        pygame.draw.circle(surface, self.GREEN, (cx, cy), 4, 2)
        
        # Draw velocity vector
        vel_len = int(np.sqrt(vel_x**2 + vel_y**2) * 30)  # Scale for visualization
        if vel_len > 1:
            tip = (int(cx + vel_x * 30), int(cy + vel_y * 30))
            pygame.draw.line(surface, self.YELLOW, (cx, cy), tip, 2)
            # Arrow tip
            pygame.draw.circle(surface, self.YELLOW, tip, 3)
    
    def draw_info_panel(self, info, done=False, current_action=0, episode_reward=0.0, step_count=0):
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
        draw_text(f"Step: {step_count}")
        draw_text(f"Reward: {episode_reward:.2f}")
        draw_text(f"Episodes: {self.episode_count}")
        
        if info is not None:
            draw_text(f"Progress: {info['r_progress'][0].item():.3f}")
            draw_text(f"Goal: {info['r_goal'][0].item():.3f}")
            draw_text(f"Hitwall: {info['hitwall'][0].item():.3f}")
            
            vel = info['vel'][0]
            draw_text(f"Vel: ({vel[0].item():.2f}, {vel[1].item():.2f})")
        
        y_offset += 8
        
        # Current action
        draw_text("=== Current Action ===", self.BLACK)
        action_name = self.action_names[current_action]
        draw_text(f"{current_action}: {action_name}", self.BLUE)
        
        y_offset += 8
        
        # Recording status
        draw_text("=== Recording ===", self.BLACK)
        draw_text("ALWAYS RECORDING", self.RED)
        saved_episodes = self.get_episode_count_in_session()
        draw_text(f"Episodes saved: {saved_episodes}")
        
        y_offset += 8
        
        # Controls
        draw_text("=== Controls ===", self.BLACK)
        controls = [
            "WASD: Move",
            "Combine for diagonals",
            "R: Skip Episode",
            "ESC: Quit"
        ]
        for control in controls:
            surface = self.small_font.render(control, True, self.BLACK)
            self.screen.blit(surface, (panel_x + 10, y_offset))
            y_offset += 16
        
        # Status
        if done:
            y_offset += 15
            status_text = "EPISODE FINISHED!"
            surface = self.font.render(status_text, True, self.RED)
            self.screen.blit(surface, (panel_x + 10, y_offset))
    
    def get_pressed_action(self, keys):
        """Get action from currently pressed keys, supporting combinations"""
        # Check for key combinations first (more specific)
        pressed_movement_keys = []
        movement_keys = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]
        
        for key in movement_keys:
            if keys[key]:
                pressed_movement_keys.append(key)
        
        if not pressed_movement_keys:
            return 0  # No action
        
        # Sort keys for consistent comparison
        pressed_tuple = tuple(sorted(pressed_movement_keys))
        
        # Try to find exact combination match
        for key_combo, action in self.key_to_action.items():
            if tuple(sorted(key_combo)) == pressed_tuple:
                return action
        
        # If more than 2 keys pressed or invalid combination, take first valid single key
        if len(pressed_movement_keys) > 2:
            for key in pressed_movement_keys:
                if (key,) in self.key_to_action:
                    return self.key_to_action[(key,)]
        
        return 0  # No valid action found
    
    def reset_environment(self):
        """Reset the environment for a new episode and return initial observation"""
        obs, info = self.env.reset()
        
        # Don't record anything here - let the main loop handle all recording
        # The first observation will be recorded when the human takes the first action
        
        print(f"Environment reset. Starting episode {self.episode_count + 1}")
        return obs, info
    
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
        
        # Verify file structure is compatible (new format with separate image/state)
        required_datasets = ['obs_images', 'obs_states', 'actions', 'rewards', 'episodes', 
                           'agent_positions']
        for dataset_name in required_datasets:
            if dataset_name not in self.hdf5_file:
                raise ValueError(f"Existing HDF5 file missing required dataset: {dataset_name}")
        
        # Check compatibility of observation shapes
        image_shape = (self.num_levels, 2, self.view_size, self.view_size)  # uint8 images
        state_shape = (2,)  # float32 velocity
        
        existing_image_shape = self.hdf5_file['obs_images'].shape[1:]
        existing_state_shape = self.hdf5_file['obs_states'].shape[1:]
        
        if existing_image_shape != image_shape:
            raise ValueError(f"Image observation shape mismatch: existing {existing_image_shape} vs required {image_shape}")
        if existing_state_shape != state_shape:
            raise ValueError(f"State observation shape mismatch: existing {existing_state_shape} vs required {state_shape}")
        
        # Get existing datasets
        self.obs_images_dataset = self.hdf5_file['obs_images']
        self.obs_states_dataset = self.hdf5_file['obs_states']
        self.actions_dataset = self.hdf5_file['actions']
        self.rewards_dataset = self.hdf5_file['rewards']
        self.episodes_dataset = self.hdf5_file['episodes']
        
        # Get world state datasets (only agent positions per step)
        self.agent_positions_dataset = self.hdf5_file['agent_positions']
        
        # Get current counts from existing data
        self.total_steps_recorded = self.hdf5_file.attrs.get('total_steps', len(self.obs_images_dataset))
        self.episode_count = self.hdf5_file.attrs.get('total_episodes', len(self.episodes_dataset))
        
        print(f"Appending to existing file - Current: {self.episode_count} episodes, {self.total_steps_recorded} steps")
    
    def _create_new_hdf5(self):
        """Create new HDF5 file"""
        # Estimate initial size (can be resized later)
        initial_episodes = 1000
        initial_steps = 100000  # Estimate steps per episode * episodes
        image_shape = (self.num_levels, 2, self.view_size, self.view_size)  # uint8 images
        state_shape = (2,)  # float32 velocity
        
        self.hdf5_file = h5py.File(self.hdf5_filename, 'w')
        
        # Create datasets with chunk_size = 1 for random access
        chunk_size = 1  # Number of steps per chunk
        
        # Image observations: [total_steps, num_levels, 2, view_size, view_size] uint8
        self.obs_images_dataset = self.hdf5_file.create_dataset(
            'obs_images', 
            shape=(initial_steps, *image_shape),
            maxshape=(None, *image_shape),  # Unlimited in first dimension
            dtype=np.uint8,
            chunks=(chunk_size, *image_shape),
            compression='gzip',  # Only compress image data
            compression_opts=1  # Light compression for speed
        )
        
        # State observations: [total_steps, 2] float32
        self.obs_states_dataset = self.hdf5_file.create_dataset(
            'obs_states',
            shape=(initial_steps, *state_shape),
            maxshape=(None, *state_shape),
            dtype=np.float32,
            chunks=(chunk_size, *state_shape)  # No compression for state data
        )
        
        # Actions: [total_steps]
        self.actions_dataset = self.hdf5_file.create_dataset(
            'actions',
            shape=(initial_steps,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(chunk_size,)  # No compression
        )
        
        # Rewards: [total_steps]
        self.rewards_dataset = self.hdf5_file.create_dataset(
            'rewards',
            shape=(initial_steps,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(chunk_size,)  # No compression
        )
        
        # Agent positions: [total_steps, 2] - Agent x,y coordinates in world space
        self.agent_positions_dataset = self.hdf5_file.create_dataset(
            'agent_positions',
            shape=(initial_steps, 2),
            maxshape=(None, 2),
            dtype=np.float32,
            chunks=(chunk_size, 2)  # No compression for positions
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
                ('timestamp', 'S20'),     # Timestamp string
                ('world_map', 'f4', (512, 512)),  # World cost map for this episode
                ('start_pos', 'i4', (2,)),        # Start position [x, y]
                ('target_pos', 'i4', (2,))        # Target position [x, y]
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
        current_steps = self.obs_images_dataset.shape[0]
        if required_steps > current_steps:
            new_size = max(required_steps, int(current_steps * 1.5))
            self.obs_images_dataset.resize((new_size, *self.obs_images_dataset.shape[1:]))
            self.obs_states_dataset.resize((new_size, *self.obs_states_dataset.shape[1:]))
            self.actions_dataset.resize((new_size,))
            self.rewards_dataset.resize((new_size,))
            
            # Resize agent positions dataset
            self.agent_positions_dataset.resize((new_size, 2))
        
        # Check and resize episode dataset
        current_episodes = self.episodes_dataset.shape[0]
        if required_episodes > current_episodes:
            new_size = max(required_episodes, int(current_episodes * 1.5))
            self.episodes_dataset.resize((new_size,))

    def save_and_reset_episode(self, episode_data):
        """Save episode data to HDF5 file and reset episode tracking"""
        
        observations = episode_data['observations']  # List of TensorDict observations
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        agent_positions = episode_data['agent_positions']
        episode_length = len(observations)
        
        # Episode-level data (constant for entire episode)
        world_map = episode_data['world_map']  # Single 512x512 map for entire episode
        start_pos = episode_data['start_pos']  # Single start position for entire episode
        target_pos = episode_data['target_pos']  # Single target position for entire episode
        
        # Skip episodes that are too short (likely corrupted or accidental)
        min_episode_length = 30
        if episode_length < min_episode_length:
            print(f"Skipping short episode with only {episode_length} steps (minimum: {min_episode_length})")
            return False  # Return False to indicate episode was not saved
        
        # Extract image and state components separately
        obs_images = []
        obs_states = []
        
        for obs_dict in observations:
            # obs_dict is a TensorDict with 'image' and 'state'
            obs_images.append(obs_dict['image'])  # (num_levels, 2, view_size, view_size) uint8
            obs_states.append(obs_dict['state'])  # (2,) float32
        
        # Convert to numpy arrays
        obs_images_array = torch.stack(obs_images).cpu().numpy().astype(np.uint8)  # [episode_length, num_levels, 2, view_size, view_size]
        obs_states_array = torch.stack(obs_states).cpu().numpy().astype(np.float32)  # [episode_length, 2]
        actions_array = np.array(actions, dtype=np.int32)
        rewards_array = np.array(rewards, dtype=np.float32)
        
        # Convert agent positions to numpy array (per-step data)
        agent_positions_array = torch.stack(agent_positions).cpu().numpy().astype(np.float32)  # [episode_length, 2]
        
        # Convert episode-level world state data to numpy arrays
        world_map_array = world_map.cpu().numpy().astype(np.float32)  # [512, 512]
        start_pos_array = start_pos.cpu().numpy().astype(np.int32)  # [2]
        target_pos_array = target_pos.cpu().numpy().astype(np.int32)  # [2]
        
        # Resize datasets if needed
        required_steps = self.total_steps_recorded + episode_length
        required_episodes = self.episode_count + 1
        self._resize_datasets_if_needed(required_steps, required_episodes)
        
        # Write step data
        start_idx = self.total_steps_recorded
        end_idx = start_idx + episode_length
        
        self.obs_images_dataset[start_idx:end_idx] = obs_images_array
        self.obs_states_dataset[start_idx:end_idx] = obs_states_array
        self.actions_dataset[start_idx:end_idx] = actions_array
        self.rewards_dataset[start_idx:end_idx] = rewards_array
        
        # Write agent positions (per-step data)
        self.agent_positions_dataset[start_idx:end_idx] = agent_positions_array
        
        # Write episode metadata (including world-level data)
        episode_record = np.array([
            (
                start_idx,
                episode_length,
                episode_data['episode_reward'],
                episode_data.get('success', False),
                time.strftime("%Y%m%d_%H%M%S").encode('utf-8'),
                world_map_array,  # Store world map in episode metadata
                start_pos_array,  # Store start position in episode metadata
                target_pos_array  # Store target position in episode metadata
            )
        ], dtype=self.episodes_dataset.dtype)
        
        self.episodes_dataset[self.episode_count] = episode_record[0]
        
        # Update counters
        self.total_steps_recorded += episode_length
        
        # Flush to disk periodically for safety
        if self.episode_count % 10 == 0:
            self.hdf5_file.flush()
        
        print(f"Saved episode {self.episode_count} with {episode_length} steps (total steps: {self.total_steps_recorded}) + world state data")
        
        # Increment episode count after successful save
        self.episode_count += 1
        
        return True  # Return True to indicate episode was saved successfully
    
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
                self.obs_images_dataset.resize((self.total_steps_recorded, *self.obs_images_dataset.shape[1:]))
                self.obs_states_dataset.resize((self.total_steps_recorded, *self.obs_states_dataset.shape[1:]))
                self.actions_dataset.resize((self.total_steps_recorded,))
                self.rewards_dataset.resize((self.total_steps_recorded,))
                
                # Trim agent positions dataset
                self.agent_positions_dataset.resize((self.total_steps_recorded, 2))
            
            if self.episode_count > 0:
                self.episodes_dataset.resize((self.episode_count,))
            
            self.hdf5_file.close()
            print(f"Closed HDF5 file with {self.episode_count} episodes and {self.total_steps_recorded} steps")
            self.hdf5_file = None
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Initialize environment - get initial observation and info
        obs, info = self.reset_environment()
        
        # Local variables for episode state
        done = False  # Local variable for episode completion
        current_episode_data = []  # Local variable for current episode data
        current_action = 0  # Local variable for current action
        episode_reward = 0.0  # Local variable for episode reward
        step_count = 0  # Local variable for step count
        
        # Episode-level world state (will be set on first step)
        episode_world_map = None
        episode_start_pos = None
        episode_target_pos = None
        
        while running:
            # Handle events first
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Skip current episode - discard data and restart
                        if current_episode_data:
                            print(f"Skipping episode - discarded {len(current_episode_data)} steps")
                            current_episode_data = []
                        obs, info = self.reset_environment()
                        done = False  # Reset done flag for new episode
                        episode_reward = 0.0  # Reset episode reward
                        step_count = 0  # Reset step count
                        current_action = 0  # Reset current action
                        # Reset episode-level world state
                        episode_world_map = None
                        episode_start_pos = None
                        episode_target_pos = None
            
            # Clear screen and render current state FIRST so player can see it
            self.screen.fill(self.WHITE)
            
            # Draw observation levels
            for level_idx in range(self.num_levels):
                # Extract image and state from TensorDict observation
                obs_image_level = obs['image'][0, level_idx]  # [0] for first env, image for this level
                obs_state = obs['state'][0]  # [0] for first env, state vector
                
                surface = self.obs_level_to_pygame_surface(obs_image_level)
                self.draw_level_info(surface, level_idx, obs_state)
                
                x_pos = level_idx * self.level_display_size
                
                # Draw border around the image
                border_rect = pygame.Rect(x_pos - 2, -2, self.level_display_size + 4, self.level_display_size + 4)
                pygame.draw.rect(self.screen, self.BORDER_COLOR, border_rect, 2)
                
                self.screen.blit(surface, (x_pos, 0))
            
            # Draw info panel
            self.draw_info_panel(info, done, current_action, episode_reward, step_count)
            
            # Update display so player can see the current state
            pygame.display.flip()
            
            # NOW get current action from keyboard after player has seen the screen
            keys = pygame.key.get_pressed()
            current_action = self.get_pressed_action(keys)
            
            # Step environment
            action_tensor = torch.tensor([current_action], device=self.device)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = self.env.step(action_tensor)
            
            # Record the observation-action-reward triplet plus agent position
            current_episode_data.append({
                'obs': obs[0].cpu().clone(),  # Current observation that human saw
                'action': current_action,     # Action human decided based on current obs
                'reward': reward[0].item(),   # Reward received for this action
                # Agent position (changes each step)
                'agent_pos': self.env.pos[0].cpu().clone(),  # Agent x,y position in world
            })
            
            # Store episode-level data only on first step
            if step_count == 1:  # First step of episode
                episode_world_map = self.env.cost_maps[0].cpu().clone()  # Full 512x512 world cost map
                episode_start_pos = self.env.starts[0].cpu().clone()     # Start position for this episode
                episode_target_pos = self.env.targets[0].cpu().clone()   # Target position for this episode
            
            # Update game state
            obs = next_obs
            info = next_info
            done = terminated[0].item() or truncated[0].item()
            episode_reward_delta = reward[0].item()
            episode_reward += episode_reward_delta
            step_count += 1
            
            # Handle episode end
            if done:
                success = info['success'][0].item()
                if success:
                    print(f"SUCCESS! Episode finished! Reward: {episode_reward:.2f}, Steps: {step_count}")
                else:
                    print(f"FAILED! Episode finished! Reward: {episode_reward:.2f}, Steps: {step_count}")
                
                if current_episode_data:
                    episode_data = {
                        'observations': [step['obs'] for step in current_episode_data],
                        'actions': [step['action'] for step in current_episode_data],
                        'rewards': [step['reward'] for step in current_episode_data],
                        'agent_positions': [step['agent_pos'] for step in current_episode_data], 
                        'world_map': episode_world_map,     # Single world map for entire episode
                        'start_pos': episode_start_pos,     # Single start position for entire episode
                        'target_pos': episode_target_pos,   # Single target position for entire episode
                        'episode_reward': episode_reward,
                        'episode_length': len(current_episode_data),
                        'success': success
                    }
                    self.save_and_reset_episode(episode_data)
                
                # Auto-restart after a brief pause
                pygame.time.wait(1000)  # Wait 1 second
                obs, info = self.reset_environment()
                done = False  # Reset done flag for new episode
                current_episode_data = []  # Reset episode data for new episode
                episode_reward = 0.0  # Reset episode reward
                step_count = 0  # Reset step count
                current_action = 0  # Reset current action
                # Reset episode-level world state
                episode_world_map = None
                episode_start_pos = None
                episode_target_pos = None
            
            clock.tick(30)  # 30 FPS
        
        # Close HDF5 file before quitting
        self.close_hdf5_file()
        pygame.quit()


@hydra.main(version_base=None, config_path=".", config_name="human_config")
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
        # Note: current_episode_data is now local to run() method, so no cleanup needed here
        
        # Ensure HDF5 file is properly closed
        if hasattr(player, 'close_hdf5_file'):
            player.close_hdf5_file()
        
        print("Human player session ended.")


if __name__ == '__main__':
    main()
