import os
import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F
from numba import jit, types
from numba.typed import List
from timer import get_timer


@jit(nopython=True, cache=True)
def fast_bfs_distance(eroded_mask, start_y, start_x, max_dist):
    """
    JIT-compiled BFS for computing distances from a start point.
    Much faster than pure Python implementation.
    
    Args:
        eroded_mask: 2D boolean array where True = drivable
        start_y, start_x: starting coordinates
        max_dist: maximum distance to compute (for early termination).
                 Use -1 for no limit (compute full distance map).
    
    Returns:
        dists: 2D float32 array with distances (-1 for unreachable)
        visited: 2D boolean array marking reachable areas
    """
    height, width = eroded_mask.shape
    visited = np.zeros_like(eroded_mask, dtype=types.boolean)
    dists = np.full_like(eroded_mask, np.inf, dtype=types.float32)

    # Cast start coordinates to int32 to avoid precision warnings
    start_y = types.int32(start_y)
    start_x = types.int32(start_x)
    
    # Initialize queue with start position
    # Using lists instead of deque for numba compatibility
    queue_y = List.empty_list(types.int32)
    queue_x = List.empty_list(types.int32)
    queue_y.append(start_y)
    queue_x.append(start_x)
    
    visited[start_y, start_x] = True
    dists[start_y, start_x] = 0.0
    
    # Precompute neighbor offsets
    dy = np.array([1, -1, 0, 0], dtype=types.int32)
    dx = np.array([0, 0, 1, -1], dtype=types.int32)
    
    queue_idx = 0
    while queue_idx < len(queue_y):
        y0 = queue_y[queue_idx]
        x0 = queue_x[queue_idx]
        queue_idx += 1
        
        current_dist = dists[y0, x0]
        
        # Early termination: no need to explore beyond max_dist (unless max_dist is -1)
        if max_dist >= 0 and current_dist >= max_dist:
            continue
            
        # Check all 4 neighbors
        for i in range(4):
            ny = y0 + dy[i]
            nx = x0 + dx[i]
            
            # Bounds check
            if 0 <= ny < height and 0 <= nx < width:
                # Check if drivable and unvisited
                if eroded_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dists[ny, nx] = current_dist + 1.0
                    queue_y.append(ny)
                    queue_x.append(nx)
    
    return dists, visited


class ParallelDrivingEnv:
    """
    Parallel 2D disk driving environment for GPU acceleration.
    Simulates multiple environments in parallel using 3D tensors.
    No Gymnasium inheritance - pure PyTorch implementation.
    """
    
    _accel_table = np.array([
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0, 0.0),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (0.0, -1.0),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0, 0.0),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    ], dtype=np.float32)

    def __init__(self,
                 num_envs,
                 view_size,
                 dt,
                 a_max,
                 v_max,
                 w_col,
                 R_goal,
                 disk_radius,
                 render_dir,
                 video_fps,
                 w_dist,
                 w_accel,
                 max_steps,
                 hitwall_cost,
                 pyramid_levels,  # Changed from pyramid_scales to pyramid_levels
                 num_levels,
                 min_start_goal_dist=50,
                 max_start_goal_dist=200,
                 device='cuda'):
        
        self.num_envs = num_envs
        self.map_h, self.map_w = 512, 512
        self.view_size = view_size
        self.dt = dt
        self.a_max = a_max
        self.v_max = v_max
        self.w_dist = w_dist
        self.w_col = w_col
        self.R_goal = R_goal
        self.disk_radius = disk_radius
        self.w_accel = w_accel
        self.hitwall_cost = hitwall_cost
        self.render_dir = render_dir
        self.video_fps = video_fps
        self.pyramid_levels = list(pyramid_levels)  # e.g., [0, 2, 4] for scales [1, 4, 16]
        self.num_levels = num_levels
        self.max_steps = max_steps
        self.min_start_goal_dist = min_start_goal_dist
        self.max_start_goal_dist = max_start_goal_dist
        self.device = device
        
        if self.render_dir:
            os.makedirs(self.render_dir, exist_ok=True)
        
        # Create action table on GPU
        self.action_table = torch.tensor(ParallelDrivingEnv._accel_table, 
                                       device=self.device, dtype=torch.float32)
        
        # Initialize parallel state tensors
        # Positions: (num_envs, 2) - x, y coordinates
        self.pos = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
        # Velocities: (num_envs, 2) - vx, vy
        self.vel = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
        # Cost maps: (num_envs, map_h, map_w)
        self.cost_maps = torch.zeros((num_envs, self.map_h, self.map_w), device=device, dtype=torch.float32)
        # Geodesic distance maps: (num_envs, map_h, map_w)
        self.geodesic_dist_maps = torch.zeros((num_envs, self.map_h, self.map_w), device=device, dtype=torch.float32)
        # Start and target positions: (num_envs, 2)
        self.starts = torch.zeros((num_envs, 2), device=device, dtype=torch.int32)
        self.targets = torch.zeros((num_envs, 2), device=device, dtype=torch.int32)
        # Episode state tracking
        self.step_counts = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.prev_dist_to_goals = torch.zeros(num_envs, device=device, dtype=torch.float32)
        
        # Pre-allocate downsampled and padded maps for efficient cropping
        self.padded_cost_maps = {}
        half_view = self.view_size // 2
        
        for level in self.pyramid_levels:
            downsample_factor = 2 ** level
            downsampled_h = self.map_h // downsample_factor
            downsampled_w = self.map_w // downsample_factor
            
            # Only need to pad by half_view at each level since we're using view_size crops
            self.padded_cost_maps[level] = torch.zeros(
                (self.num_envs, downsampled_h + 2 * half_view, downsampled_w + 2 * half_view),
                device=self.device, dtype=torch.float32
            )
        
        # Random number generator
        self.rng = np.random.default_rng(int(time.time() * 1e6) % (2**32 - 1))

        self.dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _gpu_gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur using GPU convolution for faster processing"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        coords = torch.arange(kernel_size, device=self.device, dtype=torch.float32)
        coords = coords - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable convolution for all environments at once
        x_expanded = x.unsqueeze(1)  # (num_envs, 1, H, W)
        
        # Horizontal blur
        kernel_h = kernel_1d.view(1, 1, 1, -1)  # (1, 1, 1, kernel_size)
        x_h = F.conv2d(x_expanded, kernel_h, padding=(0, kernel_size//2))
        
        # Vertical blur
        kernel_v = kernel_1d.view(1, 1, -1, 1)  # (1, 1, kernel_size, 1)
        x_blurred = F.conv2d(x_h, kernel_v, padding=(kernel_size//2, 0))
        
        return x_blurred.squeeze(1)  # Back to (num_envs, H, W)

    def _generate_maps(self, env_indices=None):
        """Generate maps for specified environments (or all if None)"""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        
        corridor_width = 3
        min_dist = self.min_start_goal_dist
        max_dist = self.max_start_goal_dist
        
        for env_idx in env_indices:
            while True:
                # Generate random map
                M = torch.rand((self.map_h, self.map_w), device=self.device, dtype=torch.float32)
                
                # Set border to strong positive value
                BORDER_VALUE = 1.5
                M[0, :] = BORDER_VALUE
                M[-1, :] = BORDER_VALUE
                M[:, 0] = BORDER_VALUE
                M[:, -1] = BORDER_VALUE
                M -= 0.5
                M *= 10
                M = self._gpu_gaussian_blur(M.unsqueeze(0), sigma=10.0).squeeze(0)
                
                T = 0.0
                drivable = (M <= T)
                cost_map = torch.zeros_like(M)
                cost_map[drivable] = 0.0
                cost_map[~drivable] = 1.0
                
                # Convert to CPU for erosion operations
                cost_map_cpu = cost_map.cpu().numpy()
                drivable_cpu = drivable.cpu().numpy()
                
                # Enforce minimum width by eroding drivable mask
                kernel_size = 2 * corridor_width + 1
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                eroded = cv2.erode(drivable_cpu.astype(np.uint8), se).astype(bool)
                
                # Pick random target and start
                ys_e, xs_e = np.where(eroded)
                if len(xs_e) < 2:
                    continue
                
                idx_t = self.rng.integers(len(xs_e))
                tx, ty = xs_e[idx_t], ys_e[idx_t]
                
                # Compute distances using JIT
                dists, visited = fast_bfs_distance(eroded, ty, tx, max_dist)
                
                # Find valid start positions
                pts = np.argwhere((visited) & (dists >= min_dist) & (dists <= max_dist))
                pts = pts[~((pts[:, 0] == ty) & (pts[:, 1] == tx))]
                if len(pts) == 0:
                    continue
                
                sy, sx = pts[self.rng.integers(len(pts))]
                
                # Store results
                self.cost_maps[env_idx] = cost_map
                self.starts[env_idx] = torch.tensor([sx, sy], device=self.device, dtype=torch.int32)
                self.targets[env_idx] = torch.tensor([tx, ty], device=self.device, dtype=torch.int32)
                
                # Compute geodesic distance map
                dist_map, _ = fast_bfs_distance(drivable_cpu, ty, tx, max_dist=-1)
                self.geodesic_dist_maps[env_idx] = torch.from_numpy(dist_map.astype(np.float32)).to(self.device)
                
                break
        
        # Update cached padded maps for efficient cropping
        self._update_padded_maps(env_indices)

    def _update_padded_maps(self, env_indices):
        """Update cached downsampled and padded maps for specified environments"""
        half_view = self.view_size // 2
        
        for level in self.pyramid_levels:
            downsample_factor = 2 ** level
            
            # Update padded maps for specified environments
            for env_idx in env_indices:
                # Downsample the cost map first
                if level == 0:
                    # No downsampling needed for level 0
                    downsampled_map = self.cost_maps[env_idx]
                else:
                    # Use average pooling for downsampling
                    downsampled_map = F.avg_pool2d(
                        self.cost_maps[env_idx].unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                        kernel_size=downsample_factor,
                        stride=downsample_factor
                    ).squeeze(0).squeeze(0)  # Back to (H/factor, W/factor)
                
                # Pad the downsampled map
                padded = F.pad(downsampled_map, (half_view, half_view, half_view, half_view), value=1.0) # 1.0 means wall
                self.padded_cost_maps[level][env_idx] = padded

    def reset(self):
        """Reset all environments"""
        env_indices = list(range(self.num_envs))
        self._reset(env_indices)

        obs = self._get_obs()
        
        # Create info dict
        info = {
            'hitwall': torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            'r_progress': torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            'r_goal': torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            'vel': self.vel.clone(),
            'success': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

        return obs, info

    def _reset(self, env_indices):
        # Generate new maps for specified environments
        self._generate_maps(env_indices)
        
        for env_idx in env_indices:
            # Set agent to start position
            start_pos = self.starts[env_idx].float() + 0.5  # +0.5 for pixel center
            self.pos[env_idx] = start_pos
            self.vel[env_idx] = 0.0
            
            # Initialize distance tracking
            x, y = int(self.pos[env_idx, 0].item()), int(self.pos[env_idx, 1].item())
            self.prev_dist_to_goals[env_idx] = self.geodesic_dist_maps[env_idx, y, x]

    def step(self, actions):
        # we use NEXT_STEP autoreset convention in gymnasium
        reset_env_ids = self.dones.nonzero(as_tuple=True)[0].tolist()

        with get_timer("reset"):
            self._reset(reset_env_ids)

        self.step_counts[reset_env_ids] = 0
        self.step_counts += 1

        obs, rewards, terminated, truncated, info = self._step(actions)

        self.dones = terminated | truncated

        return obs, rewards, terminated, truncated, info

    def _step(self, actions):
        """
        Step all environments in parallel.
        
        Args:
            actions: torch.Tensor of shape (num_envs,) with discrete action indices
        
        Returns:
            obs: observation tensor
            rewards: reward tensor (num_envs,)
            terminated: boolean tensor (num_envs,)
            truncated: boolean tensor (num_envs,)
            info: dict with various tensors
        """
        # Convert actions to acceleration vectors
        activate_mask = ~self.dones

        a = self.action_table[actions[activate_mask]] * self.a_max  # (num_envs, 2)

        
        # Update velocities
        self.vel[activate_mask] = self.vel[activate_mask] + a * self.dt
        
        # Limit velocity by norm (vectorized)
        v_norms = torch.norm(self.vel, dim=1, keepdim=True)  # (num_envs, 1)
        exceed_mask = v_norms.squeeze() > self.v_max
        self.vel[exceed_mask] = self.vel[exceed_mask] * (self.v_max / v_norms[exceed_mask])
        
        # Update positions
        self.pos[activate_mask] = self.pos[activate_mask] + self.vel[activate_mask] * self.dt

        # Boundary clipping
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, self.map_w - 1)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, self.map_h - 1)
        
        # Sample costs at current positions
        x_indices = self.pos[:, 0].long()
        y_indices = self.pos[:, 1].long()
        env_indices = torch.arange(self.num_envs, device=self.device)
        costs = self.cost_maps[env_indices, y_indices, x_indices]
        
        truncated = self.step_counts >= self.max_steps
        
        # Wall collision checks
        hit_wall = costs > 0
        terminated_by_wall = hit_wall
        
        # Sample geodesic distances
        geodesic_dists = self.geodesic_dist_maps[env_indices, y_indices, x_indices]
        
        # Goal checks
        terminated_by_goal = (geodesic_dists < self.disk_radius * 2) & ~terminated_by_wall
        
        # Combine termination conditions
        terminated = terminated_by_wall | terminated_by_goal
        
        # Compute rewards
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Wall collision penalty
        hitwall_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        hitwall_reward[terminated_by_wall] = self.hitwall_cost
        rewards += hitwall_reward

        # Goal reward
        goal_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        goal_reward[terminated_by_goal] = self.R_goal

        rewards += goal_reward

        # Progress reward (only for non-terminated environments)
        progress = self.prev_dist_to_goals - geodesic_dists
        progress_reward = self.w_dist * progress
        progress_reward[~activate_mask | terminated] = 0.0
        rewards += progress_reward
        
        # ensure zero reward for autoreset envs
        rewards[~activate_mask] = 0.0
        
        # Update previous distances
        self.prev_dist_to_goals[activate_mask] = geodesic_dists[activate_mask]
        
        # Generate observations (always next step observation due to auto-reset)
        with get_timer("get_obs"):
            obs = self._get_obs()
        
        # Create info dict
        info = {
            'hitwall': hitwall_reward,
            'r_progress': progress_reward,
            'r_goal': goal_reward,
            'vel': self.vel.clone(),
            'success': terminated_by_goal,
        }

        return obs, rewards, terminated, truncated, info

    def _batch_crop_integer(self, maps: torch.Tensor, tl_corners: torch.Tensor, crop_size: int):
        """
        Efficient batched cropping using integer alignment.
        
        Args:
            maps: (N, H, W) - padded and potentially downsampled maps 
            tl_corners: (N, 2) - top-left corner coordinates (x, y) in the coordinate space of maps
            crop_size: size of square crop
            
        Returns:
            crops: (N, crop_size, crop_size)
        """
        N = tl_corners.size(0)
        
        # Top-left corners are already integers
        tl_y = tl_corners[:, 1]  # Note: tl_corners is (x, y) but we need (y, x) for indexing
        tl_x = tl_corners[:, 0]
        
        # Generate row and column indices for cropping
        ys = tl_y[:, None] + torch.arange(crop_size, device=self.device)[None, :]  # (N, crop_size)
        xs = tl_x[:, None] + torch.arange(crop_size, device=self.device)[None, :]  # (N, crop_size)
        
        # Batch indexing
        b = torch.arange(N, device=self.device)[:, None, None]  # (N, 1, 1)
        crops = maps[b, ys[:, :, None], xs[:, None, :]]  # (N, crop_size, crop_size)
        
        return crops.contiguous()

    def _get_obs(self):
        """Generate observations for all environments in parallel using efficient batched cropping"""
        levels = []
        
        # Get agent positions as floats for all environments
        agent_pos = self.pos  # (num_envs, 2) - x, y coordinates
        target_pos = self.targets.float() + 0.5 # center in pixel
        half_view = self.view_size // 2
        
        for level in self.pyramid_levels:
            downsample_factor = 2 ** level
            
            # Scale agent positions to this level's coordinate space
            scaled_agent_pos = agent_pos / downsample_factor + half_view  # Add half_view for padding offset
            scaled_target_pos = target_pos / downsample_factor + half_view 
            
            # Calculate top-left corner for cropping (round to integers)
            tl_corners = (scaled_agent_pos.round() - half_view).long()  # (num_envs, 2)

            # Batch crop cost maps for all environments at once
            cost_crops = self._batch_crop_integer(
                self.padded_cost_maps[level], 
                tl_corners, 
                self.view_size
            )
            
            # Set drivable areas to -1
            cost_crops = torch.where(cost_crops == 0.0, -1.0, cost_crops)
            
            # Create velocity channels
            vel_x = self.vel[:, 0:1] / self.v_max  # (num_envs, 1)
            vel_y = self.vel[:, 1:2] / self.v_max  # (num_envs, 1)
            vel_x_maps = vel_x.unsqueeze(-1).expand(-1, self.view_size, self.view_size)
            vel_y_maps = vel_y.unsqueeze(-1).expand(-1, self.view_size, self.view_size)
            
            # Create target channels
            target_maps = torch.zeros((self.num_envs, self.view_size, self.view_size), device=self.device)
            
            # Mark agent centers for all environments at once
            center = self.view_size // 2
            target_maps[:, center-2:center+2, center-2:center+2] = 0.5
            
            # Target position relative to top-left corner of the crop
            target_in_crop = scaled_target_pos.long() - tl_corners  # (num_envs, 2)
            
            # Convert to integer coordinates within the crop
            tx_crop = target_in_crop[:, 0]  # (num_envs,)
            ty_crop = target_in_crop[:, 1]  # (num_envs,)
            
            # Create mask for valid targets (within view and with proper margin for 4x4 region)
            valid_mask = (tx_crop >= 1) & (tx_crop < self.view_size - 2) & \
                        (ty_crop >= 1) & (ty_crop < self.view_size - 2)
            
            # Use advanced indexing to mark all targets at once
            if valid_mask.any():
                valid_envs = torch.where(valid_mask)[0]
                valid_tx = tx_crop[valid_mask]
                valid_ty = ty_crop[valid_mask]
                
                # Create coordinate grids for 4x4 target regions
                target_offsets = torch.arange(-1, 3, device=self.device)
                dy, dx = torch.meshgrid(target_offsets, target_offsets, indexing='ij')
                dy_flat = dy.flatten()  # (16,)
                dx_flat = dx.flatten()  # (16,)
                
                # Expand for all valid environments
                num_valid = len(valid_envs)
                env_indices = valid_envs.repeat_interleave(16)  # (num_valid * 16,)
                y_indices = (valid_ty.unsqueeze(1) + dy_flat.unsqueeze(0)).flatten()  # (num_valid * 16,)
                x_indices = (valid_tx.unsqueeze(1) + dx_flat.unsqueeze(0)).flatten()  # (num_valid * 16,)
                
                # Set target values using advanced indexing
                target_maps[env_indices, y_indices, x_indices] = 1.0
            
            # Stack channels: cost, vel_x, vel_y, target
            level_obs = torch.stack([cost_crops, vel_x_maps, vel_y_maps, target_maps], dim=1)  # (num_envs, 4, view_size, view_size)
            levels.append(level_obs)
        
        # Stack pyramid: shape (num_envs, num_levels, 4, view_size, view_size)
        obs = torch.stack(levels, dim=1)
        return obs


def get_human_frame(obs, vel, v_max, action=None, info=None, a_max=None):
    """
    Generate a high-res visualization for human viewing.
    Modified to work with the new observation format.
    """
    # obs is now (num_levels, 4, view_size, view_size) for a single environment
    vis_levels = []
    canvas_size = 256
    num_levels = obs.shape[0]
    
    for i in range(num_levels):
        img = obs_level_to_image(obs[i], canvas_size=canvas_size)
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
        # Draw action arrow if provided
        if action is not None:
            if hasattr(action, 'item'):
                action_idx = int(action.item())
            elif isinstance(action, (np.ndarray, list)) and np.isscalar(action[0]):
                action_idx = int(action[0])
            else:
                action_idx = int(action)
            a = ParallelDrivingEnv._accel_table[action_idx]
            a_norm = np.linalg.norm(a)
            if a_norm > 1e-3:
                a_dir = a / (a_norm + 1e-6)
                a_len = int((a_norm / a_max) * (canvas_size // 4))
                tip = (int(cx + a_dir[0] * a_len), int(cy + a_dir[1] * a_len))
                cv2.arrowedLine(img, (cx, cy), tip, (0, 128, 255), 2, tipLength=0.3)
        vis_levels.append(img)
    
    frame = np.concatenate(vis_levels, axis=1)
    
    # Overlay text info
    if info is None:
        info = {}
    r_progress = info.get('r_progress', 0.0)
    r_goal = info.get('r_goal', 0.0)
    hitwall = info.get('hitwall', 0.0)
    info_lines = [
        f"r_progress: {r_progress:.2f}",
        f"r_goal: {r_goal:.2f}",
        f"hitwall: {hitwall:.2f}",
        f"Total reward: {(r_progress + r_goal + hitwall):.2f}",
    ]
    y0 = 30
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def obs_level_to_image(obs_level, canvas_size):
    """
    Convert a single observation level (4, H, W) to a visualization image.
    """
    # Convert to numpy if it's a tensor
    if hasattr(obs_level, 'cpu'):
        obs_level = obs_level.cpu().numpy()
    
    cost = obs_level[0]
    target_map = obs_level[3]
    # Normalize cost: -1 (drivable) -> 0 (black), 1 (non-drivable) -> 255 (white)
    cost_norm = (cost + 1) / 2  # -1..1 -> 0..1
    img = np.stack([cost_norm, cost_norm, cost_norm], axis=-1)  # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    # Overlay target as magenta heatmap
    magenta = np.zeros_like(img)
    magenta[..., 0] = 255  # Red
    magenta[..., 2] = 255  # Blue
    alpha = np.clip(target_map, 0, 1)[..., None]  # (H, W, 1)
    img = img * (1 - alpha) + magenta * alpha
    img = img.astype(np.uint8)
    # Upscale
    img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)
    return img
