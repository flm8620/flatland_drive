import os
import numpy as np
import cv2
import time
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class DrivingEnv(gym.Env):
    """
    2D disk driving env:
    - Discrete acceleration control (9 choices: zero or 8 compass directions)
    - Cost map penalty + collision + goal reward
    Observation: BEV image + velocity (2-d)
    Action: discrete (0-8)
    """
    metadata = {'render_modes': ['rgb_array']}
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
    ],
                            dtype=np.float32)

    def __init__(self,
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
                 pyramid_scales,
                 num_levels,
                 min_start_goal_dist=50,
                 max_start_goal_dist=200,
                 seed=None):
        super().__init__()
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
        self.video_writer = None
        if self.render_dir:
            os.makedirs(self.render_dir, exist_ok=True)
        # Discrete action space: 9 actions
        self.action_space = spaces.Discrete(9)
        self.pyramid_scales = list(pyramid_scales)
        self.num_levels = num_levels
        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(self.num_levels, 4,
                                                   view_size, view_size),
                                            dtype=np.float32)
        self.seed(seed)
        self.max_steps = max_steps
        self._step_count = 0
        self.min_start_goal_dist = min_start_goal_dist
        self.max_start_goal_dist = max_start_goal_dist
        self.is_done = None
        self.np_random = None

    def seed(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1e6) % (2**32 - 1)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [self._seed]

    def _generate_map(self):
        start_time = time.time()
        corridor_width = 3
        min_dist = self.min_start_goal_dist
        max_dist = self.max_start_goal_dist

        while True:
            M = self.np_random.random((self.map_h, self.map_w),
                                      dtype=np.float32)
            # set border to strong positive value as world border
            BORDER_VALUE = 1.5
            M[0, :] = BORDER_VALUE
            M[-1, :] = BORDER_VALUE
            M[:, 0] = BORDER_VALUE
            M[:, -1] = BORDER_VALUE
            M -= 0.5
            M *= 10
            M = cv2.GaussianBlur(M, (0, 0), sigmaX=10, sigmaY=10)

            T = 0.0
            drivable = (M <= T)
            self.cost_map = np.zeros_like(M)
            # Drivable area: 0, Non-drivable: 1.0 (sharp boundary)
            self.cost_map[drivable] = 0.0
            self.cost_map[~drivable] = 1.0

            # 3) Enforce minimum width by eroding drivable mask
            kernel_size = 2 * corridor_width + 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
            eroded = cv2.erode(drivable.astype(np.uint8), se).astype(bool)

            # 4) Pick a random target on eroded mask
            ys_e, xs_e = np.where(eroded)
            if len(xs_e) < 2:
                continue  # Not enough drivable area, regenerate

            # Randomly pick a target
            idx_t = self.np_random.integers(len(xs_e))
            tx, ty = xs_e[idx_t], ys_e[idx_t]

            # BFS flood-fill from (ty, tx) over the eroded mask
            visited = np.zeros_like(eroded, bool)
            queue = deque([(ty, tx)])
            visited[ty, tx] = True
            dists = np.full_like(eroded, -1, dtype=np.float32)
            dists[ty, tx] = 0
            while queue:
                y0, x0 = queue.popleft()
                for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ny, nx = y0 + dy, x0 + dx
                    if 0 <= ny < self.map_h and 0 <= nx < self.map_w \
                        and eroded[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        dists[ny, nx] = dists[y0, x0] + 1
                        queue.append((ny, nx))

            # Collect all positions within [min_dist, max_dist] (in L2 distance)
            pts = np.argwhere((visited) & (dists >= min_dist)
                              & (dists <= max_dist))
            # drop the target itself
            pts = pts[~((pts[:, 0] == ty) & (pts[:, 1] == tx))]
            if len(pts) == 0:
                continue  # No valid start, regenerate map

            # Randomly select one as start
            sy, sx = pts[self.np_random.integers(len(pts))]
            # Notice that for start and target, we use interger pixel index as coordinates
            # for float point coordinate, we think they are centered at the pixel square,
            # which coordinate should be (sx + 0.5, sy + 0.5) and (tx + 0.5, ty + 0.5)
            self.start = np.array([sx, sy], dtype=np.int32)
            self.target = np.array([tx, ty], dtype=np.int32)
            elapsed = time.time() - start_time
            # print(
            #     f"[DrivingEnv] Map generated in {elapsed:.3f} seconds. Start-goal dist: {dists[sy, sx]:.1f}"
            # )
            # Compute geodesic distance map after map generation
            self._compute_geodesic_distance_map()
            break

    def _compute_geodesic_distance_map(self):
        """
        Compute the smooth geodesic (quasi-Euclidean) distance from every drivable cell to the target using scikit-image's MCP_Geometric.
        The result is stored in self.geodesic_dist_map (float32, shape=(map_h, map_w)).
        Walls (cost_map > 0) are treated as obstacles (infinite cost).
        """
        from skimage.graph import MCP_Geometric
        # Mask: True for drivable, False for wall
        drivable_mask = (self.cost_map == 0)
        # MCP_Geometric expects costs: 0 for wall, 1 for drivable
        costs = np.where(drivable_mask, 1.0, np.inf)
        mcp = MCP_Geometric(costs)
        # Target is (row, col) = (y, x)
        ty, tx = self.target[1], self.target[0]
        # Compute distances from target to all points
        dist_map, _ = mcp.find_costs([(ty, tx)])
        self.geodesic_dist_map = dist_map.astype(np.float32)

    def visualize_map(self, filename='map.png'):
        """
        Write an RGB visualization of the entire cost_map to render_dir:
        - drivable pixels in white
        - others colored from blue (low cost) to red (high cost)
        - start in green, target in red
        """
        if not self.render_dir:
            return
        h, w = self.cost_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        # drivable mask
        drivable = (self.cost_map == 0)
        # white for drivable
        vis[drivable] = [255, 255, 255]
        vis[~drivable] = [0, 0, 0]
        # draw points
        cv2.circle(vis,
                   tuple(self.start),
                   radius=5,
                   color=(0, 255, 0),
                   thickness=-1)
        cv2.circle(vis,
                   tuple(self.target),
                   radius=5,
                   color=(0, 0, 255),
                   thickness=-1)
        path = os.path.join(self.render_dir, filename)
        cv2.imwrite(path, vis)

    def reset(self, *, seed=None, options=None):
        self.is_done = False
        if seed is not None:
            self.seed(seed)
        # Regenerate the map for each episode
        self._generate_map()
        # Set agent to the designated start position from map generation
        self.pos = self.start.astype(np.float32) + 0.5  # +0.5 for pixel center
        self.vel = np.zeros(2, dtype=np.float32)
        obs = self._get_obs()
        self._step_count = 0  # Reset step counter
        # Use geodesic distance for progress reward
        x, y = int(self.pos[0]), int(self.pos[1])
        self._prev_dist_to_goal = self.geodesic_dist_map[y, x]
        return obs, {}

    def step(self, action):
        assert self.is_done is not None
        assert not self.is_done
        # Map discrete action to acceleration vector
        a = DrivingEnv._accel_table[action]
        # Update velocity
        self.vel = self.vel + a * self.dt
        # Limit velocity by norm (ball projection)
        v_norm = np.linalg.norm(self.vel)
        if v_norm > self.v_max:
            self.vel = self.vel * (self.v_max / v_norm)
        self.pos = self.pos + self.vel * self.dt

        # boundary warp
        self.pos[0] = np.clip(self.pos[0], 0, self.map_w - 1)
        self.pos[1] = np.clip(self.pos[1], 0, self.map_h - 1)

        # costs
        x_i, y_i = int(self.pos[0]), int(self.pos[1])
        cost = self.cost_map[y_i, x_i]

        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        # Wall collision: terminate and give high negative reward
        hit_wall = cost > 0
        if hit_wall:
            reward = self.hitwall_cost
            terminated = True
            obs = self._get_obs()
            info = {
                'hitwall': reward,
                'r_progress': 0.0,
                'r_goal': 0.0,
                'r_col': 0.0,
                'success': False  # Not a success if hit wall
            }
            self.is_done = terminated or truncated
            return obs, reward, terminated, truncated, info

        # collision (single agent skip)
        r_col = 0.0

        # goal
        dist_to_goal = self.geodesic_dist_map[y_i, x_i]
        terminated = dist_to_goal < self.disk_radius * 2
        r_goal = self.R_goal if terminated else 0.0
        # Progress reward: positive if moving closer to goal (using geodesic distance)
        progress = self._prev_dist_to_goal - dist_to_goal
        r_progress = self.w_dist * progress  # per-pixel, not normalized
        self._prev_dist_to_goal = dist_to_goal
        # --- Time penalty ---
        r_time = -0.01  # Small negative reward per step
        obs = self._get_obs()
        reward = r_col + r_goal + r_progress + r_time

        info = {
            'hitwall': 0.0,
            'r_progress': r_progress,
            'r_goal': r_goal,
            'r_col': r_col,
            'r_time': r_time,
            'vel': self.vel.copy(),
            'success': terminated,
        }

        self.is_done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        levels = []
        for scale in self.pyramid_scales:
            size = self.view_size
            crop_size = int(size * scale)
            x, y = self.pos
            x = int(round(x))
            y = int(round(y))
            half = crop_size // 2
            x0 = x - half
            x1 = x0 + crop_size
            y0 = y - half
            y1 = y0 + crop_size
            # Pad if out of bounds
            pad_x0, pad_y0 = max(0, -x0), max(0, -y0)
            pad_x1, pad_y1 = max(0, x1 - self.map_w), max(0, y1 - self.map_h)
            x0c, x1c = max(0, x0), min(self.map_w, x1)
            y0c, y1c = max(0, y0), min(self.map_h, y1)
            view = np.zeros((crop_size, crop_size), dtype=np.float32)
            view[pad_y0:crop_size - pad_y1,
                 pad_x0:crop_size - pad_x1] = self.cost_map[y0c:y1c, x0c:x1c]
            # Set drivable area (cost==0) to -1.0 for visual distinction
            view[view == 0.0] = -1.0
            # Downsample to view_size
            if scale > 1:
                view = cv2.resize(view, (self.view_size, self.view_size),
                                  interpolation=cv2.INTER_AREA)
            # velocity map (two channels: vx and vy, normalized)
            vx = np.full_like(view, float(self.vel[0]) / self.v_max)
            vy = np.full_like(view, float(self.vel[1]) / self.v_max)
            # target map
            target_map = np.zeros_like(view)
            tx, ty = self.target
            # Compute target position in this crop
            if x0 <= tx < x1 and y0 <= ty < y1:
                tx_rel = int((0.5 + (tx - x0)) / scale)
                ty_rel = int((0.5 + (ty - y0)) / scale)
                if 0 <= tx_rel < self.view_size and 0 <= ty_rel < self.view_size:
                    # Draw a filled square (5x5) in numpy for target
                    y_start = max(0, ty_rel - 2)
                    y_end = min(self.view_size, ty_rel + 3)
                    x_start = max(0, tx_rel - 2)
                    x_end = min(self.view_size, tx_rel + 3)
                    target_map[y_start:y_end, x_start:x_end] = 1.0
                    
            # Always paint the center of observation, as the indicator of self
            center_x = self.view_size // 2
            center_y = self.view_size // 2
            target_map[center_y - 2:center_y + 2,
                       center_x - 2:center_x + 2] = 0.5

            # Stack: cost, vx, vy, target
            level = np.stack([view, vx, vy, target_map], axis=0)  # (4, H, W)
            levels.append(level)
        # Stack pyramid: shape (num_levels, 4, view_size, view_size)
        obs = np.stack(levels, axis=0).astype(np.float32)
        return obs

    def visualize_distance_map(self, filename='distance_map.png', period=20.0):
        """
        Visualize the geodesic distance map with a periodic colormap to show contour-like effects.
        - Uses a periodic (e.g., sine or modulo) mapping to highlight contours.
        - Saves the image to render_dir/filename.
        """
        if not self.render_dir:
            return
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        dist = self.geodesic_dist_map.copy()
        # Mask walls as nan for visualization
        dist[self.cost_map > 0] = np.nan
        # Periodic colormap: use modulo to create contours
        contour = np.mod(dist, period)
        # Normalize for colormap
        norm = mcolors.Normalize(vmin=0, vmax=period)
        # Use a cyclic colormap (e.g., 'hsv' or 'twilight')
        cmap = plt.get_cmap('twilight')
        img = cmap(norm(contour))[:, :, :3]  # Drop alpha
        # Overlay start and target
        sy, sx = self.start[1], self.start[0]
        ty, tx = self.target[1], self.target[0]
        img[sy - 3:sy + 4, sx - 3:sx + 4, :] = [0, 1, 0]  # Green for start
        img[ty - 3:ty + 4, tx - 3:tx + 4, :] = [1, 0, 0]  # Red for target
        # Save as PNG
        path = os.path.join(self.render_dir, filename)
        plt.imsave(path, img)
        plt.close()


def obs_level_to_image(obs_level, canvas_size):
    """
    Convert a single observation level (4, H, W) to a visualization image.
    - Cost channel: map to grayscale (min=-1, max=1)
    - Target: overlay in red (strong), and also as a magenta heatmap
    """
    cost = obs_level[0]
    target_map = obs_level[3]  # channel 3 is target
    # Normalize cost: -1 (drivable) -> 0 (black), 1 (non-drivable) -> 255 (white)
    cost_norm = (cost + 1) / 2  # -1..1 -> 0..1
    img = np.stack([cost_norm, cost_norm, cost_norm], axis=-1)  # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    # Overlay target as magenta heatmap (blend with cost image)
    magenta = np.zeros_like(img)
    magenta[..., 0] = 255  # Red
    magenta[..., 2] = 255  # Blue
    alpha = np.clip(target_map, 0, 1)[..., None]  # (H, W, 1)
    img = img * (1 - alpha) + magenta * alpha
    img = img.astype(np.uint8)
    # Upscale
    img = cv2.resize(img, (canvas_size, canvas_size),
                     interpolation=cv2.INTER_NEAREST)
    return img


def get_human_frame(obs, vel, v_max, action=None, info=None, a_max=None):
    """
    Generate a high-res visualization for human viewing.
    - Uses the current observation as the base image for each level
    - Draws velocity and action vectors as overlays
    - Overlays reward info from the provided info dict (if any)
    """
    vis_levels = []
    canvas_size = 256
    num_levels = len(obs)
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
            cv2.arrowedLine(img, (cx, cy),
                            tip, (255, 255, 0),
                            2,
                            tipLength=0.3)
        # Draw action arrow if provided (discrete action index)
        if action is not None:
            # If action is a tensor or numpy scalar, convert to int
            if hasattr(action, 'item'):
                action_idx = int(action.item())
            elif isinstance(action,
                            (np.ndarray, list)) and np.isscalar(action[0]):
                action_idx = int(action[0])
            else:
                action_idx = int(action)
            # Get the corresponding acceleration vector
            a = DrivingEnv._accel_table[action_idx]
            a_norm = np.linalg.norm(a)
            if a_norm > 1e-3:
                a_dir = a / (a_norm + 1e-6)
                a_len = int((a_norm / a_max) * (canvas_size // 4))
                tip = (int(cx + a_dir[0] * a_len), int(cy + a_dir[1] * a_len))
                cv2.arrowedLine(img, (cx, cy),
                                tip, (0, 128, 255),
                                2,
                                tipLength=0.3)
        vis_levels.append(img)
    frame = np.concatenate(vis_levels, axis=1)
    # --- Overlay text info ---
    # Use info dict if provided, else fallback to zeros
    if info is None:
        info = {}
    r_progress = info.get('r_progress', 0.0)
    r_goal = info.get('r_goal', 0.0)
    r_col = info.get('r_col', 0.0)
    hitwall = info.get('hitwall', 0.0)
    info_lines = [
        f"r_progress: {r_progress:.2f}",
        f"r_goal: {r_goal:.2f}",
        f"r_col: {r_col:.2f}",
        f"hitwall: {hitwall:.2f}",
        f"Total reward: {(r_progress + r_goal + r_col + hitwall):.2f}",
    ]
    y0 = 30
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
