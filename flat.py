'''
We want to create a Renforcement Learning Playground project. 
In this project, I want to train a network for autonomous driving. 
To simplify the task, I will assume each agent (car) is a disk in 2D world. 
And the driver can only control the acceleration vector in x y plane for each frame. 
The acceleration vector's norm should be limited not to exceed a maximum value.

And for the 2D world map, I think maybe we can simply generate a large random texture as a maze-like map. 
Each pixel has a value. value 0 means drivable, i.e. driving on it will not cause any punishment. 

For other positive values, driving on it per frame will cause punishment proportional to this value. 
So the agent should avoid driving on high value pixels.

For the target, there should be a clear marker on the map indicating the target. 
When reaching the target, the agent will get a large reward.

And what I want also train is the ability to avoid crashing on other agent. 
So if two agent is crashing on to each other, 
each of the agent will suffer a punishment proportional to the relative speed of crash.

And I want the entire project mainly written in pytorch. 
For the map generation and rendering, you can use other lib, 
for example OpenGL or Vulkan or other libs. 
The network should only take the current BEV image as input, together with current speed vector.
'''

import os
import numpy as np
import math
import cv2
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from opensimplex import OpenSimplex
import time
# Hydra and TensorBoard imports
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from torch.utils.tensorboard import SummaryWriter


##########################
# RoPE Implementation
##########################
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device=None):
        t = torch.arange(seq_len, device=self.inv_freq.device if device is None else device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    @staticmethod
    def apply_rotary(x, rope_emb):
        # x: (B, N, num_heads, head_dim)
        # rope_emb: (N, head_dim)
        # Only apply to the first 2*half_dim (even, odd pairs)
        head_dim = x.shape[-1]
        half_dim = head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:2*half_dim]
        sin = rope_emb[:, :half_dim].unsqueeze(0).unsqueeze(2)  # (1, N, 1, half_dim)
        cos = rope_emb[:, half_dim:2*half_dim].unsqueeze(0).unsqueeze(2)  # (1, N, 1, half_dim)
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        x_rot = torch.cat([x_rot1, x_rot2], dim=-1)
        # If head_dim > 2*half_dim, append the rest unchanged
        if head_dim > 2*half_dim:
            x_rot = torch.cat([x_rot, x[..., 2*half_dim:]], dim=-1)
        return x_rot


##########################
# 1. Environment
##########################
class DrivingEnv(gym.Env):
    """
    2D disk driving env:
    - Continuous acceleration control
    - Cost map penalty + collision + goal reward
    Observation: BEV image + velocity (2-d)
    Action: acceleration vector (2-d)
    """
    # metadata = {'render.modes': ['human']}
    metadata = {'render.modes': []}

    def __init__(self,
                 view_size=64,
                 dt=0.1,
                 a_max=2.0,
                 v_max=10.0,
                 w_cost=1.0,
                 w_col=1.0,
                 R_goal=500.0,
                 num_agents=1,
                 disk_radius=2.0,
                 render_dir=None,
                 video_fps=30,
                 w_dist=10000.0,
                 w_accel=0.1):
        super().__init__()
        self.map_h, self.map_w = 512, 512
        self.view_size = view_size
        self.dt = dt
        self.a_max = a_max
        self.v_max = v_max
        self.w_cost = w_cost
        self.w_dist = w_dist
        self.w_col = w_col
        self.R_goal = R_goal
        self.disk_radius = disk_radius
        self.num_agents = num_agents
        self.w_accel = w_accel

        # Directory to dump frames or video if set
        self.render_dir = render_dir
        self.video_fps = video_fps
        self.episode_count = 0
        self.video_writer = None
        if self.render_dir:
            os.makedirs(self.render_dir, exist_ok=True)

        # Action: 2-d acceleration
        self.action_space = spaces.Box(low=-a_max,
                                       high=a_max,
                                       shape=(2, ),
                                       dtype=np.float32)

        # Obs: image + velocity
        self.pyramid_scales = [1, 4, 16]
        self.num_levels = len(self.pyramid_scales)
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.num_levels, 3,
                                                   view_size, view_size),
                                            dtype=np.float32)
        # we'll concatenate flattened vel channels at right-most pixels

        self.seed()

        self.canvas_size = 256  # For human video rendering

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._seed = seed or np.random.randint(1e6)
        return [self._seed]

    def _generate_map(self):
        start_time = time.time()
        map_mode = 'random'  # 'random' or 'perlin'
        corridor_width = 3
        min_dist = 50

        if map_mode == 'perlin':
            perlin_scale = 50.0
            simplex = OpenSimplex(seed=self._seed)
            xs = np.arange(self.map_w)
            ys = np.arange(self.map_h)
            grid_x, grid_y = np.meshgrid(xs, ys)
            M = np.vectorize(lambda i, j: simplex.noise2d(
                i / perlin_scale, j / perlin_scale))(grid_y, grid_x)
            M = M.astype(np.float32)
        else:
            rng = np.random.RandomState(self._seed)
            M = rng.rand(self.map_h, self.map_w).astype(np.float32)
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
            # save M to tiff file for debugging
            cv2.imwrite(os.path.join(self.render_dir, 'M_map.tiff'), M)
            cv2.imwrite(os.path.join(self.render_dir, 'cost_map.tiff'),
                        self.cost_map)

        drivable = self.cost_map == 0

        # 3) Enforce minimum width by eroding drivable mask
        kernel_size = 2 * corridor_width + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
        eroded = cv2.erode(drivable.astype(np.uint8), se).astype(bool)

        # 4) Pick a random target on eroded mask
        ys_e, xs_e = np.where(eroded)
        if len(xs_e) < 2:
            raise RuntimeError(
                "Not enough wide drivable area to place start and target")

        best_pair = None  # will store (tx, ty, sx, sy)
        best_dist = -1
        chosen = False

        for _ in range(10):
            # sample a candidate target
            idx_t = np.random.randint(len(xs_e))
            tx_c, ty_c = xs_e[idx_t], ys_e[idx_t]

            # BFS flood-fill from (ty_c, tx_c) over the eroded mask
            visited = np.zeros_like(eroded, bool)
            queue = deque([(ty_c, tx_c)])
            visited[ty_c, tx_c] = True
            while queue:
                y0, x0 = queue.popleft()
                for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ny, nx = y0 + dy, x0 + dx
                    if 0 <= ny < self.map_h and 0 <= nx < self.map_w \
                    and eroded[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            pts = np.argwhere(visited)
            # drop the target itself
            pts = pts[~((pts[:, 0] == ty_c) & (pts[:, 1] == tx_c))]
            if len(pts) == 0:
                continue

            # measure distances
            dists = np.linalg.norm(pts - np.array([ty_c, tx_c]), axis=1)
            idx_max = np.argmax(dists)
            max_dist = dists[idx_max]

            # if this pair is far enough, take it immediately
            if max_dist >= min_dist:
                sy, sx = pts[idx_max]
                tx, ty = tx_c, ty_c
                chosen = True
                break

            # otherwise remember it if it's the best so far
            if max_dist > best_dist:
                best_dist = max_dist
                sy, sx = pts[idx_max]
                best_pair = (tx_c, ty_c, sx, sy)

        # if none exceeded min_dist, use the best one
        if not chosen and best_pair is not None:
            tx, ty, sx, sy = best_pair

        # Assign start and target
        self.start = np.array([sx, sy], dtype=np.int32)
        self.target = np.array([tx, ty], dtype=np.int32)
        elapsed = time.time() - start_time
        print(f"[DrivingEnv] Map generated in {elapsed:.3f} seconds.")

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
        episode_start_time = time.time()
        self.episode_count += 1
        if seed is not None:
            self.seed(seed)
        # Regenerate the map for each episode
        self._generate_map()
        # Dump global map visualization for this episode
        if self.render_dir:
            self.visualize_map(filename=f"map_ep{self.episode_count:03d}.png")
        # Set agent to the designated start position from map generation
        self.pos = self.start.astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        # Only generate video for every 10th episode
        if self.render_dir:
            video_path = os.path.join(self.render_dir,
                                      f'episode_{self.episode_count:03d}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                self.video_fps,
                (self.canvas_size * self.num_levels, self.canvas_size),
                isColor=True)
        else:
            self.video_writer = None
        obs = self._get_obs()
        self._episode_start_time = episode_start_time  # Save for timing in step
        return obs, {}

    def step(self, action):
        # Limit acceleration by norm (ball projection)
        a = np.array(action)
        a_norm = np.linalg.norm(a)
        if a_norm > self.a_max:
            a = a * (self.a_max / a_norm)
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
        r_step = -self.w_cost * cost

        # collision (single agent skip)
        r_col = 0.0

        # goal
        dist_to_goal = np.linalg.norm(self.pos - self.target)
        terminated = dist_to_goal < self.disk_radius * 2
        r_goal = self.R_goal if terminated else 0.0
        # Progress reward: positive if moving closer to goal
        max_dist = np.linalg.norm([self.map_w, self.map_h])
        if not hasattr(self, '_prev_dist_to_goal'):
            self._prev_dist_to_goal = dist_to_goal
        progress = self._prev_dist_to_goal - dist_to_goal
        r_progress = self.w_dist * (progress / max_dist)
        self._prev_dist_to_goal = dist_to_goal
        # Acceleration penalty (normalized)
        a_norm = np.linalg.norm(action)
        excess = max(0, a_norm - self.a_max)
        r_accel = -self.w_accel * (excess / self.a_max)

        obs = self._get_obs()
        reward = r_step + r_col + r_goal + r_progress + r_accel

        # Print reward details for debugging every 100 steps
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
        if self._step_count % 100 == 0:
            print(
                f"[Step {self._step_count}] cost: {cost:.2f}, r_step: {r_step:.2f}, r_col: {r_col:.2f}, r_goal: {r_goal:.2f}, r_progress: {r_progress:.2f}, r_accel: {r_accel:.2f}, total_reward: {reward:.2f}"
            )

        # Render this step: write to video writer
        if self.render_dir and self.video_writer is not None:
            frame = self.get_human_frame(action=action,
                                         canvas_size=self.canvas_size)
            self.video_writer.write(frame)
            if terminated:
                self.video_writer.release()
                self.video_writer = None
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # Pyramid levels: (scale, view_size)
        pyramid_scales = [1, 4, 16]  # 1: high-res, 4: mid, 16: low-res
        levels = []
        for scale in pyramid_scales:
            size = self.view_size
            # Compute crop size in map pixels
            crop_size = size * scale
            x, y = self.pos
            half = crop_size // 2
            x0 = int(x) - half
            x1 = x0 + crop_size
            y0 = int(y) - half
            y1 = y0 + crop_size
            # pad
            pad_x0, pad_y0 = max(0, -x0), max(0, -y0)
            pad_x1, pad_y1 = max(0, x1 - self.map_w), max(0, y1 - self.map_h)
            x0c, x1c = max(0, x0), min(self.map_w, x1)
            y0c, y1c = max(0, y0), min(self.map_h, y1)
            view = np.zeros((crop_size, crop_size), dtype=np.float32)
            view[pad_y0:crop_size-pad_y1, pad_x0:crop_size-pad_x1] = \
                self.cost_map[y0c:y1c, x0c:x1c]
            # Set drivable area (cost==0) to -1.0 for visual distinction
            view[view == 0.0] = -1.0
            # Downsample to view_size
            if scale > 1:
                view = cv2.resize(view, (self.view_size, self.view_size),
                                  interpolation=cv2.INTER_AREA)

            # velocity map (now two channels: vx and vy, normalized)
            vx = np.full_like(view, self.vel[0] / self.v_max)
            vy = np.full_like(view, self.vel[1] / self.v_max)
            # target map
            target_map = np.zeros_like(view)
            tx, ty = self.target
            # Compute target position in this crop
            if x0 <= tx < x1 and y0 <= ty < y1:
                tx_rel = int((tx - x0) / scale)
                ty_rel = int((ty - y0) / scale)
                if 0 <= tx_rel < self.view_size and 0 <= ty_rel < self.view_size:
                    cv2.circle(target_map, (tx_rel, ty_rel),
                               color=1.0,
                               radius=2,
                               thickness=-1)
            # Stack: cost, vx, vy, target
            levels.append(np.stack([view, vx, vy, target_map], axis=0))
        # Stack pyramid: shape (num_levels, 4, view_size, view_size)
        return np.stack(levels, axis=0)

    def obs_level_to_image(self, obs_level, canvas_size=256):
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
        # Overlay target (where target_map > 0.5) in strong red
        target_mask = target_map > 0.5
        img[target_mask] = [0, 0, 255]
        # Upscale
        img = cv2.resize(img, (canvas_size, canvas_size),
                         interpolation=cv2.INTER_NEAREST)
        return img

    def get_human_frame(self, action=None, canvas_size=256):
        """
        Generate a high-res visualization for human viewing.
        - Uses the current observation as the base image for each level
        - Draws velocity and action vectors as overlays
        """
        obs = self._get_obs()  # shape: (num_levels, 3, H, W)
        vis_levels = []
        for i in range(self.num_levels):
            img = self.obs_level_to_image(obs[i], canvas_size=canvas_size)
            # Draw agent (center)
            cx, cy = canvas_size // 2, canvas_size // 2
            cv2.circle(
                img, (cx, cy),
                max(3, int(self.disk_radius * (canvas_size / self.view_size))),
                (0, 255, 0), 2)
            # Draw velocity vector
            v_norm = np.linalg.norm(self.vel)
            if v_norm > 1e-3:
                v_dir = self.vel / (v_norm + 1e-6)
                v_len = int((v_norm / self.v_max) * (canvas_size // 4))
                tip = (int(cx + v_dir[0] * v_len), int(cy + v_dir[1] * v_len))
                cv2.arrowedLine(img, (cx, cy),
                                tip, (255, 255, 0),
                                2,
                                tipLength=0.3)
            # Draw action vector if provided
            if action is not None:
                a_norm = np.linalg.norm(action)
                if a_norm > 1e-3:
                    a_dir = action / (a_norm + 1e-6)
                    a_len = int((a_norm / self.a_max) * (canvas_size // 4))
                    tip = (int(cx + a_dir[0] * a_len),
                           int(cy + a_dir[1] * a_len))
                    cv2.arrowedLine(img, (cx, cy),
                                    tip, (0, 128, 255),
                                    2,
                                    tipLength=0.3)
            vis_levels.append(img)
        frame = np.concatenate(vis_levels, axis=1)
        return frame


##########################
# 2. Replay Buffer
##########################
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), np.stack(actions), np.array(rewards, dtype=np.float32), \
               np.stack(next_states), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


##########################
# 3. Networks
##########################


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.heads = heads
        self.head_dim = dim // heads
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        x_norm = self.norm1(x)
        # Prepare Q, K, V for MultiheadAttention
        q = k = v = x_norm
        # Reshape for RoPE: (B, N, heads, head_dim)
        q = q.view(B, N, self.heads, self.head_dim)
        k = k.view(B, N, self.heads, self.head_dim)
        rope_emb = self.rope(N, device=x.device)  # (N, head_dim)
        q = RotaryEmbedding.apply_rotary(q, rope_emb)
        k = RotaryEmbedding.apply_rotary(k, rope_emb)
        # Merge heads back: (B, N, D)
        q = q.reshape(B, N, D)
        k = k.reshape(B, N, D)
        # MultiheadAttention expects (B, N, D)
        attn_out, _ = self.attn(q, k, v)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self,
                 view_size,
                 patch_size=8,
                 in_chans=4,
                 embed_dim=128,
                 depth=6,
                 num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_proj = nn.ModuleList([
            nn.Linear(in_chans * patch_size * patch_size, embed_dim)
            for _ in range(num_levels)
        ])
        self.transformer = nn.ModuleList([
            nn.Sequential(*[
                TransformerBlock(embed_dim, heads=8, mlp_ratio=8)
                for _ in range(depth)
            ]) for _ in range(num_levels)
        ])
        self.fc_out = nn.Linear(embed_dim * num_levels, embed_dim * num_levels)
        self.view_size = view_size

    def forward(self, x):
        # x: (B, num_levels, 4, H, W)
        feats = []
        for i in range(self.num_levels):
            xi = x[:, i]  # (B, 4, H, W)
            B, C, H, W = xi.shape
            # Patchify
            patches = xi.unfold(2, self.patch_size, self.patch_size).unfold(
                3, self.patch_size, self.patch_size)
            patches = patches.permute(0, 2, 3, 1, 4,
                                      5).contiguous()  # (B, nH, nW, C, pH, pW)
            patches = patches.view(B, -1, C * self.patch_size *
                                   self.patch_size)  # (B, N, patch_dim)
            # Linear proj
            tokens = self.patch_proj[i](patches)  # (B, N, D)
            tokens = self.transformer[i](tokens)
            # Pool (mean)
            pooled = tokens.mean(dim=1)
            feats.append(pooled)
        out = torch.cat(feats, dim=-1)
        out = self.fc_out(out)
        return out


class Actor(nn.Module):

    def __init__(self, view_size, action_dim=2, num_levels=3):
        super().__init__()
        self.encoder = ViTEncoder(view_size,
                                  num_levels=num_levels,
                                  embed_dim=128)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.embed_dim * num_levels, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.log_std = nn.Parameter(torch.full(
            (action_dim, ), -1.0))  # Lower initial std for smoother actions

    def forward(self, obs):
        # obs shape: (B, num_levels, 3, H, W)
        h = self.encoder(obs)
        mu = self.net(h)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist


class Critic(nn.Module):

    def __init__(self, view_size, action_dim=2, num_levels=3):
        super().__init__()
        self.encoder = ViTEncoder(view_size,
                                  num_levels=num_levels,
                                  embed_dim=128)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.embed_dim * num_levels + action_dim, 128),
            nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs, action):
        h = self.encoder(obs)
        x = torch.cat([h, action], dim=1)
        return self.net(x)


##########################
# 4. SAC Trainer
##########################
def soft_update(net, target_net, tau):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.mul_(1 - tau)
        tp.data.add_(tau * p.data)


def train(cfg: DictConfig):
    # Set up run directory and TensorBoard
    run_name = cfg.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = to_absolute_path(cfg.output_dir)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DrivingEnv(
        render_dir=run_dir if cfg.env.save_viz else None,
        view_size=cfg.env.view_size,
        dt=cfg.env.dt,
        a_max=cfg.env.a_max,
        v_max=cfg.env.v_max,
        w_cost=cfg.env.w_cost,
        w_col=cfg.env.w_col,
        R_goal=cfg.env.R_goal,
        num_agents=cfg.env.num_agents,
        disk_radius=cfg.env.disk_radius,
        video_fps=cfg.env.video_fps,
        w_dist=cfg.env.w_dist,
        w_accel=cfg.env.w_accel,
    )
    replay = ReplayBuffer(cfg.train.replay_size)

    num_levels = env.num_levels
    actor = Actor(env.view_size, num_levels=num_levels).to(device)
    critic1 = Critic(env.view_size, num_levels=num_levels).to(device)
    critic2 = Critic(env.view_size, num_levels=num_levels).to(device)
    target_critic1 = Critic(env.view_size, num_levels=num_levels).to(device)
    target_critic2 = Critic(env.view_size, num_levels=num_levels).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=cfg.train.lr)
    opt_critic1 = optim.Adam(critic1.parameters(), lr=cfg.train.lr)
    opt_critic2 = optim.Adam(critic2.parameters(), lr=cfg.train.lr)

    gamma = cfg.train.gamma
    tau = cfg.train.tau
    alpha = cfg.train.alpha
    batch_size = cfg.train.batch_size
    warmup = cfg.train.warmup

    total_steps = 0
    for ep in range(1, cfg.train.episodes + 1):
        ep_start_time = time.time()
        obs, _ = env.reset()
        obs = torch.tensor(obs[None], dtype=torch.float32, device=device)
        ep_reward = 0
        ep_loss_q1 = 0
        ep_loss_q2 = 0
        ep_loss_pi = 0
        updates = 0
        for t in range(1, cfg.train.max_steps + 1):
            total_steps += 1
            # sample action
            with torch.no_grad():
                dist = actor(obs)
                action = dist.sample().cpu().numpy()[0]
            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(obs.cpu().numpy()[0], action, r, next_obs, done)
            obs = torch.tensor(next_obs[None],
                               dtype=torch.float32,
                               device=device)
            ep_reward += r

            if len(replay) > warmup:
                s, a, rew, s2, d = replay.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.float32, device=device)
                rew = torch.tensor(rew, dtype=torch.float32,
                                   device=device).unsqueeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32,
                                 device=device).unsqueeze(1)

                with torch.no_grad():
                    dist_next = actor(s2)
                    a2 = dist_next.rsample()
                    logp2 = dist_next.log_prob(a2).sum(-1, keepdim=True)
                    q1_target = target_critic1(s2, a2)
                    q2_target = target_critic2(s2, a2)
                    q_target = torch.min(q1_target, q2_target) - alpha * logp2
                    y = rew + (1 - d) * gamma * q_target

                q1 = critic1(s, a)
                q2 = critic2(s, a)
                loss_q1 = F.mse_loss(q1, y)
                loss_q2 = F.mse_loss(q2, y)
                opt_critic1.zero_grad()
                loss_q1.backward()
                opt_critic1.step()
                opt_critic2.zero_grad()
                loss_q2.backward()
                opt_critic2.step()

                # actor update
                dist_curr = actor(s)
                a_curr = dist_curr.rsample()
                logp = dist_curr.log_prob(a_curr).sum(-1, keepdim=True)
                q1_pi = critic1(s, a_curr)
                q2_pi = critic2(s, a_curr)
                loss_pi = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
                opt_actor.zero_grad()
                loss_pi.backward()
                opt_actor.step()

                # soft updates
                soft_update(critic1, target_critic1, tau)
                soft_update(critic2, target_critic2, tau)

                ep_loss_q1 += loss_q1.item()
                ep_loss_q2 += loss_q2.item()
                ep_loss_pi += loss_pi.item()
                updates += 1

            if done:
                break
        ep_time = time.time() - ep_start_time
        avg_loss_q1 = ep_loss_q1 / updates if updates > 0 else 0
        avg_loss_q2 = ep_loss_q2 / updates if updates > 0 else 0
        avg_loss_pi = ep_loss_pi / updates if updates > 0 else 0
        print(
            f"Episode {ep} Reward: {ep_reward:.2f} | Time: {ep_time:.3f} seconds | LossQ1: {avg_loss_q1:.4f} | LossQ2: {avg_loss_q2:.4f} | LossPi: {avg_loss_pi:.4f}"
        )
        writer.add_scalar('Reward/Episode', ep_reward, ep)
        writer.add_scalar('Loss/Q1', avg_loss_q1, ep)
        writer.add_scalar('Loss/Q2', avg_loss_q2, ep)
        writer.add_scalar('Loss/Policy', avg_loss_pi, ep)
        writer.add_scalar('Time/Episode', ep_time, ep)
        # save every N eps
        if ep % cfg.train.save_every == 0:
            torch.save(actor.state_dict(),
                       os.path.join(run_dir, f"actor_ep{ep}.pth"))
            torch.save(critic1.state_dict(),
                       os.path.join(run_dir, f"critic1_ep{ep}.pth"))
    writer.close()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == '__main__':
    main()
