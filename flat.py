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
import time
# Hydra and TensorBoard imports
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Add tqdm import
import imageio.v2 as imageio
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from torch.amp import autocast, GradScaler

from networks import ViTEncoder, ConvEncoder


##########################
# 1. Environment
##########################
class DrivingEnv(gym.Env):
    """
    2D disk driving env:
    - Discrete acceleration control (9 choices: zero or 8 compass directions)
    - Cost map penalty + collision + goal reward
    Observation: BEV image + velocity (2-d)
    Action: discrete (0-8)
    """
    metadata = {'render.modes': []}
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

            drivable = self.cost_map == 0

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
            self.start = np.array([sx, sy], dtype=np.int32)
            self.target = np.array([tx, ty], dtype=np.int32)
            elapsed = time.time() - start_time
            print(
                f"[DrivingEnv] Map generated in {elapsed:.3f} seconds. Start-goal dist: {dists[sy, sx]:.1f}"
            )
            break

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
        self.pos = self.start.astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        obs = self._get_obs()
        self._step_count = 0  # Reset step counter
        self._prev_dist_to_goal = np.linalg.norm(self.pos - self.target)

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
        # r_step removed since wall collision now terminates with -100

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
                'r_col': 0.0
            }
            self.is_done = terminated or truncated
            return obs, reward, terminated, truncated, info

        # collision (single agent skip)
        r_col = 0.0

        # goal
        dist_to_goal = np.linalg.norm(self.pos - self.target)
        terminated = dist_to_goal < self.disk_radius * 2
        r_goal = self.R_goal if terminated else 0.0
        # Progress reward: positive if moving closer to goal
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
        }

        self.is_done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        levels = []
        for scale in self.pyramid_scales:
            size = self.view_size
            crop_size = int(size * scale)
            x, y = self.pos
            half = crop_size // 2
            x0 = int(x) - half
            x1 = x0 + crop_size
            y0 = int(y) - half
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
                tx_rel = int((tx - x0) / scale)
                ty_rel = int((ty - y0) / scale)
                if 0 <= tx_rel < self.view_size and 0 <= ty_rel < self.view_size:
                    # Draw a filled square (5x5) in numpy for target
                    y_start = max(0, ty_rel - 2)
                    y_end = min(self.view_size, ty_rel + 3)
                    x_start = max(0, tx_rel - 2)
                    x_end = min(self.view_size, tx_rel + 3)
                    target_map[y_start:y_end, x_start:x_end] = 1.0
            # Stack: cost, vx, vy, target
            level = np.stack([view, vx, vy, target_map], axis=0)  # (4, H, W)
            levels.append(level)
        # Stack pyramid: shape (num_levels, 4, view_size, view_size)
        obs = np.stack(levels, axis=0).astype(np.float32)
        return obs


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


##########################
# 3. Networks
##########################


# --- Model selection ---
def make_encoder(cfg, view_size, num_levels):
    if cfg.model.type == 'vit':
        return ViTEncoder(view_size,
                          num_levels=num_levels,
                          embed_dim=cfg.model.vit_embed_dim)
    elif cfg.model.type == 'conv':
        return ConvEncoder(view_size,
                           num_levels=num_levels,
                           out_dim=cfg.model.conv_out_dim)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")


class Actor(nn.Module):

    def __init__(self, cfg, view_size, action_dim=9, num_levels=3):
        super().__init__()
        self.encoder = make_encoder(cfg, view_size, num_levels)
        hidden_dim = self.encoder.fc_out.out_features if hasattr(
            self.encoder, 'fc_out') else self.encoder.out_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.logits_head = nn.Linear(128, action_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        x = self.net(h)
        logits = self.logits_head(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist


class Critic(nn.Module):

    def __init__(self, cfg, view_size, num_levels=3):
        super().__init__()
        self.encoder = make_encoder(cfg, view_size, num_levels)
        self.net = nn.Sequential(
            nn.Linear((self.encoder.fc_out.out_features if hasattr(
                self.encoder, 'fc_out') else self.encoder.out_dim), 128),
            nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs):
        x = self.encoder(obs)
        return self.net(x)


##########################
# 4. PPO
##########################
class RolloutBuffer:

    def __init__(self, n_steps, n_envs, obs_shape, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = torch.zeros((n_steps, n_envs, *obs_shape),
                               dtype=torch.float32,
                               device=device)
        self.actions = torch.zeros((n_steps, n_envs),
                                   dtype=torch.long,
                                   device=device)
        self.rewards = torch.zeros((n_steps, n_envs),
                                   dtype=torch.float32,
                                   device=device)
        self.is_terminals = torch.zeros((n_steps, n_envs),
                                        dtype=torch.float32,
                                        device=device)
        self.logprobs = torch.zeros((n_steps, n_envs),
                                    dtype=torch.float32,
                                    device=device)
        self.values = torch.zeros((n_steps, n_envs),
                                  dtype=torch.float32,
                                  device=device)

    def add(self, obs, action, reward, is_terminal, logprob, value):
        # obs, action, reward, is_terminal, logprob, value: shape (n_envs, ...)
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr].copy_(action)
        self.rewards[self.ptr].copy_(reward)
        self.is_terminals[self.ptr].copy_(is_terminal)
        self.logprobs[self.ptr].copy_(logprob)
        self.values[self.ptr].copy_(value)
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def get(self):
        assert self.full, "Buffer not full yet!"
        # Return as (n_steps * n_envs, ...) for training
        obs = self.obs.reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1)
        rewards = self.rewards.reshape(-1)
        is_terminals = self.is_terminals.reshape(-1)
        logprobs = self.logprobs.reshape(-1)
        values = self.values.reshape(-1)
        return obs, actions, rewards, is_terminals, logprobs, values

    def get_raw(self):
        # Return as (n_steps, n_envs, ...)
        return self.obs, self.actions, self.rewards, self.is_terminals, self.logprobs, self.values

    def reset(self):
        self.ptr = 0
        self.full = False


def compute_gae(rewards, values, is_terminals, gamma, lam, the_extra_value,
                is_next_terminal):
    # rewards, values, is_terminals: (n_steps, n_envs)
    # the_extra_value: (n_envs,)
    # is_next_terminal: (n_envs,)
    n_steps, n_envs = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(n_envs, device=rewards.device)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            episode_continues = 1.0 - is_next_terminal
            next_value = the_extra_value
        else:
            episode_continues = 1.0 - is_terminals[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * episode_continues - values[t]
        lastgaelam = delta + gamma * lam * episode_continues * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def make_env(cfg, run_dir, episode_offset=0):
    env_seed = int(time.time() * 1e6) % (2**32 - 1) + episode_offset * 10000

    def _thunk():
        return DrivingEnv(render_dir=run_dir,
                          view_size=cfg.env.view_size,
                          dt=cfg.env.dt,
                          a_max=cfg.env.a_max,
                          v_max=cfg.env.v_max,
                          w_col=cfg.env.w_col,
                          R_goal=cfg.env.R_goal,
                          disk_radius=cfg.env.disk_radius,
                          video_fps=cfg.env.video_fps,
                          w_dist=cfg.env.w_dist,
                          w_accel=cfg.env.w_accel,
                          max_steps=cfg.train.max_steps,
                          hitwall_cost=cfg.env.hitwall_cost,
                          pyramid_scales=cfg.env.pyramid_scales,
                          num_levels=cfg.env.num_levels,
                          min_start_goal_dist=cfg.env.min_start_goal_dist,
                          max_start_goal_dist=cfg.env.max_start_goal_dist,
                          seed=env_seed)

    return _thunk


def train(cfg: DictConfig):
    run_name = cfg.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = to_absolute_path(cfg.output_dir)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    device = torch.device("cuda")
    num_levels = cfg.env.num_levels
    actor = Actor(cfg, cfg.env.view_size, num_levels=num_levels).to(device)
    critic = Critic(cfg, cfg.env.view_size, num_levels=num_levels).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=cfg.train.lr)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.train.lr)

    use_fp16 = cfg.train.fp16
    scaler = GradScaler() if use_fp16 else None

    # === Resume logic ===
    start_rollout_idx = 1
    if cfg.train.resume:
        import glob
        # Find latest actor and critic files
        actor_files = sorted(glob.glob(os.path.join(run_dir, 'actor_ep*.pth')))
        critic_files = sorted(
            glob.glob(os.path.join(run_dir, 'critic1_ep*.pth')))
        if actor_files and critic_files:
            # Get the highest episode number (use only the filename part)
            def extract_ep(fname):
                import re, os
                base = os.path.basename(fname)
                m = re.search(r'ep(\d+)', base)
                return int(m.group(1)) if m else -1

            latest_actor = max(actor_files, key=extract_ep)
            latest_critic = max(critic_files, key=extract_ep)
            latest_ep = extract_ep(latest_actor)
            print(
                f"[RESUME] Loading actor from {latest_actor}, critic from {latest_critic}, starting at rollout {latest_ep+1}"
            )
            actor.load_state_dict(torch.load(latest_actor,
                                             map_location=device))
            critic.load_state_dict(
                torch.load(latest_critic, map_location=device))
            start_rollout_idx = latest_ep + 1
        else:
            print(
                f"[RESUME] No checkpoint found in {run_dir}, starting from scratch."
            )

    # === Load from snapshot if specified ===
    if cfg.load_actor_path:
        actor_path = cfg.load_actor_path
        print(f"[INFO] Loading actor weights from {actor_path}")
        actor.load_state_dict(torch.load(actor_path, map_location=device))
    if cfg.load_critic_path:
        critic_path = cfg.load_critic_path
        print(f"[INFO] Loading critic weights from {critic_path}")
        critic.load_state_dict(torch.load(critic_path, map_location=device))

    gamma = cfg.train.gamma
    lam = cfg.train.gae_lambda
    rollout_steps = cfg.train.rollout_steps
    ppo_epochs = cfg.train.ppo_epochs
    minibatch_size = cfg.train.minibatch_size
    clip_eps = cfg.train.clip_eps

    num_envs = cfg.env.num_envs
    # Vectorized env with switch
    vector_type = cfg.env.vector_type
    if vector_type == 'async':
        VecEnvClass = AsyncVectorEnv
    elif vector_type == 'sync':
        VecEnvClass = SyncVectorEnv
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")
    env_fns = [
        make_env(cfg, run_dir, episode_offset=i) for i in range(num_envs)
    ]
    envs = VecEnvClass(env_fns)
    obs_shape = (num_levels, 4, cfg.env.view_size, cfg.env.view_size)
    buffer = RolloutBuffer(rollout_steps, num_envs, obs_shape, device)

    num_rollouts = cfg.train.num_rollouts
    rollout_steps = cfg.train.rollout_steps
    for rollout_idx in range(start_rollout_idx, num_rollouts + 1):
        rollout_start_time = time.time()
        print(f"[INFO] Rollout {rollout_idx}: Start recording transitions...")
        buffer.reset()
        record_start_time = time.time()
        # --- Video recording for the whole rollout ---
        render_this_rollout = (cfg.env.render_every > 0
                               and rollout_idx % cfg.env.render_every == 0)
        video_frames = None
        if render_this_rollout:
            video_frames = [[] for _ in range(cfg.env.num_envs)]
        # video_path = os.path.join(run_dir, f"rollout_{rollout_idx:05d}.mp4")  # Remove old single-env path
        next_obs, _ = envs.reset()
        next_obs = torch.as_tensor(next_obs,
                                   device=device,
                                   dtype=torch.float32)
        is_next_terminal = np.zeros(num_envs, dtype=bool)
        episode_rewards = [[] for _ in range(num_envs)]
        cur_episode_rewards = [[] for _ in range(num_envs)]
        gamma_pow = np.ones(num_envs, dtype=np.float32)
        gamma = cfg.train.gamma
        t_infer = t_env = t_render = t_cv = t_buffer = t_append = 0.0
        for step in range(rollout_steps):
            cur_obs = next_obs
            is_current_terminal = is_next_terminal.copy()
            t0 = time.time()
            with torch.no_grad():
                dist = actor(cur_obs)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = critic(cur_obs).squeeze(-1)
            t_infer += time.time() - t0

            t0 = time.time()
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy())
            next_obs = torch.as_tensor(next_obs,
                                       device=device,
                                       dtype=torch.float32)
            t_env += time.time() - t0

            if render_this_rollout:
                t0 = time.time()
                # Record video for all envs
                for env_id in cfg.env.render_env_ids:
                    info_env = {key: infos[key][env_id] for key in infos}
                    env_vel = infos['vel'][env_id]
                    frame = get_human_frame(obs=cur_obs[env_id].cpu().numpy(),
                                            vel=env_vel,
                                            v_max=cfg.env.v_max,
                                            action=action[env_id],
                                            info=info_env,
                                            a_max=cfg.env.a_max)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_frames[env_id].append(frame_rgb)
                t_render += time.time() - t0

            t0 = time.time()
            # Add batched data to buffer
            buffer.add(cur_obs, action, torch.as_tensor(reward, device=device),
                       torch.as_tensor(is_current_terminal, device=device),
                       logprob, value)
            t_buffer += time.time() - t0

            is_next_terminal = np.logical_or(terminated, truncated)
            for i in range(num_envs):
                # Only update if current step is not a placeholder (not terminal)
                if not is_current_terminal[i]:
                    cur_episode_rewards[i].append(reward[i] * gamma_pow[i])
                    gamma_pow[i] *= gamma
                if is_next_terminal[i]:
                    if len(cur_episode_rewards[i]) > 0:
                        episode_rewards[i].append(sum(cur_episode_rewards[i]))
                    cur_episode_rewards[i] = []
                    gamma_pow[i] = 1.0
        with torch.no_grad():
            next_value = critic(next_obs).squeeze(-1)
        # --- Save video for this rollout ---
        if render_this_rollout:
            for env_id in cfg.env.render_env_ids:
                video_path = os.path.join(
                    run_dir, f"rollout_{rollout_idx:05d}_env{env_id}.mp4")
                imageio.mimsave(video_path,
                                video_frames[env_id],
                                fps=cfg.env.video_fps,
                                codec='libx264')
        record_time = time.time() - record_start_time
        print(
            f"[INFO] Rollout {rollout_idx}: Record took {record_time:.3f} s.")
        print(
            f"[TIMER] Rollout {rollout_idx}: infer={t_infer:.3f}s, env={t_env:.3f}s, render={t_render:.3f}s, cvtColor={t_cv:.3f}s, buffer={t_buffer:.3f}s, append={t_append:.3f}s"
        )
        update_start_time = time.time()
        # Compute GAE and returns
        obs_buf, act_buf, rew_buf, terminal_buf, logp_buf, val_buf = buffer.get(
        )
        # For GAE, use unflattened shapes
        obs_raw, act_raw, rew_raw, terminal_raw, logp_raw, val_raw = buffer.get_raw(
        )
        adv_buf, ret_buf = compute_gae(
            rew_raw, val_raw, terminal_raw, gamma, lam, next_value,
            torch.as_tensor(is_next_terminal,
                            device=device,
                            dtype=torch.float32))
        adv_buf = adv_buf.reshape(-1)
        ret_buf = ret_buf.reshape(-1)
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        # PPO update
        inds = torch.randperm(rollout_steps * num_envs)
        actor_losses = []
        critic_losses = []
        for _ in range(ppo_epochs):
            for start in range(0, rollout_steps * num_envs, minibatch_size):
                mb_inds = inds[start:start + minibatch_size]
                mb_obs = obs_buf[mb_inds]
                mb_act = act_buf[mb_inds]
                mb_adv = adv_buf[mb_inds]
                mb_ret = ret_buf[mb_inds]
                mb_logp_old = logp_buf[mb_inds]
                # Actor update
                with autocast(device_type='cuda', enabled=use_fp16):
                    dist = actor(mb_obs)
                    logp = dist.log_prob(mb_act)
                    ratio = (logp - mb_logp_old).exp()
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                                        1.0 + clip_eps) * mb_adv
                    loss_pi = -torch.min(surr1, surr2).mean()
                opt_actor.zero_grad()
                if use_fp16:
                    scaler.scale(loss_pi).backward()
                    scaler.step(opt_actor)
                    scaler.update()
                else:
                    loss_pi.backward()
                    opt_actor.step()
                actor_losses.append(loss_pi.item())
                # Critic update
                with autocast(device_type='cuda', enabled=use_fp16):
                    value = critic(mb_obs).squeeze(-1)
                    loss_v = F.mse_loss(value, mb_ret)
                opt_critic.zero_grad()
                if use_fp16:
                    scaler.scale(loss_v).backward()
                    scaler.step(opt_critic)
                    scaler.update()
                else:
                    loss_v.backward()
                    opt_critic.step()
                critic_losses.append(loss_v.item())
        # Log losses
        writer.add_scalar('Loss/Actor', np.mean(actor_losses), rollout_idx)
        writer.add_scalar('Loss/Critic', np.mean(critic_losses), rollout_idx)
        # Log average discounted episode reward (only for complete episodes)
        flat_episode_rewards = [
            r for sublist in episode_rewards for r in sublist
        ]
        if flat_episode_rewards:
            avg_discounted_reward = np.mean(flat_episode_rewards)
            writer.add_scalar('Reward/AvgDiscountedEpisode',
                              avg_discounted_reward, rollout_idx)
        update_time = time.time() - update_start_time
        rollout_time = time.time() - rollout_start_time
        print(
            f"[INFO] Rollout {rollout_idx}: Network update took {update_time:.3f} s."
        )
        print(
            f"[INFO] Rollout {rollout_idx}: TotalTime: {rollout_time:.3f} s\n")
        if rollout_idx % cfg.train.save_every == 0:
            torch.save(actor.state_dict(),
                       os.path.join(run_dir, f"actor_ep{rollout_idx}.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(run_dir, f"critic1_ep{rollout_idx}.pth"))
    writer.close()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == '__main__':
    main()
