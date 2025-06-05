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
from tqdm import tqdm  # Add tqdm import
import imageio.v2 as imageio

from networks import ViTEncoder, ConvEncoder


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
                 max_start_goal_dist=200):
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
        self.action_space = spaces.Box(low=-a_max,
                                       high=a_max,
                                       shape=(2, ),
                                       dtype=np.float32)
        self.pyramid_scales = list(pyramid_scales)
        self.num_levels = num_levels
        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(self.num_levels, 4,
                                                   view_size, view_size),
                                            dtype=np.float32)
        self.seed()
        self.canvas_size = 256  # For human video rendering
        self.max_steps = max_steps
        self._step_count = 0
        self.min_start_goal_dist = min_start_goal_dist
        self.max_start_goal_dist = max_start_goal_dist
        self.is_done = None

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._seed = seed or np.random.randint(1e6)
        return [self._seed]

    def _generate_map(self):
        start_time = time.time()
        corridor_width = 3
        min_dist = self.min_start_goal_dist
        max_dist = self.max_start_goal_dist

        while True:
            M = np.random.rand(self.map_h, self.map_w).astype(np.float32)
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
            idx_t = np.random.randint(len(xs_e))
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
            sy, sx = pts[np.random.randint(len(pts))]
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

    def reset(self, *, seed=None, options=None, episode_num=None):
        self.is_done = False
        if seed is not None:
            self.seed(seed)
        # Regenerate the map for each episode
        self._generate_map()
        # Use episode_num for debug file naming
        self._current_episode_num = episode_num if episode_num is not None else 1
        # Dump global map visualization for this episode
        # if self.render_dir:
        #     self.visualize_map(
        #         filename=f"map_ep{self._current_episode_num:05d}.png")
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
                'r_accel': 0.0,
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
        # Acceleration penalty (normalized)
        a_norm = np.linalg.norm(action)
        excess = max(0, a_norm - self.a_max)
        r_accel = -self.w_accel * (excess / self.a_max)

        obs = self._get_obs()
        reward = r_col + r_goal + r_progress + r_accel

        info = {
            'hitwall': 0.0,
            'r_accel': r_accel,
            'r_progress': r_progress,
            'r_goal': r_goal,
            'r_col': r_col
        }

        self.is_done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Pyramid levels: (scale, view_size)
        pyramid_scales = self.pyramid_scales  # Use instance variable from config
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

    def get_human_frame(self, action=None, canvas_size=256, info=None):
        """
        Generate a high-res visualization for human viewing.
        - Uses the current observation as the base image for each level
        - Draws velocity and action vectors as overlays
        - Overlays reward info from the provided info dict (if any)
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
                # assert action is not NaN nor Inf
                if not np.isfinite(action).all():
                    raise ValueError("Action contains NaN or Inf values.")
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
        # --- Overlay text info ---
        # Use info dict if provided, else fallback to zeros
        if info is None:
            info = {}
        r_accel = info.get('r_accel', 0.0)
        r_progress = info.get('r_progress', 0.0)
        r_goal = info.get('r_goal', 0.0)
        r_col = info.get('r_col', 0.0)
        hitwall = info.get('hitwall', 0.0)
        info_lines = [
            f"r_progress: {r_progress:.2f}",
            f"r_accel: {r_accel:.2f}",
            f"r_goal: {r_goal:.2f}",
            f"r_col: {r_col:.2f}",
            f"hitwall: {hitwall:.2f}",
            f"Total reward: {(r_accel + r_progress + r_goal + r_col + hitwall):.2f}",
        ]
        y0 = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, line, (10, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1,
                        cv2.LINE_AA)
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

    def __init__(self, cfg, view_size, action_dim=2, num_levels=3):
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
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2

    def forward(self, obs):
        h = self.encoder(obs)
        x = self.net(h)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX -
                                            self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
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

    def __init__(self, size, obs_shape, action_dim, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = torch.zeros((size, *obs_shape),
                               dtype=torch.float32,
                               device=device)
        self.actions = torch.zeros((size, action_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.is_terminals = torch.zeros(size,
                                        dtype=torch.float32,
                                        device=device)
        self.logprobs = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)

    def add(self, obs, action, reward, is_terminal, logprob, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.is_terminals[self.ptr] = is_terminal
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def get(self):
        assert self.full, "Buffer not full yet!"
        return self.obs, self.actions, self.rewards, self.is_terminals, self.logprobs, self.values

    def reset(self):
        self.ptr = 0
        self.full = False


def compute_gae(rewards, values, is_terminals, gamma, lam, the_extra_value, is_next_terminal):
    adv = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            episode_continues = 1.0 - is_next_terminal
            next_value = the_extra_value
        else:
            episode_continues = 1.0 - is_terminals[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * episode_continues - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * episode_continues * lastgaelam
    returns = adv + values
    return adv, returns


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
    lam = getattr(cfg.train, 'gae_lambda', 0.95)
    rollout_steps = cfg.train.rollout_steps
    ppo_epochs = getattr(cfg.train, 'ppo_epochs', 4)
    minibatch_size = getattr(cfg.train, 'minibatch_size', 64)
    clip_eps = getattr(cfg.train, 'clip_eps', 0.2)

    obs_shape = (num_levels, 4, cfg.env.view_size, cfg.env.view_size)
    action_dim = 2
    buffer = RolloutBuffer(rollout_steps, obs_shape, action_dim, device)

    env = DrivingEnv(render_dir=run_dir,
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
                     max_start_goal_dist=cfg.env.max_start_goal_dist)

    num_rollouts = cfg.train.num_rollouts
    rollout_steps = cfg.train.rollout_steps
    episode_num = 1  # For video/logging
    for rollout_idx in range(1, num_rollouts + 1):
        rollout_start_time = time.time()
        print(f"[INFO] Rollout {rollout_idx}: Start recording transitions...")
        rollout_reward = 0
        buffer.reset()
        record_start_time = time.time()
        # --- Video recording for the whole rollout ---
        video_frames = []
        video_path = os.path.join(run_dir, f"rollout_{rollout_idx:05d}.mp4")
        next_obs, _ = env.reset(episode_num=episode_num)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        episode_num += 1
        is_next_terminal = False

        for _ in range(rollout_steps):
            cur_obs = next_obs
            is_current_terminal = is_next_terminal
            with torch.no_grad():
                dist = actor(cur_obs.unsqueeze(0))
                action = dist.sample()[0]
                logprob = dist.log_prob(action).sum()
                value = critic(cur_obs.unsqueeze(0))[0, 0]

            if is_current_terminal:
                next_obs, _ = env.reset(episode_num=episode_num)
                episode_num += 1
                next_obs = torch.tensor(next_obs,
                                        dtype=torch.float32,
                                        device=device)
                reward = 0.0
                terminated = False
                truncated = False
                info = None
            else:
                next_obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy())
                next_obs = torch.tensor(next_obs,
                                        dtype=torch.float32,
                                        device=device)

            # Store reward info for visualization
            # --- Collect frame for video ---
            frame = env.get_human_frame(action=action.cpu().numpy(),
                                        canvas_size=env.canvas_size,
                                        info=info)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame_rgb)
            # ---

            buffer.add(cur_obs, action, reward, float(is_current_terminal),
                       logprob, value)

            is_next_terminal = terminated or truncated

            rollout_reward += reward
        
        with torch.no_grad():
            next_value = critic(next_obs.unsqueeze(0))[0, 0]

        # --- Save video for this rollout ---
        imageio.mimsave(video_path,
                        video_frames,
                        fps=cfg.env.video_fps,
                        codec='libx264')
        record_time = time.time() - record_start_time
        update_start_time = time.time()
        # Compute GAE and returns
        obs_buf, act_buf, rew_buf, terminal_buf, logp_buf, val_buf = buffer.get(
        )
        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, terminal_buf, gamma,
                                       lam, next_value, is_next_terminal)
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        # PPO update
        inds = torch.randperm(rollout_steps)
        for _ in range(ppo_epochs):
            for start in range(0, rollout_steps, minibatch_size):
                mb_inds = inds[start:start + minibatch_size]
                mb_obs = obs_buf[mb_inds]
                mb_act = act_buf[mb_inds]
                mb_adv = adv_buf[mb_inds]
                mb_ret = ret_buf[mb_inds]
                mb_logp_old = logp_buf[mb_inds]
                # Actor update
                dist = actor(mb_obs)
                logp = dist.log_prob(mb_act).sum(-1)
                ratio = (logp - mb_logp_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                                    1.0 + clip_eps) * mb_adv
                loss_pi = -torch.min(surr1, surr2).mean()
                opt_actor.zero_grad()
                loss_pi.backward()
                opt_actor.step()
                # Critic update
                value = critic(mb_obs, mb_act).squeeze(-1)
                loss_v = F.mse_loss(value, mb_ret)
                opt_critic.zero_grad()
                loss_v.backward()
                opt_critic.step()
        update_time = time.time() - update_start_time
        rollout_time = time.time() - rollout_start_time
        print(
            f"[INFO] Rollout {rollout_idx}: Network update took {update_time:.3f} s."
        )
        print(
            f"[INFO] Rollout {rollout_idx}: TotalReward: {rollout_reward:.2f} | TotalTime: {rollout_time:.3f} s\n"
        )
        writer.add_scalar('Reward/Rollout', rollout_reward, rollout_idx)
        writer.add_scalar('Time/Rollout', rollout_time, rollout_idx)
        writer.add_scalar('Time/Recording', record_time, rollout_idx)
        writer.add_scalar('Time/Update', update_time, rollout_idx)
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
