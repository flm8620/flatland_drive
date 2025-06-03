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

    def __init__(self, view_size, dt, a_max, v_max, w_col, R_goal, disk_radius,
                 render_dir, video_fps, w_dist, w_accel, max_steps,
                 hitwall_cost, pyramid_scales, num_levels,
                 min_start_goal_dist=50, max_start_goal_dist=200):
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
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.num_levels, 3,
                                                   view_size, view_size),
                                            dtype=np.float32)
        self.seed()
        self.canvas_size = 256  # For human video rendering
        self.max_steps = max_steps
        self._step_count = 0
        self.min_start_goal_dist = min_start_goal_dist
        self.max_start_goal_dist = max_start_goal_dist

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
            pts = np.argwhere((visited) & (dists >= min_dist) & (dists <= max_dist))
            # drop the target itself
            pts = pts[~((pts[:, 0] == ty) & (pts[:, 1] == tx))]
            if len(pts) == 0:
                continue  # No valid start, regenerate map

            # Randomly select one as start
            sy, sx = pts[np.random.randint(len(pts))]
            self.start = np.array([sx, sy], dtype=np.int32)
            self.target = np.array([tx, ty], dtype=np.int32)
            elapsed = time.time() - start_time
            print(f"[DrivingEnv] Map generated in {elapsed:.3f} seconds. Start-goal dist: {dists[sy, sx]:.1f}")
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

    def reset(self,
              *,
              seed=None,
              options=None,
              render_video=None,
              episode_num=None):
        episode_start_time = time.time()
        if seed is not None:
            self.seed(seed)
        # Regenerate the map for each episode
        self._generate_map()
        # Use episode_num for debug file naming
        self._current_episode_num = episode_num if episode_num is not None else 1
        # Dump global map visualization for this episode
        if self.render_dir:
            self.visualize_map(
                filename=f"map_ep{self._current_episode_num:03d}.png")
        # Set agent to the designated start position from map generation
        self.pos = self.start.astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)

        # Use render_video argument to decide video_writer
        if render_video:
            video_path = os.path.join(
                self.render_dir,
                f'episode_{self._current_episode_num:03d}.mp4')
            self._video_path = video_path
            self._video_frames = []
        else:
            self._video_path = None
            self._video_frames = None
        obs = self._get_obs()
        self._episode_start_time = episode_start_time  # Save for timing in step
        self._step_count = 0  # Reset step counter
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
            # Render this step: collect frame for imageio
            if self.render_dir and self._video_frames is not None:
                frame = self.get_human_frame(action=action, canvas_size=self.canvas_size)
                # Convert BGR (OpenCV) to RGB for imageio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._video_frames.append(frame_rgb)
                if terminated or truncated:
                    imageio.mimsave(self._video_path, self._video_frames, fps=self.video_fps, codec='libx264')
                    self._video_frames = None
            return obs, reward, terminated, truncated, info

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
        reward = r_col + r_goal + r_progress + r_accel

        info = {
            'hitwall': 0.0,
            'r_accel': r_accel,
            'r_progress': r_progress,
            'r_goal': r_goal,
            'r_col': r_col
        }


        # Render this step: collect frame for imageio
        if self.render_dir and self._video_frames is not None:
            frame = self.get_human_frame(action=action, canvas_size=self.canvas_size)
            # Convert BGR (OpenCV) to RGB for imageio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._video_frames.append(frame_rgb)
            # Save video if episode ends by termination or truncation
            if terminated or truncated:
                imageio.mimsave(self._video_path, self._video_frames, fps=self.video_fps, codec='libx264')
                self._video_frames = None
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
        self.net = nn.Sequential(
            nn.Linear(
                self.encoder.fc_out.out_features if hasattr(
                    self.encoder, 'fc_out') else self.encoder.out_dim, 128),
            nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim, ), -2.0))

    def forward(self, obs):
        h = self.encoder(obs)
        mu = self.net(h)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist


class Critic(nn.Module):

    def __init__(self, cfg, view_size, action_dim=2, num_levels=3):
        super().__init__()
        self.encoder = make_encoder(cfg, view_size, num_levels)
        self.net = nn.Sequential(
            nn.Linear((self.encoder.fc_out.out_features if hasattr(
                self.encoder, 'fc_out') else self.encoder.out_dim) +
                      action_dim, 128), nn.ReLU(), nn.Linear(128, 128),
            nn.ReLU(), nn.Linear(128, 1))

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

    device = torch.device("cuda")
    replay = ReplayBuffer(cfg.train.replay_size)

    num_levels = cfg.env.num_levels
    actor = Actor(cfg, cfg.env.view_size, num_levels=num_levels).to(device)
    critic1 = Critic(cfg, cfg.env.view_size, num_levels=num_levels).to(device)
    critic2 = Critic(cfg, cfg.env.view_size, num_levels=num_levels).to(device)
    target_critic1 = Critic(cfg, cfg.env.view_size,
                            num_levels=num_levels).to(device)
    target_critic2 = Critic(cfg, cfg.env.view_size,
                            num_levels=num_levels).to(device)
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

    use_amp = getattr(cfg.train, 'use_amp',
                      True)  # Default to True for backward compatibility

    # Dummy GradScaler for non-AMP mode
    class DummyScaler:

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    scaler = torch.amp.GradScaler(device=device) if use_amp else DummyScaler()

    def backward_and_step(loss, optimizer):
        scaler.scale(loss).backward()
        scaler.step(optimizer)

    total_steps = 0

    # --- Load latest checkpoint if exists ---
    latest_ep = 0
    import re
    actor_ckpts = [
        f for f in os.listdir(run_dir) if re.match(r"actor_ep(\\d+)\\.pth", f)
    ]
    if actor_ckpts:
        # Find the highest episode number
        ep_nums = [
            int(re.findall(r"actor_ep(\\d+)\\.pth", f)[0]) for f in actor_ckpts
        ]
        latest_ep = max(ep_nums)
        actor_path = os.path.join(run_dir, f"actor_ep{latest_ep}.pth")
        critic1_path = os.path.join(run_dir, f"critic1_ep{latest_ep}.pth")
        if os.path.exists(actor_path):
            print(f"[INFO] Loading actor weights from {actor_path}")
            actor.load_state_dict(torch.load(actor_path))
        if os.path.exists(critic1_path):
            print(f"[INFO] Loading critic1 weights from {critic1_path}")
            critic1.load_state_dict(torch.load(critic1_path))

    # Use autocast context inline, and wrap scaler logic in a helper for elegance
    for ep in range(latest_ep + 1, cfg.train.episodes + 1):
        ep_start_time = time.time()
        # Create a new environment for each episode to avoid state carryover
        env = DrivingEnv(render_dir=run_dir if cfg.env.save_viz else None,
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
        render_video = cfg.env.save_viz and (ep % cfg.env.render_every == 0)
        obs, _ = env.reset(render_video=render_video, episode_num=ep)
        obs = torch.tensor(obs[None], dtype=torch.float32, device=device)
        ep_reward_discounted = 0
        ep_loss_q1 = 0
        ep_loss_q2 = 0
        ep_loss_pi = 0
        updates = 0
        done = False
        step_count = 0
        # --- Only accumulators for discounted reward components ---
        total_hitwall = 0.0
        total_r_accel = 0.0
        total_r_progress = 0.0
        total_r_goal = 0.0
        total_r_col = 0.0
        gamma_pow = 1.0
        with tqdm(total=cfg.train.max_steps, desc=f"Episode {ep}",
                  leave=False) as pbar:
            while not done:
                total_steps += 1
                with torch.no_grad():
                    dist = actor(obs)
                    # Assert actor output is valid
                    mu = dist.mean
                    std = dist.stddev
                    assert torch.isfinite(mu).all(), "Actor mean contains NaN or Inf"
                    assert torch.isfinite(std).all(), "Actor stddev contains NaN or Inf"
                    action = dist.sample().cpu().numpy()[0]
                next_obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay.push(obs.cpu().numpy()[0], action, r, next_obs, done)
                obs = torch.tensor(next_obs[None],
                                   dtype=torch.float32,
                                   device=device)
                ep_reward_discounted += gamma_pow * r
                step_count += 1
                pbar.update(1)
                # --- Use info dict for discounted reward components ---
                total_r_col += gamma_pow * info.get('r_col', 0.0)
                total_r_goal += gamma_pow * info.get('r_goal', 0.0)
                total_r_progress += gamma_pow * info.get('r_progress', 0.0)
                total_r_accel += gamma_pow * info.get('r_accel', 0.0)
                total_hitwall += gamma_pow * info.get('hitwall', 0.0)
                gamma_pow *= gamma

                if len(replay) > warmup:
                    s, a, rew, s2, d = replay.sample(batch_size)
                    s = torch.tensor(s, dtype=torch.float32, device=device)
                    a = torch.tensor(a, dtype=torch.float32, device=device)
                    rew = torch.tensor(rew, dtype=torch.float32,
                                       device=device).unsqueeze(1)
                    s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                    d = torch.tensor(d, dtype=torch.float32,
                                     device=device).unsqueeze(1)

                    with torch.amp.autocast(device_type='cuda',
                                            enabled=use_amp):
                        with torch.no_grad():
                            dist_next = actor(s2)
                            # Assert actor output is valid
                            mu_next = dist_next.mean
                            std_next = dist_next.stddev
                            assert torch.isfinite(mu_next).all(), "Actor mean (next) contains NaN or Inf"
                            assert torch.isfinite(std_next).all(), "Actor stddev (next) contains NaN or Inf"
                            a2 = dist_next.rsample()
                            logp2 = dist_next.log_prob(a2).sum(-1,
                                                               keepdim=True)
                            q1_target = target_critic1(s2, a2)
                            q2_target = target_critic2(s2, a2)
                            # Assert critic outputs are valid
                            assert torch.isfinite(q1_target).all(), "Critic1 target output contains NaN or Inf"
                            assert torch.isfinite(q2_target).all(), "Critic2 target output contains NaN or Inf"
                            q_target = torch.min(q1_target,
                                                 q2_target) - alpha * logp2
                            y = rew + (1 - d) * gamma * q_target
                        q1 = critic1(s, a)
                        q2 = critic2(s, a)
                        # Assert critic outputs are valid
                        assert torch.isfinite(q1).all(), "Critic1 output contains NaN or Inf"
                        assert torch.isfinite(q2).all(), "Critic2 output contains NaN or Inf"
                        loss_q1 = F.mse_loss(q1, y)
                        loss_q2 = F.mse_loss(q2, y)
                    opt_critic1.zero_grad()
                    backward_and_step(loss_q1, opt_critic1)
                    opt_critic2.zero_grad()
                    backward_and_step(loss_q2, opt_critic2)
                    scaler.update()
                    # actor update
                    with torch.amp.autocast(device_type='cuda',
                                            enabled=use_amp):
                        dist_curr = actor(s)
                        # Assert actor output is valid
                        mu_curr = dist_curr.mean
                        std_curr = dist_curr.stddev
                        assert torch.isfinite(mu_curr).all(), "Actor mean (curr) contains NaN or Inf"
                        assert torch.isfinite(std_curr).all(), "Actor stddev (curr) contains NaN or Inf"
                        a_curr = dist_curr.rsample()
                        logp = dist_curr.log_prob(a_curr).sum(-1, keepdim=True)
                        q1_pi = critic1(s, a_curr)
                        q2_pi = critic2(s, a_curr)
                        # Assert critic outputs are valid
                        assert torch.isfinite(q1_pi).all(), "Critic1 pi output contains NaN or Inf"
                        assert torch.isfinite(q2_pi).all(), "Critic2 pi output contains NaN or Inf"
                        loss_pi = (alpha * logp -
                                   torch.min(q1_pi, q2_pi)).mean()
                    opt_actor.zero_grad()
                    backward_and_step(loss_pi, opt_actor)
                    scaler.update()
                    soft_update(critic1, target_critic1, tau)
                    soft_update(critic2, target_critic2, tau)
                    ep_loss_q1 += loss_q1.item()
                    ep_loss_q2 += loss_q2.item()
                    ep_loss_pi += loss_pi.item()
                    updates += 1

        ep_time = time.time() - ep_start_time
        avg_loss_q1 = ep_loss_q1 / updates if updates > 0 else 0
        avg_loss_q2 = ep_loss_q2 / updates if updates > 0 else 0
        avg_loss_pi = ep_loss_pi / updates if updates > 0 else 0
        print(
            f"Episode {ep} Discounted Reward: {ep_reward_discounted:.2f} | Time: {ep_time:.3f} seconds | LossQ1: {avg_loss_q1:.4f} | LossQ2: {avg_loss_q2:.4f} | LossPi: {avg_loss_pi:.4f}")
        # --- Print only discounted costs for each kind ---
        print(f"[Episode {ep} Discounted Totals] hitwall: {total_hitwall:.2f}, accel: {total_r_accel:.2f}, progress: {total_r_progress:.2f}, goal: {total_r_goal:.2f}, col: {total_r_col:.2f}")
        writer.add_scalar('Reward/Episode', ep_reward_discounted, ep)
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
