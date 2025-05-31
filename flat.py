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
                 a_max=1.0,
                 v_max=2.0,
                 w_cost=1.0,
                 w_col=1.0,
                 R_goal=50.0,
                 num_agents=1,
                 disk_radius=2.0,
                 render_dir=None,
                 video_fps=30):
        super().__init__()
        self.map_h, self.map_w = 512, 512
        self.view_size = view_size
        self.dt = dt
        self.a_max = a_max
        self.v_max = v_max
        self.w_cost = w_cost
        self.w_col = w_col
        self.R_goal = R_goal
        self.disk_radius = disk_radius
        self.num_agents = num_agents

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
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(3, view_size, view_size),
                                            dtype=np.float32)
        # we'll concatenate flattened vel channels at right-most pixels

        # Generate cost map and target
        self.seed()
        self._generate_map()
        if self.render_dir:
            self.visualize_map()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._seed = seed or np.random.randint(1e6)
        return [self._seed]

    def _generate_map(self):
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
            COST_MAX = 0.2
            drivable = (M <= T)
            M[M > COST_MAX] = COST_MAX
            # save M to tiff file
            cv2.imwrite(os.path.join(self.render_dir, 'M_map.tiff'), M)
            self.cost_map = np.zeros_like(M)
            mask = ~drivable
            self.cost_map[mask] = (M[mask] - T) / (COST_MAX - T)
            cv2.imwrite(os.path.join(self.render_dir, 'cost_map.tiff'), M)

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
        # color cost
        idx = np.where(~drivable)
        costs = np.clip(self.cost_map[idx], 0, 1)
        # BGR: blue->red
        vis[idx] = np.stack([(255 * (1 - costs)).astype(np.uint8),
                             np.zeros_like(costs, dtype=np.uint8),
                             (255 * costs).astype(np.uint8)],
                            axis=1)
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
        self.episode_count += 1
        if seed is not None:
            self.seed(seed)
        mask = (self.cost_map == 0)
        ys, xs = np.where(mask)
        idx = np.random.randint(len(xs))
        self.pos = np.array([xs[idx], ys[idx]], dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        # Initialize video writer for this episode
        if self.render_dir:
            video_path = os.path.join(self.render_dir,
                                      f'episode_{self.episode_count:03d}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                self.video_fps, (self.view_size, self.view_size),
                isColor=False)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # clip
        a = np.clip(action, -self.a_max, self.a_max)
        # dynamics
        self.vel = np.clip(self.vel + a * self.dt, -self.v_max, self.v_max)
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

        obs = self._get_obs()
        reward = r_step + r_col + r_goal

        # Render this step: write to video writer
        if self.render_dir and self.video_writer is not None:
            frame = (obs[0] * 255).astype(np.uint8)
            self.video_writer.write(frame)
            if terminated:
                self.video_writer.release()
                self.video_writer = None

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # crop view
        x, y = self.pos
        half = self.view_size // 2
        x0 = int(x) - half
        x1 = x0 + self.view_size
        y0 = int(y) - half
        y1 = y0 + self.view_size
        # pad
        pad_x0, pad_y0 = max(0, -x0), max(0, -y0)
        pad_x1, pad_y1 = max(0, x1 - self.map_w), max(0, y1 - self.map_h)
        x0c, x1c = max(0, x0), min(self.map_w, x1)
        y0c, y1c = max(0, y0), min(self.map_h, y1)
        view = np.zeros((self.view_size, self.view_size), dtype=np.float32)
        view[pad_y0:self.view_size-pad_y1, pad_x0:self.view_size-pad_x1] = \
            self.cost_map[y0c:y1c, x0c:x1c]
        # velocity map
        vel_map = np.full_like(view, np.linalg.norm(self.vel) / self.v_max)
        # target map
        target_map = np.zeros_like(view)
        tx, ty = self.target
        if x0 <= tx < x1 and y0 <= ty < y1:
            tx_rel = int(tx - x0)
            ty_rel = int(ty - y0)
            cv2.circle(target_map, (tx_rel, ty_rel), int(self.disk_radius),
                       1.0, -1)
        return np.stack([view, vel_map, target_map], axis=0)


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
class ConvEncoder(nn.Module):

    def __init__(self, view_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_in = 128 * (view_size // 8) * (view_size // 8)

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class Actor(nn.Module):

    def __init__(self, view_size, action_dim=2):
        super().__init__()
        self.encoder = ConvEncoder(view_size)
        self.net = nn.Sequential(nn.Linear(self.encoder.fc_in, 256), nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        # obs shape: (B,3,H,W)
        h = self.encoder(obs)
        mu = self.net(h)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist


class Critic(nn.Module):

    def __init__(self, view_size, action_dim=2):
        super().__init__()
        self.encoder = ConvEncoder(view_size)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.fc_in + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 1))

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


def train():
    render_dir = 'viz'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DrivingEnv(render_dir=render_dir)
    replay = ReplayBuffer(100000)

    actor = Actor(env.view_size).to(device)
    critic1 = Critic(env.view_size).to(device)
    critic2 = Critic(env.view_size).to(device)
    target_critic1 = Critic(env.view_size).to(device)
    target_critic2 = Critic(env.view_size).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic1 = optim.Adam(critic1.parameters(), lr=3e-4)
    opt_critic2 = optim.Adam(critic2.parameters(), lr=3e-4)

    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 64
    warmup = 1000

    total_steps = 0
    for ep in range(1, 501):
        obs, _ = env.reset()
        obs = torch.tensor(obs[None], dtype=torch.float32, device=device)
        ep_reward = 0
        for t in range(1, 1001):
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

            if done:
                break

        print(f"Episode {ep} Reward: {ep_reward:.2f}")
        # save every 50 eps
        if ep % 50 == 0:
            torch.save(actor.state_dict(), f"actor_ep{ep}.pth")
            torch.save(critic1.state_dict(), f"critic1_ep{ep}.pth")


if __name__ == '__main__':
    train()
