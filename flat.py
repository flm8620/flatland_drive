'''
We want to create a Reinforcement Learning Playground project. 
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
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
# Hydra and TensorBoard imports
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from torch.utils.tensorboard import SummaryWriter
import imageio.v2 as imageio
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from torch.amp import autocast, GradScaler

from env import DrivingEnv, get_human_frame
from networks import ViTEncoder, ConvEncoder


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

    def __init__(self, n_steps, n_envs, obs_shape):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ptr = 0
        self.full = False
        self.obs = torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.float32, device='cpu')
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device='cpu')
        self.rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.is_terminals = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.values = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')

    def add(self, obs, action, reward, is_terminal, logprob, value):
        self.obs[self.ptr].copy_(obs.cpu())
        self.actions[self.ptr].copy_(action.cpu())
        self.rewards[self.ptr].copy_(reward.cpu())
        self.is_terminals[self.ptr].copy_(is_terminal.cpu())
        self.logprobs[self.ptr].copy_(logprob.cpu())
        self.values[self.ptr].copy_(value.cpu())
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def get_raw(self):
        assert self.full, "Buffer not full yet!"
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


def make_env(cfg, run_dir, episode_offset=0, min_start_goal_dist=None, max_start_goal_dist=None, hitwall_cost=None):
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
                          hitwall_cost=hitwall_cost,
                          pyramid_scales=cfg.env.pyramid_scales,
                          num_levels=cfg.env.num_levels,
                          min_start_goal_dist=min_start_goal_dist,
                          max_start_goal_dist=max_start_goal_dist,
                          seed=env_seed)

    return _thunk



def collect_rollout(cfg, device, rollout_idx, run_dir, actor, critic, writer):
    """
    Handles curriculum, creates envs, collects a rollout, logs stats, saves video, and returns only the buffer and next_value needed for PPO update.
    """
    # === Curriculum learning: select env difficulty from config ===
    curriculum = cfg.env.curriculum
    num_envs = cfg.env.num_envs
    num_levels = cfg.env.num_levels
    rollout_steps = cfg.train.rollout_steps
    obs_shape = (num_levels, 4, cfg.env.view_size, cfg.env.view_size)
    if curriculum:
        for stage in curriculum:
            if rollout_idx <= stage['until_rollout']:
                min_start_goal_dist = stage['min_start_goal_dist']
                max_start_goal_dist = stage['max_start_goal_dist']
                hitwall_cost = stage['hitwall_cost']
                break
    else:
        raise RuntimeError('No curriculum defined in config!')
    env_fns = [
        make_env(cfg, run_dir, episode_offset=i,
                 min_start_goal_dist=min_start_goal_dist,
                 max_start_goal_dist=max_start_goal_dist,
                 hitwall_cost=hitwall_cost)
        for i in range(num_envs)
    ]
    if cfg.env.vector_type == 'async':
        VecEnvClass = AsyncVectorEnv
    elif cfg.env.vector_type == 'sync':
        VecEnvClass = SyncVectorEnv
    else:
        raise ValueError(f"Unknown vector type: {cfg.env.vector_type}")
    envs = VecEnvClass(env_fns)
    print(f"[INFO] Rollout {rollout_idx}: Start recording transitions... (min/max dist: {min_start_goal_dist}-{max_start_goal_dist}, hitwall: {hitwall_cost})")
    buffer = RolloutBuffer(rollout_steps, num_envs, obs_shape)
    gamma = cfg.train.gamma
    render_this_rollout = (cfg.env.render_every > 0 and rollout_idx % cfg.env.render_every == 0 or rollout_idx == 1)
    video_frames = None
    if render_this_rollout:
        video_frames = [[] for _ in range(cfg.env.num_envs)]
    next_obs, _ = envs.reset()
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    is_next_terminal = np.zeros(num_envs, dtype=bool)
    episode_rewards = [[] for _ in range(num_envs)]
    cur_episode_rewards = [[] for _ in range(num_envs)]
    gamma_pow = np.ones(num_envs, dtype=np.float32)
    t_infer = t_env = t_render = t_cv = t_buffer = t_append = 0.0
    success_count = 0
    failure_count = 0
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
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        t_env += time.time() - t0
        if render_this_rollout:
            t0 = time.time()
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
        buffer.add(cur_obs, action, torch.as_tensor(reward, device=device),
                   torch.as_tensor(is_current_terminal, device=device),
                   logprob, value)
        t_buffer += time.time() - t0
        is_next_terminal = np.logical_or(terminated, truncated)
        for i in range(num_envs):
            if not is_current_terminal[i]:
                cur_episode_rewards[i].append(reward[i] * gamma_pow[i])
                gamma_pow[i] *= gamma
            if is_next_terminal[i]:
                if len(cur_episode_rewards[i]) > 0:
                    episode_rewards[i].append(sum(cur_episode_rewards[i]))
                cur_episode_rewards[i] = []
                gamma_pow[i] = 1.0
                if terminated[i]:
                    if infos['success'][i]:
                        success_count += 1
                    else:
                        failure_count += 1
    with torch.no_grad():
        next_value = critic(next_obs).squeeze(-1)
    # --- Logging and video saving here ---
    flat_episode_rewards = [r for sublist in episode_rewards for r in sublist]
    print(f"[TIMER] Rollout {rollout_idx}: infer={t_infer:.3f}s, env={t_env:.3f}s, render={t_render:.3f}s, cvtColor={t_cv:.3f}s, buffer={t_buffer:.3f}s, append={t_append:.3f}s")
    if flat_episode_rewards:
        avg_discounted_reward = np.mean(flat_episode_rewards)
        writer.add_scalar('Reward/AvgDiscountedEpisode', avg_discounted_reward, rollout_idx)
    avg_successes_per_env = success_count / num_envs
    avg_failures_per_env = failure_count / num_envs
    writer.add_scalar('Reward/AvgSuccess', np.mean(avg_successes_per_env), rollout_idx)
    writer.add_scalar('Reward/AvgFailure', np.mean(avg_failures_per_env), rollout_idx)
    if render_this_rollout and video_frames is not None:
        for env_id in cfg.env.render_env_ids:
            video_path = os.path.join(run_dir, f"rollout_{rollout_idx:05d}_env{env_id}.mp4")
            imageio.mimsave(video_path, video_frames[env_id], fps=cfg.env.video_fps, codec='libx264')
    # Only return what is needed for PPO update
    return buffer, is_next_terminal, next_value


def ppo_update(buffer, critic, actor, cfg, device, is_next_terminal, next_value, scaler, use_fp16, writer, rollout_idx, opt_actor, opt_critic):
    gamma = cfg.train.gamma
    lam = cfg.train.gae_lambda
    ppo_epochs = cfg.train.ppo_epochs
    minibatch_size = cfg.train.minibatch_size
    clip_eps = cfg.train.clip_eps
    obs_raw, act_raw, rew_raw, terminal_raw, logp_raw, val_raw = buffer.get_raw()
    adv_buf, ret_buf = compute_gae(
        rew_raw, val_raw, terminal_raw, gamma, lam,
        next_value.cpu(),
        torch.as_tensor(is_next_terminal, device='cpu', dtype=torch.float32))
    is_valid_buf = (terminal_raw == 0).reshape(-1)
    obs_buf = obs_raw.reshape(-1, *obs_raw.shape[2:])[is_valid_buf]
    act_buf = act_raw.reshape(-1)[is_valid_buf]
    logp_buf = logp_raw.reshape(-1)[is_valid_buf]
    adv_buf = adv_buf.reshape(-1)[is_valid_buf]
    ret_buf = ret_buf.reshape(-1)[is_valid_buf]
    obs_buf = obs_buf.to(device)
    act_buf = act_buf.to(device)
    logp_buf = logp_buf.to(device)
    adv_buf = adv_buf.to(device)
    ret_buf = ret_buf.to(device)
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
    inds = torch.randperm(adv_buf.shape[0], device=device)
    actor_losses = []
    critic_losses = []
    for _ in range(ppo_epochs):
        for start in range(0, adv_buf.shape[0], minibatch_size):
            mb_inds = inds[start:start + minibatch_size]
            mb_obs = obs_buf[mb_inds]
            mb_act = act_buf[mb_inds]
            mb_adv = adv_buf[mb_inds]
            mb_ret = ret_buf[mb_inds]
            mb_logp_old = logp_buf[mb_inds]
            with autocast(device_type='cuda', enabled=use_fp16):
                dist = actor(mb_obs)
                logp = dist.log_prob(mb_act)
                ratio = (logp - mb_logp_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
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
    writer.add_scalar('Loss/Actor', np.mean(actor_losses), rollout_idx)
    writer.add_scalar('Loss/Critic', np.mean(critic_losses), rollout_idx)


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
        # Find latest actor and critic files
        actor_files = sorted(glob.glob(os.path.join(run_dir, 'actor_ep*.pth')))
        critic_files = sorted(
            glob.glob(os.path.join(run_dir, 'critic1_ep*.pth')))
        if actor_files and critic_files:
            # Get the highest episode number (use only the filename part)
            def extract_ep(fname):
                import re
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



    num_rollouts = cfg.train.num_rollouts
    for rollout_idx in range(start_rollout_idx, num_rollouts + 1):
        rollout_start_time = time.time()
        # --- Collect rollout (now includes env creation, logging, and video saving) ---
        buffer, is_next_terminal, next_value = collect_rollout(cfg, device, rollout_idx, run_dir, actor, critic, writer)
        update_start_time = time.time()
        print(f"[INFO] Rollout {rollout_idx}: Record took {update_start_time - rollout_start_time:.3f} s.")

        writer.add_scalar('Time/Record', update_start_time - rollout_start_time, rollout_idx)
        # --- PPO update ---
        ppo_update(buffer, critic, actor, cfg, device, is_next_terminal, next_value, scaler, use_fp16, writer, rollout_idx, opt_actor, opt_critic)
        update_time = time.time() - update_start_time
        rollout_time = time.time() - rollout_start_time
        writer.add_scalar('Time/Update', update_time, rollout_idx)
        writer.add_scalar('Time/Rollout', rollout_time, rollout_idx)
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