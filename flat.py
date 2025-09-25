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
from tqdm import tqdm
import multiprocessing
# Hydra and TensorBoard imports
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from torch.utils.tensorboard import SummaryWriter
import imageio.v2 as imageio
from torch.amp import autocast, GradScaler
from tensordict import TensorDict

# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)


from env import ParallelDrivingEnv, get_human_frame
from networks import ViTEncoder, ConvEncoder
from timer import get_timer, set_timing_enabled, reset_timer, print_timing_report


def make_encoder(cfg, view_size, num_levels):
    if cfg.model.type == 'vit':
        return ViTEncoder(view_size,
                          num_levels=num_levels,
                          embed_dim=cfg.model.vit_embed_dim,
                          in_chans=2,
                          state_dim=2)
    elif cfg.model.type == 'conv':
        return ConvEncoder(view_size,
                           num_levels=num_levels,
                           out_dim=cfg.model.conv_out_dim,
                           in_chans=2,
                           state_dim=2)
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

    def __init__(self, n_steps, n_envs, image_shape, state_shape):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ptr = 0
        self.full = False
        
        # TensorDict storage for observations
        self.obs = TensorDict({
            'image': torch.zeros((n_steps, n_envs, *image_shape), dtype=torch.uint8, device='cpu'),
            'state': torch.zeros((n_steps, n_envs, *state_shape), dtype=torch.float32, device='cpu')
        }, batch_size=[n_steps, n_envs])
        
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device='cpu')
        self.rewards = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.is_terminated = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.is_truncated = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.values = torch.zeros((n_steps, n_envs), dtype=torch.float32, device='cpu')
        self.final_values = None
        self.final_is_terminated = None
        self.final_is_truncated = None

    def add(self, obs_dict, action, reward, is_terminated, is_truncated, logprob, value):
        self.obs[self.ptr] = obs_dict.cpu()
        self.actions[self.ptr].copy_(action.cpu())
        self.rewards[self.ptr].copy_(reward.cpu())
        self.is_terminated[self.ptr].copy_(is_terminated.cpu())
        self.is_truncated[self.ptr].copy_(is_truncated.cpu())
        self.logprobs[self.ptr].copy_(logprob.cpu())
        self.values[self.ptr].copy_(value.cpu())
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def add_final_step_info(self, final_values, final_is_terminated, final_is_truncated):
        assert self.full, "Buffer not full yet!"
        self.final_values = final_values.cpu()
        self.final_is_terminated = final_is_terminated.float().cpu()
        self.final_is_truncated = final_is_truncated.float().cpu()

    def reset(self):
        self.ptr = 0
        self.full = False


def compute_gae(buffer, gamma, lam):
    rewards = buffer.rewards
    terminated = buffer.is_terminated
    truncated = buffer.is_truncated
    values = buffer.values
    
    # rewards, values, is_terminals: (n_steps, n_envs)
    # the_extra_value: (n_envs,)
    # is_next_terminal: (n_envs,)
    n_steps, n_envs = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(n_envs, device=rewards.device)
    done = (terminated.bool() | truncated.bool()).float()
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            need_next_value = 1.0 - buffer.final_is_terminated
            next_value = buffer.final_values
            step_continues = torch.zeros(n_envs, device=rewards.device)
        else:
            need_next_value = 1.0 - terminated[t + 1]
            next_value = values[t + 1]
            step_continues = 1.0 - done[t + 1]

        delta = rewards[t] + gamma * next_value * need_next_value - values[t]
        lastgaelam = delta + gamma * lam * step_continues * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    valid_mask = ~done.bool()
    return adv, returns, valid_mask


def collect_rollout(cfg, device, rollout_idx, run_dir, actor, critic, writer):
    """
    Handles curriculum, creates parallel envs, collects a rollout, logs stats, saves video, and returns only the buffer and next_value needed for PPO update.
    """
    # === Curriculum learning: select env difficulty from config ===
    curriculum = cfg.env.curriculum
    num_envs = cfg.env.num_envs
    num_levels = cfg.env.num_levels
    rollout_steps = cfg.train.rollout_steps
    
    # Define observation shapes for dict format
    image_shape = (num_levels, 2, cfg.env.view_size, cfg.env.view_size)  # uint8 images
    state_shape = (2,)  # float32 velocity
    
    if curriculum:
        for stage in curriculum:
            if rollout_idx <= stage['until_rollout']:
                min_start_goal_dist = stage['min_start_goal_dist']
                max_start_goal_dist = stage['max_start_goal_dist']
                hitwall_cost = stage['hitwall_cost']
                break
    else:
        raise RuntimeError('No curriculum defined in config!')
    
    # Create parallel environment
    env = ParallelDrivingEnv(
        num_envs=num_envs,
        view_size=cfg.env.view_size,
        dt=cfg.env.dt,
        a_max=cfg.env.a_max,
        v_max=cfg.env.v_max,
        w_col=cfg.env.w_col,
        R_goal=cfg.env.R_goal,
        disk_radius=cfg.env.disk_radius,
        render_dir=run_dir,
        video_fps=cfg.env.video_fps,
        w_dist=cfg.env.w_dist,
        w_accel=cfg.env.w_accel,
        max_steps=cfg.train.max_steps,
        hitwall_cost=hitwall_cost,
        pyramid_levels=cfg.env.pyramid_levels,
        num_levels=cfg.env.num_levels,
        min_start_goal_dist=min_start_goal_dist,
        max_start_goal_dist=max_start_goal_dist,
        device=device
    )
    
    print(f"[INFO] Rollout {rollout_idx}: Start recording transitions... (min/max dist: {min_start_goal_dist}-{max_start_goal_dist}, hitwall: {hitwall_cost})")
    buffer = RolloutBuffer(rollout_steps, num_envs, image_shape, state_shape)
    gamma = cfg.train.gamma
    render_this_rollout = (cfg.env.render_every > 0 and rollout_idx % cfg.env.render_every == 0 or rollout_idx == 1)
    video_frames = None
    if render_this_rollout:
        video_frames = [[] for _ in range(cfg.env.num_envs)]
    
    next_obs_dict, _ = env.reset()
    # Move observation dict to device using TensorDict
    next_obs_dict = next_obs_dict.to(device)
    is_next_terminated = torch.zeros(num_envs, dtype=torch.bool)
    is_next_truncated = torch.zeros(num_envs, dtype=torch.bool)
    episode_rewards = []
    cur_episode_rewards = [[] for _ in range(num_envs)]
    gamma_pow = np.ones(num_envs, dtype=np.float32)
    success_count = 0
    failure_count = 0
    
    for step in tqdm(range(rollout_steps), desc=f"Rollout {rollout_idx} steps", smoothing=0.01):
        cur_obs_dict = next_obs_dict
        is_current_terminated = is_next_terminated.clone()
        is_current_truncated = is_next_truncated.clone()

        with get_timer("inference"):
            with torch.no_grad():
                dist = actor(cur_obs_dict)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = critic(cur_obs_dict).squeeze(-1)

        with get_timer("step"):
            next_obs_dict, reward, terminated, truncated, infos = env.step(action)
            next_obs_dict = next_obs_dict.to(device)
        
        if render_this_rollout:
            with get_timer("render"):
                for env_id in cfg.env.render_env_ids:
                    # Extract single environment observation using TensorDict indexing
                    env_obs_dict = cur_obs_dict[env_id]
                    vel_np = infos['vel'][env_id].cpu().numpy()
                    action_item = action[env_id]
                    
                    # Extract info for this environment
                    info_env = {
                        'r_progress': infos['r_progress'][env_id].item(),
                        'r_goal': infos['r_goal'][env_id].item(),
                        'hitwall': infos['hitwall'][env_id].item(),
                    }
                    
                    frame = get_human_frame(obs_dict=env_obs_dict,
                                            vel=vel_np,
                                            v_max=cfg.env.v_max,
                                            action=action_item,
                                            info=info_env,
                                            a_max=cfg.env.a_max)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_frames[env_id].append(frame_rgb)
        
        buffer.add(cur_obs_dict, action, reward,
                   is_current_terminated,
                   is_current_truncated,
                   logprob, value)
        
        is_next_terminated = terminated
        is_next_truncated = truncated
        
        for i in range(num_envs):
            current_done = is_next_terminated[i] or is_next_truncated[i]
            if not current_done:
                cur_episode_rewards[i].append(reward[i].item() * gamma_pow[i])
                gamma_pow[i] *= gamma

            next_done = is_next_terminated[i] or is_next_truncated[i]
            if next_done:
                if len(cur_episode_rewards[i]) > 0:
                    episode_rewards.append(sum(cur_episode_rewards[i]))
                cur_episode_rewards[i] = []
                gamma_pow[i] = 1.0
                if is_next_terminated[i]:
                    if infos['success'][i].item():
                        success_count += 1
                    else:
                        failure_count += 1
    
    with torch.no_grad():
        next_value = critic(next_obs_dict).squeeze(-1)
    buffer.add_final_step_info(next_value, is_next_terminated, is_next_truncated)

    if episode_rewards:
        avg_discounted_reward = np.mean(episode_rewards)
        writer.add_scalar('Reward/AvgDiscountedEpisode', avg_discounted_reward, rollout_idx)
    avg_successes_per_env = success_count / num_envs
    avg_failures_per_env = failure_count / num_envs
    writer.add_scalar('Reward/AvgSuccess', np.mean(avg_successes_per_env), rollout_idx)
    writer.add_scalar('Reward/AvgFailure', np.mean(avg_failures_per_env), rollout_idx)
    if render_this_rollout and video_frames is not None:
        for env_id in cfg.env.render_env_ids:
            video_path = os.path.join(run_dir, f"rollout_{rollout_idx:05d}_env{env_id}.mp4")
            imageio.mimsave(video_path, video_frames[env_id], fps=cfg.env.video_fps, codec='libx264')
    
    return buffer

def process_buffer(buffer, gamma, lam):
    adv_buf, ret_buf, is_valid_buf = compute_gae(buffer, gamma, lam)

    is_valid_buf = is_valid_buf.reshape(-1)

    # Use TensorDict reshape and indexing for cleaner code
    obs_buf = buffer.obs.reshape(-1)[is_valid_buf]  # TensorDict with flattened batch dimension
    act_buf = buffer.actions.reshape(-1)[is_valid_buf]
    logp_buf = buffer.logprobs.reshape(-1)[is_valid_buf]
    adv_buf = adv_buf.reshape(-1)[is_valid_buf]
    ret_buf = ret_buf.reshape(-1)[is_valid_buf]

    return obs_buf, act_buf, logp_buf, adv_buf, ret_buf


def ppo_update(buffer, critic, actor, cfg, device, scaler, use_fp16, writer, rollout_idx, opt_actor, opt_critic):
    gamma = cfg.train.gamma
    lam = cfg.train.gae_lambda
    ppo_epochs = cfg.train.ppo_epochs
    minibatch_size = cfg.train.minibatch_size
    clip_eps = cfg.train.clip_eps

    # Process buffer on CPU first
    obs_buf, act_buf, logp_buf, adv_buf, ret_buf = process_buffer(buffer, gamma, lam)

    # Keep data on CPU and normalize advantages
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
    
    # Create indices for shuffling on CPU
    total_samples = adv_buf.shape[0]
    inds = torch.randperm(total_samples)
    
    # Determine chunk size, otherwise GPU memory may blow up
    chunk_size = min(minibatch_size * 16, total_samples)  # Process 16 minibatches

    actor_losses = []
    critic_losses = []
    
    for _ in range(ppo_epochs):
        # Process data in chunks to balance memory usage and transfer efficiency
        for chunk_start in tqdm(range(0, total_samples, chunk_size), desc="PPO chunks"):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_inds = inds[chunk_start:chunk_end]
            
            # Transfer chunk to GPU once - TensorDict handles device transfer automatically
            chunk_obs = obs_buf[chunk_inds].to(device)
            chunk_act = act_buf[chunk_inds].to(device)
            chunk_adv = adv_buf[chunk_inds].to(device)
            chunk_ret = ret_buf[chunk_inds].to(device)
            chunk_logp_old = logp_buf[chunk_inds].to(device)
            
            # Process minibatches within this chunk (all data already on GPU)
            chunk_samples = chunk_end - chunk_start
            for mb_start in tqdm(range(0, chunk_samples, minibatch_size), desc="PPO minibatches", leave=False):
                mb_end = min(mb_start + minibatch_size, chunk_samples)
                
                # Extract minibatch using TensorDict indexing
                mb_obs_dict = chunk_obs[mb_start:mb_end]
                mb_act = chunk_act[mb_start:mb_end]
                mb_adv = chunk_adv[mb_start:mb_end]
                mb_ret = chunk_ret[mb_start:mb_end]
                mb_logp_old = chunk_logp_old[mb_start:mb_end]
                mb_act = chunk_act[mb_start:mb_end]
                mb_adv = chunk_adv[mb_start:mb_end]
                mb_ret = chunk_ret[mb_start:mb_end]
                mb_logp_old = chunk_logp_old[mb_start:mb_end]
                
                # Actor update
                with autocast(device_type='cuda', enabled=use_fp16):
                    dist = actor(mb_obs_dict)
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
                
                # Critic update
                with autocast(device_type='cuda', enabled=use_fp16):
                    value = critic(mb_obs_dict).squeeze(-1)
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
            
            # Clear GPU memory for this chunk
            del chunk_obs, chunk_act, chunk_adv, chunk_ret, chunk_logp_old
            torch.cuda.empty_cache()
    
    writer.add_scalar('Loss/Actor', np.mean(actor_losses), rollout_idx)
    writer.add_scalar('Loss/Critic', np.mean(critic_losses), rollout_idx)


def train(cfg: DictConfig):
    run_name = cfg.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = to_absolute_path(cfg.output_dir)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    if cfg.enable_timer:
        set_timing_enabled(True)

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
        reset_timer()

        with get_timer("collect_rollout") as rollout_timer:
            buffer = collect_rollout(cfg, device, rollout_idx, run_dir, actor, critic, writer)
        
        # Get the rollout collection time from the timer instance
        rollout_collection_time = rollout_timer.elapsed()
        print(f"[INFO] Rollout {rollout_idx}: Record took {rollout_collection_time:.3f} s.")

        writer.add_scalar('Time/Record', rollout_collection_time, rollout_idx)
        
        with get_timer("ppo_update") as update_timer:
            ppo_update(buffer, critic, actor, cfg, device, scaler, use_fp16, writer, rollout_idx, opt_actor, opt_critic)
        
        # Get the update time from the timer instance
        update_time = update_timer.elapsed()
        total_rollout_time = rollout_collection_time + update_time
        
        writer.add_scalar('Time/Update', update_time, rollout_idx)
        writer.add_scalar('Time/Rollout', total_rollout_time, rollout_idx)
        print(f"[INFO] Rollout {rollout_idx}: Network update took {update_time:.3f} s.")
        print(f"[INFO] Rollout {rollout_idx}: TotalTime: {total_rollout_time:.3f} s\n")
        if cfg.enable_timer:
            print_timing_report(title=f"Timing Report after Rollout {rollout_idx}", show_exclusive=True)
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