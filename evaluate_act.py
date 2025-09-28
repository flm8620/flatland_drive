#!/usr/bin/env python3
"""
ACT Evaluation and Inference for Flatland Driving
Test the trained ACT model in the environment
"""

import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import cv2
from tqdm import tqdm

from flatland_act import FlatlandACT
from env import ParallelDrivingEnv, get_human_frame


class ACTAgent:
    """Agent that uses trained ACT model for action prediction"""
    
    def __init__(self, checkpoint_path: str, num_levels: int, view_size: int, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint['config']['model']
        
        # Create model with same config as training
        self.model = FlatlandACT(
            num_levels=num_levels,  # From environment config
            view_size=view_size,  # From environment config
            action_dim=model_config['action_dim'],
            chunk_size=model_config['chunk_size'],
            hidden_dim=model_config['hidden_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers'],
            num_decoder_layers=model_config['num_decoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            spatial_size=model_config['spatial_size'],
            use_cvae=model_config['use_cvae'],
            latent_dim=model_config['latent_dim']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Action chunking state
        self.chunk_size = model_config['chunk_size']
        self.action_queue = []
        
        print(f"Loaded ACT model from {checkpoint_path}")
        print(f"Model epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    def predict_action(self, observation):
        """
        Predict next action given observation.
        Uses action chunking - predicts multiple actions at once and queues them.
        
        Args:
            observation: Dictionary with 'image' and 'state' keys
            
        Returns:
            action: integer action index
        """
        # If we have queued actions, use them first
        if self.action_queue:
            return self.action_queue.pop(0)
        
        # Generate new action chunk
        with torch.no_grad():
            obs_batch = {
                'image': observation['image'].unsqueeze(0).to(self.device),
                'state': observation['state'].unsqueeze(0).to(self.device)
            }
            
            actions = self.model.generate(obs_batch)  # (1, chunk_size)
            action_list = actions[0].cpu().tolist()  # Convert to list
        
        # Take first action and queue the rest
        first_action = action_list[0]
        self.action_queue = action_list[1:]
        
        return first_action
    
    def reset(self):
        """Reset agent state (clear action queue)"""
        self.action_queue = []


def evaluate_agent(cfg: DictConfig, agent: ACTAgent, num_episodes: int = 10, render: bool = True, run_name: str = None):
    """Evaluate the ACT agent in the environment"""
    
    # Create evaluation output directory
    if run_name is None:
        run_name = "evaluation"
    
    eval_dir = os.path.join("eval_outputs", run_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    if render:
        video_dir = os.path.join(eval_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    
    # Create environment using config parameters
    env = ParallelDrivingEnv(
        num_envs=cfg.env.num_envs,
        view_size=cfg.env.view_size,
        dt=cfg.env.dt,
        a_max=cfg.env.a_max,
        v_max=cfg.env.v_max,
        w_col=cfg.env.w_col,
        R_goal=cfg.env.R_goal,
        disk_radius=cfg.env.disk_radius,
        render_dir=cfg.env.render_dir,
        video_fps=cfg.env.video_fps,
        w_dist=cfg.env.w_dist,
        w_accel=cfg.env.w_accel,
        max_steps=cfg.env.max_steps,
        hitwall_cost=cfg.env.hitwall_cost,
        pyramid_levels=cfg.env.pyramid_levels,
        num_levels=cfg.env.num_levels,
        min_start_goal_dist=cfg.env.min_start_goal_dist,
        max_start_goal_dist=cfg.env.max_start_goal_dist,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = {
        'success_rate': 0,
        'avg_episode_reward': 0,
        'avg_episode_length': 0,
        'collision_rate': 0
    }
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    collisions = 0
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        agent.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        frames = []
        
        while not done:
            # Get action from agent
            obs_single = obs[0]  # Single environment observation
            obs_dict = {
                'image': obs_single['image'],
                'state': obs_single['state']
            }
            
            action = agent.predict_action(obs_dict)
            
            # Step environment - ensure action tensor is on correct device
            action_tensor = torch.tensor([action], device=env.device)
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            
            episode_reward += reward[0].item()
            episode_length += 1
            done = terminated[0] or truncated[0]
            
            # Render frame if requested
            if render:
                frame = get_human_frame(
                    obs_single,
                    info['vel'][0].cpu(),
                    cfg.env.v_max,
                    action=action,
                    info={k: v[0] for k, v in info.items() if k != 'vel'},
                    a_max=cfg.env.a_max
                )
                frames.append(frame)
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info['success'][0]:
            successes += 1
        if info['hitwall'][0] < -1:  # Collision detected
            collisions += 1
        
        # Save video if rendering
        if render and frames:
            video_path = os.path.join(video_dir, f'episode_{episode:03d}.avi')
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
        
        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, Length={episode_length}, "
              f"Success={'Yes' if info['success'][0] else 'No'}")
    
    # Calculate final statistics
    results['success_rate'] = successes / num_episodes
    results['avg_episode_reward'] = np.mean(episode_rewards)
    results['avg_episode_length'] = np.mean(episode_lengths)
    results['collision_rate'] = collisions / num_episodes
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Average Episode Reward: {results['avg_episode_reward']:.2f}")
    print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
    print(f"Collision Rate: {results['collision_rate']:.1%}")
    print("="*50)
    
    return results, eval_dir


@hydra.main(version_base=None, config_path=".", config_name="eval_config")
def main(cfg: DictConfig):
    """Main evaluation function"""
    
    import json
    
    # Check for checkpoint_path and run_name in config
    checkpoint_path = cfg.checkpoint_path
    eval_run_name = cfg.run_name

    if not checkpoint_path:
        print("ERROR: checkpoint_path must be provided!")
        print("Usage: python evaluate_act.py checkpoint_path=/path/to/checkpoint.pth")
        print("Optional: python evaluate_act.py checkpoint_path=/path/to/checkpoint.pth run_name=my_eval")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Determine run name for output organization
    if eval_run_name:
        run_name = eval_run_name
        print(f"Using provided run name: {run_name}")
    else:
        # Extract run name from checkpoint path for organized output
        # e.g., "checkpoints/act_flatland/run_20250926_151024/best_model.pth" -> "run_20250926_151024"
        path_parts = checkpoint_path.split(os.sep)
        run_name = None
        for part in path_parts:
            if part.startswith('run_') or part.startswith('0'):  # Handle both run_* and date formats like 0928
                run_name = part
                break
        
        if run_name is None:
            # Fallback to checkpoint filename without extension
            run_name = os.path.basename(checkpoint_path).replace('.pth', '')
        
        print(f"Auto-detected run name: {run_name}")
    
    agent = ACTAgent(checkpoint_path, cfg.env.num_levels, cfg.env.view_size)
    results, eval_dir = evaluate_agent(cfg, agent, num_episodes=1, render=True, run_name=run_name)
    
    # Save results in the evaluation directory
    results_path = os.path.join(eval_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print(f"Videos saved in {os.path.join(eval_dir, 'videos')}")


if __name__ == '__main__':
    main()
