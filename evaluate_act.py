#!/usr/bin/env python3
"""
ACT Evaluation and Inference for Flatland Driving
Test the trained ACT model in the environment
"""

import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import cv2
from tqdm import tqdm

from flatland_act import FlatlandACT, create_act_config
from env import ParallelDrivingEnv, get_human_frame


class ACTAgent:
    """Agent that uses trained ACT model for action prediction"""
    
    def __init__(self, checkpoint_path: str, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint['config']['model']
        
        # Create model with same config as training
        self.model = FlatlandACT(
            num_levels=3,  # From environment config
            view_size=64,  # From environment config
            action_dim=model_config['action_dim'],
            chunk_size=model_config['chunk_size'],
            hidden_dim=model_config['hidden_dim'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers'],
            num_decoder_layers=model_config['num_decoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout']
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
            observation: (num_levels, 4, view_size, view_size) tensor
            
        Returns:
            action: integer action index
        """
        # If we have queued actions, use them first
        if self.action_queue:
            return self.action_queue.pop(0)
        
        # Generate new action chunk
        with torch.no_grad():
            obs_batch = observation.unsqueeze(0).to(self.device)  # Add batch dim
            actions = self.model.generate(obs_batch)  # (1, chunk_size)
            action_list = actions[0].cpu().tolist()  # Convert to list
        
        # Take first action and queue the rest
        first_action = action_list[0]
        self.action_queue = action_list[1:]
        
        return first_action
    
    def reset(self):
        """Reset agent state (clear action queue)"""
        self.action_queue = []


def evaluate_agent(cfg: DictConfig, agent: ACTAgent, num_episodes: int = 10, render: bool = True):
    """Evaluate the ACT agent in the environment"""
    
    # Create environment (single environment for evaluation)
    env = ParallelDrivingEnv(
        num_envs=1,
        view_size=cfg.env.view_size,
        dt=cfg.env.dt,
        a_max=cfg.env.a_max,
        v_max=cfg.env.v_max,
        w_col=cfg.env.w_col,
        R_goal=cfg.env.R_goal,
        disk_radius=cfg.env.disk_radius,
        render_dir=None,  # Don't save environment videos
        video_fps=cfg.env.video_fps,
        w_dist=cfg.env.w_dist,
        w_accel=cfg.env.w_accel,
        max_steps=cfg.train.max_steps,
        hitwall_cost=-50.0,  # Use final curriculum level
        pyramid_levels=cfg.env.pyramid_levels,
        num_levels=cfg.env.num_levels,
        min_start_goal_dist=50,  # Challenging scenarios
        max_start_goal_dist=200,
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
    
    if render:
        os.makedirs('eval_videos', exist_ok=True)
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        agent.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        frames = []
        
        while not done:
            # Get action from agent
            action = agent.predict_action(obs[0])  # obs is (1, num_levels, 4, H, W)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(torch.tensor([action]))
            
            episode_reward += reward[0].item()
            episode_length += 1
            done = terminated[0] or truncated[0]
            
            # Render frame if requested
            if render:
                # Convert observation to human viewable format
                frame = get_human_frame(
                    obs[0].cpu(),  # Single environment observation
                    info['vel'][0].cpu(),  # Velocity
                    cfg.env.v_max,
                    action=action,
                    info={k: v[0] for k, v in info.items() if k != 'vel'},  # Single env info
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
            video_path = f'eval_videos/episode_{episode:03d}.mp4'
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    
    return results


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function"""
    
    # Find best checkpoint
    checkpoint_dir = "checkpoints/act_flatland"
    checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"  {f}")
        else:
            print("  Checkpoint directory doesn't exist")
        return
    
    # Create agent
    agent = ACTAgent(checkpoint_path)
    
    # Run evaluation
    results = evaluate_agent(cfg, agent, num_episodes=20, render=True)
    
    # Save results
    import json
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to eval_results.json")


if __name__ == '__main__':
    main()
