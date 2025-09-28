#!/usr/bin/env python3
"""
Evaluation script for ConvNet-based imitation learning model
"""

import os
import torch
import numpy as np
import h5py
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import time
from collections import defaultdict
import cv2

from train_convnet import FlatlandConvNet, FlatlandConvNetDataset
from env.env import ParallelDrivingEnv


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained ConvNet model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = FlatlandConvNet(
        num_levels=3,  # Default from dataset
        view_size=64,  # Default from dataset  
        action_dim=config['model']['action_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['loss']:.4f}")
    
    return model, config


def evaluate_on_dataset(model, dataset, device, num_samples=1000):
    """
    Evaluate model on validation dataset
    
    Args:
        model: Trained ConvNet model
        dataset: FlatlandConvNetDataset instance
        device: Device to run on
        num_samples: Number of samples to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Limit samples if dataset is smaller
    num_samples = min(num_samples, len(dataset))
    
    correct_predictions = 0
    total_predictions = 0
    action_counts = defaultdict(int)
    action_correct = defaultdict(int)
    
    print(f"Evaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Progress: {i}/{num_samples}")
                
            observations, true_action = dataset[i]
            
            # Add batch dimension and move to device
            obs_batch = {
                'image': observations['image'].unsqueeze(0).to(device),
                'state': observations['state'].unsqueeze(0).to(device)
            }
            
            # Predict
            logits = model(obs_batch)
            pred_action = torch.argmax(logits, dim=-1).item()
            true_action = true_action.item()
            
            # Update metrics
            total_predictions += 1
            action_counts[true_action] += 1
            
            if pred_action == true_action:
                correct_predictions += 1
                action_correct[true_action] += 1
    
    # Compute metrics
    overall_accuracy = correct_predictions / total_predictions
    
    action_names = ["No-op", "Up", "Up-Right", "Right", "Down-Right", 
                   "Down", "Down-Left", "Left", "Up-Left"]
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print(f"\nPer-Action Results:")
    print(f"{'Action':<12} {'Count':<8} {'Accuracy':<10} {'Frequency':<10}")
    print("-" * 45)
    
    for action in range(9):
        count = action_counts[action]
        if count > 0:
            acc = action_correct[action] / count
            freq = count / total_predictions
            print(f"{action_names[action]:<12} {count:<8} {acc:<10.3f} {freq:<10.3f}")
        else:
            print(f"{action_names[action]:<12} {0:<8} {'N/A':<10} {0.0:<10.3f}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_predictions,
        'action_counts': dict(action_counts),
        'action_accuracies': {k: action_correct[k] / action_counts[k] if action_counts[k] > 0 else 0 
                             for k in range(9)}
    }


def evaluate_in_environment(model, env_config_path, device, num_episodes=10, max_steps=500):
    """
    Evaluate model by running episodes in the environment
    
    Args:
        model: Trained ConvNet model
        env_config_path: Path to environment config
        device: Device to run on
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        dict: Environment evaluation metrics
    """
    model.eval()
    
    # Load environment config
    env_config = OmegaConf.load(env_config_path)
    
    # Create environment
    env = ParallelDrivingEnv(
        batch_size=1,  # Single environment
        view_size=env_config.view_size,
        num_levels=env_config.num_levels,
        max_episode_steps=max_steps,
        device=device
    )
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\n=== ENVIRONMENT EVALUATION ===")
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        with torch.no_grad():
            while not done and episode_length < max_steps:
                # Model prediction
                logits = model(obs)
                action = torch.argmax(logits, dim=-1)  # (1,) - single action
                
                # Environment step
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward.item()
                episode_length += 1
                
                # Check if episode completed successfully
                if done and reward.item() > 0:  # Positive reward indicates success
                    success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length) 
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}, "
              f"Success={'Yes' if done and episode_reward > 0 else 'No'}")
    
    # Compute metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print(f"\n=== ENVIRONMENT RESULTS ===")
    print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Success Rate: {success_rate:.3f} ({success_count}/{num_episodes})")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ConvNet imitation learning model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default='human_data/human_data_250928-3.h5',
                       help='Path to evaluation dataset')
    parser.add_argument('--env_config', type=str,
                       default='config/env_config.yaml',
                       help='Path to environment config')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run evaluation on')
    parser.add_argument('--eval_samples', type=int, default=1000,
                       help='Number of dataset samples to evaluate on')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of environment episodes to run')
    parser.add_argument('--no_dataset_eval', action='store_true',
                       help='Skip dataset evaluation')
    parser.add_argument('--no_env_eval', action='store_true', 
                       help='Skip environment evaluation')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Dataset evaluation
    if not args.no_dataset_eval:
        if os.path.exists(args.data_path):
            # Create validation dataset (using same split logic as training)
            with h5py.File(args.data_path, 'r') as h5_file:
                total_episodes = h5_file.attrs['total_episodes']
                
            # Use same split logic as training
            np.random.seed(42)  # Same seed as training
            episode_indices = np.arange(total_episodes)
            np.random.shuffle(episode_indices)
            
            split_idx = int(total_episodes * 0.8)  # Same split as training
            val_episode_indices = sorted(episode_indices[split_idx:].tolist())
            
            val_dataset = FlatlandConvNetDataset(
                args.data_path,
                episode_indices=val_episode_indices,
                mode='val'
            )
            
            dataset_metrics = evaluate_on_dataset(model, val_dataset, device, args.eval_samples)
        else:
            print(f"Dataset not found: {args.data_path}. Skipping dataset evaluation.")
    
    # Environment evaluation  
    if not args.no_env_eval:
        if os.path.exists(args.env_config):
            env_metrics = evaluate_in_environment(model, args.env_config, device, args.eval_episodes)
        else:
            print(f"Environment config not found: {args.env_config}. Skipping environment evaluation.")
    
    print("\n=== EVALUATION COMPLETE ===")


if __name__ == '__main__':
    main()
