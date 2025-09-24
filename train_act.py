#!/usr/bin/env python3
"""
Training script for Flatland ACT
Run this to start training the ACT model on human demonstration data
"""

import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

from flatland_act import ACTTrainer


def main():
    """Main training function"""
    print("Starting Flatland ACT Training")
    print("="*50)
    
    # Check if config exists, create if not
    config_path = 'act_config.yaml'
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Check if data file exists
    if not os.path.exists(cfg.data.h5_file_path):
        print(f"ERROR: Data file not found at {cfg.data.h5_file_path}")
        print("Please make sure you have recorded human demonstration data first.")
        print("Run: python human_player.py")
        return
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*50)
    
    # Create trainer and start training
    try:
        trainer = ACTTrainer(cfg)
        trainer.train()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Training finished. Checkpoints saved in:", cfg.logging.save_dir)
    print("To evaluate the model, run: python evaluate_act.py")


if __name__ == '__main__':
    main()
