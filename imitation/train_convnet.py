#!/usr/bin/env python3
"""
ConvNet-based Imitation Learning for Flatland Driving
A simpler, smaller alternative to the ACT transformer model
"""

import os
import h5py
import logging
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Import timer utilities
from utils.timer import get_timer, set_timing_enabled, reset_timer, print_timing_report

# Import shared dataloader
from utils.dataloader import create_flatland_dataloader

# Import shared losses
from utils.losses import FocalLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlatlandConvBackbone(nn.Module):
    """
    CNN backbone for extracting spatial features from a single zoom level.
    
    Takes a single level with 2 channels (map + target markers) and outputs fixed-size features.
    The outer model will call this multiple times for different zoom levels.
    """
    
    def __init__(self, 
                 view_size: int = 64,
                 feature_dim: int = 64):
        super().__init__()
        
        self.view_size = view_size
        self.feature_dim = feature_dim
        
        # CNN that processes 2-channel input (map + target markers)
        
        # First conv block - reduce spatial size by 4x (64x64 -> 32x32)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2)  # 2 channels per level
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        
        # Second conv block - reduce by another 4x (32x32 -> 16x16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        
        # Third conv block - reduce by another 4x (16x16 -> 8x8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        # Final conv block - reduce to 4x4 (8x8 -> 4x4)
        self.conv4 = nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1)  # Fixed feature dimension
        self.bn4 = nn.BatchNorm2d(feature_dim)
        
        # Global average pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (batch, feature_dim, 1, 1) -> (batch, feature_dim)
    
    def forward(self, x):
        """
        Process a single zoom level through the CNN.
        
        Args:
            x: (batch_size, 2, view_size, view_size) - single level with map + target channels
            
        Returns:
            features: (batch_size, feature_dim) - features for this level
        """
        # First conv block (64x64 -> 32x32)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn1_2(self.conv1_2(x)), inplace=True)
        
        # Second conv block (32x32 -> 16x16)  
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn2_2(self.conv2_2(x)), inplace=True)
        
        # Third conv block (16x16 -> 8x8)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn3_2(self.conv3_2(x)), inplace=True)
        
        # Final conv block (8x8 -> 4x4)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        
        # Global average pooling and flatten
        features = self.global_pool(x)  # (B, feature_dim, 1, 1)
        features = features.view(x.size(0), -1)  # (B, feature_dim)
        
        return features
    

class FlatlandConvNet(nn.Module):
    """
    Lightweight ConvNet for Flatland navigation
    
    Much smaller than the ACT transformer:
    - Simple CNN backbone for spatial features  
    - Single-step action prediction (no chunking)
    - Roughly 100x smaller than ACT model
    """
    
    def __init__(self, 
                 num_levels: int = 3,
                 view_size: int = 64, 
                 action_dim: int = 9,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_levels = num_levels
        self.view_size = view_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # CNN backbone for spatial feature extraction (processes single zoom level)
        backbone_feature_dim = 64  # Fixed output size from backbone
        self.conv_backbone = FlatlandConvBackbone(
            view_size=view_size,
            feature_dim=backbone_feature_dim
        )
        
        # MLP to map concatenated features from all levels to desired hidden_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(backbone_feature_dim * num_levels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # State encoder for velocity information
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 32),  # Input: (2,) velocity
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer to combine spatial and state features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Add layer normalization before final prediction to stabilize logits
        self.pre_action_norm = nn.LayerNorm(hidden_dim // 2)
        
        # Action prediction head - using smaller hidden dimension helps keep logits reasonable
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)
    
    def forward(self, observations):
        """
        Args:
            observations: Dictionary with 'image' and 'state' keys
                - image: (batch_size, num_levels, 2, view_size, view_size)
                - state: (batch_size, 2)
                
        Returns:
            logits: (batch_size, action_dim) - single-step action predictions
        """
        batch_size = observations['image'].size(0)
        
        # Process spatial features from each zoom level separately
        images = observations['image']  # (B, num_levels, 2, H, W)
        level_features = []
        
        # Process each zoom level through the same backbone
        for level_idx in range(self.num_levels):
            level_input = images[:, level_idx, :, :, :]  # (B, 2, H, W)
            level_feat = self.conv_backbone(level_input)  # (B, backbone_feature_dim)
            level_features.append(level_feat)
        
        # Concatenate features from all levels
        concatenated_features = torch.cat(level_features, dim=1)  # (B, backbone_feature_dim * num_levels)
        
        # Map to desired hidden_dim using MLP
        spatial_features = self.feature_fusion(concatenated_features)  # (B, hidden_dim)
        
        # Process state features
        state_features = self.state_encoder(observations['state'])  # (B, 64)
        
        # Fuse spatial and state features
        combined_features = torch.cat([spatial_features, state_features], dim=1)  # (B, hidden_dim + 64)
        fused_features = self.fusion(combined_features)  # (B, hidden_dim // 2)
        
        # Apply layer normalization before final prediction
        normalized_features = self.pre_action_norm(fused_features)
        
        # Predict action with normalized features
        logits = self.action_head(normalized_features)  # (B, action_dim)
        
        return logits
    
    def generate(self, observations, temperature=1.0):
        """
        Generate actions for inference with sampling
        
        Args:
            observations: Dictionary with 'image' and 'state' keys
                - image: (batch_size, num_levels, 2, view_size, view_size)
                - state: (batch_size, 2)
            temperature: Temperature for softmax sampling (1.0 = normal, >1.0 = more random, <1.0 = more confident)
                
        Returns:
            actions: (batch_size,) - sampled actions for ConvNet (single action per sample)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(observations)  # (batch_size, action_dim)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (batch_size, action_dim)
            
            # Sample from the probability distribution
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)
            
        return actions


class ConvNetTrainer:
    """Trainer for the Flatland ConvNet model"""
    
    def __init__(self, cfg: DictConfig):
        reset_timer()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up run directories
        run_name = cfg.run_name or f"convnet_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = to_absolute_path(cfg.logging.output_dir)
        self.run_dir = os.path.join(output_dir, run_name)
        
        # Create subdirectories
        self.log_dir = os.path.join(self.run_dir, "log")
        self.save_dir = os.path.join(self.run_dir, "checkpoints")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"Run directory: {self.run_dir}")
        logger.info(f"Tensorboard logs: {self.log_dir}")
        logger.info(f"Model checkpoints: {self.save_dir}")
        
        # Enable timing if requested
        if hasattr(cfg, 'enable_timer') and cfg.enable_timer:
            set_timing_enabled(True)
        
        # Create datasets using shared dataloader
        with get_timer("dataset_creation"):
            # Get filtering parameters from config (with defaults for ConvNet)
            max_no_op_ratio = cfg.data.max_no_op_ratio
            stride = cfg.data.stride
            no_op_filter_percentage = cfg.data.no_op_filter_percentage

            self.train_dataset, self.val_dataset = create_flatland_dataloader(
                h5_file_path=cfg.data.h5_file_path,
                history_frames=1,  # ConvNet currently uses 1 frame
                action_chunk_size=1,  # ConvNet predicts single actions
                train_split=cfg.data.train_split,
                random_seed=cfg.data.split_seed,
                max_no_op_ratio=max_no_op_ratio,
                stride=stride,
                no_op_filter_percentage=no_op_filter_percentage
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn
        )
        
        # Create model
        self.model = FlatlandConvNet(
            num_levels=self.train_dataset.num_levels,  # Shared dataloader preserves these attributes
            view_size=self.train_dataset.view_size,
            action_dim=cfg.model.action_dim,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout
        ).to(self.device)
        
        # Optimizer and scheduler  
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        
        # Calculate warmup steps (configurable ratio of total training steps)
        warmup_ratio = cfg.training.warmup_ratio
        warmup_epochs = max(1, int(cfg.training.epochs * warmup_ratio))
        total_steps = len(self.train_loader) * cfg.training.epochs
        warmup_steps = len(self.train_loader) * warmup_epochs
        
        # Create warmup + cosine annealing scheduler (same pattern as ACT)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))  # Min LR is 1% of initial
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss function
        self.criterion = FocalLoss(
            alpha=cfg.training.focal_alpha,
            gamma=cfg.training.focal_gamma,
            reduction='mean'
        )
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Global step counter
        self.global_step = 0
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def _worker_init_fn(self, worker_id):
        """Initialize worker processes for DataLoader multiprocessing."""
        # Disable timers in worker processes to avoid CUDA synchronization issues
        set_timing_enabled(False)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract data from shared dataloader format
            observations = batch['observations']  # Dict with 'image' and 'state' keys
            actions = batch['actions'].squeeze(-1)  # (batch_size, 1) -> (batch_size,)
            
            # Move to device
            observations = {k: v.to(self.device) for k, v in observations.items()}
            actions = actions.to(self.device)
            
            # Forward pass
            logits = self.model(observations)  # (batch_size, action_dim)
            loss = self.criterion(logits, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # More aggressive gradient clipping and monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Update learning rate scheduler (per step, like ACT)
            self.scheduler.step()
            
            # Calculate accuracy
            pred_actions = torch.argmax(logits, dim=-1)
            accuracy = (pred_actions == actions).float().mean()
            
            # Calculate accuracy without no-op (action 0) for better understanding of model performance
            non_noop_mask = actions != 0  # Actions that are not no-op
            if non_noop_mask.sum() > 0:  # Only if there are non-no-op actions in this batch
                non_noop_accuracy = ((pred_actions == actions) & non_noop_mask).float().sum() / non_noop_mask.sum()
            else:
                non_noop_accuracy = torch.tensor(0.0)  # No non-no-op actions in this batch
            
            # Accumulate metrics
            total_loss += loss.item()
            total_acc += accuracy.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy.item():.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', accuracy.item(), self.global_step)
                self.writer.add_scalar('train/accuracy_non_noop', non_noop_accuracy.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/grad_norm', grad_norm, self.global_step)
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.3f}")
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_non_noop_acc = 0
        non_noop_batches = 0  # Count batches that have non-noop actions
        num_batches = 0
        
        all_pred_actions = []
        all_true_actions = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} Validation")
            
            for batch in pbar:
                # Extract data from shared dataloader format
                observations = batch['observations']  # Dict with 'image' and 'state' keys
                actions = batch['actions'].squeeze(-1)  # (batch_size, 1) -> (batch_size,)
                
                # Move to device
                observations = {k: v.to(self.device) for k, v in observations.items()}
                actions = actions.to(self.device)
                
                # Forward pass
                logits = self.model(observations)
                loss = self.criterion(logits, actions)
                
                # Calculate accuracy
                pred_actions = torch.argmax(logits, dim=-1)
                accuracy = (pred_actions == actions).float().mean()
                
                # Calculate accuracy without no-op (action 0)
                non_noop_mask = actions != 0
                if non_noop_mask.sum() > 0:
                    non_noop_accuracy = ((pred_actions == actions) & non_noop_mask).float().sum() / non_noop_mask.sum()
                else:
                    non_noop_accuracy = torch.tensor(0.0)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_acc += accuracy.item()
                if non_noop_mask.sum() > 0:  # Only count batches with non-noop actions
                    total_non_noop_acc += non_noop_accuracy.item()
                    non_noop_batches += 1
                num_batches += 1
                
                # Store predictions for detailed analysis
                all_pred_actions.append(pred_actions.cpu().numpy())
                all_true_actions.append(actions.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy.item():.3f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_non_noop_acc = total_non_noop_acc / max(1, non_noop_batches)  # Avoid division by zero
        
        # Detailed analysis
        all_pred_actions = np.concatenate(all_pred_actions)
        all_true_actions = np.concatenate(all_true_actions)
        
        # Action distribution analysis
        action_names = ["No-op", "Up", "Up-Right", "Right", "Down-Right", 
                       "Down", "Down-Left", "Left", "Up-Left"]
        
        # Log action distribution (similar to ACT training)
        if epoch >= 0:  # Only for actual training epochs
            pred_hist = np.bincount(all_pred_actions, minlength=9)
            true_hist = np.bincount(all_true_actions, minlength=9)
            
            pred_dist = pred_hist / pred_hist.sum()
            true_dist = true_hist / true_hist.sum()
            
            # Create distribution comparison text
            dist_lines = ["Action | True Prob | Pred Prob | Difference"]
            dist_lines.append("-" * 45)
            for action_idx in range(9):
                action_name = action_names[action_idx]
                true_prob = true_dist[action_idx]
                pred_prob = pred_dist[action_idx]
                diff = pred_prob - true_prob
                dist_lines.append(f"{action_name:<8} | {true_prob:.3f}    | {pred_prob:.3f}    | {diff:+.3f}")
            
            # Log action distribution with HTML formatting for monospace display
            html_dist_content = f'<pre style="font-family: monospace; font-size: 12px;">' + '\n'.join(dist_lines) + '</pre>'
            self.writer.add_text(
                'val/action_distribution',
                html_dist_content,
                self.global_step
            )
            
            # Log action distributions as histograms for better visualization
            # Create weighted samples for histogram visualization
            true_samples = np.repeat(np.arange(9), (true_hist * 1000).astype(int))  # Scale up for better visualization
            pred_samples = np.repeat(np.arange(9), (pred_hist * 1000).astype(int))
            
            if len(true_samples) > 0:  # Only log if we have samples
                self.writer.add_histogram('val/action_dist/ground_truth', true_samples, self.global_step)
            if len(pred_samples) > 0:
                self.writer.add_histogram('val/action_dist/predictions', pred_samples, self.global_step)
        
        if epoch == -1:
            logger.info(f"Initial validation - Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.3f}, Non-NoOp Acc: {avg_non_noop_acc:.3f}")
        else:
            logger.info(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.3f}, Non-NoOp Acc: {avg_non_noop_acc:.3f}")
        
        # Log to tensorboard (only for actual training epochs, not initial validation)
        if epoch >= 0:
            self.writer.add_scalar('val/loss', avg_loss, self.global_step)
            self.writer.add_scalar('val/accuracy', avg_acc, self.global_step)
            self.writer.add_scalar('val/accuracy_non_noop', avg_non_noop_acc, self.global_step)
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with val loss: {loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.cfg.logging.save_interval == 0:
            periodic_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting ConvNet training...")
        
        # Log network architecture details
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("=" * 60)
        logger.info("CONVNET MODEL ARCHITECTURE")
        logger.info("=" * 60)
        logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(f"Model size estimate: ~{total_params * 4 / 1e6:.1f} MB (FP32)")
        logger.info("=" * 60)
        
        # Run initial validation
        logger.info("Running initial validation...")
        initial_val_loss, initial_val_acc = self.validate(epoch=-1)
        
        # Log initial validation to tensorboard for reference
        self.writer.add_scalar('val/loss_initial', initial_val_loss, 0)
        self.writer.add_scalar('val/accuracy_initial', initial_val_acc, 0)
        
        best_val_loss = initial_val_loss
        
        for epoch in range(self.cfg.training.epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            logger.info(f"Epoch {epoch+1}/{self.cfg.training.epochs} completed")
            logger.info(f"Best val loss so far: {best_val_loss:.4f}")
        
        logger.info("Training completed!")
        self.writer.close()
        
        if hasattr(self.cfg, 'enable_timer') and self.cfg.enable_timer:
            print_timing_report()


@hydra.main(version_base=None, config_path="../config", config_name="convnet_config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    
    if not os.path.exists(cfg.data.h5_file_path):
        raise FileNotFoundError(f"H5 file not found: {cfg.data.h5_file_path}")
    
    # Create trainer and start training
    trainer = ConvNetTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
