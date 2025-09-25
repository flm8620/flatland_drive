#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers) for Flatland Driving
Adapted for the maze navigation task with multi-scale observations
"""

import os
import h5py
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf

# Import timer utilities
from timer import get_timer, set_timing_enabled, reset_timer, print_timing_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlatlandDataset(Dataset):
    """Dataset for loading Flatland human demonstration data from H5 files."""
    
    def __init__(self, h5_file_path: str, chunk_size: int = 32, 
                 train_split: float = 0.8, mode: str = 'train'):
        """
        Args:
            h5_file_path: Path to the H5 file containing demonstrations
            chunk_size: Number of actions to predict per observation
            train_split: Fraction of episodes to use for training
            mode: 'train' or 'val'
        """
        self.h5_file_path = h5_file_path
        self.chunk_size = chunk_size
        self.mode = mode
        
        # Open HDF5 file and keep it open for the dataset lifetime
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        # Get metadata
        total_episodes = self.h5_file.attrs['total_episodes']
        self.view_size = self.h5_file.attrs['view_size']
        self.num_levels = self.h5_file.attrs['num_levels']
        
        # Load episode metadata
        episodes = self.h5_file['episodes'][:total_episodes]
        
        # Split episodes
        split_idx = int(total_episodes * train_split)
        if mode == 'train':
            self.episodes = episodes[:split_idx]
        else:  # val
            self.episodes = episodes[split_idx:]
        
        print(f"{mode.capitalize()} split: {len(self.episodes)} episodes")
        
        # Build list of valid starting positions for chunks
        self.valid_starts = []
        for ep in self.episodes:
            start_idx = ep['start_idx']
            length = ep['length']
            # Can start a chunk at any position where we have enough future actions
            for i in range(length - chunk_size + 1):
                self.valid_starts.append(start_idx + i)
        
        print(f"Total valid chunk starts: {len(self.valid_starts)}")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        """
        Returns:
            observation: TensorDict with 'image' and 'state' keys
            actions: (chunk_size,) - future actions to predict
            mask: (chunk_size,) - padding mask (all True for now)
        """
        with get_timer("dataset_getitem"):
            start_idx = self.valid_starts[idx]
            
            with get_timer("h5_file_read"):
                # Load observations at start position - NEW FORMAT
                # Images: (num_levels, 2, view_size, view_size) uint8
                obs_images = self.h5_file['obs_images'][start_idx]  
                # State: (2,) float32 - velocity information
                obs_states = self.h5_file['obs_states'][start_idx]
                
                # Load next chunk_size actions
                actions = self.h5_file['actions'][start_idx:start_idx + self.chunk_size]
            
            with get_timer("tensor_conversion"):
                # Convert to tensors
                obs_images_tensor = torch.from_numpy(obs_images).float() / 255.0  # Convert uint8 to float [0,1]
                obs_states_tensor = torch.from_numpy(obs_states).float()
                actions_tensor = torch.from_numpy(actions).long()
                
                # Create observation dictionary (plain dict, no batch dimension for single sample)
                obs_dict = {
                    'image': obs_images_tensor,  # (num_levels, 2, view_size, view_size)
                    'state': obs_states_tensor   # (2,) velocity
                }
                
                # Create mask (no padding for now)
                mask = torch.ones(self.chunk_size, dtype=torch.bool)
            
            return {
                'observations': obs_dict,
                'actions': actions_tensor,
                'mask': mask,
            }
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoidal positional encoding"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class MultiScaleObsEncoder(nn.Module):
    """Encoder for multi-scale observations from the Flatland environment."""
    
    def __init__(self, num_levels: int = 3, view_size: int = 64, 
                 feature_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.num_levels = num_levels
        self.view_size = view_size
        self.feature_dim = feature_dim
        
        # Individual encoders for each scale level
        # Note: New format has 2 channels instead of 4 (cost + target maps)
        self.level_encoders = nn.ModuleList()
        for level in range(num_levels):
            encoder = nn.Sequential(
                # Input: (2, view_size, view_size) - cost map + target map
                nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),  # Fixed spatial size
                nn.Flatten(),  # 128 * 4 * 4 = 2048
                nn.Linear(128 * 4 * 4, feature_dim),
                nn.ReLU(inplace=True)
            )
            self.level_encoders.append(encoder)
        
        # State encoder for velocity information
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 64),  # Input: (2,) velocity
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion network to combine multi-scale features + state
        fusion_input_dim = num_levels * feature_dim + 128  # Image features + state features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, obs_dict):
        """
        Args:
            obs_dict: Dictionary with keys:
                - 'image': (batch_size, num_levels, 2, view_size, view_size)
                - 'state': (batch_size, 2) - velocity information
        Returns:
            features: (batch_size, hidden_dim)
        """
        with get_timer("obs_encoder_forward"):
            images = obs_dict['image']
            states = obs_dict['state']
            batch_size = images.size(0)
            
            # Encode each level separately
            level_features = []
            with get_timer("level_encoding"):
                for level in range(self.num_levels):
                    level_input = images[:, level]  # (batch_size, 2, view_size, view_size)
                    level_feat = self.level_encoders[level](level_input)  # (batch_size, feature_dim)
                    level_features.append(level_feat)
            
            # Encode state information
            state_features = self.state_encoder(states)  # (batch_size, 128)
            
            # Concatenate and fuse
            with get_timer("feature_fusion"):
                combined = torch.cat(level_features + [state_features], dim=1)  # (batch_size, fusion_input_dim)
                output = self.fusion(combined)  # (batch_size, hidden_dim)
            
            return output


class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # Self attention
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Standard transformer decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # Self attention on target sequence
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention with memory
        tgt2 = self.multihead_attn(query=tgt if query_pos is None else tgt + query_pos,
                                   key=memory if pos is None else memory + pos,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class FlatlandACT(nn.Module):
    """ACT model adapted for Flatland navigation task"""
    
    def __init__(self, 
                 num_levels: int = 3,
                 view_size: int = 64,
                 action_dim: int = 9,
                 chunk_size: int = 32,
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # Observation encoder
        self.obs_encoder = MultiScaleObsEncoder(
            num_levels=num_levels,
            view_size=view_size, 
            feature_dim=hidden_dim//2,
            hidden_dim=hidden_dim
        )
        
        # Transformer encoder (for observations) - manual implementation to avoid built-in bugs
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.num_encoder_layers = num_encoder_layers
        
        # Transformer decoder (for action sequence generation) - manual implementation
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.num_decoder_layers = num_decoder_layers
        
        # Action queries (learnable embeddings for each position in action sequence)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))
        
        # Positional encodings
        self.register_buffer(
            'pos_encoding', 
            get_sinusoid_encoding_table(chunk_size, hidden_dim)
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, observations, actions=None, mask=None):
        """
        Args:
            observations: Dictionary with 'image' and 'state' keys
            actions: (batch_size, chunk_size) - ground truth actions for training
            mask: (batch_size, chunk_size) - padding mask
            
        Returns:
            logits: (batch_size, chunk_size, action_dim)
            
        For training, actions should be provided.
        For inference, actions should be None.
        """
        with get_timer("act_forward"):
            # Get batch size from the observations dictionary
            batch_size = observations['image'].size(0)
            device = observations['image'].device
            
            # Encode observations
            with get_timer("observation_encoding"):
                obs_features = self.obs_encoder(observations)  # (batch_size, hidden_dim)
            
            # Add batch dimension and transpose for transformer: (1, batch_size, hidden_dim)
            with get_timer("memory_preparation"):
                memory = obs_features.unsqueeze(0)
                
                # Pass through encoder layers manually
                for encoder_layer in self.encoder_layers:
                    memory = encoder_layer(memory)  # (1, batch_size, hidden_dim)
            
            # Prepare action queries
            with get_timer("query_preparation"):
                # action_queries: (chunk_size, hidden_dim) -> (chunk_size, batch_size, hidden_dim)
                tgt = self.action_queries.unsqueeze(1).repeat(1, batch_size, 1)
                
                # Add positional encoding to queries
                query_pos = self.pos_encoding[0, :self.chunk_size].unsqueeze(1).repeat(1, batch_size, 1)
                
                # Generate causal mask for decoder (autoregressive generation)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.chunk_size).to(device)
            
            # Pass through decoder layers manually
            with get_timer("decoder_forward"):
                output = tgt
                for decoder_layer in self.decoder_layers:
                    output = decoder_layer(
                        tgt=output,
                        memory=memory,
                        tgt_mask=tgt_mask,
                        query_pos=query_pos
                    )  # (chunk_size, batch_size, hidden_dim)
            
            # Transpose back: (batch_size, chunk_size, hidden_dim)
            with get_timer("output_projection"):
                output = output.transpose(0, 1)
                
                # Generate action logits
                logits = self.action_head(output)  # (batch_size, chunk_size, action_dim)
            
            return logits
    
    def generate(self, observations):
        """Generate actions for inference (non-autoregressive for now)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(observations)
            actions = torch.argmax(logits, dim=-1)  # (batch_size, chunk_size)
        return actions


class ACTTrainer:
    """Trainer for the Flatland ACT model"""
    
    def __init__(self, cfg: DictConfig):
        reset_timer()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable timing if requested
        if hasattr(cfg, 'enable_timer') and cfg.enable_timer:
            set_timing_enabled(True)
        
        # Create datasets
        with get_timer("dataset_creation"):
            self.train_dataset = FlatlandDataset(
                h5_file_path=cfg.data.h5_file_path,
                chunk_size=cfg.model.chunk_size,
                train_split=cfg.data.train_split,
                mode='train'
            )
            
            self.val_dataset = FlatlandDataset(
                h5_file_path=cfg.data.h5_file_path,
                chunk_size=cfg.model.chunk_size,
                train_split=cfg.data.train_split,
                mode='val'
            )
        
        # Create data loaders
        with get_timer("dataloader_creation"):
            # Configure timing for multiprocessing
            if cfg.training.num_workers > 0:
                # With multiprocessing, we need to configure timers for worker processes
                # The main process keeps timers enabled, workers will disable them
                worker_init_fn = self._worker_init_fn
            else:
                # Single-threaded: keep all timers enabled
                worker_init_fn = None
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn
            )
        
        # Create model
        with get_timer("model_creation"):
            self.model = FlatlandACT(
                num_levels=self.train_dataset.num_levels,
                view_size=self.train_dataset.view_size,
                action_dim=cfg.model.action_dim,
                chunk_size=cfg.model.chunk_size,
                hidden_dim=cfg.model.hidden_dim,
                nhead=cfg.model.nhead,
                num_encoder_layers=cfg.model.num_encoder_layers,
                num_decoder_layers=cfg.model.num_decoder_layers,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout
            ).to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.lr * 0.01
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Don't reduce yet
        
        # Logging
        self.log_dir = cfg.logging.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Model saving
        self.save_dir = cfg.logging.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def _worker_init_fn(self, worker_id):
        """Initialize worker processes for DataLoader multiprocessing.
        
        This function disables timers in worker processes to avoid CUDA issues,
        while keeping them enabled in the main process.
        """
        # Disable timers in worker processes to avoid CUDA synchronization issues
        set_timing_enabled(False)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0
    
        dataloader_iter = iter(self.train_loader)
        pbar = tqdm(range(len(self.train_loader)), desc=f"Epoch {epoch+1} Training")
        
        for batch_idx in pbar:
            
            with get_timer("fetch_data"):
                batch = next(dataloader_iter)

            with get_timer("data_to_device"):
                # Transfer dictionary of observations to device
                observations = batch['observations']
                observations = {
                    'image': observations['image'].to(self.device),
                    'state': observations['state'].to(self.device)
                }
                actions = batch['actions'].to(self.device)  # (B, chunk_size)
                mask = batch['mask'].to(self.device)  # (B, chunk_size)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            with get_timer("forward_pass"):
                logits = self.model(observations, actions, mask)  # (B, chunk_size, action_dim)
            
            # Compute loss (only on non-padded positions)
            with get_timer("loss_computation"):
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),  # (B * chunk_size, action_dim) 
                    actions.reshape(-1)  # (B * chunk_size,)
                )
                loss = loss.reshape(actions.shape)  # (B, chunk_size)
                
                # Apply mask and average
                valid_loss = (loss * mask.float()).sum() / mask.sum()
            
            # Backward pass
            with get_timer("backward_pass"):
                valid_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Calculate accuracy
            with get_timer("accuracy_computation"):
                pred_actions = torch.argmax(logits, dim=-1)
                correct = (pred_actions == actions) * mask
                accuracy = correct.sum().float() / mask.sum()
            
            total_loss += valid_loss.item()
            total_acc += accuracy.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{valid_loss.item():.4f}",
                'acc': f"{accuracy.item():.3f}"
            })
            
            # Log to tensorboard
            with get_timer("tensorboard_logging"):
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', valid_loss.item(), global_step)
                self.writer.add_scalar('train/accuracy', accuracy.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

            print_timing_report()
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.3f}")
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with get_timer(f"val_epoch_{epoch}"):
            with torch.no_grad():
                dataloader_iter = iter(self.val_loader)
                pbar = tqdm(range(len(self.val_loader)), desc=f"Epoch {epoch+1} Validation")
                
                for batch_idx in pbar:
                    with get_timer("val_batch_processing"):
                        
                        batch = next(dataloader_iter)
                        
                        with get_timer("val_data_transfer"):
                            # Transfer dictionary of observations to device
                            observations = batch['observations']
                            observations = {
                                'image': observations['image'].to(self.device),
                                'state': observations['state'].to(self.device)
                            }
                            actions = batch['actions'].to(self.device)
                            mask = batch['mask'].to(self.device)
                        
                        # Forward pass
                        with get_timer("val_forward_pass"):
                            logits = self.model(observations, actions, mask)
                        
                        # Compute loss
                        with get_timer("val_loss_computation"):
                            loss = self.criterion(
                                logits.reshape(-1, logits.size(-1)),
                                actions.reshape(-1)
                            )
                            loss = loss.reshape(actions.shape)
                            valid_loss = (loss * mask.float()).sum() / mask.sum()
                        
                        # Calculate accuracy
                        with get_timer("val_accuracy_computation"):
                            pred_actions = torch.argmax(logits, dim=-1)
                            correct = (pred_actions == actions) * mask
                            accuracy = correct.sum().float() / mask.sum()
                        
                        total_loss += valid_loss.item()
                        total_acc += accuracy.item()
                        num_batches += 1
                        
                        pbar.set_postfix({
                            'loss': f"{valid_loss.item():.4f}",
                            'acc': f"{accuracy.item():.3f}"
                        })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Val Acc: {avg_acc:.3f}")
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', avg_acc, epoch)
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, loss, is_best=False):
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
            logger.info(f"Saved best checkpoint at epoch {epoch+1}")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.cfg.logging.save_interval == 0:
            periodic_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
    
    def train(self):
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.cfg.training.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log epoch summary
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
        
        logger.info("Training completed!")
        self.writer.close()


@hydra.main(version_base=None, config_path=".", config_name="act_config")
def main(cfg: DictConfig):

    if not os.path.exists(cfg.data.h5_file_path):
        print(f"ERROR: Data file not found at {cfg.data.h5_file_path}")
        print("Please make sure you have recorded human demonstration data first.")
        print("Run: python human_player.py")
        return
    
    # Create trainer and start training
    trainer = ACTTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
