#!/usr/bin/env python3
"""
ACT (Action Chunking with Transformers)
based on https://github.com/tonyzhaozh/act
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import os
import math
import copy
from typing import Optional, List
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared data loader
from .shared_data_loader import (
    UniversalDataset, 
    DataConfig, 
    ChunkGenerator, 
    get_clips_metadata,
    create_data_config_from_cfg
)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     query_pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None,
                    query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, visual_features, proprio_features, latent_features, mask, query_embed, pos_embed):
        """
        Args:
            visual_features: (bs, c, h, w) vision feature
            proprio_features: (bs, hidden_dim) robot state feature
            latent_features: (bs, hidden_dim) latent variable feature
            mask: attention mask
            query_embed: query embedding
            pos_embed: position encoding
        """
        bs, c, h, w = visual_features.shape
        visual_tokens = visual_features.flatten(2).permute(2, 0, 1)  # (h*w, bs, hidden_dim)
        
        visual_pos_original = pos_embed.flatten(2).permute(2, 0, 1)  # (h*w, 1, hidden_dim)
        if visual_pos_original.shape[1] == 1:
            # Expand batch dimension
            visual_pos = visual_pos_original.expand(-1, bs, -1)  # (h*w, bs, hidden_dim)
        else:
            visual_pos = visual_pos_original
        
        # process proprio and latent features
        # add proprio and latent features as special tokens
        proprio_token = proprio_features.unsqueeze(0)  # (1, bs, hidden_dim)
        latent_token = latent_features.unsqueeze(0)    # (1, bs, hidden_dim)
        
        # create special position encoding for proprio and latent tokens
        special_pos = torch.zeros(2, bs, self.d_model, device=visual_features.device)
        
        # merge all tokens: [latent, proprio, visual_patches]
        src = torch.cat([latent_token, proprio_token, visual_tokens], dim=0)  # (2+h*w, bs, hidden_dim)
        pos_embed_full = torch.cat([special_pos, visual_pos], dim=0)  # (2+h*w, bs, hidden_dim)
        
        # prepare query embedding
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (num_queries, bs, hidden_dim)
        
        # Transformer encoding-decoding
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed_full)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed_full, query_pos=query_embed)
        hs = hs.transpose(1, 2)  # (bs, num_queries, hidden_dim)
        return hs


def build_transformer(args):
    return Transformer(
        d_model=args['hidden_dim'],
        dropout=args['dropout'],
        nhead=args['nheads'],
        dim_feedforward=args['dim_feedforward'],
        num_encoder_layers=args['enc_layers'],
        num_decoder_layers=args['dec_layers'],
        normalize_before=args['pre_norm'],
        return_intermediate_dec=True,
    )


def build_encoder(args):
    """Build CVAE"""
    d_model = args['hidden_dim']
    dropout = args['dropout']
    nhead = args['nheads']
    dim_feedforward = args['dim_feedforward']
    num_encoder_layers = args['enc_layers']
    normalize_before = args['pre_norm']
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        batch_size, _, h, w = x.shape
        
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        pos = pos.expand(batch_size, -1, -1, -1)
        return pos


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # only train the last layers
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        if return_interm_layers:
            from torchvision.models._utils import IntermediateLayerGetter
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            from torchvision.models._utils import IntermediateLayerGetter
            return_layers = {'layer4': "0"}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        import torchvision
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor):
        xs = self[0](tensor)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def build_position_encoding(args):
    N_steps = args['hidden_dim'] // 2
    if args.get('position_embedding', 'sine') in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.get('position_embedding', 'sine')}")
    return position_embedding


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.get('lr_backbone', 0) > 0
    return_interm_layers = args.get('masks', False)
    backbone_name = args.get('backbone', 'resnet18')
    dilation = args.get('dilation', False)
    
    backbone = Backbone(backbone_name, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class DETRVAE(nn.Module):
    """The DETR-VAE as used in the official ACT implementation."""
    
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries, camera_names):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.backbones = nn.ModuleList(backbones)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # CVAE
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)  # CLS token
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
        
        # [CLS], qpos, action_sequence
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Args:
            qpos: (batch, state_dim) robot state
            image: (batch, num_cam, channel, height, width) image input
            actions: (batch, seq, action_dim) action sequence (during training)
            is_pad: (batch, seq) padding mask (during training)
        """
        is_training = actions is not None
        bs, _ = qpos.shape

        # === CVAE ===
        if is_training:
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)        # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight                 # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            
            # combine input for encoder: [CLS, qpos, action_sequence]
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+2, bs, hidden_dim)
            
            # prepare filling mask
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2)
            
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # Take CLS token output
            
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # === decoder ===
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id])
            features = features[0]  # Take last layer features
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        
        # merge all camera features (if multiple cameras, concatenate along width dimension)
        visual_features = torch.cat(all_cam_features, axis=3)  # (bs, hidden_dim, h, w*num_cams)
        visual_pos = torch.cat(all_cam_pos, axis=3)            # (bs, hidden_dim, h, w*num_cams)
        
        proprio_features = self.input_proj_robot_state(qpos)  # (bs, hidden_dim)
        
        hs = self.transformer(
            visual_features=visual_features,
            proprio_features=proprio_features,
            latent_features=latent_input,
            mask=None,
            query_embed=self.query_embed.weight,
            pos_embed=visual_pos
        )[0]  # (bs, num_queries, hidden_dim)
        
        a_hat = self.action_head(hs)      # (bs, num_queries, action_dim)
        is_pad_hat = self.is_pad_head(hs) # (bs, num_queries, 1)
        
        return a_hat, is_pad_hat, [mu, logvar]


def build_ACT_model(args):
    state_dim = args['state_dim']
    action_dim = args['action_dim']
    
    backbone = build_backbone(args)
    backbones = [backbone]

    transformer = build_transformer(args)
    
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        num_queries=args['num_queries'],
        camera_names=args['image_keys'],  # Use image_keys from data_config
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params/1e6:.2f}M")

    return model


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTPolicy(nn.Module):
    def __init__(self, args_override, data_config):
        super().__init__()
        self.model = build_ACT_model(args_override)
        self.kl_weight = args_override['kl_weight']
        self.data_config = data_config
        logger.info(f'KL weight: {self.kl_weight}')

    def __call__(self, batch):
        """
        Forward pass with simplified tensor format
        Input batch format:
        - sensor_obs: (batch, n_obs_steps, sensor_dim)
        - image_obs: (batch, n_obs_steps, cameras, 3, H, W)
        - action: action sequence  
        - is_pad: padding mask
        """
        # Extract tensors from simplified format
        sensor_obs = batch['sensor_obs']
        image_obs = batch['image_obs']
        actions = batch['action']
        is_pad = batch['is_pad']
        
        # For ACT: use only the latest observation (index -1)
        # Extract sensor data (qpos) - take the latest sensor observation
        qpos = sensor_obs[:, -1, :]  # (batch, sensor_dim)
        
        # Extract image data - use the latest image from first camera (index 0)
        # image_obs shape: (batch, n_obs_steps, cameras, 3, H, W)
        image = image_obs[:, -1, :, :, :, :]  # (batch, cameras, 3, H, W)
        
        # ImageNet normalization
        if image is not None:
            normalize = lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
            image = normalize(image)
        
        if actions is not None:  # training
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference
            a_hat, _, (_, _) = self.model(qpos, image)
            return a_hat

    def configure_optimizers(self, lr=1e-5, weight_decay=1e-4):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


class ACTTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if cfg.device == "auto" and torch.cuda.is_available() 
            else cfg.device if cfg.device != "auto" 
            else "cpu"
        )

        # Create data config to get actual dimensions
        from .shared_data_loader import create_data_config_from_cfg
        self.data_config = create_data_config_from_cfg(cfg)
        
        num_queries = cfg.model.horizon - cfg.model.n_obs_steps + 1
        
        args_override = {
            'state_dim': self.data_config.get_lowdim_dimension(),  # Get from data_config instead of cfg
            'action_dim': cfg.model.action_dim,
            'num_queries': num_queries,
            'hidden_dim': cfg.model.hidden_dim,
            'dropout': cfg.model.dropout,
            'nheads': cfg.model.nheads,
            'dim_feedforward': cfg.model.dim_feedforward,
            'enc_layers': cfg.model.enc_layers,
            'dec_layers': cfg.model.dec_layers,
            'pre_norm': cfg.model.pre_norm,
            'kl_weight': cfg.model.kl_weight,
            'image_keys': self.data_config.image_keys,
        }
        
        self.policy = ACTPolicy(args_override, self.data_config).to(self.device)
        
        if cfg.training.optimizer.type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=cfg.training.optimizer.lr,
                weight_decay=cfg.training.optimizer.weight_decay,
                betas=cfg.training.optimizer.betas,
                eps=cfg.training.optimizer.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {cfg.training.optimizer.type}")

        # Learning rate scheduler - created later
        self.scheduler_config = cfg.training.scheduler
        self.scheduler = None

        # TensorBoard settings
        if cfg.logging.tensorboard:
            tensorboard_dir = os.path.join(cfg.output_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            self.global_step = 0
            logger.info(f"TensorBoard log directory: {tensorboard_dir}")
        else:
            self.writer = None
            self.global_step = 0

    def find_latest_checkpoint(self, save_dir):
        if not os.path.exists(save_dir):
            return None
            
        checkpoint_pattern = os.path.join(save_dir, "*.ckpt")
        import glob
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
            
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint

    def should_resume_training(self, save_dir):
        if self.cfg.training.get('resume_from_checkpoint'):
            checkpoint_path = self.cfg.training.resume_from_checkpoint
            if os.path.exists(checkpoint_path):
                return checkpoint_path
            else:
                logger.warning(f"Specified checkpoint file does not exist: {checkpoint_path}")
                return None

        # Check for auto resume
        if self.cfg.training.get('auto_resume', False):
            return self.find_latest_checkpoint(save_dir)
            
        return None

    def _create_scheduler(self, steps_per_epoch, current_epoch=0):
        if self.scheduler_config.type.lower() == 'cosine':
            from torch.optim.lr_scheduler import LambdaLR
            
            total_steps = self.cfg.training.num_epochs * steps_per_epoch
            warmup_steps = self.scheduler_config.get('warmup_steps', 0)
            min_lr_ratio = self.scheduler_config.min_lr / self.cfg.training.optimizer.lr

            logger.info(f"Creating cosine annealing learning rate scheduler:")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Total training steps: {total_steps}")
            logger.info(f"  Warmup steps: {warmup_steps}")
            logger.info(f"  Minimum learning rate ratio: {min_lr_ratio}")
            logger.info(f"  Current epoch: {current_epoch}")
            
            def lr_lambda(step):
                actual_step = step + current_epoch * steps_per_epoch
                
                if actual_step < warmup_steps:
                    return 0.1 + 0.9 * actual_step / warmup_steps
                elif actual_step < total_steps:
                    progress = (actual_step - warmup_steps) / (total_steps - warmup_steps)
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    return min_lr_ratio + (1 - min_lr_ratio) * cosine_factor
                else:
                    # Keep minimum learning rate after training
                    return min_lr_ratio
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            
        elif self.scheduler_config.type.lower() == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                self.optimizer,
                step_size=5000,
                gamma=0.9
            )
        else:
            self.scheduler = None

    def train_epoch(self, train_loader):
        self.policy.train()
        total_loss = 0.0
        total_l1_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch tensors to device
            sensor_obs = batch['sensor_obs'].to(self.device)
            image_obs = batch['image_obs'].to(self.device)
            action = batch['action'].to(self.device)
            is_pad = batch['is_pad'].to(self.device)
            
            # Create batch dict for model
            batch_dict = {
                'sensor_obs': sensor_obs,
                'image_obs': image_obs,
                'action': action,
                'is_pad': is_pad
            }
            
            loss_dict = self.policy(batch_dict)
            
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            
            total_norm = 0
            param_count = 0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            total_norm = total_norm ** (1. / 2)
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.training.grad_clip_norm)
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            if batch_idx % self.cfg.training.gpu_memory_clear_interval == 0:
                torch.cuda.empty_cache()
            
            total_loss += loss_dict['loss'].item()
            total_l1_loss += loss_dict['l1'].item()
            total_kl_loss += loss_dict['kl'].item()
            num_batches += 1
            
            if self.writer and batch_idx % self.cfg.training.log_interval == 0:
                self.writer.add_scalar('Train/Loss_Step', loss_dict['loss'].item(), self.global_step)
                self.writer.add_scalar('Train/L1_Loss_Step', loss_dict['l1'].item(), self.global_step)
                self.writer.add_scalar('Train/KL_Loss_Step', loss_dict['kl'].item(), self.global_step)
                self.writer.add_scalar('Train/Gradient_Norm', total_norm, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss_dict["loss"].item():.4f}',
                'l1': f'{loss_dict["l1"].item():.4f}',
                'kl': f'{loss_dict["kl"].item():.4f}',
                'grad_norm': f'{total_norm:.3f}'
            })
        
        return (total_loss / num_batches, 
                total_l1_loss / num_batches,
                total_kl_loss / num_batches)

    def validate(self, val_loader):
        self.policy.eval()
        total_loss = 0.0
        total_l1_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Move batch tensors to device
                sensor_obs = batch['sensor_obs'].to(self.device)
                image_obs = batch['image_obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_pad = batch['is_pad'].to(self.device)
                
                # Create batch dict for model
                batch_dict = {
                    'sensor_obs': sensor_obs,
                    'image_obs': image_obs,
                    'action': action,
                    'is_pad': is_pad
                }
                
                loss_dict = self.policy(batch_dict)
                
                total_loss += loss_dict['loss'].item()
                total_l1_loss += loss_dict['l1'].item()
                total_kl_loss += loss_dict['kl'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_l1_loss, avg_kl_loss

    def train(self, save_dir="models"):
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Record absolute path of save directory
        save_dir = os.path.abspath(save_dir)
        logger.info(f"Model will be saved to: {save_dir}")

        start_epoch = 0
        best_val_loss = float('inf')
        
        checkpoint_path = self.should_resume_training(save_dir)
        if checkpoint_path:
            try:
                training_state = self.load_checkpoint(checkpoint_path)
                start_epoch = training_state['epoch'] + 1 # train from next epoch
                best_val_loss = training_state['best_val_loss']
                logger.info(f"Resuming training from checkpoint: epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")
                start_epoch = 0
                best_val_loss = float('inf')
        else:
            logger.info("Starting new training")

        clips_metadata = get_clips_metadata(self.cfg.data_dir)
        
        num_clips = len(clips_metadata)
        train_ratio = 1.0 - self.cfg.training.val_split
        num_train_clips = int(num_clips * train_ratio)
        
        import random
        random.seed(42)
        shuffled_clips = clips_metadata.copy()
        random.shuffle(shuffled_clips)
        
        train_clips_meta = shuffled_clips[:num_train_clips]
        val_clips_meta = shuffled_clips[num_train_clips:]
        
        logger.info(f"Training clips: {len(train_clips_meta)}")
        logger.info(f"Validation clips: {len(val_clips_meta)}")
        
        # generate training chunks with random offset and local randomization (no shuffle for determinism)
        val_chunks = ChunkGenerator.generate_chunks_for_epoch(
            val_clips_meta, 
            self.cfg.training.chunk_size,
            max_offset=0,
            clip_group_size=1,
            chunk_stride=self.cfg.training.chunk_stride,
            n_obs_steps=1  # ACT only needs current observation
        )
        
        val_dataset = UniversalDataset(
            val_chunks, 
            data_config=create_data_config_from_cfg(self.cfg),
            chunk_size=self.cfg.training.chunk_size,
            cache_size=self.cfg.training.cache_size,
            lookahead_chunks=self.cfg.training.lookahead_chunks,
            cache_management_interval=self.cfg.training.cache_management_interval
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.cfg.training.batch_size,
            shuffle=False, # no shuffle, generate_chunks_for_epoch already settled the order
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            persistent_workers=self.cfg.training.num_workers > 0
        )
        
        logger.info(f"validation chunks: {len(val_chunks)}")
        
        if self.writer:
            config_text = OmegaConf.to_yaml(self.cfg)
            self.writer.add_text('Config/Parameters', config_text)
        
        for epoch in range(start_epoch, self.cfg.training.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.cfg.training.num_epochs}")

            # regenerate training chunks with random offset and local randomization (no shuffle for determinism)
            train_chunks = ChunkGenerator.generate_chunks_for_epoch(
                train_clips_meta, 
                self.cfg.training.chunk_size,
                max_offset=self.cfg.training.chunk_offset,
                clip_group_size=self.cfg.training.clip_group_size,
                chunk_stride=self.cfg.training.chunk_stride,
                n_obs_steps=1  # ACT only needs current observation
            )
            
            train_dataset = UniversalDataset(
                train_chunks, 
                data_config=create_data_config_from_cfg(self.cfg),
                chunk_size=self.cfg.training.chunk_size,
                cache_size=self.cfg.training.cache_size,
                lookahead_chunks=self.cfg.training.lookahead_chunks,
                cache_management_interval=self.cfg.training.cache_management_interval
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.cfg.training.batch_size,
                shuffle=False, # no shuffle, generate_chunks_for_epoch already settled the order
                num_workers=self.cfg.training.num_workers,
                pin_memory=self.cfg.training.pin_memory,
                prefetch_factor=self.cfg.training.prefetch_factor if self.cfg.training.num_workers > 0 else None,
                persistent_workers=self.cfg.training.num_workers > 0
            )
            
            logger.info(f"training chunks: {len(train_chunks)}")
            
            if epoch == 0 or (start_epoch > 0 and epoch == start_epoch):
                self._create_scheduler(len(train_loader), current_epoch=epoch)
            
            train_loss, train_l1, train_kl = self.train_epoch(train_loader)
            
            val_loss, val_l1, val_kl = self.validate(val_loader)
            
            if self.writer:
                self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                self.writer.add_scalar('Epoch/Train_L1_Loss', train_l1, epoch)
                self.writer.add_scalar('Epoch/Train_KL_Loss', train_kl, epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Val_L1_Loss', val_l1, epoch)
                self.writer.add_scalar('Epoch/Val_KL_Loss', val_kl, epoch)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
                
                self.writer.add_scalar('Epoch/Train_Chunks', len(train_chunks), epoch)
                self.writer.add_scalar('Epoch/Val_Chunks', len(val_chunks), epoch)
                
                if train_loss > 0:
                    l1_ratio = train_l1 / train_loss
                    kl_ratio = train_kl / train_loss
                    self.writer.add_scalar('Epoch/L1_Loss_Ratio', l1_ratio, epoch)
                    self.writer.add_scalar('Epoch/KL_Loss_Ratio', kl_ratio, epoch)
        
            
            logger.info(f"Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, KL: {train_kl:.4f})")
            logger.info(f"Val Loss: {val_loss:.4f} (L1: {val_l1:.4f}, KL: {val_kl:.4f})")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(save_dir, "best_act_model.ckpt"),
                    epoch=epoch,
                    best_val_loss=best_val_loss
                )
                logger.info("Save best model checkpoint")
                if self.writer:
                    self.writer.add_scalar('Epoch/Best_Val_Loss', best_val_loss, epoch)
            
            if (epoch + 1) % self.cfg.training.save_interval == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"act_model_epoch_{epoch+1}.ckpt"),
                    epoch=epoch,
                    best_val_loss=best_val_loss
                )
            
            self.save_checkpoint(
                os.path.join(save_dir, "latest_checkpoint.ckpt"),
                epoch=epoch,
                best_val_loss=best_val_loss
            )
        
        if self.writer:
            self.writer.close()

        logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved in: {save_dir}")
        if self.writer:
            tensorboard_dir = os.path.join(self.cfg.output_dir, "tensorboard")
            logger.info(f"TensorBoard logs saved in: {tensorboard_dir}")
            logger.info(f"Use the following command to view:")
            logger.info(f"tensorboard --logdir={tensorboard_dir}")

    def save_checkpoint(self, filepath, epoch=None, best_val_loss=None):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'global_step': self.global_step,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        temp_filepath = filepath + '.tmp'
        torch.save(checkpoint, temp_filepath)
        os.rename(temp_filepath, filepath)

        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        logger.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.global_step = checkpoint.get('global_step', 0)
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'global_step': self.global_step
        }

class ACTAgent:
    def __init__(self, cfg: DictConfig, model_path: str = None, device=None):
        if cfg is None:
            raise ValueError("cfg is required to initialize ACTAgent")
            
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if cfg.device == "auto" and torch.cuda.is_available() 
            else cfg.device if cfg.device != "auto" 
            else "cpu"
        ) if device is None else device
        
        self.model_chunk_size = cfg.training.chunk_size
        inference_chunk_size = getattr(cfg.inference, 'action_chunk_size', None)
        if inference_chunk_size is not None and inference_chunk_size > 0:
            self.chunk_size = inference_chunk_size
            logger.info(f"using custom inference action chunk size: {self.chunk_size}")
        else:
            self.chunk_size = self.model_chunk_size
            logger.info(f"using model default action chunk size: {self.chunk_size}")

        # Create data config for format conversion
        from .shared_data_loader import create_data_config_from_cfg
        self.data_config = create_data_config_from_cfg(cfg)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'config' in checkpoint:
            model_cfg = OmegaConf.create(checkpoint['config'])
            num_queries = model_cfg.model.horizon - model_cfg.model.n_obs_steps + 1
            args_override = {
                'state_dim': model_cfg.model.lowdim_dim,
                'action_dim': model_cfg.model.action_dim,
                'num_queries': num_queries,
                'hidden_dim': model_cfg.model.hidden_dim,
                'dropout': model_cfg.model.dropout,
                'nheads': model_cfg.model.nheads,
                'dim_feedforward': model_cfg.model.dim_feedforward,
                'enc_layers': model_cfg.model.enc_layers,
                'dec_layers': model_cfg.model.dec_layers,
                'pre_norm': model_cfg.model.pre_norm,
                'kl_weight': model_cfg.model.kl_weight,
                'image_keys': self.data_config.image_keys,
            }
        
        self.policy = ACTPolicy(args_override, self.data_config).to(self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        
        self.action_buffer = None
        self.buffer_idx = 0

    def predict_action(self, observation):
        """
        Predict action given tensor observation data
        
        Args:
            observation: dict with tensor observations
                - 'sensor_obs': (n_obs_steps, sensor_dim) tensor
                - 'image_obs': (n_obs_steps, cameras, 3, H, W) tensor
        
        Returns:
            action: numpy array of shape (action_dim,)
        """
        # Check if we need to generate new action chunk
        if self.action_buffer is None or self.buffer_idx >= self.chunk_size:
            # Convert single observation to batch format for model inference
            # observation should contain 'sensor_obs' and 'image_obs' tensors
            batch = {
                'sensor_obs': None,
                'image_obs': None,
                'action': None,  # No action needed for inference
                'is_pad': None
            }
            
            # Extract sensor and image data from observation
            if 'sensor_obs' in observation:
                sensor_obs = observation['sensor_obs']
                if isinstance(sensor_obs, np.ndarray):
                    tensor_value = torch.FloatTensor(sensor_obs).to(self.device)
                elif isinstance(sensor_obs, torch.Tensor):
                    tensor_value = sensor_obs.to(self.device)
                else:
                    raise ValueError(f"Unsupported sensor_obs type: {type(sensor_obs)}")
                batch['sensor_obs'] = tensor_value.unsqueeze(0)  # Add batch dim
            else:
                raise ValueError("observation must contain 'sensor_obs'")
            
            if 'image_obs' in observation:
                image_obs = observation['image_obs']
                if isinstance(image_obs, np.ndarray):
                    tensor_value = torch.FloatTensor(image_obs).to(self.device)
                elif isinstance(image_obs, torch.Tensor):
                    tensor_value = image_obs.to(self.device)
                else:
                    raise ValueError(f"Unsupported image_obs type: {type(image_obs)}")
                batch['image_obs'] = tensor_value.unsqueeze(0)  # Add batch dim
            else:
                raise ValueError("observation must contain 'image_obs'")
            
            with torch.no_grad():
                full_action_chunk = self.policy(batch)  # (1, model_chunk_size, action_dim)
            
            if self.chunk_size != self.model_chunk_size:
                if self.chunk_size < self.model_chunk_size:
                    action_chunk = full_action_chunk[:, :self.chunk_size, :]
                else:
                    raise ValueError("Inference chunk size cannot be larger than model chunk size")
            else:
                action_chunk = full_action_chunk
            
            self.action_buffer = action_chunk.squeeze(0).cpu().numpy()
            self.buffer_idx = 0
        
        action = self.action_buffer[self.buffer_idx]
        self.buffer_idx += 1
        
        return action

    def reset(self):
        self.action_buffer = None
        self.buffer_idx = 0
