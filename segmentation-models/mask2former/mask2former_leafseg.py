"""
Mask2Former — Leaf Segmentation
================================
Full Mask2Former architecture for "SEMANTIC SEGMENTATION" task
on leaf-seg dataset (images + ground-truth masks).

Architecture:
    Image -> Swin-L Backbone -> Pixel Decoder (FPN)
    -> Transformer Decoder (Masked Cross-Attention)
    -> Per-query (class prediction + mask prediction)
    -> Hungarian Matching Loss (CE + BCE + Dice)

Metrics:
    mIoU, Dice Coefficient, Pixel Accuracy (per-class & mean)

Usage:
python mask2former_leafseg_with_swin_backbone.py \
  --data_dir "/mnt/d/NCKH/Run_Models/version_4.0.0/dataset/leaf-seg" \
  --output_dir "./output_mask2former" \
  --epochs 50 \
  --batch_size 2 \
  --img_size 384 \
  --num_workers 2 \
  --lr 1e-4 \
  --backbone_lr 1e-5 \
  --hidden_dim 256 \
  --num_queries 50 \
  --num_decoder_layers 4

Requirements:
    pip install torch torchvision timm numpy pillow scikit-learn matplotlib seaborn tqdm opencv-python scipy
"""

import os
import sys
import csv
import math
import json
import time
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None

from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

warnings.filterwarnings('ignore')


# ============================================================
# 1. JOINT TRANSFORMS  (image + mask đồng bộ)
# ============================================================

class JointResize:
    """Resize cả image và mask về cùng kích thước."""
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        image = image.resize(self.size, Image.BILINEAR)
        mask  = mask.resize(self.size, Image.NEAREST)
        return image, mask


class JointRandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class JointRandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask  = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask


class JointRandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, Image.BILINEAR, fillcolor=0)
        mask  = mask.rotate(angle, Image.NEAREST,  fillcolor=0)
        return image, mask


class JointColorJitter:
    """Chỉ áp dụng color jitter cho image, mask giữ nguyên."""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask


class JointToTensor:
    """Chuyển image+mask sang tensor; normalize image; map mask values."""
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, value_mapping=None):
        self.normalize = transforms.Normalize(self.MEAN, self.STD)
        self.value_mapping = value_mapping

    def __call__(self, image, mask):
        image = transforms.ToTensor()(image)
        image = self.normalize(image)

        mask_np = np.array(mask, dtype=np.int64)
        if self.value_mapping:
            mapped = np.zeros_like(mask_np)
            for old_v, new_v in self.value_mapping.items():
                mapped[mask_np == old_v] = new_v
            mask_np = mapped

        mask_tensor = torch.from_numpy(mask_np).long()
        return image, mask_tensor


class JointCompose:
    """Chain nhiều joint transforms."""
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, image, mask):
        for t in self.tfms:
            image, mask = t(image, mask)
        return image, mask


# ============================================================
# 2. DATASET
# ============================================================

class LeafSegDataset(Dataset):
    """Leaf segmentation dataset.

    Structure:
        root_dir/
        ├── images/   (*.jpg)
        ├── masks/    (*.png)
        └── train.csv (imageid,maskid)
    """

    def __init__(self, root_dir, joint_transform=None,
                 indices=None, value_mapping=None, verbose=True):
        self.root_dir   = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir  = os.path.join(root_dir, 'masks')
        self.csv_path   = os.path.join(root_dir, 'train.csv')
        self.joint_transform = joint_transform

        # ---------- Read CSV ----------
        self.pairs = []
        with open(self.csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
            except csv.Error:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self.pairs.append((row[0].strip(), row[1].strip()))

        # Sub-select by indices
        if indices is not None:
            self.pairs = [self.pairs[i] for i in indices]

        # ---------- Auto-detect classes from mask values ----------
        if value_mapping is None:
            self.value_mapping, self.num_classes, self.class_names = \
                self._detect_classes()
        else:
            self.value_mapping = value_mapping
            self.num_classes = len(set(value_mapping.values()))
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]

        if verbose:
            print(f"  Dataset: {len(self.pairs)} images, "
                  f"{self.num_classes} classes  "
                  f"{list(self.value_mapping.items())[:6]}")

    def _detect_classes(self):
        """Scan a sample of masks to discover unique pixel values."""
        unique_vals = set()
        n_sample = min(100, len(self.pairs))
        for i in range(n_sample):
            mpath = os.path.join(self.masks_dir, self.pairs[i][1])
            if not os.path.exists(mpath):
                continue
            m = np.array(Image.open(mpath).convert('L'))
            if m.size == 0:
                continue
            unique_vals.update(np.unique(m).tolist())

        sorted_vals = sorted(unique_vals)

        if not sorted_vals:
            raise RuntimeError(
                'Could not detect any class values from masks. '
                'Please verify train.csv paths and mask files.'
            )

        # If values are like {0, 255} → map to {0, 1}
        if max(sorted_vals) >= len(sorted_vals):
            mapping = {v: i for i, v in enumerate(sorted_vals)}
        else:
            mapping = {v: v for v in sorted_vals}

        num_classes = len(sorted_vals)
        class_names = [f'class_{mapping[v]}' for v in sorted_vals]

        return mapping, num_classes, class_names

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img_path  = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask_np = np.array(mask, dtype=np.int64)
            if hasattr(self, 'value_mapping') and self.value_mapping:
                mapped = np.zeros_like(mask_np)
                for old_v, new_v in self.value_mapping.items():
                    mapped[mask_np == old_v] = new_v
                mask_np = mapped
            mask = torch.from_numpy(mask_np).long()

        return image, mask


# ============================================================
# 3. POSITIONAL EMBEDDING
# ============================================================

class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal positional embedding (DETR / Mask2Former)."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        B, C, H, W = x.shape

        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=x.device)

        if self.normalize:
            y_embed = y_embed / H * self.scale
            x_embed = x_embed / W * self.scale

        y_embed = y_embed.unsqueeze(1).repeat(1, W)
        x_embed = x_embed.unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, 0::2].sin(),
                             pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(),
                             pos_y[:, :, 1::2].cos()], dim=3).flatten(2)

        pos = torch.cat([pos_y, pos_x], dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


# ============================================================
# 4. BACKBONE — Swin-L
# ============================================================

class Mask2FormerBackbone(nn.Module):
    """Multi-scale feature extraction: C2 (stride 4) … C5 (stride 32)."""

    FEATURE_CHANNELS = [192, 384, 768, 1536]

    def __init__(self, pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError(
                "Swin-L backbone requires timm. Please install: pip install timm"
            )

        self.swin = timm.create_model(
            'swin_large_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

    def forward(self, x):
        feats = self.swin(x)
        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 feature maps from Swin-L, got {len(feats)}")

        norm_feats = []
        for i, feat in enumerate(feats):
            if feat.ndim != 4:
                raise RuntimeError(f"Feature map at level {i} must be 4D, got {feat.ndim}D")

            expected_c = self.FEATURE_CHANNELS[i]
            if feat.shape[1] == expected_c:
                norm_feats.append(feat)
            elif feat.shape[-1] == expected_c:
                norm_feats.append(feat.permute(0, 3, 1, 2).contiguous())
            else:
                raise RuntimeError(
                    f"Unexpected Swin-L feature shape at level {i}: {tuple(feat.shape)}"
                )

        c2, c3, c4, c5 = norm_feats
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


# ============================================================
# 5. PIXEL DECODER — FPN
# ============================================================

class PixelDecoder(nn.Module):
    """FPN pixel decoder.

    Returns:
        mask_features        : (B, D, H/4, W/4)
        multi_scale_features : [P5, P4, P3]
    """

    def __init__(self, feature_channels=None, hidden_dim=256):
        super().__init__()
        if feature_channels is None:
            feature_channels = [256, 512, 1024, 2048]

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, hidden_dim, 1) for ch in feature_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            ) for _ in feature_channels
        ])
        self.mask_proj = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        c2, c3, c4, c5 = (features["c2"], features["c3"],
                           features["c4"], features["c5"])
        feat_list = [c2, c3, c4, c5]

        laterals = [self.lateral_convs[i](feat_list[i]) for i in range(4)]

        for i in range(2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=False)

        outputs = [self.output_convs[i](laterals[i]) for i in range(4)]

        mask_features = self.mask_proj(outputs[0])
        multi_scale_features = [outputs[3], outputs[2], outputs[1]]  # P5, P4, P3

        return mask_features, multi_scale_features


# ============================================================
# 6. TRANSFORMER DECODER LAYER (Masked Cross-Attention)
# ============================================================

class Mask2FormerDecoderLayer(nn.Module):
    """Masked Cross-Attn -> Self-Attn -> FFN  (all pre-norm)."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.nhead = nhead

        self.cross_attn    = nn.MultiheadAttention(d_model, nhead,
                                                    dropout=dropout, batch_first=False)
        self.norm_cross    = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        self.self_attn     = nn.MultiheadAttention(d_model, nhead,
                                                    dropout=dropout, batch_first=False)
        self.norm_self     = nn.LayerNorm(d_model)
        self.dropout_self  = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, queries, features, query_pos, pos_embed, attn_mask=None):
        # 1. Masked Cross-Attention
        q2 = self.norm_cross(queries)
        cross_out, cross_w = self.cross_attn(
            query=q2 + query_pos, key=features + pos_embed,
            value=features, attn_mask=attn_mask)
        queries = queries + self.dropout_cross(cross_out)

        # 2. Self-Attention
        q2 = self.norm_self(queries)
        self_out = self.self_attn(
            query=q2 + query_pos, key=q2 + query_pos, value=q2)[0]
        queries = queries + self.dropout_self(self_out)

        # 3. FFN
        q2 = self.norm_ffn(queries)
        queries = queries + self.ffn(q2)

        return queries, cross_w


# ============================================================
# 7. TRANSFORMER DECODER (Multi-scale, Masked Attention)
# ============================================================

class Mask2FormerTransformerDecoder(nn.Module):
    """Returns per-layer queries & mask predictions (for auxiliary loss)."""

    NUM_FEATURE_LEVELS = 3

    def __init__(self, hidden_dim=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, num_queries=100, dropout=0.0):
        super().__init__()
        self.num_layers  = num_layers
        self.num_queries = num_queries
        self.hidden_dim  = hidden_dim
        self.nhead       = nhead

        self.layers = nn.ModuleList([
            Mask2FormerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat  = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(self.NUM_FEATURE_LEVELS, hidden_dim)
        self.pos_enc     = PositionEmbeddingSine(hidden_dim // 2)

        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
            ) for _ in range(self.NUM_FEATURE_LEVELS)
        ])

        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, multi_scale_features, mask_features):
        """
        Returns:
            all_queries    : list[num_layers] of (B, Q, C)
            all_mask_preds : list[num_layers] of (B, Q, H_mask, W_mask)
        """
        B = mask_features.shape[0]

        # Prepare multi-scale sources
        src_list, pos_list, size_list = [], [], []
        for i, feat in enumerate(multi_scale_features):
            src = self.input_projs[i](feat)
            pos = self.pos_enc(src)
            size_list.append(src.shape[-2:])
            src = src.flatten(2).permute(2, 0, 1)
            pos = pos.flatten(2).permute(2, 0, 1)
            src = src + self.level_embed.weight[i].view(1, 1, -1)
            src_list.append(src)
            pos_list.append(pos)

        # Init queries
        queries   = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        all_queries    = []
        all_mask_preds = []

        for layer_idx in range(self.num_layers):
            level_idx = layer_idx % self.NUM_FEATURE_LEVELS
            src = src_list[level_idx]
            pos = pos_list[level_idx]
            spatial_size = size_list[level_idx]

            # Build attention mask from previous predictions
            if layer_idx == 0:
                attn_mask = None
            else:
                prev_mask = all_mask_preds[-1].detach()
                prev_resized = F.interpolate(
                    prev_mask, size=spatial_size,
                    mode='bilinear', align_corners=False)
                attn_mask_bool = (prev_resized.sigmoid().flatten(2) < 0.5)
                all_masked = attn_mask_bool.all(dim=-1, keepdim=True)
                attn_mask_bool = attn_mask_bool.masked_fill(all_masked, False)
                attn_mask_bool = attn_mask_bool.unsqueeze(1) \
                    .repeat(1, self.nhead, 1, 1).flatten(0, 1)
                attn_mask = torch.zeros_like(attn_mask_bool, dtype=src.dtype)
                attn_mask.masked_fill_(attn_mask_bool, float('-inf'))

            queries, _ = self.layers[layer_idx](
                queries, src, query_pos, pos, attn_mask)

            # Mask prediction via dot product
            q_perm = queries.permute(1, 0, 2)                     # (B, Q, C)
            mask_emb = self.mask_embed(q_perm)                     # (B, Q, C)
            mask_pred = torch.einsum('bqc,bchw->bqhw',
                                     mask_emb, mask_features)      # (B, Q, H, W)

            all_queries.append(q_perm)
            all_mask_preds.append(mask_pred)

        return all_queries, all_mask_preds


# ============================================================
# 8. MASK2FORMER SEGMENTOR
# ============================================================

class Mask2FormerSegmentor(nn.Module):
    """Full Mask2Former for semantic segmentation.

    Per-query output:
        - class logits  (K+1 classes, +1 for 'no object')
        - binary mask   (H/4 x W/4)
    """

    def __init__(self, num_classes, hidden_dim=256, num_queries=100,
                 nheads=8, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim  = hidden_dim

        self.backbone = Mask2FormerBackbone(pretrained=pretrained_backbone)
        self.pixel_decoder = PixelDecoder(
            Mask2FormerBackbone.FEATURE_CHANNELS, hidden_dim)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            hidden_dim, nheads, num_decoder_layers,
            dim_feedforward, num_queries, dropout)

        # Per-query class head:  K semantic classes + 1 "no object"
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x):
        """
        Returns dict:
            pred_logits : (B, Q, K+1)
            pred_masks  : (B, Q, H/4, W/4)
            aux_outputs : list[dict] for intermediate layers
        """
        features = self.backbone(x)
        mask_features, multi_scale = self.pixel_decoder(features)
        all_queries, all_mask_preds = self.transformer_decoder(
            multi_scale, mask_features)

        outputs = {
            'pred_logits': self.class_embed(all_queries[-1]),
            'pred_masks':  all_mask_preds[-1],
        }

        # Auxiliary outputs (intermediate layers)
        outputs['aux_outputs'] = [
            {
                'pred_logits': self.class_embed(all_queries[i]),
                'pred_masks':  all_mask_preds[i],
            }
            for i in range(len(all_queries) - 1)
        ]

        return outputs


# ============================================================
# 9. SEMANTIC INFERENCE
# ============================================================

@torch.no_grad()
def semantic_inference(pred_logits, pred_masks, img_size):
    """Combine per-query predictions into a semantic segmentation map.

    pred_logits : (B, Q, K+1)
    pred_masks  : (B, Q, H_pred, W_pred)

    Returns: (B, H, W) — predicted class per pixel.
    """
    # Upsample masks to original resolution
    pred_masks_up = F.interpolate(
        pred_masks, size=(img_size, img_size),
        mode='bilinear', align_corners=False)

    mask_probs = pred_masks_up.sigmoid()                   # (B, Q, H, W)
    cls_probs  = F.softmax(pred_logits, dim=-1)[..., :-1]  # (B, Q, K)  drop 'no object'

    # semseg[b,k,h,w] = sum_q  P(class=k|q) * sigma(mask_q[h,w])
    semseg = torch.einsum('bqc,bqhw->bchw', cls_probs, mask_probs)

    return semseg.argmax(dim=1)   # (B, H, W)


# ============================================================
# 10. HUNGARIAN MATCHER
# ============================================================

class HungarianMatcher(nn.Module):
    """Bipartite matching between predicted queries and GT segments.

    Cost = lambda_cls * C_cls + lambda_bce * C_bce + lambda_dice * C_dice
    """

    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask  = cost_mask
        self.cost_dice  = cost_dice

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs : dict with 'pred_logits' (B,Q,K+1), 'pred_masks' (B,Q,H,W)
        targets : list[B] of dict { 'labels': (N,), 'masks': (N,H,W) }

        Returns: list[B] of (pred_indices, gt_indices)
        """
        B, Q, _ = outputs['pred_logits'].shape
        indices = []

        for b in range(B):
            pred_logits = outputs['pred_logits'][b]       # (Q, K+1)
            pred_masks  = outputs['pred_masks'][b]        # (Q, H, W)
            gt_labels   = targets[b]['labels']             # (N,)
            gt_masks    = targets[b]['masks']              # (N, H, W)

            N = len(gt_labels)
            if N == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            # ---- Classification cost ----
            pred_probs = pred_logits.softmax(-1)           # (Q, K+1)
            class_cost = -pred_probs[:, gt_labels]         # (Q, N)

            # ---- Resize GT masks to prediction resolution if needed ----
            if gt_masks.shape[-2:] != pred_masks.shape[-2:]:
                gt_masks = F.interpolate(
                    gt_masks.unsqueeze(1).float(),
                    size=pred_masks.shape[-2:],
                    mode='nearest').squeeze(1)

            # ---- Mask BCE cost (matrix multiplication trick) ----
            pred_flat = pred_masks.flatten(1)              # (Q, HW)
            gt_flat   = gt_masks.flatten(1).float()        # (N, HW)
            HW = pred_flat.shape[1]

            out_sigmoid = pred_flat.sigmoid()
            neg_cost = -(1 - out_sigmoid + 1e-8).log()     # (Q, HW)
            pos_cost = -(out_sigmoid + 1e-8).log()         # (Q, HW)

            bce_cost = (torch.matmul(pos_cost, gt_flat.t()) +
                        torch.matmul(neg_cost, (1 - gt_flat).t())) / HW   # (Q, N)

            # ---- Mask Dice cost ----
            numerator   = 2.0 * torch.matmul(out_sigmoid, gt_flat.t())     # (Q, N)
            denominator = (out_sigmoid.sum(-1, keepdim=True) +
                           gt_flat.sum(-1, keepdim=True).t())              # (Q, N)
            dice_cost   = 1.0 - (numerator + 1.0) / (denominator + 1.0)   # (Q, N)

            # ---- Total cost ----
            C = (self.cost_class * class_cost +
                 self.cost_mask  * bce_cost   +
                 self.cost_dice  * dice_cost)

            # Guard against NaN/Inf values to keep Hungarian matching stable
            # (can happen when a forward pass produces extreme logits)
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

            C = C.cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_idx, dtype=torch.long),
                            torch.as_tensor(col_idx, dtype=torch.long)))

        return indices


# ============================================================
# 11. SEGMENTATION CRITERION  (CE + BCE + Dice)
# ============================================================

def dice_loss(pred, target, num_masks):
    """Per-mask Dice loss.
    pred   : (N_matched, HW)  logits (NOT sigmoid)
    target : (N_matched, HW)  binary
    """
    pred = pred.sigmoid()
    numerator   = 2.0 * (pred * target).sum(-1)
    denominator = pred.sum(-1) + target.sum(-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return loss.sum() / max(num_masks, 1)


def sigmoid_ce_loss(pred, target, num_masks):
    """Per-mask sigmoid binary cross-entropy loss."""
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return loss.mean(1).sum() / max(num_masks, 1)


class SetCriterion(nn.Module):
    """Mask2Former loss with Hungarian matching.

    Total loss = lambda_cls * L_cls + lambda_bce * L_bce + lambda_dice * L_dice
    Applied at final + all auxiliary decoder layers.
    """

    def __init__(self, num_classes, matcher, eos_coef=0.1,
                 weight_cls=2.0, weight_bce=5.0, weight_dice=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.weight_cls  = weight_cls
        self.weight_bce  = weight_bce
        self.weight_dice = weight_dice

        # Down-weight "no object" class in CE
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        """Flatten matched indices across batch."""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification CE loss."""
        pred_logits = outputs['pred_logits']         # (B, Q, K+1)
        B, Q, _ = pred_logits.shape

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][J] for t, (_, J) in zip(targets, indices)])

        # Default target = "no object" (last class)
        target_classes = torch.full((B, Q), self.num_classes,
                                     dtype=torch.long,
                                     device=pred_logits.device)
        target_classes[idx] = target_classes_o.to(pred_logits.device)

        loss = F.cross_entropy(pred_logits.transpose(1, 2),
                               target_classes,
                               weight=self.empty_weight)
        return loss

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Mask BCE + Dice loss (matched pairs only)."""
        src_idx = self._get_src_permutation_idx(indices)

        pred_masks = outputs['pred_masks']           # (B, Q, H, W)
        pred_masks = pred_masks[src_idx]              # (N_matched, H, W)

        target_masks = torch.cat([
            t['masks'][J] for t, (_, J) in zip(targets, indices)
        ]).to(pred_masks.device)                      # (N_matched, H_gt, W_gt)

        # Down-sample GT masks to prediction resolution
        if target_masks.shape[-2:] != pred_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode='nearest').squeeze(1)

        pred_flat   = pred_masks.flatten(1)            # (N, HW)
        target_flat = target_masks.flatten(1).float()  # (N, HW)

        l_bce  = sigmoid_ce_loss(pred_flat, target_flat, num_masks)
        l_dice = dice_loss(pred_flat, target_flat, num_masks)

        return l_bce, l_dice

    def forward(self, outputs, targets):
        """Compute total loss (final layer + aux layers)."""

        # Match final layer
        indices = self.matcher(outputs, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = max(num_masks, 1)

        # Final layer losses
        l_cls = self.loss_labels(outputs, targets, indices, num_masks)
        l_bce, l_dice = self.loss_masks(outputs, targets, indices, num_masks)

        total = (self.weight_cls  * l_cls +
                 self.weight_bce  * l_bce +
                 self.weight_dice * l_dice)

        # Auxiliary layer losses
        if 'aux_outputs' in outputs:
            for aux in outputs['aux_outputs']:
                aux_indices = self.matcher(aux, targets)
                l_cls_a = self.loss_labels(aux, targets, aux_indices, num_masks)
                l_bce_a, l_dice_a = self.loss_masks(
                    aux, targets, aux_indices, num_masks)
                total += (self.weight_cls  * l_cls_a +
                          self.weight_bce  * l_bce_a +
                          self.weight_dice * l_dice_a)

        losses = {
            'total':     total,
            'loss_cls':  l_cls.detach(),
            'loss_bce':  l_bce.detach(),
            'loss_dice': l_dice.detach(),
        }
        return losses


# ============================================================
# 12. PREPARE TARGETS
# ============================================================

def prepare_targets(gt_masks, device):
    """Decompose semantic masks into per-class binary masks.

    gt_masks : (B, H, W) long tensor  (pixel values = class IDs)

    Returns: list[B] of dict { 'labels': (N,), 'masks': (N, H, W) }
    """
    targets = []
    for b in range(gt_masks.shape[0]):
        m = gt_masks[b]                                    # (H, W)
        classes = m.unique()                                # e.g. [0, 1]
        labels  = classes.long().to(device)                 # (N,)
        masks   = (m.unsqueeze(0) == classes.view(-1, 1, 1)).float()  # (N, H, W)
        targets.append({'labels': labels, 'masks': masks.to(device)})
    return targets


# ============================================================
# 13. SEGMENTATION METRICS
# ============================================================

def compute_metrics(pred, gt, num_classes):
    """Compute mIoU, mean Dice, Pixel Accuracy  (numpy arrays).

    Returns dict with per-class + mean values.
    """
    assert pred.shape == gt.shape

    per_class_iou  = []
    per_class_dice = []
    valid_classes  = []

    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c   = (gt == c)
        intersection = (pred_c & gt_c).sum()
        union        = (pred_c | gt_c).sum()

        if union == 0:
            continue   # skip classes not present in GT or pred

        valid_classes.append(c)
        iou = intersection / union
        per_class_iou.append(float(iou))

        dice_denom = pred_c.sum() + gt_c.sum()
        dice_val   = 2.0 * intersection / dice_denom if dice_denom > 0 else 0.0
        per_class_dice.append(float(dice_val))

    pixel_acc = float((pred == gt).sum() / pred.size) if pred.size > 0 else 0.0

    return {
        'mIoU':           np.mean(per_class_iou)  if per_class_iou  else 0.0,
        'mean_Dice':      np.mean(per_class_dice) if per_class_dice else 0.0,
        'pixel_accuracy': pixel_acc,
        'per_class_iou':  dict(zip(valid_classes, per_class_iou)),
        'per_class_dice': dict(zip(valid_classes, per_class_dice)),
    }


# ============================================================
# 14. TRAINING
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer,
                    device, epoch, img_size, num_classes, scaler=None):
    model.train()
    running_loss = 0.0
    running_cls  = 0.0
    running_bce  = 0.0
    running_dice = 0.0
    total_pixels = 0
    correct      = 0
    n_batches    = 0

    pbar = tqdm(dataloader, desc=f'  Epoch {epoch:>3d} [Train]', ncols=120)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        targets = prepare_targets(masks, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                losses  = criterion(outputs, targets)
                loss    = losses['total']
            if not torch.isfinite(loss):
                tqdm.write(f"[Warning] Skip batch: non-finite loss = {loss.item()}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            losses  = criterion(outputs, targets)
            loss    = losses['total']
            if not torch.isfinite(loss):
                tqdm.write(f"[Warning] Skip batch: non-finite loss = {loss.item()}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

        # Quick pixel accuracy for monitoring
        with torch.no_grad():
            pred_seg = semantic_inference(
                outputs['pred_logits'].detach(),
                outputs['pred_masks'].detach(), img_size)
            correct      += (pred_seg == masks).sum().item()
            total_pixels += masks.numel()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_cls  += losses['loss_cls'].item()  * bs
        running_bce  += losses['loss_bce'].item()  * bs
        running_dice += losses['loss_dice'].item() * bs
        n_batches    += bs

        pbar.set_postfix(
            loss=f'{loss.item():.4f}',
            pAcc=f'{100.0 * correct / total_pixels:.1f}%')

    return {
        'loss':      running_loss / n_batches,
        'loss_cls':  running_cls  / n_batches,
        'loss_bce':  running_bce  / n_batches,
        'loss_dice': running_dice / n_batches,
        'pixel_acc': 100.0 * correct / total_pixels,
    }


# ============================================================
# 15. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, criterion, device,
             img_size, num_classes):
    model.eval()
    running_loss = 0.0
    n_batches    = 0

    all_preds  = []
    all_labels = []

    pbar = tqdm(dataloader, desc='  Evaluating', ncols=120)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        targets = prepare_targets(masks, device)
        outputs = model(images)
        losses  = criterion(outputs, targets)

        pred_seg = semantic_inference(
            outputs['pred_logits'], outputs['pred_masks'], img_size)

        all_preds.append(pred_seg.cpu().numpy())
        all_labels.append(masks.cpu().numpy())

        bs = images.size(0)
        running_loss += losses['total'].item() * bs
        n_batches    += bs

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics['loss'] = running_loss / n_batches

    return metrics


# ============================================================
# 16. PLOTTING
# ============================================================

def plot_training_curves(history, save_path):
    """Loss, mIoU, Dice, Pixel Accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', ms=3, lw=2, label='Train')
    axes[0, 0].plot(epochs, history['val_loss'],   'r-o', ms=3, lw=2, label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # mIoU
    axes[0, 1].plot(epochs, history['val_miou'],   'r-o', ms=3, lw=2, label='Val mIoU')
    axes[0, 1].set_title('mIoU (%)')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Dice
    axes[1, 0].plot(epochs, history['val_dice'],   'r-o', ms=3, lw=2, label='Val Dice')
    axes[1, 0].set_title('Mean Dice (%)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # Pixel Accuracy
    axes[1, 1].plot(epochs, history['train_pacc'], 'b-o', ms=3, lw=2, label='Train')
    axes[1, 1].plot(epochs, history['val_pacc'],   'r-o', ms=3, lw=2, label='Val')
    axes[1, 1].set_title('Pixel Accuracy (%)')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Epoch')

    fig.suptitle('Mask2Former — Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Curves saved -> {save_path}")


def plot_sample_predictions(model, dataset, device, img_size,
                            num_classes, save_path, n=6):
    """Visualize sample predictions: Image | GT | Pred."""
    model.eval()
    n = min(n, len(dataset))
    sample_indices = random.sample(range(len(dataset)), n)

    PALETTE = np.array([
        [0,   0,   0],    # class 0 — background (black)
        [255, 50,  50],   # class 1
        [50,  255, 50],   # class 2
        [50,  50,  255],  # class 3
        [255, 255, 50],   # class 4
        [255, 50,  255],  # class 5
        [50,  255, 255],  # class 6
        [255, 140,  0],   # class 7
    ], dtype=np.uint8)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    for row, idx in enumerate(sample_indices):
        img_t, mask_t = dataset[idx]
        inp = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
            pred = semantic_inference(
                out['pred_logits'], out['pred_masks'], img_size)
            pred = pred[0].cpu().numpy()

        # De-normalize image for display
        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * STD + MEAN, 0, 1)

        gt_np  = mask_t.cpu().numpy()

        # Coloured masks
        gt_color   = PALETTE[gt_np   % len(PALETTE)]
        pred_color = PALETTE[pred % len(PALETTE)]

        axes[row, 0].imshow(img_np);          axes[row, 0].set_title('Image')
        axes[row, 1].imshow(gt_color);        axes[row, 1].set_title('GT Mask')
        axes[row, 2].imshow(pred_color);      axes[row, 2].set_title('Prediction')

        for c in range(3):
            axes[row, c].axis('off')

    fig.suptitle('Sample Predictions — Mask2Former',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sample predictions saved -> {save_path}")


# ============================================================
# 17. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mask2Former — Leaf Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = parser.add_argument_group('Data')
    g.add_argument('--data_dir',    type=str, required=True,
                   help='Root of leaf-seg dataset (contains images/, masks/, train.csv)')
    g.add_argument('--output_dir',  type=str, default='./output_mask2former')
    g.add_argument('--img_size',    type=int, default=512)
    g.add_argument('--train_ratio', type=float, default=0.7)
    g.add_argument('--val_ratio',   type=float, default=0.15)

    g = parser.add_argument_group('Model')
    g.add_argument('--hidden_dim',         type=int,   default=256)
    g.add_argument('--num_queries',        type=int,   default=100)
    g.add_argument('--nheads',             type=int,   default=8)
    g.add_argument('--num_decoder_layers', type=int,   default=6)
    g.add_argument('--dim_feedforward',    type=int,   default=1024)
    g.add_argument('--dropout',            type=float, default=0.1)

    g = parser.add_argument_group('Training')
    g.add_argument('--epochs',       type=int,   default=50)
    g.add_argument('--batch_size',   type=int,   default=4)
    g.add_argument('--lr',           type=float, default=1e-4)
    g.add_argument('--backbone_lr',  type=float, default=1e-5)
    g.add_argument('--weight_decay', type=float, default=0.01)
    g.add_argument('--patience',     type=int,   default=10)
    g.add_argument('--num_workers',  type=int,   default=4)
    g.add_argument('--seed',         type=int,   default=42)

    g = parser.add_argument_group('Loss')
    g.add_argument('--weight_cls',  type=float, default=2.0)
    g.add_argument('--weight_bce',  type=float, default=5.0)
    g.add_argument('--weight_dice', type=float, default=5.0)
    g.add_argument('--eos_coef',    type=float, default=0.1,
                   help='Weight for "no object" class in CE loss')

    args = parser.parse_args()

    # ---------- Setup ----------
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  Mask2Former — Leaf Segmentation")
    print(f"{'='*65}")
    print(f"  Device     : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
    print(f"  Output dir : {args.output_dir}")

    # ---------- Discover dataset ----------
    print(f"\n  Loading dataset from: {args.data_dir}")

    # Read CSV to get total count for splitting
    csv_path = os.path.join(args.data_dir, 'train.csv')
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        _ = next(reader, None)
        all_rows = list(reader)
    total = len(all_rows)

    indices = list(range(total))
    random.Random(args.seed).shuffle(indices)
    train_end = int(total * args.train_ratio)
    val_end   = train_end + int(total * args.val_ratio)
    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    print(f"  Total: {total} | Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ---------- Build datasets ----------
    # Create a temp dataset to detect classes & get mapping
    tmp_ds = LeafSegDataset(args.data_dir, verbose=False)
    num_classes = tmp_ds.num_classes
    value_mapping = tmp_ds.value_mapping
    class_names = tmp_ds.class_names
    del tmp_ds

    print(f"  Classes: {num_classes}   mapping: {value_mapping}")

    train_transform = JointCompose([
        JointResize(args.img_size),
        JointRandomHFlip(0.5),
        JointRandomVFlip(0.3),
        JointRandomRotation(15),
        JointColorJitter(0.3, 0.3, 0.3, 0.1),
        JointToTensor(value_mapping),
    ])

    val_transform = JointCompose([
        JointResize(args.img_size),
        JointToTensor(value_mapping),
    ])

    train_ds = LeafSegDataset(args.data_dir, joint_transform=train_transform,
                              indices=train_idx, value_mapping=value_mapping,
                              verbose=False)
    val_ds   = LeafSegDataset(args.data_dir, joint_transform=val_transform,
                              indices=val_idx, value_mapping=value_mapping,
                              verbose=False)
    test_ds  = LeafSegDataset(args.data_dir, joint_transform=val_transform,
                              indices=test_idx, value_mapping=value_mapping,
                              verbose=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---------- Model ----------
    model = Mask2FormerSegmentor(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        nheads=args.nheads,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_backbone=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters: {total_params:,} total | {train_params:,} trainable")

    # ---------- Loss / Optimizer / Scheduler ----------
    matcher   = HungarianMatcher(args.weight_cls, args.weight_bce, args.weight_dice)
    criterion = SetCriterion(num_classes, matcher, args.eos_coef,
                             args.weight_cls, args.weight_bce, args.weight_dice).to(device)

    backbone_params = list(model.backbone.parameters())
    other_params    = [p for n, p in model.named_parameters()
                       if not n.startswith('backbone')]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': other_params,    'lr': args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ---------- Training loop ----------
    history = {
        'train_loss': [], 'val_loss': [],
        'val_miou': [], 'val_dice': [],
        'train_pacc': [], 'val_pacc': [],
    }
    best_miou = 0.0
    patience_counter = 0
    checkpoint = None

    print(f"\n{'='*65}")
    print(f"  Training for up to {args.epochs} epochs  (patience={args.patience})")
    print(f"{'='*65}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_info = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args.img_size, num_classes, scaler)

        val_metrics = evaluate(
            model, val_loader, criterion, device,
            args.img_size, num_classes)

        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_info['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_miou'].append(val_metrics['mIoU'] * 100)
        history['val_dice'].append(val_metrics['mean_Dice'] * 100)
        history['train_pacc'].append(train_info['pixel_acc'])
        history['val_pacc'].append(val_metrics['pixel_accuracy'] * 100)

        v_miou = val_metrics['mIoU'] * 100
        v_dice = val_metrics['mean_Dice'] * 100
        v_pacc = val_metrics['pixel_accuracy'] * 100

        print(f"  Epoch {epoch:>3d}/{args.epochs}  |  "
              f"Loss: {train_info['loss']:.4f}/{val_metrics['loss']:.4f}  |  "
              f"mIoU: {v_miou:.2f}%  Dice: {v_dice:.2f}%  "
              f"pAcc: {v_pacc:.2f}%  |  {elapsed:.1f}s")

        if v_miou > best_miou:
            best_miou = v_miou
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'num_classes': num_classes,
                'value_mapping': value_mapping,
                'class_names': class_names,
                'model_config': {
                    'num_classes': num_classes,
                    'hidden_dim': args.hidden_dim,
                    'num_queries': args.num_queries,
                    'nheads': args.nheads,
                    'num_decoder_layers': args.num_decoder_layers,
                    'dim_feedforward': args.dim_feedforward,
                    'dropout': args.dropout,
                    'img_size': args.img_size,
                },
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"    * Best model saved  (mIoU: {best_miou:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Save last model
    last_ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'value_mapping': value_mapping,
        'class_names': class_names,
        'model_config': checkpoint['model_config'] if checkpoint else {},
    }
    torch.save(last_ckpt, os.path.join(args.output_dir, 'last_model.pth'))

    # Training curves
    plot_training_curves(history, os.path.join(args.output_dir, 'training_curves.png'))

    # ==================== TEST EVALUATION ====================
    print(f"\n{'='*65}")
    print(f"  Evaluating on Test Set  ({len(test_idx)} images)")
    print(f"{'='*65}\n")

    best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = evaluate(
        model, test_loader, criterion, device,
        args.img_size, num_classes)

    print(f"\n  {'='*50}")
    print(f"  TEST RESULTS  (Mask2Former — Segmentation)")
    print(f"  {'='*50}")
    print(f"    mIoU           : {test_metrics['mIoU'] * 100:.2f}%")
    print(f"    Mean Dice      : {test_metrics['mean_Dice'] * 100:.2f}%")
    print(f"    Pixel Accuracy : {test_metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"    Loss           : {test_metrics['loss']:.4f}")

    print(f"\n  Per-class IoU:")
    for c, iou in test_metrics['per_class_iou'].items():
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        print(f"    {name:<20s}  IoU: {iou * 100:.2f}%")

    print(f"\n  Per-class Dice:")
    for c, d in test_metrics['per_class_dice'].items():
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        print(f"    {name:<20s}  Dice: {d * 100:.2f}%")

    # Sample predictions
    plot_sample_predictions(
        model, test_ds, device, args.img_size, num_classes,
        os.path.join(args.output_dir, 'sample_predictions.png'), n=6)

    # Save results
    results_json = {
        'mIoU':           float(test_metrics['mIoU'] * 100),
        'mean_Dice':      float(test_metrics['mean_Dice'] * 100),
        'pixel_accuracy': float(test_metrics['pixel_accuracy'] * 100),
        'loss':           float(test_metrics['loss']),
        'best_val_miou':  float(best_miou),
        'epochs_trained': epoch,
        'per_class_iou':  {str(k): float(v * 100)
                           for k, v in test_metrics['per_class_iou'].items()},
        'per_class_dice': {str(k): float(v * 100)
                           for k, v in test_metrics['per_class_dice'].items()},
        'num_classes':    num_classes,
        'class_names':    class_names,
    }
    with open(os.path.join(args.output_dir, 'test_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\n  All results saved to: {args.output_dir}/")
    print(f"    - best_model.pth")
    print(f"    - last_model.pth")
    print(f"    - training_curves.png")
    print(f"    - sample_predictions.png")
    print(f"    - test_results.json")
    print(f"\n  Done!")


if __name__ == '__main__':
    main()
