import os
import random
import argparse
from collections import defaultdict, Counter
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

# For pHash duplicate detection
try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("[WARNING] imagehash not installed. pHash duplicate detection disabled.")
    print("          Install with: pip install imagehash")

# ------------------------------
# Utils: Dataset builder
# ------------------------------
PRIOR_ORG_ORDER = ["hand", "leaf", "flower", "fruit"]   # first priority
SECOND_ORG_ORDER = ["seed", "root"]                    # second priority
# `available` images are files directly under class folder

def collect_all_images(data_root, verbose=True):
    """
    Collect ALL available images from the dataset without any limit.
    Returns: dict class_name -> list of ALL image paths
    """
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    all_class_imgs = {}

    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        all_imgs = []

        # Collect from all subfolders
        for sub in PRIOR_ORG_ORDER + SECOND_ORG_ORDER:
            subdir = os.path.join(cls_dir, sub)
            if os.path.isdir(subdir):
                files = [os.path.join(subdir, f) for f in os.listdir(subdir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))]
                all_imgs.extend(sorted(files))

        # Collect available images (files in class root)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        all_imgs.extend(sorted(files))

        # Deduplicate and keep order
        seen = set()
        uniq = []
        for p in all_imgs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        all_class_imgs[cls] = uniq
        if verbose:
            print(f"[collect_all] class={cls} -> {len(uniq)} total images available")

    return all_class_imgs

def collect_images_per_class(data_root, max_per_class=100, verbose=True):
    """
    Select images per class according to priority rules:
     1) First collect ALL available images
     2) Then select up to max_per_class images with priority:
        - hand, leaf, flower, fruit (first priority)
        - seed, root (second priority)
        - available images (files in class root)
     3) Cap at max_per_class

    Returns:
        - selected_imgs: dict class_name -> list of selected image paths (max max_per_class)
        - all_imgs: dict class_name -> list of ALL available image paths
    """
    # First, collect ALL available images
    all_class_imgs = collect_all_images(data_root, verbose=False)

    classes = sorted(all_class_imgs.keys())
    selected_class_imgs = {}

    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        selected = []

        # 1) First priority subfolders: hand, leaf, flower, fruit
        for sub in PRIOR_ORG_ORDER:
            subdir = os.path.join(cls_dir, sub)
            if os.path.isdir(subdir):
                files = [os.path.join(subdir, f) for f in os.listdir(subdir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))]
                selected.extend(sorted(files))
                if len(selected) >= max_per_class:
                    selected = selected[:max_per_class]
                    break

        # 2) Second priority: seed, root (if still < max_per_class)
        if len(selected) < max_per_class:
            for sub in SECOND_ORG_ORDER:
                subdir = os.path.join(cls_dir, sub)
                if os.path.isdir(subdir):
                    files = [os.path.join(subdir, f) for f in os.listdir(subdir)
                            if f.lower().endswith(('.jpg','.jpeg','.png'))]
                    selected.extend(sorted(files))
                    if len(selected) >= max_per_class:
                        selected = selected[:max_per_class]
                        break

        # 3) Available images (files in class root) - if still < max_per_class
        if len(selected) < max_per_class:
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]
            selected.extend(sorted(files))
            if len(selected) >= max_per_class:
                selected = selected[:max_per_class]

        # Deduplicate and keep order
        seen = set()
        uniq = []
        for p in selected:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        selected_class_imgs[cls] = uniq

        if verbose:
            total_available = len(all_class_imgs[cls])
            selected_count = len(uniq)
            unselected_count = total_available - selected_count
            print(f"[select] class={cls} -> Selected: {selected_count}/{total_available} "
                  f"(Unselected: {unselected_count})")

    return selected_class_imgs, all_class_imgs

# ------------------------------
# Dataset class returning image, species_label, filename
# Also supports generating pseudo organ labels via later function
# ------------------------------
class BigPlantsDataset(Dataset):
    def __init__(self, class_to_imgs, class_to_idx, transform=None, pseudo_org=None):
        """
        class_to_imgs: dict class -> list of image paths
        pseudo_org: dict img_path -> organ_prior vector (numpy)
        """
        self.samples = []
        for cls, imgs in class_to_imgs.items():
            idx = class_to_idx[cls]
            for p in imgs:
                self.samples.append((p, idx, cls))
        self.transform = transform
        self.pseudo_org = pseudo_org or {}
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        p, idx, cls = self.samples[i]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)
        org_prior = self.pseudo_org.get(p, None)
        if org_prior is None:
            # default uniform prior over organs: hand/leaf/flower/fruit/other
            org_prior = np.array([0.2]*5, dtype=np.float32)
        return img_t, idx, p, torch.from_numpy(org_prior.astype(np.float32))

# ------------------------------
# O1: Pseudo-label organ mining via clustering on features
# We'll extract global features or patch features using a pretrained backbone, then KMeans into M clusters (=organ buckets)
# Output: dict img_path -> organ_prior (soft one-hot or probabilities)
# ------------------------------
def extract_features_for_paths(paths, feature_extractor, device, batch_size=64):
    """
    Extract features for a list of image paths.
    Returns: features (N, D), paths list
    """
    transform = T.Compose([T.Resize((224,224)), T.ToTensor(),
                           T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
    class DummyDataset(Dataset):
        def __init__(self, paths): self.paths=paths
        def __len__(self): return len(self.paths)
        def __getitem__(self,i):
            p=self.paths[i]; img=Image.open(p).convert('RGB'); return transform(img), p
    
    loader = DataLoader(DummyDataset(paths), batch_size=batch_size, shuffle=False, num_workers=4)
    feats=[]
    paths_out=[]
    feature_extractor.eval()
    with torch.no_grad():
        for batch, ps in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            f = feature_extractor(batch)
            if isinstance(f, tuple) or isinstance(f, list):
                f = f[0]
            f = f.detach().cpu().numpy()
            feats.append(f)
            paths_out.extend(ps)
    feats = np.vstack(feats)
    return feats, paths_out


def generate_pseudo_organs_train_only(train_map, feature_extractor, device, n_clusters=5, batch_size=64):
    """
    [FIX DATA LEAKAGE] Generate pseudo organ priors using ONLY training data.
    
    This function fits KMeans ONLY on training images to prevent data leakage.
    Returns:
        - priors: dict img_path -> organ_prior vector (for train images)
        - kmeans: fitted KMeans model (to transform val/test later)
    """
    # Collect only TRAIN paths
    train_paths = []
    for imgs in train_map.values():
        train_paths.extend(imgs)
    
    print(f"[O1] Extracting features from {len(train_paths)} TRAIN images only (no data leakage)...")
    feats, paths = extract_features_for_paths(train_paths, feature_extractor, device, batch_size)
    
    # KMeans on TRAIN data only
    print(f"[O1] KMeans clustering on TRAIN features -> {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
    labels = kmeans.labels_
    
    # Create one-hot priors for train
    priors = {}
    for p, l in zip(paths, labels):
        vec = np.zeros(n_clusters, dtype=np.float32)
        vec[l] = 1.0
        priors[p] = vec
    
    return priors, kmeans


def apply_kmeans_to_split(split_map, kmeans, feature_extractor, device, n_clusters=5, batch_size=64):
    """
    [FIX DATA LEAKAGE] Apply pre-fitted KMeans to val/test data.
    
    Uses kmeans.predict() instead of fit() to avoid leakage.
    """
    # Collect paths from split
    all_paths = []
    for imgs in split_map.values():
        all_paths.extend(imgs)
    
    if len(all_paths) == 0:
        return {}
    
    print(f"[O1] Transforming {len(all_paths)} images using pre-fitted KMeans...")
    feats, paths = extract_features_for_paths(all_paths, feature_extractor, device, batch_size)
    
    # PREDICT (not fit!) using pre-fitted kmeans
    labels = kmeans.predict(feats)
    
    # Create one-hot priors
    priors = {}
    for p, l in zip(paths, labels):
        vec = np.zeros(n_clusters, dtype=np.float32)
        vec[l] = 1.0
        priors[p] = vec
    
    return priors


# Keep old function for backward compatibility but mark as deprecated
def generate_pseudo_orginals(class_to_imgs, feature_extractor, device, n_clusters=5, batch_size=64):
    """
    [DEPRECATED - HAS DATA LEAKAGE]
    Use generate_pseudo_organs_train_only() instead.
    
    This function clusters ALL images including val/test, causing data leakage.
    """
    print("[WARNING] Using deprecated function with potential data leakage!")
    print("[WARNING] Consider using generate_pseudo_organs_train_only() instead.")
    
    all_paths=[]
    for imgs in class_to_imgs.values():
        all_paths.extend(imgs)
    # DataLoader to extract features
    transform = T.Compose([T.Resize((224,224)), T.ToTensor(),
                           T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
    class DummyDataset(Dataset):
        def __init__(self, paths): self.paths=paths
        def __len__(self): return len(self.paths)
        def __getitem__(self,i):
            p=self.paths[i]; img=Image.open(p).convert('RGB'); return transform(img), p
    loader = DataLoader(DummyDataset(all_paths), batch_size=batch_size, shuffle=False, num_workers=4)
    feats=[]
    paths=[]
    feature_extractor.eval()
    with torch.no_grad():
        for batch, ps in tqdm(loader, desc="extract feats for clustering"):
            batch = batch.to(device)
            f = feature_extractor(batch)  # expect (B, D)
            if isinstance(f, tuple) or isinstance(f, list):
                f = f[0]
            f = f.detach().cpu().numpy()
            feats.append(f)
            paths.extend(ps)
    feats = np.vstack(feats)  # (N, D)
    # KMeans
    print("[O1] KMeans clustering features -> clusters:", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
    labels = kmeans.labels_
    # create one-hot soft prior (one-hot)
    priors = {}
    for p, l in zip(paths, labels):
        vec = np.zeros(n_clusters, dtype=np.float32)
        vec[l]=1.0
        priors[p]=vec
    return priors, kmeans

# ------------------------------
# Model components
# - backbone: timm ViT (we'll use forward_features to get patch tokens)
# - OrganAuxHead: predict organ cluster from image (global)
# - Router: take patch token and organ_prior -> produce logits over experts
# - SwitchMoE: experts are small FFNs; naive dispatch
# ------------------------------
class OrganAuxHead(nn.Module):
    def __init__(self, in_dim, n_org):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_org)
    def forward(self, x):
        # x is global feature (B, D) or CLS token
        return self.fc(x)

class Router(nn.Module):
    def __init__(self, token_dim, organ_dim, n_experts):
        super().__init__()
        self.linear = nn.Linear(token_dim + organ_dim, n_experts)
    def forward(self, token, organ_prior):
        # token (B*T, D), organ_prior (B*T, organ_dim) (we will broadcast per token)
        inp = torch.cat([token, organ_prior], dim=-1)
        logits = self.linear(inp)  # (B*T, n_experts)
        return logits

class FFNExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x): return self.net(x)

class SwitchMoE(nn.Module):
    def __init__(self, d_model, organ_dim, n_experts=8, d_ff=2048, capacity_factor=1.25, top_k=1, entropy_threshold=1.5):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.capacity_factor = capacity_factor
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold  # O5: entropy fallback threshold
        self.router = Router(d_model, organ_dim, n_experts)
        self.experts = nn.ModuleList([FFNExpert(d_model, d_ff) for _ in range(n_experts)])

        # O4: Expert specialization - track expert usage for regularization
        self.register_buffer('expert_usage', torch.zeros(n_experts))

    def forward(self, tokens, organ_priors, training=True):
        """
        tokens: (B, T, D)
        organ_priors: (B, T, organ_dim)  -- organ prior repeated per patch
        Returns: out tokens same shape
        """
        B,T,D = tokens.shape
        flat = tokens.reshape(B*T, D)
        flat_prior = organ_priors.reshape(B*T, -1)
        logits = self.router(flat, flat_prior)  # (B*T, E)
        probs = F.softmax(logits, dim=-1)

        # O5: Coarse-to-fine routing with entropy fallback
        entropy = -(probs * (probs+1e-12).log()).sum(dim=-1)  # (B*T,)

        # Adaptive top_k based on entropy (high entropy = uncertain -> use top-2)
        effective_top_k = torch.where(entropy > self.entropy_threshold,
                                       torch.tensor(min(2, self.n_experts), device=entropy.device),
                                       torch.tensor(self.top_k, device=entropy.device))

        # For simplicity, use max top_k for all
        max_k = min(2, self.n_experts) if training else self.top_k
        topk_vals, topk_idx = torch.topk(probs, max_k, dim=-1)  # (B*T, k)

        # Normalize topk probabilities
        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)

        # Process with weighted combination for uncertain tokens
        outputs = torch.zeros_like(flat, device=flat.device)
        usage_counts = torch.zeros(self.n_experts, device=flat.device)
        capacity = math.ceil((B*T / self.n_experts) * self.capacity_factor)

        # Track which tokens are processed
        processed_mask = torch.zeros(B*T, dtype=torch.bool, device=flat.device)

        # Process each expert
        for e in range(self.n_experts):
            # Find tokens that selected this expert (in top-k)
            mask = (topk_idx == e).any(dim=-1)  # (B*T,)
            idx = mask.nonzero(as_tuple=True)[0]

            if idx.numel() == 0:
                continue

            # Capacity enforcement
            if idx.numel() > capacity:
                # Prioritize by routing probability
                expert_probs = probs[idx, e]
                _, sorted_idx = torch.sort(expert_probs, descending=True)
                idx = idx[sorted_idx[:capacity]]

            selected = flat[idx]  # (n_sel, D)
            out_sel = self.experts[e](selected)

            # Get weights for this expert
            weights = torch.zeros(idx.shape[0], device=flat.device)
            for i, token_idx in enumerate(idx):
                expert_mask = (topk_idx[token_idx] == e)
                if expert_mask.any():
                    k_pos = expert_mask.nonzero(as_tuple=True)[0][0]
                    weights[i] = topk_probs[token_idx, k_pos]

            # Weighted output accumulation
            outputs[idx] += out_sel * weights.unsqueeze(-1)
            processed_mask[idx] = True
            usage_counts[e] = idx.numel()

        # O4: Update expert usage statistics
        if training:
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * usage_counts

        # Residual connection for unprocessed tokens (should not happen)
        outputs[~processed_mask] = flat[~processed_mask]

        outputs = outputs.reshape(B, T, D)
        return outputs, probs.reshape(B, T, -1), entropy.reshape(B, T)

# ------------------------------
# Organ-Aware V-MoE model (backbone + some MoE layers + heads)
# We'll keep architecture simple: vit backbone -> extract patch tokens -> pass through one SwitchMoE -> pool CLS -> classification head
# plus organ aux head on CLS
# ------------------------------
class OrganAwareSwitchViT(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', n_classes=100, organ_dim=5, n_experts=8, d_ff_expert=1024, top_k=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)  # returns features via forward_features
        # probe an input to discover dims (or use known)
        dummy = torch.randn(1,3,224,224)
        with torch.no_grad():
            # try to call forward_features if exists
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(dummy)  # often returns (B, N+1, D)
            else:
                feat = self.backbone(dummy)
            # If feat shape (B, D) then backbone returns pooled; we'll instead rely on backbone.patch_embed if exists
        # many timm ViT returns (B, N+1, D)
        # We'll assume forward_features output shape (B, N+1, D)
        # So pick D:
        if isinstance(feat, torch.Tensor) and feat.dim()==3:
            _, NT, D = feat.shape
            self.token_dim = D
            self.num_tokens = NT
        else:
            # fallback: set dims
            self.token_dim = 768
            self.num_tokens = 197
        self.organ_dim = organ_dim
        # Switch MoE layer (operate on patch tokens excluding CLS)
        # We'll operate on tokens 1: (patch tokens)
        self.switch = SwitchMoE(d_model=self.token_dim, organ_dim=self.organ_dim, n_experts=n_experts, d_ff=d_ff_expert, top_k=top_k)
        # classification head: pool CLS token after adding MoE outputs back
        self.cls_head = nn.Linear(self.token_dim, n_classes)
        # organ aux head from CLS
        self.aux_head = OrganAuxHead(self.token_dim, organ_dim)
        # small layernorm
        self.ln = nn.LayerNorm(self.token_dim)
    def forward(self, x, organ_priors_image, training=True, capacity_factor=None):
        """
        x: (B,3,224,224)
        organ_priors_image: dict or tensor providing per-image organ prior distribution (B, organ_dim)
         - We'll expand to per-patch inside
        """
        # backbone forward_features -> tokens
        # Many timm vit models have forward_features returning (B, N+1, D)
        if hasattr(self.backbone, 'forward_features'):
            tokens = self.backbone.forward_features(x)  # (B, N+1, D)
        else:
            tokens = self.backbone(x)
        # split cls and patches
        cls = tokens[:,0:1,:]  # (B,1,D)
        patches = tokens[:,1:,:]  # (B, T, D)
        B,T,D = patches.shape
        # organ_priors_image: (B, organ_dim) -> expand per patch
        if isinstance(organ_priors_image, torch.Tensor):
            org_prior_tokens = organ_priors_image.unsqueeze(1).expand(B, T, -1)  # (B,T,organ_dim)
        else:
            # if given as numpy or list convert
            org_prior_tokens = torch.stack([torch.from_numpy(organ_priors_image[i]).float() for i in range(len(organ_priors_image))]).to(x.device)
            org_prior_tokens = org_prior_tokens.unsqueeze(1).expand(B, T, -1)
        # pass through SwitchMoE
        outputs, probs, entropy = self.switch(patches, org_prior_tokens, training=training)
        # combine: replace patch tokens with outputs; recombine with cls token
        tokens2 = torch.cat([cls, outputs], dim=1)
        # pool cls (simple take tokens2[:,0])
        cls_final = tokens2[:,0,:]  # (B, D)
        cls_final = self.ln(cls_final)
        logits = self.cls_head(cls_final)
        aux_org = self.aux_head(cls_final)  # (B, organ_dim)
        return logits, aux_org, probs, entropy

# ------------------------------
# Augmentation: OrganMix simple cut-paste (O6)
# ------------------------------
def organmix(img1, img2, alpha=0.5):
    """
    OrganMix: Mix two images with random rectangular regions
    img1, img2: torch tensors (C, H, W)
    Returns mixed image
    """
    _, h, w = img1.shape
    # Random bbox for mixing
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    img_mixed = img1.clone()
    img_mixed[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]

    return img_mixed, lam

# ------------------------------
# Training & Evaluation routines
# ------------------------------
def train_epoch(model, dataloader, optimizer, device, epoch, aux_weight=0.5, balance_weight=0.01,
                capacity_factor=1.25, organ_dim=5, use_organmix=True, organmix_prob=0.5):
    model.train()
    total_loss, total_acc = 0.0, 0
    total_cls_loss, total_aux_loss, total_balance_loss = 0.0, 0.0, 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()

    epoch_start = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        imgs, labels, paths, org_priors = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        org_priors = org_priors.to(device)

        # O6: OrganMix augmentation
        if use_organmix and np.random.rand() < organmix_prob and imgs.size(0) > 1:
            # Mix pairs within batch
            indices = torch.randperm(imgs.size(0))
            imgs_mixed = []
            labels_a, labels_b, lam_list = [], [], []

            for i in range(imgs.size(0)):
                if np.random.rand() < 0.5:  # Apply mixing
                    j = indices[i]
                    mixed_img, lam = organmix(imgs[i], imgs[j], alpha=0.5)
                    imgs_mixed.append(mixed_img)
                    labels_a.append(labels[i])
                    labels_b.append(labels[j])
                    lam_list.append(lam)
                else:
                    imgs_mixed.append(imgs[i])
                    labels_a.append(labels[i])
                    labels_b.append(labels[i])
                    lam_list.append(1.0)

            imgs = torch.stack(imgs_mixed)
            labels_a = torch.stack(labels_a)
            labels_b = torch.stack(labels_b)
            lam_tensor = torch.tensor(lam_list, device=device)
        else:
            labels_a = labels
            labels_b = labels
            lam_tensor = torch.ones(imgs.size(0), device=device)

        optimizer.zero_grad()
        logits, aux_org, probs, entropy = model(imgs, org_priors, training=True)

        # Mixed loss for OrganMix
        loss_cls_a = criterion(logits, labels_a)
        loss_cls_b = criterion(logits, labels_b)
        loss_cls = (lam_tensor * loss_cls_a + (1 - lam_tensor) * loss_cls_b).mean()

        # O2: Aux organ loss with calibration
        if aux_org is not None:
            logp = F.log_softmax(aux_org, dim=-1)
            target = org_priors
            loss_aux = F.kl_div(logp, target, reduction='batchmean')
        else:
            loss_aux = torch.tensor(0.0, device=device)

        # O4: Expert specialization regularization
        # Encourage balanced expert usage
        p_mean = probs.mean(dim=(0, 1))  # (E,)
        L_balance = (p_mean * p_mean).sum() * balance_weight

        # Total loss
        loss = loss_cls + aux_weight * loss_aux + L_balance
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        total_loss += loss.item() * imgs.size(0)
        total_cls_loss += loss_cls.item() * imgs.size(0)
        total_aux_loss += loss_aux.item() * imgs.size(0)
        total_balance_loss += L_balance.item() * imgs.size(0)

        # Accuracy with original labels
        preds = logits.argmax(dim=-1)
        total_acc += (preds == labels).sum().item()
        n += imgs.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_acc/n:.4f}',
            'cls': f'{loss_cls.item():.4f}',
            'aux': f'{loss_aux.item():.4f}'
        })

    epoch_time = time.time() - epoch_start

    return {
        'loss': total_loss / n,
        'acc': total_acc / n,
        'cls_loss': total_cls_loss / n,
        'aux_loss': total_aux_loss / n,
        'balance_loss': total_balance_loss / n,
        'time': epoch_time
    }

def evaluate(model, dataloader, device, class_names, phase='val'):
    model.eval()
    y_true = []
    y_pred = []
    paths_all = []
    all_probs = []

    eval_start = time.time()

    with torch.no_grad():
        for imgs, labels, paths, org_priors in tqdm(dataloader, desc=f"[{phase.upper()}]"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            org_priors = org_priors.to(device)

            logits, aux_org, probs, entropy = model(imgs, org_priors, training=False)

            pred_probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            paths_all.extend(paths)
            all_probs.extend(pred_probs.cpu().numpy())

    eval_time = time.time() - eval_start

    # O8: Comprehensive metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    # Accuracy
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Classification report
    cls_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0,
        output_dict=True
    )

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'per_precision': per_p,
        'per_recall': per_r,
        'per_f1': per_f1,
        'per_support': per_support,
        'y_true': y_true,
        'y_pred': y_pred,
        'paths': paths_all,
        'probs': all_probs,
        'confusion_matrix': cm,
        'classification_report': cls_report,
        'time': eval_time
    }

# ------------------------------
# Utilities: build DataLoaders, train/val/test split
# ------------------------------
def build_loaders(class_to_imgs, max_per_class, batch_size, val_split=0.1, test_split=0.1, num_workers=4, seed=42):
    classes = sorted(list(class_to_imgs.keys()))
    class_to_idx = {c:i for i,c in enumerate(classes)}

    # Split within each class: train / val / test
    train_map = {}
    val_map = {}
    test_map = {}

    for c, imgs in class_to_imgs.items():
        n = len(imgs)
        n_test = int(math.ceil(test_split * n))
        n_val = int(math.ceil(val_split * n))

        # Shuffle deterministic
        random.Random(hash(c) & 0xffffffff).shuffle(imgs)

        test = imgs[:n_test]
        val = imgs[n_test:n_test+n_val]
        train = imgs[n_test+n_val:]

        train_map[c] = train
        val_map[c] = val
        test_map[c] = test

    # Transforms
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Initial pseudo orgs empty -> default uniform
    train_ds = BigPlantsDataset(train_map, class_to_idx, transform=train_tf, pseudo_org={})
    val_ds = BigPlantsDataset(val_map, class_to_idx, transform=val_tf, pseudo_org={})
    test_ds = BigPlantsDataset(test_map, class_to_idx, transform=val_tf, pseudo_org={})

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx, train_map, val_map, test_map

# ------------------------------
# Helper function to save dataset splits to CSV
# ------------------------------
def save_dataset_splits(out_dir, all_class_imgs, class_to_imgs, train_map, val_map, test_map, class_to_idx):
    """Save selected, unselected, train, val, test splits to CSV files

    Args:
        all_class_imgs: dict with ALL available images per class
        class_to_imgs: dict with selected images per class (max 100)
        train_map, val_map, test_map: train/val/test splits
    """

    # Collect all selected images (from train/val/test splits)
    selected_images = set()
    for cls_map in [train_map, val_map, test_map]:
        for imgs in cls_map.values():
            selected_images.update(imgs)

    # Dataset selected
    with open(os.path.join(out_dir, 'dataset_selected.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'class_idx', 'image_path', 'split'])

        for cls, imgs in train_map.items():
            for img in imgs:
                writer.writerow([cls, class_to_idx[cls], img, 'train'])
        for cls, imgs in val_map.items():
            for img in imgs:
                writer.writerow([cls, class_to_idx[cls], img, 'val'])
        for cls, imgs in test_map.items():
            for img in imgs:
                writer.writerow([cls, class_to_idx[cls], img, 'test'])

    # Dataset unselected (all available images that were NOT selected)
    with open(os.path.join(out_dir, 'dataset_unselected.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'class_idx', 'image_path'])

        # Use all_class_imgs (all available) to find unselected ones
        for cls, all_imgs in all_class_imgs.items():
            for img in all_imgs:
                if img not in selected_images:
                    writer.writerow([cls, class_to_idx[cls], img])

    # Train split
    with open(os.path.join(out_dir, 'train.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'class_idx', 'image_path'])
        for cls, imgs in train_map.items():
            for img in imgs:
                writer.writerow([cls, class_to_idx[cls], img])

    # Val split
    with open(os.path.join(out_dir, 'val.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'class_idx', 'image_path'])
        for cls, imgs in val_map.items():
            for img in imgs:
                writer.writerow([cls, class_to_idx[cls], img])

    # Test split
    with open(os.path.join(out_dir, 'test.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'class_idx', 'image_path'])
        for cls, imgs in test_map.items():
            for img in imgs:
                writer.writerow([cls, class_to_idx[cls], img])

    # Count statistics
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = len(selected_images)
    total_unselected = total_available - total_selected

    print(f"[INFO] Saved dataset splits to {out_dir}")
    print(f"  - Total available: {total_available} images")
    print(f"  - Selected: {total_selected} images")
    print(f"  - Unselected: {total_unselected} images")

def save_confusion_matrix(cm, class_names, out_dir, phase='test'):
    """Save and visualize confusion matrix"""
    # Save as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, f'confusion_matrix_{phase}.csv'))

    # Visualize (only for smaller matrices to avoid huge images)
    if len(class_names) <= 50:
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {phase.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'confusion_matrix_{phase}.png'), dpi=150)
        plt.close()

    # Save normalized version
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_norm_df.to_csv(os.path.join(out_dir, f'confusion_matrix_{phase}_normalized.csv'))

def save_classification_report(cls_report, class_names, out_dir, phase='test'):
    """Save classification report to CSV"""
    # Convert to DataFrame
    report_data = []
    for cls_name in class_names:
        if cls_name in cls_report:
            metrics = cls_report[cls_name]
            report_data.append({
                'class': cls_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })

    # Add macro/micro averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in cls_report:
            metrics = cls_report[avg_type]
            report_data.append({
                'class': avg_type,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })

    df = pd.DataFrame(report_data)
    df.to_csv(os.path.join(out_dir, f'classification_report_{phase}.csv'), index=False)
    print(f"[INFO] Saved classification report to {out_dir}/classification_report_{phase}.csv")


# ------------------------------
# pHash Data Leakage Detection
# ------------------------------
def compute_phash(img_path, hash_size=8):
    """
    Compute perceptual hash for an image.
    Returns hex string of hash, or None if error.
    """
    if not PHASH_AVAILABLE:
        return None
    try:
        img = Image.open(img_path).convert('RGB')
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception as e:
        print(f"[WARNING] Could not compute pHash for {img_path}: {e}")
        return None


def check_data_leakage_phash(train_map, val_map, test_map, out_dir, hash_size=8, threshold=5):
    """
    Check for potential data leakage between train/val/test using pHash.
    
    This function:
    1. Computes pHash for all images in train, val, test
    2. Finds exact duplicates (same hash)
    3. Finds near-duplicates (hamming distance <= threshold)
    4. Reports any cross-split duplicates as potential leakage
    
    Args:
        train_map, val_map, test_map: dicts of class -> list of image paths
        out_dir: output directory for leakage report
        hash_size: pHash size (default 8 -> 64-bit hash)
        threshold: max hamming distance for near-duplicates
    
    Returns:
        dict with leakage statistics
    """
    if not PHASH_AVAILABLE:
        print("[WARNING] imagehash not available. Skipping pHash leakage check.")
        print("          Install with: pip install imagehash")
        return {'status': 'skipped', 'reason': 'imagehash not installed'}
    
    print("\n" + "=" * 80)
    print("[DATA LEAKAGE CHECK] Using pHash to detect duplicate images across splits")
    print("=" * 80)
    
    # Collect all paths with split labels
    def collect_paths(split_map, split_name):
        paths = []
        for cls, imgs in split_map.items():
            for img in imgs:
                paths.append((img, split_name, cls))
        return paths
    
    train_paths = collect_paths(train_map, 'train')
    val_paths = collect_paths(val_map, 'val')
    test_paths = collect_paths(test_map, 'test')
    
    all_items = train_paths + val_paths + test_paths
    print(f"[INFO] Total images to check: {len(all_items)}")
    print(f"  - Train: {len(train_paths)}")
    print(f"  - Val: {len(val_paths)}")
    print(f"  - Test: {len(test_paths)}")
    
    # Compute hashes
    print("\n[STEP 1] Computing pHash for all images...")
    hash_map = {}  # path -> (hash_hex, split, class)
    hash_to_paths = {}  # hash -> list of (path, split, class)
    
    for path, split, cls in tqdm(all_items, desc="Computing pHash"):
        h = compute_phash(path, hash_size=hash_size)
        if h is not None:
            hash_map[path] = (h, split, cls)
            if h not in hash_to_paths:
                hash_to_paths[h] = []
            hash_to_paths[h].append((path, split, cls))
    
    print(f"[INFO] Successfully computed {len(hash_map)} hashes")
    
    # Find exact duplicates (same hash)
    print("\n[STEP 2] Finding EXACT duplicates (same pHash)...")
    exact_duplicates = []
    exact_cross_split = []  # Cross-split duplicates = DATA LEAKAGE!
    
    for h, items in hash_to_paths.items():
        if len(items) > 1:
            splits = set(item[1] for item in items)
            if len(splits) > 1:
                # Cross-split duplicate - LEAKAGE!
                exact_cross_split.append({
                    'hash': h,
                    'items': items,
                    'splits': list(splits)
                })
            exact_duplicates.append({
                'hash': h,
                'count': len(items),
                'items': items
            })
    
    # Find near-duplicates (hamming distance <= threshold)
    print(f"\n[STEP 3] Finding NEAR duplicates (hamming distance <= {threshold})...")
    near_duplicates = []
    near_cross_split = []
    
    # Group by prefix for efficiency
    hash_bits = hash_size * hash_size
    prefix_bits = min(16, hash_bits)
    shift = hash_bits - prefix_bits
    
    buckets = {}
    for path, (h, split, cls) in hash_map.items():
        try:
            int_hash = int(h, 16)
            prefix = int_hash >> shift
            if prefix not in buckets:
                buckets[prefix] = []
            buckets[prefix].append((path, h, int_hash, split, cls))
        except:
            continue
    
    checked_pairs = set()
    for bucket_items in tqdm(buckets.values(), desc="Checking near-duplicates"):
        n = len(bucket_items)
        for i in range(n):
            for j in range(i + 1, n):
                p1, h1, int1, s1, c1 = bucket_items[i]
                p2, h2, int2, s2, c2 = bucket_items[j]
                
                pair_key = tuple(sorted([p1, p2]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                # Hamming distance
                dist = (int1 ^ int2).bit_count()
                
                if 0 < dist <= threshold:  # Near but not exact
                    record = {
                        'path1': p1, 'split1': s1, 'class1': c1,
                        'path2': p2, 'split2': s2, 'class2': c2,
                        'distance': dist
                    }
                    near_duplicates.append(record)
                    
                    if s1 != s2:
                        near_cross_split.append(record)
    
    # Generate report
    print("\n" + "=" * 80)
    print("[DATA LEAKAGE CHECK] RESULTS")
    print("=" * 80)
    
    leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total exact duplicate groups: {len(exact_duplicates)}")
    print(f"  Total near-duplicate pairs: {len(near_duplicates)}")
    print(f"")
    
    if leakage_found:
        print(f"  🚨 CROSS-SPLIT EXACT DUPLICATES: {len(exact_cross_split)} groups")
        print(f"  🚨 CROSS-SPLIT NEAR DUPLICATES: {len(near_cross_split)} pairs")
        print(f"")
        print(f"  ⚠️  DATA LEAKAGE DETECTED! ⚠️")
    else:
        print(f"  ✅ NO CROSS-SPLIT DUPLICATES FOUND")
        print(f"  ✅ No data leakage detected between train/val/test splits")
    
    # Save detailed report
    report_path = os.path.join(out_dir, 'data_leakage_check.csv')
    report_data = []
    
    # Exact cross-split duplicates
    for group in exact_cross_split:
        for item in group['items']:
            report_data.append({
                'type': 'exact_duplicate',
                'hash': group['hash'],
                'path': item[0],
                'split': item[1],
                'class': item[2],
                'distance': 0,
                'is_cross_split': True
            })
    
    # Near cross-split duplicates
    for record in near_cross_split:
        report_data.append({
            'type': 'near_duplicate',
            'hash': '',
            'path': record['path1'],
            'split': record['split1'],
            'class': record['class1'],
            'distance': record['distance'],
            'is_cross_split': True,
            'paired_with': record['path2'],
            'paired_split': record['split2']
        })
        report_data.append({
            'type': 'near_duplicate',
            'hash': '',
            'path': record['path2'],
            'split': record['split2'],
            'class': record['class2'],
            'distance': record['distance'],
            'is_cross_split': True,
            'paired_with': record['path1'],
            'paired_split': record['split1']
        })
    
    if report_data:
        df = pd.DataFrame(report_data)
        df.to_csv(report_path, index=False)
        print(f"\n[INFO] Detailed leakage report saved to: {report_path}")
    
    # Print examples if leakage found
    if exact_cross_split:
        print(f"\n[EXAMPLES] Exact cross-split duplicates:")
        for i, group in enumerate(exact_cross_split[:5]):  # Show first 5
            print(f"  Group {i+1} (hash={group['hash'][:16]}...):")
            for item in group['items']:
                print(f"    - [{item[1]:5s}] {os.path.basename(item[0])} (class: {item[2]})")
    
    if near_cross_split:
        print(f"\n[EXAMPLES] Near cross-split duplicates (distance <= {threshold}):")
        for i, record in enumerate(near_cross_split[:5]):  # Show first 5
            print(f"  Pair {i+1} (distance={record['distance']}):")
            print(f"    - [{record['split1']:5s}] {os.path.basename(record['path1'])} (class: {record['class1']})")
            print(f"    - [{record['split2']:5s}] {os.path.basename(record['path2'])} (class: {record['class2']})")
    
    print("\n" + "=" * 80)
    
    return {
        'status': 'completed',
        'leakage_found': leakage_found,
        'exact_duplicate_groups': len(exact_duplicates),
        'near_duplicate_pairs': len(near_duplicates),
        'exact_cross_split': len(exact_cross_split),
        'near_cross_split': len(near_cross_split),
        'exact_cross_split_details': exact_cross_split,
        'near_cross_split_details': near_cross_split,
        'report_path': report_path if report_data else None,
        'hash_map': hash_map  # Return hash_map for further processing
    }


def compute_phash_for_paths(paths, hash_size=8):
    """
    Compute pHash for a list of image paths.
    Returns: dict path -> hash_hex
    """
    if not PHASH_AVAILABLE:
        return {}
    
    hash_map = {}
    for path in tqdm(paths, desc="Computing pHash"):
        h = compute_phash(path, hash_size=hash_size)
        if h is not None:
            hash_map[path] = h
    return hash_map


def check_image_leakage_with_train(candidate_path, train_paths, train_hashes, threshold=5):
    """
    Check if a candidate image has leakage with any train image.
    Returns: True if leakage found, False otherwise
    """
    if not PHASH_AVAILABLE:
        return False
    
    candidate_hash = compute_phash(candidate_path)
    if candidate_hash is None:
        return False
    
    candidate_int = int(candidate_hash, 16)
    
    for train_path in train_paths:
        if train_path in train_hashes:
            train_hash = train_hashes[train_path]
            train_int = int(train_hash, 16)
            dist = (candidate_int ^ train_int).bit_count()
            if dist <= threshold:
                return True  # Leakage found
    
    return False  # No leakage


def handle_leakage_minor(train_map, val_map, test_map, all_class_imgs, 
                          leakage_result, out_dir, threshold=5):
    """
    Handle minor leakage (<5%): 
    1. Move leaked images from val/test to train
    2. Replace with non-leaked images from unselected pool
    3. Verify no new leakage
    
    Returns: updated train_map, val_map, test_map, success flag
    """
    print("\n" + "=" * 80)
    print("[LEAKAGE FIX] Minor leakage detected (<5%). Applying fix strategy...")
    print("=" * 80)
    
    # Collect all leaked images info
    leaked_from_val = {}  # class -> list of paths
    leaked_from_test = {}  # class -> list of paths
    
    # Process exact cross-split duplicates
    for group in leakage_result.get('exact_cross_split_details', []):
        for path, split, cls in group['items']:
            if split == 'val':
                if cls not in leaked_from_val:
                    leaked_from_val[cls] = []
                if path not in leaked_from_val[cls]:
                    leaked_from_val[cls].append(path)
            elif split == 'test':
                if cls not in leaked_from_test:
                    leaked_from_test[cls] = []
                if path not in leaked_from_test[cls]:
                    leaked_from_test[cls].append(path)
    
    # Process near cross-split duplicates
    for record in leakage_result.get('near_cross_split_details', []):
        if record['split1'] == 'val':
            cls = record['class1']
            path = record['path1']
            if cls not in leaked_from_val:
                leaked_from_val[cls] = []
            if path not in leaked_from_val[cls]:
                leaked_from_val[cls].append(path)
        elif record['split1'] == 'test':
            cls = record['class1']
            path = record['path1']
            if cls not in leaked_from_test:
                leaked_from_test[cls] = []
            if path not in leaked_from_test[cls]:
                leaked_from_test[cls].append(path)
        
        if record['split2'] == 'val':
            cls = record['class2']
            path = record['path2']
            if cls not in leaked_from_val:
                leaked_from_val[cls] = []
            if path not in leaked_from_val[cls]:
                leaked_from_val[cls].append(path)
        elif record['split2'] == 'test':
            cls = record['class2']
            path = record['path2']
            if cls not in leaked_from_test:
                leaked_from_test[cls] = []
            if path not in leaked_from_test[cls]:
                leaked_from_test[cls].append(path)
    
    total_leaked_val = sum(len(v) for v in leaked_from_val.values())
    total_leaked_test = sum(len(v) for v in leaked_from_test.values())
    print(f"\n[INFO] Leaked images to move:")
    print(f"  - From val: {total_leaked_val}")
    print(f"  - From test: {total_leaked_test}")
    
    # Get currently selected images
    selected_images = set()
    for cls_map in [train_map, val_map, test_map]:
        for imgs in cls_map.values():
            selected_images.update(imgs)
    
    # Get unselected images per class
    unselected_per_class = {}
    for cls, all_imgs in all_class_imgs.items():
        unselected_per_class[cls] = [p for p in all_imgs if p not in selected_images]
    
    # Compute hashes for all train images (for checking new candidates)
    print("\n[STEP 1] Computing pHash for all train images...")
    all_train_paths = []
    for imgs in train_map.values():
        all_train_paths.extend(imgs)
    train_hashes = compute_phash_for_paths(all_train_paths)
    
    # Process each class
    replacement_stats = {'val': 0, 'test': 0, 'failed': 0}
    
    # Fix VAL leakage
    print("\n[STEP 2] Fixing VAL leakage...")
    for cls, leaked_paths in leaked_from_val.items():
        print(f"\n  Class '{cls}': {len(leaked_paths)} leaked images")
        
        for leaked_path in leaked_paths:
            # Move leaked image to train
            if leaked_path in val_map.get(cls, []):
                val_map[cls].remove(leaked_path)
                train_map[cls].append(leaked_path)
                train_hashes[leaked_path] = compute_phash(leaked_path)
                print(f"    → Moved to train: {os.path.basename(leaked_path)}")
                
                # Find replacement from unselected
                found_replacement = False
                candidates = unselected_per_class.get(cls, [])
                
                for candidate in candidates:
                    # Check if candidate has leakage with train
                    if not check_image_leakage_with_train(candidate, 
                                                          list(train_map[cls]) + all_train_paths,
                                                          train_hashes, threshold):
                        # No leakage - use this candidate
                        val_map[cls].append(candidate)
                        unselected_per_class[cls].remove(candidate)
                        selected_images.add(candidate)
                        replacement_stats['val'] += 1
                        found_replacement = True
                        print(f"    ← Replaced with: {os.path.basename(candidate)}")
                        break
                
                if not found_replacement:
                    print(f"    ⚠ No suitable replacement found for val")
                    replacement_stats['failed'] += 1
    
    # Fix TEST leakage
    print("\n[STEP 3] Fixing TEST leakage...")
    # Update train_hashes with newly added images
    for cls in train_map:
        for img in train_map[cls]:
            if img not in train_hashes:
                h = compute_phash(img)
                if h:
                    train_hashes[img] = h
    
    for cls, leaked_paths in leaked_from_test.items():
        print(f"\n  Class '{cls}': {len(leaked_paths)} leaked images")
        
        for leaked_path in leaked_paths:
            # Move leaked image to train
            if leaked_path in test_map.get(cls, []):
                test_map[cls].remove(leaked_path)
                train_map[cls].append(leaked_path)
                train_hashes[leaked_path] = compute_phash(leaked_path)
                print(f"    → Moved to train: {os.path.basename(leaked_path)}")
                
                # Find replacement from unselected
                found_replacement = False
                candidates = unselected_per_class.get(cls, [])
                
                for candidate in candidates:
                    # Check if candidate has leakage with train
                    if not check_image_leakage_with_train(candidate,
                                                          list(train_map[cls]),
                                                          train_hashes, threshold):
                        # No leakage - use this candidate
                        test_map[cls].append(candidate)
                        unselected_per_class[cls].remove(candidate)
                        selected_images.add(candidate)
                        replacement_stats['test'] += 1
                        found_replacement = True
                        print(f"    ← Replaced with: {os.path.basename(candidate)}")
                        break
                
                if not found_replacement:
                    print(f"    ⚠ No suitable replacement found for test")
                    replacement_stats['failed'] += 1
    
    print(f"\n[SUMMARY] Replacement statistics:")
    print(f"  - Val replacements: {replacement_stats['val']}")
    print(f"  - Test replacements: {replacement_stats['test']}")
    print(f"  - Failed replacements: {replacement_stats['failed']}")
    
    return train_map, val_map, test_map, replacement_stats['failed'] == 0


def hamming_distance_int(int1, int2):
    """Compute hamming distance between two integers."""
    return (int1 ^ int2).bit_count()


def build_similarity_groups(imgs, hash_size=8, threshold=5):
    """
    Build groups of similar images using Union-Find algorithm.
    Images with pHash distance <= threshold are grouped together.
    
    Returns: list of groups, each group is a list of image paths
    """
    if not PHASH_AVAILABLE or len(imgs) == 0:
        # No grouping possible, each image is its own group
        return [[img] for img in imgs]
    
    # Compute hashes
    hashes = {}
    for img in imgs:
        h = compute_phash(img, hash_size=hash_size)
        if h is not None:
            hashes[img] = (h, int(h, 16))
    
    # Union-Find
    parent = {img: img for img in imgs}
    rank = {img: 0 for img in imgs}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    # Compare all pairs and union similar ones
    img_list = list(imgs)
    for i in range(len(img_list)):
        img1 = img_list[i]
        if img1 not in hashes:
            continue
        h1, int1 = hashes[img1]
        
        for j in range(i + 1, len(img_list)):
            img2 = img_list[j]
            if img2 not in hashes:
                continue
            h2, int2 = hashes[img2]
            
            dist = hamming_distance_int(int1, int2)
            if dist <= threshold:
                union(img1, img2)
    
    # Group by parent
    groups_dict = {}
    for img in imgs:
        root = find(img)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(img)
    
    return list(groups_dict.values())


def group_aware_split(class_to_imgs, val_split=0.1, test_split=0.2, 
                      hash_size=8, threshold=5, seed=42):
    """
    Split data by GROUPS instead of individual images.
    This ensures similar images always stay in the same split,
    preventing data leakage.
    
    Algorithm:
    1. For each class, group similar images together using pHash
    2. Shuffle groups (not images)
    3. Assign groups to train/val/test splits
    
    Returns: train_map, val_map, test_map
    """
    print("\n" + "=" * 80)
    print("[GROUP-AWARE SPLIT] Building train/val/test splits by image groups")
    print("=" * 80)
    
    train_map = {}
    val_map = {}
    test_map = {}
    
    total_groups = 0
    total_images = 0
    
    for cls, imgs in tqdm(class_to_imgs.items(), desc="Processing classes"):
        # Build similarity groups for this class
        groups = build_similarity_groups(imgs, hash_size=hash_size, threshold=threshold)
        
        n_groups = len(groups)
        n_images = len(imgs)
        total_groups += n_groups
        total_images += n_images
        
        # Calculate split sizes (by number of groups)
        n_test_groups = max(1, int(round(n_groups * test_split)))
        n_val_groups = max(1, int(round(n_groups * val_split)))
        n_train_groups = n_groups - n_test_groups - n_val_groups
        
        # Ensure we have at least 1 group for train
        if n_train_groups < 1:
            n_train_groups = 1
            n_val_groups = max(0, n_groups - n_train_groups - n_test_groups)
            if n_val_groups < 0:
                n_test_groups = max(0, n_groups - n_train_groups)
                n_val_groups = 0
        
        # Shuffle groups deterministically
        random.Random(hash(cls) & 0xffffffff ^ seed).shuffle(groups)
        
        # Assign groups to splits
        test_groups = groups[:n_test_groups]
        val_groups = groups[n_test_groups:n_test_groups + n_val_groups]
        train_groups = groups[n_test_groups + n_val_groups:]
        
        # Flatten groups to image lists
        train_map[cls] = [img for group in train_groups for img in group]
        val_map[cls] = [img for group in val_groups for img in group]
        test_map[cls] = [img for group in test_groups for img in group]
    
    # Statistics
    train_total = sum(len(imgs) for imgs in train_map.values())
    val_total = sum(len(imgs) for imgs in val_map.values())
    test_total = sum(len(imgs) for imgs in test_map.values())
    
    print(f"\n[INFO] Group-aware split completed:")
    print(f"  - Total classes: {len(class_to_imgs)}")
    print(f"  - Total images: {total_images}")
    print(f"  - Total groups: {total_groups}")
    print(f"  - Average images per group: {total_images / max(1, total_groups):.2f}")
    print(f"\n[INFO] Split sizes:")
    print(f"  - Train: {train_total} images ({100*train_total/max(1,total_images):.1f}%)")
    print(f"  - Val: {val_total} images ({100*val_total/max(1,total_images):.1f}%)")
    print(f"  - Test: {test_total} images ({100*test_total/max(1,total_images):.1f}%)")
    
    return train_map, val_map, test_map


def handle_leakage_major(class_to_imgs, val_split=0.1, test_split=0.2,
                          hash_size=8, threshold=5, seed=42):
    """
    Handle major leakage (>=5%):
    Rebuild train/val/test using group-aware splitting.
    
    Returns: train_map, val_map, test_map
    """
    print("\n" + "=" * 80)
    print("[LEAKAGE FIX] Major leakage detected (>=5%). Rebuilding with group-aware split...")
    print("=" * 80)
    
    train_map, val_map, test_map = group_aware_split(
        class_to_imgs, 
        val_split=val_split, 
        test_split=test_split,
        hash_size=hash_size,
        threshold=threshold,
        seed=seed
    )
    
    return train_map, val_map, test_map


def handle_data_leakage(train_map, val_map, test_map, class_to_imgs, all_class_imgs,
                        out_dir, val_split=0.1, test_split=0.2,
                        hash_size=8, threshold=5, max_iterations=3):
    """
    Main function to detect and handle data leakage.
    
    Strategy:
    - If leakage < 5%: Move leaked images to train, replace from unselected pool
    - If leakage >= 5%: Rebuild using group-aware split
    
    Returns: final train_map, val_map, test_map, leakage_report
    """
    print("\n" + "=" * 80)
    print("[DATA LEAKAGE HANDLER] Starting leakage detection and fixing...")
    print("=" * 80)
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'─' * 40}")
        print(f"[Iteration {iteration}/{max_iterations}]")
        print(f"{'─' * 40}")
        
        # Check for leakage
        leakage_result = check_data_leakage_phash(
            train_map, val_map, test_map, out_dir,
            hash_size=hash_size, threshold=threshold
        )
        
        if not leakage_result.get('leakage_found', False):
            print("\n✅ No data leakage detected. Dataset is clean!")
            return train_map, val_map, test_map, leakage_result
        
        # Calculate leakage percentage
        total_val = sum(len(imgs) for imgs in val_map.values())
        total_test = sum(len(imgs) for imgs in test_map.values())
        total_eval = total_val + total_test
        
        n_leaked = leakage_result.get('exact_cross_split', 0) + \
                   leakage_result.get('near_cross_split', 0)
        
        leakage_pct = (n_leaked / max(1, total_eval)) * 100
        
        print(f"\n[INFO] Leakage statistics:")
        print(f"  - Total evaluation images: {total_eval}")
        print(f"  - Leaked images: {n_leaked}")
        print(f"  - Leakage percentage: {leakage_pct:.2f}%")
        
        if leakage_pct < 5.0:
            # Minor leakage: fix by replacement
            print(f"\n[DECISION] Leakage < 5% → Applying minor fix (replacement strategy)")
            train_map, val_map, test_map, success = handle_leakage_minor(
                train_map, val_map, test_map, all_class_imgs,
                leakage_result, out_dir, threshold=threshold
            )
            
            if not success:
                print("\n⚠ Some replacements failed. May need to use group-aware split.")
        else:
            # Major leakage: rebuild with group-aware split
            print(f"\n[DECISION] Leakage >= 5% → Applying major fix (group-aware split)")
            train_map, val_map, test_map = handle_leakage_major(
                class_to_imgs,
                val_split=val_split,
                test_split=test_split,
                hash_size=hash_size,
                threshold=threshold
            )
    
    # Final check
    print(f"\n{'─' * 40}")
    print(f"[Final Check after {max_iterations} iterations]")
    print(f"{'─' * 40}")
    
    final_result = check_data_leakage_phash(
        train_map, val_map, test_map, out_dir,
        hash_size=hash_size, threshold=threshold
    )
    
    if final_result.get('leakage_found', False):
        print("\n⚠ WARNING: Some leakage still remains after fixing attempts.")
        print("  Consider manual review or increasing threshold.")
    else:
        print("\n✅ All leakage has been successfully resolved!")
    
    return train_map, val_map, test_map, final_result

# ------------------------------
# CLI / main orchestration
# ------------------------------
def main(args):
    # Timestamp start
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Collect images per class by user rules
    print("\n[STEP 1] Collecting images per class...")
    print("[STEP 1a] First, collecting ALL available images...")
    class_to_imgs, all_class_imgs = collect_images_per_class(args.data_root, max_per_class=args.max_per_class, verbose=True)

    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = sum(len(imgs) for imgs in class_to_imgs.values())
    print(f"\n[STEP 1b] Summary:")
    print(f"  - Total images available in dataset: {total_available}")
    print(f"  - Total images selected (with priority rules): {total_selected}")
    print(f"  - Images not selected: {total_available - total_selected}")

    # 2) Build loaders (without pseudo priors yet)
    print("\n[STEP 2] Building data loaders...")
    train_loader, val_loader, test_loader, class_to_idx, train_map, val_map, test_map = build_loaders(
        class_to_imgs, args.max_per_class, args.batch_size,
        val_split=args.val_split, test_split=args.test_split, num_workers=args.num_workers
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = sorted(class_to_idx.keys())
    n_classes = len(class_to_idx)

    print(f"[INFO] Number of classes: {n_classes}")
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples: {len(val_loader.dataset)}")
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")

    # Save dataset splits
    save_dataset_splits(args.out_dir, all_class_imgs, class_to_imgs, train_map, val_map, test_map, class_to_idx)

    # 2b) Check and handle data leakage using pHash
    print("\n[STEP 2b] Checking and handling data leakage using pHash...")
    train_map, val_map, test_map, leakage_result = handle_data_leakage(
        train_map, val_map, test_map, 
        class_to_imgs, all_class_imgs,
        args.out_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        hash_size=8, 
        threshold=5,
        max_iterations=3
    )
    
    # Update class_to_idx and rebuild loaders if data was modified
    classes = sorted(train_map.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Save updated dataset splits after leakage fix
    print("\n[INFO] Saving updated dataset splits after leakage fix...")
    save_dataset_splits(args.out_dir, all_class_imgs, class_to_imgs, train_map, val_map, test_map, class_to_idx)
    
    # Update counts
    print(f"\n[INFO] Updated split sizes after leakage fix:")
    print(f"  - Train: {sum(len(imgs) for imgs in train_map.values())} images")
    print(f"  - Val: {sum(len(imgs) for imgs in val_map.values())} images")
    print(f"  - Test: {sum(len(imgs) for imgs in test_map.values())} images")

    # 3) Instantiate backbone for pseudo label mining
    print("\n[STEP 3] Creating feature extractor for O1...")
    feat_model = timm.create_model('resnet18', pretrained=True, num_classes=0)

    class FeatureExtractor(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            f = self.backbone(x)
            if f.dim() == 4:
                f = f.mean(dim=(2, 3))
            return f

    feat_extractor = FeatureExtractor(feat_model).to(device)

    # 4) O1: Generate pseudo organ priors via clustering (TRAIN ONLY - NO LEAKAGE)
    print("\n[STEP 4 - O1] Pseudo organ mining (KMeans clustering on TRAIN only)...")
    print("[INFO] FIX: Clustering only on training data to prevent data leakage")
    
    # Step 4a: Fit KMeans on TRAIN data only
    priors_train, kmeans = generate_pseudo_organs_train_only(
        train_map, feat_extractor, device,
        n_clusters=args.n_org_clusters, batch_size=args.cluster_bs
    )
    
    # Step 4b: Apply pre-fitted KMeans to VAL data (no fitting, just transform)
    print("\n[STEP 4b] Applying KMeans to VAL data (transform only, no fitting)...")
    priors_val = apply_kmeans_to_split(
        val_map, kmeans, feat_extractor, device,
        n_clusters=args.n_org_clusters, batch_size=args.cluster_bs
    )
    
    # Step 4c: Apply pre-fitted KMeans to TEST data (no fitting, just transform)
    print("\n[STEP 4c] Applying KMeans to TEST data (transform only, no fitting)...")
    priors_test = apply_kmeans_to_split(
        test_map, kmeans, feat_extractor, device,
        n_clusters=args.n_org_clusters, batch_size=args.cluster_bs
    )
    
    # Merge all priors
    priors = {}
    priors.update(priors_train)
    priors.update(priors_val)
    priors.update(priors_test)

    organ_dim = args.n_org_clusters
    print(f"\n[INFO] Organ dimension: {organ_dim}")
    print(f"[INFO] Total organ priors: {len(priors)} (Train: {len(priors_train)}, Val: {len(priors_val)}, Test: {len(priors_test)})")

    # 5) Rebuild datasets with pseudo organ priors
    print("\n[STEP 5] Rebuilding datasets with organ priors...")
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    pseudo_org_tensor_map = {p: vec.astype(np.float32) for p, vec in priors.items()}

    train_ds = BigPlantsDataset(train_map, class_to_idx, transform=train_tf, pseudo_org=pseudo_org_tensor_map)
    val_ds = BigPlantsDataset(val_map, class_to_idx, transform=val_tf, pseudo_org=pseudo_org_tensor_map)
    test_ds = BigPlantsDataset(test_map, class_to_idx, transform=val_tf, pseudo_org=pseudo_org_tensor_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 6) Instantiate OrganAwareSwitchViT model
    print("\n[STEP 6] Creating Organ-Aware V-MoE model...")
    model = OrganAwareSwitchViT(
        vit_name=args.vit_name, n_classes=n_classes, organ_dim=organ_dim,
        n_experts=args.n_experts, d_ff_expert=args.d_ff_expert,
        top_k=args.top_k, pretrained=True
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # O7: Capacity scheduling
    capacity_schedule = np.linspace(args.capacity_initial, args.capacity_final, args.epochs)

    # Training loop
    print("\n[STEP 7] Starting training loop...")
    print("=" * 80)

    best_macro = -1.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_cls_loss': [],
        'train_aux_loss': [],
        'train_balance_loss': [],
        'train_time': [],
        'val_macro_f1': [],
        'val_micro_f1': [],
        'val_weighted_f1': [],
        'val_accuracy': [],
        'val_time': [],
        'capacity_factor': []
    }

    for epoch in range(1, args.epochs + 1):
        cf = float(capacity_schedule[epoch - 1])
        model.switch.capacity_factor = cf

        epoch_start = datetime.now()
        print(f"\n[Epoch {epoch}/{args.epochs}] Started at: {epoch_start.strftime('%H:%M:%S')}")
        print(f"Capacity factor: {cf:.3f}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            aux_weight=args.aux_weight, balance_weight=args.balance_weight,
            capacity_factor=cf, organ_dim=organ_dim,
            use_organmix=args.use_organmix, organmix_prob=args.organmix_prob
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, classes, phase='val')

        epoch_end = datetime.now()
        epoch_duration = (epoch_end - epoch_start).total_seconds()

        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_aux_loss'].append(train_metrics['aux_loss'])
        history['train_balance_loss'].append(train_metrics['balance_loss'])
        history['train_time'].append(train_metrics['time'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_micro_f1'].append(val_metrics['micro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_time'].append(val_metrics['time'])
        history['capacity_factor'].append(cf)

        print(f"\n[Epoch {epoch}] Results:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, Time: {train_metrics['time']:.1f}s")
        print(f"  Val   - Macro F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Time: {val_metrics['time']:.1f}s")
        print(f"  Epoch duration: {epoch_duration:.1f}s")
        print(f"  Finished at: {epoch_end.strftime('%H:%M:%S')}")

        # Save best model
        if val_metrics['macro_f1'] > best_macro:
            best_macro = val_metrics['macro_f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_macro,
                'class_to_idx': class_to_idx,
                'organ_dim': organ_dim,
                'n_experts': args.n_experts,
                'args': vars(args)
            }, os.path.join(args.out_dir, "best_model.pt"))
            print(f"  ★ New best model saved! (Macro F1: {best_macro:.4f})")

    # Save training history
    torch.save(history, os.path.join(args.out_dir, "training_history.pt"))
    print(f"\n[INFO] Training history saved to {args.out_dir}/training_history.pt")

    # Final evaluation on test set
    print("\n[STEP 8 - O8] Final evaluation on test set...")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load(os.path.join(args.out_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, classes, phase='test')

    print(f"\n[TEST RESULTS]")
    print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Micro F1:    {test_metrics['micro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")

    # Save classification report
    save_classification_report(test_metrics['classification_report'], classes, args.out_dir, phase='test')

    # Save confusion matrix
    save_confusion_matrix(test_metrics['confusion_matrix'], classes, args.out_dir, phase='test')

    # Print per-class F1 scores
    print("\n[Per-Class F1 Scores]")
    print("-" * 60)
    for i, cls in enumerate(classes):
        print(f"{cls:40s} F1: {test_metrics['per_f1'][i]:.4f}  "
              f"P: {test_metrics['per_precision'][i]:.4f}  "
              f"R: {test_metrics['per_recall'][i]:.4f}  "
              f"Support: {int(test_metrics['per_support'][i])}")

    # End timestamp
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print(f"[END] Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {total_duration/3600:.2f} hours ({total_duration:.0f} seconds)")
    print(f"[INFO] Best epoch: {best_epoch} with Macro F1: {best_macro:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organ-Aware V-MoE for BigPlants-100')

    # Data parameters
    parser.add_argument("--data_root", type=str, required=True,
                       help="path to dataset root containing 100 class folders")
    parser.add_argument("--out_dir", type=str, default="./outputs",
                       help="output directory for models and results")
    parser.add_argument("--max_per_class", type=int, default=100,
                       help="maximum images per class")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                       help="number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="learning rate")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.2,
                       help="test split ratio")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="number of data loader workers")

    # O1: Pseudo organ mining
    parser.add_argument("--n_org_clusters", type=int, default=5,
                       help="number of clusters for O1 pseudo organ mining")
    parser.add_argument("--cluster_bs", type=int, default=64,
                       help="batch size for clustering feature extraction")

    # Model architecture
    parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224",
                       help="ViT backbone model name")
    parser.add_argument("--n_experts", type=int, default=8,
                       help="number of experts in MoE")
    parser.add_argument("--d_ff_expert", type=int, default=1024,
                       help="hidden dimension of expert FFN")
    parser.add_argument("--top_k", type=int, default=1,
                       help="top-k experts to route to")

    # Loss weights
    parser.add_argument("--aux_weight", type=float, default=0.5,
                       help="weight for auxiliary organ loss")
    parser.add_argument("--balance_weight", type=float, default=0.01,
                       help="weight for expert balance loss")

    # O6: Augmentation
    parser.add_argument("--use_organmix", action='store_true', default=True,
                       help="use OrganMix augmentation")
    parser.add_argument("--organmix_prob", type=float, default=0.5,
                       help="probability of applying OrganMix")

    # O7: Capacity scheduling
    parser.add_argument("--capacity_initial", type=float, default=1.25,
                       help="initial capacity factor")
    parser.add_argument("--capacity_final", type=float, default=1.0,
                       help="final capacity factor")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
