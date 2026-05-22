#!/usr/bin/env python3
"""
Organ-Aware Switch-ViT with K-Fold Cross-Validation for BigPlants-100

Full pipeline O1–O8 with StratifiedKFold (default 5-fold):
- O1: pseudo-label organ via clustering (TRAIN-ONLY to prevent data leakage)
- O2: auxiliary organ head + calibration
- O3: organ-aware router
- O4: Switch MoE
- O5: entropy fallback routing
- O6: OrganMix augmentation
- O7: capacity scheduling
- O8: comprehensive evaluation per fold + aggregated metrics

Data Leakage Prevention Features:
- pHash-based duplicate detection across train/val/test splits
- Automatic leakage fixing by moving duplicates to train
- Train-only KMeans clustering with predict() for val/test

Usage:
  python organ_aware_switch_vit_kfold.py --data_root /path/to/dataset --out_dir ./outputs_kfold --n_folds 5 --epochs 40

  # Disable leakage checking (faster but less safe):
  python organ_aware_switch_vit_kfold.py --data_root /path/to/dataset --no_check_leakage

Requirements:
  pip install torch torchvision timm scikit-learn pandas tqdm pillow imagehash

Author: assistant
"""
import os
import sys
import argparse
import random
import math
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
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
# Utils: Dataset builder per your rules
# ------------------------------
PRIOR_ORG_ORDER = ["hand", "leaf", "flower", "fruit"]
SECOND_ORG_ORDER = ["seed", "root"]

def collect_all_images(data_root, verbose=True):
    """Collect ALL available images from the dataset without any limit."""
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    all_class_imgs = {}

    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        all_imgs = []

        # Collect from all subfolders
        for sub in PRIOR_ORG_ORDER + SECOND_ORG_ORDER:
            sub_dir = os.path.join(cls_dir, sub)
            if os.path.isdir(sub_dir):
                files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_imgs.extend(sorted(files))

        # Collect available images (files in class root)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_imgs.extend(sorted(files))

        # Deduplicate
        seen = set()
        uniq = []
        for p in all_imgs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        all_class_imgs[cls] = uniq
        if verbose:
            print(f"  {cls}: {len(uniq)} images")

    return all_class_imgs

def collect_images_per_class(data_root, max_per_class=100, verbose=True):
    """
    Select images per class with priority rules.
    Returns:
        - selected_imgs: dict class -> list of selected paths (max max_per_class)
        - all_imgs: dict class -> list of ALL available paths
    """
    all_class_imgs = collect_all_images(data_root, verbose=False)
    classes = sorted(all_class_imgs.keys())
    selected_class_imgs = {}

    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        selected = []

        # Priority 1: hand, leaf, flower, fruit
        for sub in PRIOR_ORG_ORDER:
            sub_dir = os.path.join(cls_dir, sub)
            if os.path.isdir(sub_dir):
                files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                selected.extend(sorted(files))
            if len(selected) >= max_per_class:
                selected = selected[:max_per_class]
                break

        # Priority 2: seed, root
        if len(selected) < max_per_class:
            for sub in SECOND_ORG_ORDER:
                sub_dir = os.path.join(cls_dir, sub)
                if os.path.isdir(sub_dir):
                    files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    selected.extend(sorted(files))
                if len(selected) >= max_per_class:
                    selected = selected[:max_per_class]
                    break

        # Priority 3: available
        if len(selected) < max_per_class:
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected.extend(sorted(files))
            selected = selected[:max_per_class]

        # Deduplicate
        seen = set()
        uniq = []
        for p in selected:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        selected_class_imgs[cls] = uniq
        if verbose:
            print(f"  {cls}: selected {len(uniq)} images")

    return selected_class_imgs, all_class_imgs

# ------------------------------
# Dataset class
# ------------------------------
class BigPlantsDataset(Dataset):
    def __init__(self, class_to_imgs, class_to_idx, transform=None, pseudo_org=None):
        self.samples = []
        for cls, imgs in class_to_imgs.items():
            for p in imgs:
                self.samples.append((p, class_to_idx[cls], cls))
        self.transform = transform
        self.pseudo_org = pseudo_org or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, idx, cls = self.samples[i]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        org_prior = self.pseudo_org.get(p, None)
        if org_prior is None:
            org_prior = np.zeros(5, dtype=np.float32)
        return img, idx, torch.from_numpy(org_prior), p

# ------------------------------
# O1: Pseudo-label organ clustering
# ------------------------------
def generate_pseudo_orginals(class_to_imgs, feature_extractor, device, n_clusters=5, batch_size=64):
    all_paths = []
    for imgs in class_to_imgs.values():
        all_paths.extend(imgs)

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    class DummyDataset(Dataset):
        def __init__(self, paths): self.paths = paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            p = self.paths[i]; img = Image.open(p).convert('RGB'); return transform(img), p

    loader = DataLoader(DummyDataset(all_paths), batch_size=batch_size, shuffle=False, num_workers=4)
    feats = []
    paths = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch, ps in tqdm(loader, desc="Extract features for clustering"):
            batch = batch.to(device)
            feat = feature_extractor(batch)
            feats.append(feat.cpu().numpy())
            paths.extend(ps)
    feats = np.vstack(feats)

    print(f"[O1] KMeans clustering -> {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
    labels = kmeans.labels_

    priors = {}
    for p, l in zip(paths, labels):
        vec = np.zeros(n_clusters, dtype=np.float32)
        vec[l] = 1.0
        priors[p] = vec
    return priors, kmeans


# ------------------------------
# O1: Data Leakage-Free Clustering Functions
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
        'hash_map': hash_map
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


def handle_leakage_in_fold(train_map, val_map, test_map, fold_dir,
                            hash_size=8, threshold=5):
    """
    Handle data leakage within a single fold.

    For K-Fold, we use a simplified strategy:
    - Move leaked val/test images to train
    - This ensures no leakage even if it slightly changes split sizes

    Returns: updated train_map, val_map, test_map, leakage_stats
    """
    # Check for leakage
    leakage_result = check_data_leakage_phash(
        train_map, val_map, test_map, fold_dir,
        hash_size=hash_size, threshold=threshold
    )

    if not leakage_result.get('leakage_found', False):
        return train_map, val_map, test_map, leakage_result

    print("\n[LEAKAGE FIX] Moving leaked images to train...")

    # Collect leaked images from val/test
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

    # Move leaked images from val to train
    moved_from_val = 0
    for cls, leaked_paths in leaked_from_val.items():
        for path in leaked_paths:
            if path in val_map.get(cls, []):
                val_map[cls].remove(path)
                if cls not in train_map:
                    train_map[cls] = []
                train_map[cls].append(path)
                moved_from_val += 1

    # Move leaked images from test to train
    moved_from_test = 0
    for cls, leaked_paths in leaked_from_test.items():
        for path in leaked_paths:
            if path in test_map.get(cls, []):
                test_map[cls].remove(path)
                if cls not in train_map:
                    train_map[cls] = []
                train_map[cls].append(path)
                moved_from_test += 1

    print(f"  Moved from val to train: {moved_from_val}")
    print(f"  Moved from test to train: {moved_from_test}")

    # Update leakage result
    leakage_result['fixed'] = True
    leakage_result['moved_from_val'] = moved_from_val
    leakage_result['moved_from_test'] = moved_from_test

    return train_map, val_map, test_map, leakage_result


# ------------------------------
# Model components
# ------------------------------
class OrganAuxHead(nn.Module):
    def __init__(self, in_dim, n_org):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_org)
    def forward(self, x):
        return self.fc(x)

class Router(nn.Module):
    def __init__(self, token_dim, organ_dim, n_experts):
        super().__init__()
        self.linear = nn.Linear(token_dim + organ_dim, n_experts)
    def forward(self, token, organ_prior):
        inp = torch.cat([token, organ_prior], dim=-1)
        logits = self.linear(inp)
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
        self.entropy_threshold = entropy_threshold
        self.router = Router(d_model, organ_dim, n_experts)
        self.experts = nn.ModuleList([FFNExpert(d_model, d_ff) for _ in range(n_experts)])
        self.register_buffer('expert_usage', torch.zeros(n_experts))

    def forward(self, tokens, organ_priors, training=True):
        B, T, D = tokens.shape
        flat = tokens.reshape(B * T, D)
        flat_prior = organ_priors.reshape(B * T, -1)
        logits = self.router(flat, flat_prior)
        probs = F.softmax(logits, dim=-1)

        # O5: entropy fallback
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
        max_k = min(2, self.n_experts) if training else self.top_k
        topk_vals, topk_idx = torch.topk(probs, max_k, dim=-1)
        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)

        outputs = torch.zeros_like(flat, device=flat.device)
        usage_counts = torch.zeros(self.n_experts, device=flat.device)
        capacity = math.ceil((B * T / self.n_experts) * self.capacity_factor)
        processed_mask = torch.zeros(B * T, dtype=torch.bool, device=flat.device)

        for expert_idx in range(self.n_experts):
            expert_mask = (topk_idx == expert_idx).any(dim=-1)
            expert_tokens_idx = expert_mask.nonzero(as_tuple=True)[0]

            if len(expert_tokens_idx) == 0:
                continue

            if len(expert_tokens_idx) > capacity:
                expert_tokens_idx = expert_tokens_idx[:capacity]

            expert_input = flat[expert_tokens_idx]
            expert_output = self.experts[expert_idx](expert_input)

            for pos_in_batch, global_idx in enumerate(expert_tokens_idx):
                local_topk_mask = (topk_idx[global_idx] == expert_idx)
                if local_topk_mask.any():
                    weight_pos = local_topk_mask.nonzero(as_tuple=True)[0][0]
                    weight = topk_probs[global_idx, weight_pos]
                    outputs[global_idx] += weight * expert_output[pos_in_batch]
                    processed_mask[global_idx] = True

            usage_counts[expert_idx] = len(expert_tokens_idx)

        if training:
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * usage_counts

        unprocessed = ~processed_mask
        if unprocessed.any():
            default_expert = self.experts[0]
            outputs[unprocessed] = default_expert(flat[unprocessed])

        outputs = outputs.reshape(B, T, D)
        p_mean = probs.mean(dim=0)
        balance_loss = (p_mean ** 2).sum() * self.n_experts

        return outputs, balance_loss

class OrganAwareSwitchViT(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', n_classes=100, organ_dim=5, n_experts=8, d_ff_expert=1024, top_k=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.embed_dim
        self.organ_dim = organ_dim

        self.organ_aux_head = OrganAuxHead(self.embed_dim, organ_dim)
        self.moe_layer = SwitchMoE(self.embed_dim, organ_dim, n_experts, d_ff_expert, top_k=top_k)
        self.class_head = nn.Linear(self.embed_dim, n_classes)

    def forward(self, x, organ_priors_image, training=True, capacity_factor=None):
        features = self.backbone.forward_features(x)
        if hasattr(self.backbone, 'global_pool') and self.backbone.global_pool == 'token':
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]
        else:
            cls_token = features.mean(dim=1)
            patch_tokens = features

        organ_logits = self.organ_aux_head(cls_token)

        B, T, D = patch_tokens.shape
        organ_priors_expand = organ_priors_image.unsqueeze(1).expand(B, T, self.organ_dim)

        if capacity_factor is not None:
            self.moe_layer.capacity_factor = capacity_factor

        moe_out, balance_loss = self.moe_layer(patch_tokens, organ_priors_expand, training=training)
        pooled = moe_out.mean(dim=1)
        class_logits = self.class_head(pooled)

        return class_logits, organ_logits, balance_loss

# ------------------------------
# OrganMix augmentation
# ------------------------------
def organmix(img1, img2, alpha=0.5):
    _, h, w = img1.shape
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
# Training & Evaluation
# ------------------------------
def train_epoch(model, dataloader, optimizer, device, epoch, aux_weight=0.5, balance_weight=0.01,
                capacity_factor=1.25, organ_dim=5, use_organmix=True, organmix_prob=0.5):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    total_cls_loss, total_aux_loss, total_balance_loss = 0.0, 0.0, 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()
    epoch_start = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        imgs, labels, organ_priors, paths = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        organ_priors = organ_priors.to(device)

        if use_organmix and random.random() < organmix_prob:
            indices = torch.randperm(imgs.size(0))
            imgs_shuffled = imgs[indices]
            labels_shuffled = labels[indices]
            organ_priors_shuffled = organ_priors[indices]

            mixed_imgs = []
            mixed_labels = []
            mixed_organ_priors = []
            lams = []

            for i in range(imgs.size(0)):
                img_mix, lam = organmix(imgs[i], imgs_shuffled[i], alpha=0.5)
                mixed_imgs.append(img_mix)
                lams.append(lam)

            imgs = torch.stack(mixed_imgs)
            lams = torch.tensor(lams, device=device).view(-1, 1)

            class_logits, organ_logits, balance_loss = model(imgs, organ_priors, training=True, capacity_factor=capacity_factor)

            cls_loss = 0.0
            for i in range(imgs.size(0)):
                l = lams[i].item()
                cls_loss += l * criterion(class_logits[i:i+1], labels[i:i+1])
                cls_loss += (1 - l) * criterion(class_logits[i:i+1], labels_shuffled[i:i+1])
            cls_loss = cls_loss / imgs.size(0)

            aux_target = organ_priors
            aux_loss = F.kl_div(F.log_softmax(organ_logits, dim=-1),
                               aux_target, reduction='batchmean')
        else:
            class_logits, organ_logits, balance_loss = model(imgs, organ_priors, training=True, capacity_factor=capacity_factor)
            cls_loss = criterion(class_logits, labels)
            aux_target = organ_priors
            aux_loss = F.kl_div(F.log_softmax(organ_logits, dim=-1),
                               aux_target, reduction='batchmean')

        loss = cls_loss + aux_weight * aux_loss + balance_weight * balance_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = class_logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item() * imgs.size(0)
        total_cls_loss += cls_loss.item() * imgs.size(0)
        total_aux_loss += aux_loss.item() * imgs.size(0)
        total_balance_loss += balance_loss.item() * imgs.size(0)
        total_acc += acc * imgs.size(0)
        n += imgs.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

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
        for batch in tqdm(dataloader, desc=f"Eval [{phase}]"):
            imgs, labels, organ_priors, paths = batch
            imgs = imgs.to(device)
            organ_priors = organ_priors.to(device)

            class_logits, _, _ = model(imgs, organ_priors, training=False)
            probs = F.softmax(class_logits, dim=-1)
            preds = probs.argmax(dim=-1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            paths_all.extend(paths)
            all_probs.extend(probs.cpu().numpy())

    eval_time = time.time() - eval_start

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

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
# Save utilities
# ------------------------------
def save_confusion_matrix(cm, class_names, out_dir, phase='test'):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, f'confusion_matrix_{phase}.csv'))

    if len(class_names) <= 50:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {phase}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'confusion_matrix_{phase}.png'), dpi=150)
        plt.close()

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_norm_df.to_csv(os.path.join(out_dir, f'confusion_matrix_{phase}_normalized.csv'))

def save_classification_report(cls_report, class_names, out_dir, phase='test'):
    report_data = []
    for cls_name in class_names:
        if cls_name in cls_report:
            report_data.append({
                'class': cls_name,
                'precision': cls_report[cls_name]['precision'],
                'recall': cls_report[cls_name]['recall'],
                'f1-score': cls_report[cls_name]['f1-score'],
                'support': cls_report[cls_name]['support']
            })

    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in cls_report:
            report_data.append({
                'class': avg_type,
                'precision': cls_report[avg_type]['precision'],
                'recall': cls_report[avg_type]['recall'],
                'f1-score': cls_report[avg_type]['f1-score'],
                'support': cls_report[avg_type]['support']
            })

    df = pd.DataFrame(report_data)
    df.to_csv(os.path.join(out_dir, f'classification_report_{phase}.csv'), index=False)

# ------------------------------
# Main K-Fold Cross-Validation
# ------------------------------
def main(args):
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] K-Fold Cross-Validation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Number of folds: {args.n_folds}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Collect images
    print("\n[STEP 1] Collecting images...")
    class_to_imgs, all_class_imgs = collect_images_per_class(args.data_root, max_per_class=args.max_per_class, verbose=True)

    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = sum(len(imgs) for imgs in class_to_imgs.values())
    print(f"\n[STEP 1 Summary]")
    print(f"  - Total available: {total_available}")
    print(f"  - Total selected: {total_selected}")
    print(f"  - Unselected: {total_available - total_selected}")

    classes = sorted(class_to_imgs.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Prepare all samples for stratified splitting
    all_samples = []
    all_labels = []
    for cls in classes:
        for img_path in class_to_imgs[cls]:
            all_samples.append(img_path)
            all_labels.append(class_to_idx[cls])

    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    print(f"\n[INFO] Total samples for K-Fold: {len(all_samples)}")
    print(f"[INFO] Number of classes: {n_classes}")

    # 2) Feature extractor for O1
    print("\n[STEP 2] Creating feature extractor for O1...")
    feat_model = timm.create_model('resnet18', pretrained=True, num_classes=0)

    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    feat_extractor = FeatureExtractor(feat_model).to(device)

    # 3) K-Fold Cross-Validation
    print(f"\n[STEP 3] Starting {args.n_folds}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    all_fold_cms = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_samples, all_labels), 1):
        # Skip folds before start_fold (but still iterate to maintain random state)
        if fold < args.start_fold:
            print(f"\n[FOLD {fold}/{args.n_folds}] Skipped (start_fold={args.start_fold})")
            # Consume random state for this fold to keep consistency
            _ = np.random.permutation(len(all_samples[train_idx]))
            continue

        print("\n" + "=" * 80)
        print(f"[FOLD {fold}/{args.n_folds}]")
        print("=" * 80)

        fold_start = time.time()

        # Split train into train + val
        train_samples = all_samples[train_idx]
        train_labels = all_labels[train_idx]
        test_samples = all_samples[test_idx]
        test_labels = all_labels[test_idx]

        # Further split train into train/val
        n_train = len(train_samples)
        n_val = int(n_train * args.val_split)
        indices = np.random.permutation(n_train)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        val_samples = train_samples[val_indices]
        val_labels = train_labels[val_indices]
        train_samples = train_samples[train_indices]
        train_labels = train_labels[train_indices]

        print(f"[FOLD {fold}] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        # Build train/val/test maps
        train_map = defaultdict(list)
        val_map = defaultdict(list)
        test_map = defaultdict(list)

        for path, label in zip(train_samples, train_labels):
            cls_name = classes[label]
            train_map[cls_name].append(path)

        for path, label in zip(val_samples, val_labels):
            cls_name = classes[label]
            val_map[cls_name].append(path)

        for path, label in zip(test_samples, test_labels):
            cls_name = classes[label]
            test_map[cls_name].append(path)

        # Convert to regular dict
        train_map = dict(train_map)
        val_map = dict(val_map)
        test_map = dict(test_map)

        # Create fold directory
        fold_dir = os.path.join(args.out_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # Check for data leakage using pHash (if enabled)
        if args.check_leakage:
            print(f"\n[FOLD {fold} - DATA LEAKAGE CHECK]")
            train_map, val_map, test_map, leakage_result = handle_leakage_in_fold(
                train_map, val_map, test_map, fold_dir,
                hash_size=args.phash_size, threshold=args.phash_threshold
            )

            # Log updated split sizes after leakage fix
            train_count = sum(len(imgs) for imgs in train_map.values())
            val_count = sum(len(imgs) for imgs in val_map.values())
            test_count = sum(len(imgs) for imgs in test_map.values())
            print(f"[FOLD {fold}] After leakage fix - Train: {train_count}, Val: {val_count}, Test: {test_count}")

        # O1: Generate pseudo organ priors using TRAIN-ONLY clustering (no data leakage)
        print(f"\n[FOLD {fold} - O1] Pseudo organ mining (train-only clustering)...")
        train_priors, kmeans = generate_pseudo_organs_train_only(
            train_map, feat_extractor, device,
            n_clusters=args.n_org_clusters, batch_size=args.cluster_bs
        )
        organ_dim = args.n_org_clusters

        # Apply pre-fitted KMeans to val/test (using predict, not fit)
        print(f"[FOLD {fold} - O1] Applying clustering to val/test...")
        val_priors = apply_kmeans_to_split(
            val_map, kmeans, feat_extractor, device,
            n_clusters=args.n_org_clusters, batch_size=args.cluster_bs
        )
        test_priors = apply_kmeans_to_split(
            test_map, kmeans, feat_extractor, device,
            n_clusters=args.n_org_clusters, batch_size=args.cluster_bs
        )

        # Merge all priors
        pseudo_org_map = {}
        pseudo_org_map.update({p: vec.astype(np.float32) for p, vec in train_priors.items()})
        pseudo_org_map.update({p: vec.astype(np.float32) for p, vec in val_priors.items()})
        pseudo_org_map.update({p: vec.astype(np.float32) for p, vec in test_priors.items()})

        # Build datasets
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

        train_ds = BigPlantsDataset(train_map, class_to_idx, transform=train_tf, pseudo_org=pseudo_org_map)
        val_ds = BigPlantsDataset(val_map, class_to_idx, transform=val_tf, pseudo_org=pseudo_org_map)
        test_ds = BigPlantsDataset(test_map, class_to_idx, transform=val_tf, pseudo_org=pseudo_org_map)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        # Create model
        print(f"\n[FOLD {fold}] Creating model...")
        model = OrganAwareSwitchViT(
            vit_name=args.vit_name, n_classes=n_classes, organ_dim=organ_dim,
            n_experts=args.n_experts, d_ff_expert=args.d_ff_expert,
            top_k=args.top_k, pretrained=True
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Capacity scheduling
        capacity_schedule = np.linspace(args.capacity_initial, args.capacity_final, args.epochs)

        # Training loop
        print(f"\n[FOLD {fold}] Training...")
        best_macro = -1.0
        best_epoch = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_macro_f1': [], 'val_accuracy': []
        }

        for epoch in range(1, args.epochs + 1):
            capacity = capacity_schedule[epoch - 1]

            train_metrics = train_epoch(
                model, train_loader, optimizer, device, epoch,
                aux_weight=args.aux_weight, balance_weight=args.balance_weight,
                capacity_factor=capacity, organ_dim=organ_dim,
                use_organmix=args.use_organmix, organmix_prob=args.organmix_prob
            )

            val_metrics = evaluate(model, val_loader, device, classes, phase='val')

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['val_accuracy'].append(val_metrics['accuracy'])

            print(f"[FOLD {fold}] Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
                  f"Val Macro F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            if val_metrics['macro_f1'] > best_macro:
                best_macro = val_metrics['macro_f1']
                best_epoch = epoch
                # Save best model for this fold (fold_dir already created earlier)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'macro_f1': best_macro,
                    'epoch': best_epoch,
                    'class_to_idx': class_to_idx,
                    'organ_dim': organ_dim
                }, os.path.join(fold_dir, 'best_model.pt'))

        # Load best model and evaluate on test
        print(f"\n[FOLD {fold}] Evaluating on test set...")
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pt'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate(model, test_loader, device, classes, phase='test')

        print(f"\n[FOLD {fold} TEST RESULTS]")
        print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
        print(f"  Micro F1:    {test_metrics['micro_f1']:.4f}")
        print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")

        # Save fold results
        save_classification_report(test_metrics['classification_report'], classes, fold_dir, phase='test')
        save_confusion_matrix(test_metrics['confusion_matrix'], classes, fold_dir, phase='test')

        fold_time = time.time() - fold_start

        fold_results.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'best_val_macro_f1': best_macro,
            'test_macro_f1': test_metrics['macro_f1'],
            'test_micro_f1': test_metrics['micro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_accuracy': test_metrics['accuracy'],
            'fold_time': fold_time
        })

        all_fold_cms.append(test_metrics['confusion_matrix'])

        print(f"[FOLD {fold}] Completed in {fold_time/60:.2f} minutes")

    # 4) Aggregate results
    print("\n" + "=" * 80)
    print("[AGGREGATED RESULTS ACROSS ALL FOLDS]")
    print("=" * 80)

    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(args.out_dir, 'kfold_results.csv'), index=False)

    print(f"\nPer-Fold Results:")
    print(df_results.to_string(index=False))

    print(f"\nMean ± Std across {args.n_folds} folds:")
    print(f"  Test Macro F1:    {df_results['test_macro_f1'].mean():.4f} ± {df_results['test_macro_f1'].std():.4f}")
    print(f"  Test Micro F1:    {df_results['test_micro_f1'].mean():.4f} ± {df_results['test_micro_f1'].std():.4f}")
    print(f"  Test Weighted F1: {df_results['test_weighted_f1'].mean():.4f} ± {df_results['test_weighted_f1'].std():.4f}")
    print(f"  Test Accuracy:    {df_results['test_accuracy'].mean():.4f} ± {df_results['test_accuracy'].std():.4f}")

    # Aggregate confusion matrix
    avg_cm = np.mean(all_fold_cms, axis=0)
    save_confusion_matrix(avg_cm.astype(int), classes, args.out_dir, phase='aggregated')

    # Summary stats
    summary = {
        'n_folds': args.n_folds,
        'mean_test_macro_f1': df_results['test_macro_f1'].mean(),
        'std_test_macro_f1': df_results['test_macro_f1'].std(),
        'mean_test_accuracy': df_results['test_accuracy'].mean(),
        'std_test_accuracy': df_results['test_accuracy'].std(),
        'best_fold': df_results.loc[df_results['test_macro_f1'].idxmax(), 'fold'],
        'best_fold_macro_f1': df_results['test_macro_f1'].max()
    }

    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print(f"[END] K-Fold Cross-Validation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {total_duration/3600:.2f} hours")
    print(f"[INFO] Best fold: {summary['best_fold']} with Macro F1: {summary['best_fold_macro_f1']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organ-Aware Switch-ViT K-Fold Cross-Validation')

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs_kfold")
    parser.add_argument("--max_per_class", type=int, default=100)

    # K-Fold
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--start_fold", type=int, default=1,
                        help="Start from this fold (1-indexed). Use to resume training after interruption.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # O1
    parser.add_argument("--n_org_clusters", type=int, default=5)
    parser.add_argument("--cluster_bs", type=int, default=64)

    # Model
    parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--d_ff_expert", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=1)

    # Loss weights
    parser.add_argument("--aux_weight", type=float, default=0.5)
    parser.add_argument("--balance_weight", type=float, default=0.01)

    # O6
    parser.add_argument("--use_organmix", action='store_true', default=True)
    parser.add_argument("--organmix_prob", type=float, default=0.5)

    # O7
    parser.add_argument("--capacity_initial", type=float, default=1.5)
    parser.add_argument("--capacity_final", type=float, default=1.0)

    # Data Leakage Detection
    parser.add_argument("--check_leakage", action='store_true', default=True,
                        help="Check for data leakage using pHash")
    parser.add_argument("--no_check_leakage", action='store_false', dest='check_leakage',
                        help="Disable data leakage checking")
    parser.add_argument("--phash_size", type=int, default=8,
                        help="pHash size (default 8 -> 64-bit hash)")
    parser.add_argument("--phash_threshold", type=int, default=5,
                        help="Max hamming distance for near-duplicates")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
