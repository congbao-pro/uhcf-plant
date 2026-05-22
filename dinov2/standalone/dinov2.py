import os
import random
import argparse
from collections import defaultdict
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
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
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

# pHash duplicate detection
try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("[WARNING] imagehash not installed. pHash duplicate detection disabled.")
    print("          Install with: pip install imagehash")

# ──────────────────────────────────────────────
# Utils: Dataset builder per priority rules
# ──────────────────────────────────────────────
PRIOR_ORG_ORDER = ["hand", "leaf", "flower", "fruit"]
SECOND_ORG_ORDER = ["seed", "root"]


def collect_all_images(data_root, verbose=True):
    """Collect ALL available images from the dataset without any limit."""
    classes = sorted([d for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d))])
    all_class_imgs = {}
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        all_imgs = []
        for sub in PRIOR_ORG_ORDER + SECOND_ORG_ORDER:
            subdir = os.path.join(cls_dir, sub)
            if os.path.isdir(subdir):
                files = [os.path.join(subdir, f) for f in os.listdir(subdir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_imgs.extend(sorted(files))
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_imgs.extend(sorted(files))
        seen = set(); uniq = []
        for p in all_imgs:
            if p not in seen: seen.add(p); uniq.append(p)
        all_class_imgs[cls] = uniq
        if verbose:
            print(f"[collect_all] class={cls} -> {len(uniq)} total images available")
    return all_class_imgs


def collect_images_per_class(data_root, max_per_class=100, verbose=True):
    """Select images per class with priority rules, capped at max_per_class."""
    all_class_imgs = collect_all_images(data_root, verbose=False)
    classes = sorted(all_class_imgs.keys())
    selected_class_imgs = {}
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        selected = []
        for sub in PRIOR_ORG_ORDER:
            subdir = os.path.join(cls_dir, sub)
            if os.path.isdir(subdir):
                files = [os.path.join(subdir, f) for f in os.listdir(subdir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                selected.extend(sorted(files))
                if len(selected) >= max_per_class:
                    selected = selected[:max_per_class]; break
        if len(selected) < max_per_class:
            for sub in SECOND_ORG_ORDER:
                subdir = os.path.join(cls_dir, sub)
                if os.path.isdir(subdir):
                    files = [os.path.join(subdir, f) for f in os.listdir(subdir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    selected.extend(sorted(files))
                    if len(selected) >= max_per_class:
                        selected = selected[:max_per_class]; break
        if len(selected) < max_per_class:
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected.extend(sorted(files))
            if len(selected) >= max_per_class:
                selected = selected[:max_per_class]
        seen = set(); uniq = []
        for p in selected:
            if p not in seen: seen.add(p); uniq.append(p)
        selected_class_imgs[cls] = uniq
        if verbose:
            total_available = len(all_class_imgs[cls])
            print(f"[select] class={cls} -> Selected: {len(uniq)}/{total_available} "
                  f"(Unselected: {total_available - len(uniq)})")
    return selected_class_imgs, all_class_imgs


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class BigPlantsDataset(Dataset):
    def __init__(self, class_to_imgs, class_to_idx, transform=None):
        self.samples = []
        for cls, imgs in class_to_imgs.items():
            idx = class_to_idx[cls]
            for p in imgs:
                self.samples.append((p, idx, cls))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, idx, cls = self.samples[i]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)
        return img_t, idx, p


# ──────────────────────────────────────────────
# DINOv2 Classifier Model
# ──────────────────────────────────────────────
class DINOv2Classifier(nn.Module):
    """DINOv2 backbone (frozen by default) + linear classification head."""

    def __init__(self, model_name='dinov2_vitb14', n_classes=100,
                 pretrained=True, freeze_backbone=True, dropout=0.1):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Load DINOv2 backbone via torch.hub
        print(f"[MODEL] Loading DINOv2 backbone: {model_name}")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)

        # Probe embedding dimension
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone(dummy)
        self.embed_dim = feat.shape[-1]
        print(f"[MODEL] DINOv2 embedding dim: {self.embed_dim}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[MODEL] Backbone frozen (only classifier head is trainable)")
        else:
            print("[MODEL] Backbone unfrozen (full fine-tuning)")

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, n_classes)
        )

    def forward(self, x):
        """Forward pass. Returns logits (B, n_classes)."""
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ──────────────────────────────────────────────
# Train / Val / Test split
# ──────────────────────────────────────────────
def build_loaders(class_to_imgs, batch_size, val_split=0.1, test_split=0.2,
                  num_workers=4, seed=42):
    classes = sorted(list(class_to_imgs.keys()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_map, val_map, test_map = {}, {}, {}

    for c, imgs in class_to_imgs.items():
        n = len(imgs)
        n_test = int(math.ceil(test_split * n))
        n_val = int(math.ceil(val_split * n))
        random.Random(hash(c) & 0xffffffff).shuffle(imgs)
        test = imgs[:n_test]
        val = imgs[n_test:n_test + n_val]
        train = imgs[n_test + n_val:]
        train_map[c] = train
        val_map[c] = val
        test_map[c] = test

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

    train_ds = BigPlantsDataset(train_map, class_to_idx, transform=train_tf)
    val_ds = BigPlantsDataset(val_map, class_to_idx, transform=val_tf)
    test_ds = BigPlantsDataset(test_map, class_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, class_to_idx, train_map, val_map, test_map


# ──────────────────────────────────────────────
# Save dataset splits to CSV
# ──────────────────────────────────────────────
def save_dataset_splits(out_dir, all_class_imgs, class_to_imgs,
                        train_map, val_map, test_map, class_to_idx):
    selected_images = set()
    for cls_map in [train_map, val_map, test_map]:
        for imgs in cls_map.values():
            selected_images.update(imgs)

    # dataset_selected.csv
    with open(os.path.join(out_dir, 'dataset_selected.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['class_name', 'class_idx', 'image_path', 'split'])
        for cls, imgs in train_map.items():
            for img in imgs: w.writerow([cls, class_to_idx[cls], img, 'train'])
        for cls, imgs in val_map.items():
            for img in imgs: w.writerow([cls, class_to_idx[cls], img, 'val'])
        for cls, imgs in test_map.items():
            for img in imgs: w.writerow([cls, class_to_idx[cls], img, 'test'])

    # dataset_unselected.csv
    with open(os.path.join(out_dir, 'dataset_unselected.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['class_name', 'class_idx', 'image_path'])
        for cls, all_imgs in all_class_imgs.items():
            for img in all_imgs:
                if img not in selected_images:
                    w.writerow([cls, class_to_idx[cls], img])

    # train.csv / val.csv / test.csv
    for split_name, split_map in [('train', train_map), ('val', val_map), ('test', test_map)]:
        with open(os.path.join(out_dir, f'{split_name}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['class_name', 'class_idx', 'image_path'])
            for cls, imgs in split_map.items():
                for img in imgs: w.writerow([cls, class_to_idx[cls], img])

    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = len(selected_images)
    print(f"[INFO] Saved dataset splits to {out_dir}")
    print(f"  - Total available: {total_available} images")
    print(f"  - Selected: {total_selected} images")
    print(f"  - Unselected: {total_available - total_selected} images")


# ──────────────────────────────────────────────
# Confusion matrix & classification report
# ──────────────────────────────────────────────
def save_confusion_matrix(cm, class_names, out_dir, phase='test'):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, f'confusion_matrix_{phase}.csv'))
    if len(class_names) <= 50:
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(f'Confusion Matrix - {phase.upper()}')
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
            m = cls_report[cls_name]
            report_data.append({'class': cls_name, 'precision': m['precision'],
                                'recall': m['recall'], 'f1-score': m['f1-score'],
                                'support': m['support']})
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in cls_report:
            m = cls_report[avg_type]
            report_data.append({'class': avg_type, 'precision': m['precision'],
                                'recall': m['recall'], 'f1-score': m['f1-score'],
                                'support': m['support']})
    df = pd.DataFrame(report_data)
    df.to_csv(os.path.join(out_dir, f'classification_report_{phase}.csv'), index=False)
    print(f"[INFO] Saved classification report to {out_dir}/classification_report_{phase}.csv")


# ──────────────────────────────────────────────
# pHash Data Leakage Detection & Fixing
# ──────────────────────────────────────────────
def compute_phash(img_path, hash_size=8):
    if not PHASH_AVAILABLE:
        return None
    try:
        img = Image.open(img_path).convert('RGB')
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception as e:
        print(f"[WARNING] Could not compute pHash for {img_path}: {e}")
        return None


def compute_phash_for_paths(paths, hash_size=8):
    if not PHASH_AVAILABLE:
        return {}
    hash_map = {}
    for path in tqdm(paths, desc="Computing pHash"):
        h = compute_phash(path, hash_size=hash_size)
        if h is not None:
            hash_map[path] = h
    return hash_map


def check_data_leakage_phash(train_map, val_map, test_map, out_dir,
                              hash_size=8, threshold=5):
    if not PHASH_AVAILABLE:
        print("[WARNING] imagehash not available. Skipping pHash leakage check.")
        return {'status': 'skipped', 'reason': 'imagehash not installed'}

    print("\n" + "=" * 80)
    print("[DATA LEAKAGE CHECK] Using pHash to detect duplicate images across splits")
    print("=" * 80)

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
    print(f"  - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    print("\n[STEP 1] Computing pHash for all images...")
    hash_map = {}
    hash_to_paths = {}
    for path, split, cls in tqdm(all_items, desc="Computing pHash"):
        h = compute_phash(path, hash_size=hash_size)
        if h is not None:
            hash_map[path] = (h, split, cls)
            if h not in hash_to_paths:
                hash_to_paths[h] = []
            hash_to_paths[h].append((path, split, cls))
    print(f"[INFO] Successfully computed {len(hash_map)} hashes")

    print("\n[STEP 2] Finding EXACT duplicates (same pHash)...")
    exact_duplicates = []
    exact_cross_split = []
    for h, items in hash_to_paths.items():
        if len(items) > 1:
            splits = set(item[1] for item in items)
            if len(splits) > 1:
                exact_cross_split.append({'hash': h, 'items': items, 'splits': list(splits)})
            exact_duplicates.append({'hash': h, 'count': len(items), 'items': items})

    print(f"\n[STEP 3] Finding NEAR duplicates (hamming distance <= {threshold})...")
    near_duplicates = []
    near_cross_split = []
    hash_bits = hash_size * hash_size
    prefix_bits = min(16, hash_bits)
    shift = hash_bits - prefix_bits
    buckets = {}
    for path, (h, split, cls) in hash_map.items():
        try:
            int_hash = int(h, 16)
            prefix = int_hash >> shift
            if prefix not in buckets: buckets[prefix] = []
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
                if pair_key in checked_pairs: continue
                checked_pairs.add(pair_key)
                dist = (int1 ^ int2).bit_count()
                if 0 < dist <= threshold:
                    record = {'path1': p1, 'split1': s1, 'class1': c1,
                              'path2': p2, 'split2': s2, 'class2': c2, 'distance': dist}
                    near_duplicates.append(record)
                    if s1 != s2:
                        near_cross_split.append(record)

    leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0
    print("\n" + "=" * 80)
    print("[DATA LEAKAGE CHECK] RESULTS")
    print("=" * 80)
    print(f"\n📊 SUMMARY:")
    print(f"  Total exact duplicate groups: {len(exact_duplicates)}")
    print(f"  Total near-duplicate pairs: {len(near_duplicates)}")
    if leakage_found:
        print(f"  🚨 CROSS-SPLIT EXACT DUPLICATES: {len(exact_cross_split)} groups")
        print(f"  🚨 CROSS-SPLIT NEAR DUPLICATES: {len(near_cross_split)} pairs")
        print(f"  ⚠️  DATA LEAKAGE DETECTED! ⚠️")
    else:
        print(f"  ✅ NO CROSS-SPLIT DUPLICATES FOUND")
        print(f"  ✅ No data leakage detected between train/val/test splits")

    # Save report
    report_path = os.path.join(out_dir, 'data_leakage_check.csv')
    report_data = []
    for group in exact_cross_split:
        for item in group['items']:
            report_data.append({'type': 'exact_duplicate', 'hash': group['hash'],
                                'path': item[0], 'split': item[1], 'class': item[2],
                                'distance': 0, 'is_cross_split': True})
    for record in near_cross_split:
        report_data.append({'type': 'near_duplicate', 'hash': '', 'path': record['path1'],
                            'split': record['split1'], 'class': record['class1'],
                            'distance': record['distance'], 'is_cross_split': True,
                            'paired_with': record['path2'], 'paired_split': record['split2']})
        report_data.append({'type': 'near_duplicate', 'hash': '', 'path': record['path2'],
                            'split': record['split2'], 'class': record['class2'],
                            'distance': record['distance'], 'is_cross_split': True,
                            'paired_with': record['path1'], 'paired_split': record['split1']})
    if report_data:
        pd.DataFrame(report_data).to_csv(report_path, index=False)
        print(f"\n[INFO] Detailed leakage report saved to: {report_path}")

    if exact_cross_split:
        print(f"\n[EXAMPLES] Exact cross-split duplicates:")
        for i, group in enumerate(exact_cross_split[:5]):
            print(f"  Group {i+1} (hash={group['hash'][:16]}...):")
            for item in group['items']:
                print(f"    - [{item[1]:5s}] {os.path.basename(item[0])} (class: {item[2]})")

    if near_cross_split:
        print(f"\n[EXAMPLES] Near cross-split duplicates (distance <= {threshold}):")
        for i, record in enumerate(near_cross_split[:5]):
            print(f"  Pair {i+1} (distance={record['distance']}):")
            print(f"    - [{record['split1']:5s}] {os.path.basename(record['path1'])} (class: {record['class1']})")
            print(f"    - [{record['split2']:5s}] {os.path.basename(record['path2'])} (class: {record['class2']})")

    print("\n" + "=" * 80)
    return {
        'status': 'completed', 'leakage_found': leakage_found,
        'exact_duplicate_groups': len(exact_duplicates),
        'near_duplicate_pairs': len(near_duplicates),
        'exact_cross_split': len(exact_cross_split),
        'near_cross_split': len(near_cross_split),
        'exact_cross_split_details': exact_cross_split,
        'near_cross_split_details': near_cross_split,
        'report_path': report_path if report_data else None,
        'hash_map': hash_map
    }


def check_image_leakage_with_train(candidate_path, train_paths, train_hashes, threshold=5):
    if not PHASH_AVAILABLE:
        return False
    candidate_hash = compute_phash(candidate_path)
    if candidate_hash is None:
        return False
    candidate_int = int(candidate_hash, 16)
    for train_path in train_paths:
        if train_path in train_hashes:
            train_int = int(train_hashes[train_path], 16)
            dist = (candidate_int ^ train_int).bit_count()
            if dist <= threshold:
                return True
    return False


def hamming_distance_int(int1, int2):
    return (int1 ^ int2).bit_count()


def build_similarity_groups(imgs, hash_size=8, threshold=5):
    if not PHASH_AVAILABLE or len(imgs) == 0:
        return [[img] for img in imgs]
    hashes = {}
    for img in imgs:
        h = compute_phash(img, hash_size=hash_size)
        if h is not None:
            hashes[img] = (h, int(h, 16))
    parent = {img: img for img in imgs}
    rank = {img: 0 for img in imgs}
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px == py: return
        if rank[px] < rank[py]: px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]: rank[px] += 1
    img_list = list(imgs)
    for i in range(len(img_list)):
        if img_list[i] not in hashes: continue
        _, int1 = hashes[img_list[i]]
        for j in range(i + 1, len(img_list)):
            if img_list[j] not in hashes: continue
            _, int2 = hashes[img_list[j]]
            if hamming_distance_int(int1, int2) <= threshold:
                union(img_list[i], img_list[j])
    groups_dict = {}
    for img in imgs:
        root = find(img)
        if root not in groups_dict: groups_dict[root] = []
        groups_dict[root].append(img)
    return list(groups_dict.values())


def group_aware_split(class_to_imgs, val_split=0.1, test_split=0.2,
                      hash_size=8, threshold=5, seed=42):
    print("\n" + "=" * 80)
    print("[GROUP-AWARE SPLIT] Building train/val/test splits by image groups")
    print("=" * 80)
    train_map, val_map, test_map = {}, {}, {}
    total_groups, total_images = 0, 0
    for cls, imgs in tqdm(class_to_imgs.items(), desc="Processing classes"):
        groups = build_similarity_groups(imgs, hash_size=hash_size, threshold=threshold)
        n_groups = len(groups); n_images = len(imgs)
        total_groups += n_groups; total_images += n_images
        n_test_groups = max(1, int(round(n_groups * test_split)))
        n_val_groups = max(1, int(round(n_groups * val_split)))
        n_train_groups = n_groups - n_test_groups - n_val_groups
        if n_train_groups < 1:
            n_train_groups = 1
            n_val_groups = max(0, n_groups - n_train_groups - n_test_groups)
            if n_val_groups < 0:
                n_test_groups = max(0, n_groups - n_train_groups)
                n_val_groups = 0
        random.Random(hash(cls) & 0xffffffff ^ seed).shuffle(groups)
        test_groups = groups[:n_test_groups]
        val_groups = groups[n_test_groups:n_test_groups + n_val_groups]
        train_groups = groups[n_test_groups + n_val_groups:]
        train_map[cls] = [img for group in train_groups for img in group]
        val_map[cls] = [img for group in val_groups for img in group]
        test_map[cls] = [img for group in test_groups for img in group]
    train_total = sum(len(imgs) for imgs in train_map.values())
    val_total = sum(len(imgs) for imgs in val_map.values())
    test_total = sum(len(imgs) for imgs in test_map.values())
    print(f"\n[INFO] Group-aware split completed:")
    print(f"  - Total classes: {len(class_to_imgs)}, Total images: {total_images}, Total groups: {total_groups}")
    print(f"  - Train: {train_total} ({100*train_total/max(1,total_images):.1f}%)")
    print(f"  - Val: {val_total} ({100*val_total/max(1,total_images):.1f}%)")
    print(f"  - Test: {test_total} ({100*test_total/max(1,total_images):.1f}%)")
    return train_map, val_map, test_map


def handle_leakage_minor(train_map, val_map, test_map, all_class_imgs,
                          leakage_result, out_dir, threshold=5):
    print("\n[LEAKAGE FIX] Minor leakage detected (<5%). Applying fix strategy...")
    leaked_from_val, leaked_from_test = {}, {}
    for group in leakage_result.get('exact_cross_split_details', []):
        for path, split, cls in group['items']:
            if split == 'val':
                leaked_from_val.setdefault(cls, [])
                if path not in leaked_from_val[cls]: leaked_from_val[cls].append(path)
            elif split == 'test':
                leaked_from_test.setdefault(cls, [])
                if path not in leaked_from_test[cls]: leaked_from_test[cls].append(path)
    for record in leakage_result.get('near_cross_split_details', []):
        for skey, ckey, pkey in [('split1','class1','path1'), ('split2','class2','path2')]:
            if record[skey] == 'val':
                leaked_from_val.setdefault(record[ckey], [])
                if record[pkey] not in leaked_from_val[record[ckey]]:
                    leaked_from_val[record[ckey]].append(record[pkey])
            elif record[skey] == 'test':
                leaked_from_test.setdefault(record[ckey], [])
                if record[pkey] not in leaked_from_test[record[ckey]]:
                    leaked_from_test[record[ckey]].append(record[pkey])

    selected_images = set()
    for cls_map in [train_map, val_map, test_map]:
        for imgs in cls_map.values(): selected_images.update(imgs)
    unselected_per_class = {}
    for cls, all_imgs in all_class_imgs.items():
        unselected_per_class[cls] = [p for p in all_imgs if p not in selected_images]

    all_train_paths = [p for imgs in train_map.values() for p in imgs]
    train_hashes = compute_phash_for_paths(all_train_paths)
    replacement_stats = {'val': 0, 'test': 0, 'failed': 0}

    for split_name, leaked_dict, split_map in [('val', leaked_from_val, val_map),
                                                 ('test', leaked_from_test, test_map)]:
        for cls, leaked_paths in leaked_dict.items():
            for leaked_path in leaked_paths:
                if leaked_path in split_map.get(cls, []):
                    split_map[cls].remove(leaked_path)
                    train_map[cls].append(leaked_path)
                    train_hashes[leaked_path] = compute_phash(leaked_path)
                    found = False
                    for candidate in unselected_per_class.get(cls, []):
                        if not check_image_leakage_with_train(
                                candidate, list(train_map[cls]), train_hashes, threshold):
                            split_map[cls].append(candidate)
                            unselected_per_class[cls].remove(candidate)
                            replacement_stats[split_name] += 1
                            found = True; break
                    if not found:
                        replacement_stats['failed'] += 1

    print(f"[SUMMARY] Replacements - Val: {replacement_stats['val']}, "
          f"Test: {replacement_stats['test']}, Failed: {replacement_stats['failed']}")
    return train_map, val_map, test_map, replacement_stats['failed'] == 0


def handle_data_leakage(train_map, val_map, test_map, class_to_imgs, all_class_imgs,
                        out_dir, val_split=0.1, test_split=0.2,
                        hash_size=8, threshold=5, max_iterations=3):
    print("\n[DATA LEAKAGE HANDLER] Starting leakage detection and fixing...")
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'─' * 40}\n[Iteration {iteration}/{max_iterations}]\n{'─' * 40}")
        leakage_result = check_data_leakage_phash(
            train_map, val_map, test_map, out_dir, hash_size=hash_size, threshold=threshold)
        if not leakage_result.get('leakage_found', False):
            print("\n✅ No data leakage detected. Dataset is clean!")
            return train_map, val_map, test_map, leakage_result
        total_eval = sum(len(imgs) for imgs in val_map.values()) + \
                     sum(len(imgs) for imgs in test_map.values())
        n_leaked = leakage_result.get('exact_cross_split', 0) + \
                   leakage_result.get('near_cross_split', 0)
        leakage_pct = (n_leaked / max(1, total_eval)) * 100
        print(f"[INFO] Leakage: {n_leaked}/{total_eval} ({leakage_pct:.2f}%)")
        if leakage_pct < 5.0:
            train_map, val_map, test_map, success = handle_leakage_minor(
                train_map, val_map, test_map, all_class_imgs,
                leakage_result, out_dir, threshold=threshold)
        else:
            train_map, val_map, test_map = group_aware_split(
                class_to_imgs, val_split=val_split, test_split=test_split,
                hash_size=hash_size, threshold=threshold)

    final_result = check_data_leakage_phash(
        train_map, val_map, test_map, out_dir, hash_size=hash_size, threshold=threshold)
    if final_result.get('leakage_found', False):
        print("\n⚠ WARNING: Some leakage still remains after fixing attempts.")
    else:
        print("\n✅ All leakage has been successfully resolved!")
    return train_map, val_map, test_map, final_result


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    # If backbone is frozen, keep it in eval mode for BN/dropout
    if hasattr(model, 'freeze_backbone') and model.freeze_backbone:
        model.backbone.eval()

    total_loss, total_acc, n = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    epoch_start = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for imgs, labels, paths in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=-1)
        total_acc += (preds == labels).sum().item()
        n += imgs.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{total_acc/n:.4f}'})

    if scheduler is not None:
        scheduler.step()

    epoch_time = time.time() - epoch_start
    return {'loss': total_loss / n, 'acc': total_acc / n, 'time': epoch_time}


def evaluate(model, dataloader, device, class_names, phase='val'):
    model.eval()
    y_true, y_pred, paths_all, all_probs = [], [], [], []
    eval_start = time.time()
    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc=f"[{phase.upper()}]"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            pred_probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            paths_all.extend(paths)
            all_probs.extend(pred_probs.cpu().numpy())
    eval_time = time.time() - eval_start

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cls_report = classification_report(
        y_true, y_pred, target_names=class_names,
        labels=range(len(class_names)), zero_division=0, output_dict=True)

    return {
        'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1,
        'accuracy': accuracy, 'per_precision': per_p, 'per_recall': per_r,
        'per_f1': per_f1, 'per_support': per_support,
        'y_true': y_true, 'y_pred': y_pred, 'paths': paths_all, 'probs': all_probs,
        'confusion_matrix': cm, 'classification_report': cls_report, 'time': eval_time
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(args):
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Collect images
    print("\n[STEP 1] Collecting images per class...")
    class_to_imgs, all_class_imgs = collect_images_per_class(
        args.data_root, max_per_class=args.max_per_class, verbose=True)
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = sum(len(imgs) for imgs in class_to_imgs.values())
    print(f"\n[STEP 1] Summary: Available={total_available}, Selected={total_selected}, "
          f"Unselected={total_available - total_selected}")

    # 2) Build loaders
    print("\n[STEP 2] Building data loaders...")
    train_loader, val_loader, test_loader, class_to_idx, train_map, val_map, test_map = \
        build_loaders(class_to_imgs, args.batch_size,
                      val_split=args.val_split, test_split=args.test_split,
                      num_workers=args.num_workers)

    classes = sorted(class_to_idx.keys())
    n_classes = len(class_to_idx)
    print(f"[INFO] Classes: {n_classes}, Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Save dataset splits
    save_dataset_splits(args.out_dir, all_class_imgs, class_to_imgs,
                        train_map, val_map, test_map, class_to_idx)

    # 2b) Check and handle data leakage
    print("\n[STEP 2b] Checking and handling data leakage using pHash...")
    train_map, val_map, test_map, leakage_result = handle_data_leakage(
        train_map, val_map, test_map, class_to_imgs, all_class_imgs,
        args.out_dir, val_split=args.val_split, test_split=args.test_split,
        hash_size=8, threshold=5, max_iterations=3)

    # Update class_to_idx and rebuild loaders after leakage fix
    classes = sorted(train_map.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print("\n[INFO] Saving updated dataset splits after leakage fix...")
    save_dataset_splits(args.out_dir, all_class_imgs, class_to_imgs,
                        train_map, val_map, test_map, class_to_idx)

    print(f"\n[INFO] Updated splits - Train: {sum(len(v) for v in train_map.values())}, "
          f"Val: {sum(len(v) for v in val_map.values())}, "
          f"Test: {sum(len(v) for v in test_map.values())}")

    # Rebuild datasets with updated splits
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

    train_ds = BigPlantsDataset(train_map, class_to_idx, transform=train_tf)
    val_ds = BigPlantsDataset(val_map, class_to_idx, transform=val_tf)
    test_ds = BigPlantsDataset(test_map, class_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 3) Create DINOv2 model
    print(f"\n[STEP 3] Creating DINOv2 model ({args.model_name})...")
    freeze_backbone = not args.unfreeze_backbone
    model = DINOv2Classifier(
        model_name=args.model_name,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        dropout=args.dropout
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4) Training loop
    print("\n[STEP 4] Starting training loop...")
    print("=" * 80)

    best_macro = -1.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_time': [],
        'val_macro_f1': [], 'val_micro_f1': [], 'val_weighted_f1': [],
        'val_accuracy': [], 'val_time': [], 'lr': []
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = datetime.now()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch}/{args.epochs}] Started at: {epoch_start.strftime('%H:%M:%S')}, LR: {current_lr:.6f}")

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = evaluate(model, val_loader, device, classes, phase='val')

        epoch_end = datetime.now()
        epoch_duration = (epoch_end - epoch_start).total_seconds()

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_time'].append(train_metrics['time'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_micro_f1'].append(val_metrics['micro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_time'].append(val_metrics['time'])
        history['lr'].append(current_lr)

        print(f"\n[Epoch {epoch}] Results:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, Time: {train_metrics['time']:.1f}s")
        print(f"  Val   - Macro F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Time: {val_metrics['time']:.1f}s")
        print(f"  Epoch duration: {epoch_duration:.1f}s | Finished at: {epoch_end.strftime('%H:%M:%S')}")

        if val_metrics['macro_f1'] > best_macro:
            best_macro = val_metrics['macro_f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_macro,
                'class_to_idx': class_to_idx,
                'embed_dim': model.embed_dim,
                'model_name': args.model_name,
                'freeze_backbone': freeze_backbone,
                'dropout': args.dropout,
                'n_classes': n_classes,
                'args': vars(args)
            }, os.path.join(args.out_dir, "best_model.pt"))
            print(f"  ★ New best model saved! (Macro F1: {best_macro:.4f})")

    # Save training history
    torch.save(history, os.path.join(args.out_dir, "training_history.pt"))
    print(f"\n[INFO] Training history saved to {args.out_dir}/training_history.pt")

    # 5) Final evaluation on test set
    print("\n[STEP 5] Final evaluation on test set...")
    print("=" * 80)

    checkpoint = torch.load(os.path.join(args.out_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, classes, phase='test')

    print(f"\n[TEST RESULTS]")
    print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Micro F1:    {test_metrics['micro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")

    save_classification_report(test_metrics['classification_report'], classes, args.out_dir, phase='test')
    save_confusion_matrix(test_metrics['confusion_matrix'], classes, args.out_dir, phase='test')

    print("\n[Per-Class F1 Scores]")
    print("-" * 60)
    for i, cls in enumerate(classes):
        print(f"{cls:40s} F1: {test_metrics['per_f1'][i]:.4f}  "
              f"P: {test_metrics['per_precision'][i]:.4f}  "
              f"R: {test_metrics['per_recall'][i]:.4f}  "
              f"Support: {int(test_metrics['per_support'][i])}")

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 80)
    print(f"[END] Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {total_duration/3600:.2f} hours ({total_duration:.0f} seconds)")
    print(f"[INFO] Best epoch: {best_epoch} with Macro F1: {best_macro:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINOv2 for BigPlants-100')

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root containing 100 class folders")
    parser.add_argument("--out_dir", type=str, default="./outputs",
                        help="Output directory for models and results")
    parser.add_argument("--max_per_class", type=int, default=100,
                        help="Maximum images per class")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (higher default since backbone is frozen)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in classifier head")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                                 "dinov2_vitg14", "dinov2_vits14_reg",
                                 "dinov2_vitb14_reg", "dinov2_vitl14_reg",
                                 "dinov2_vitg14_reg"],
                        help="DINOv2 model variant")
    parser.add_argument("--unfreeze_backbone", action='store_true', default=False,
                        help="Unfreeze backbone for full fine-tuning")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
