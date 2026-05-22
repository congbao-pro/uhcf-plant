import os
import json
import argparse
import random
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Set
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

# For pHash duplicate detection
try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("[WARNING] imagehash not installed. pHash duplicate detection disabled.")
    print("          Install with: pip install imagehash")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import timm
from torch.amp import autocast, GradScaler

DATA_ROOT_DEFAULT = "path/to/bigplants_dataset_100_resized"

# -----------------------------
# Repro & small utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}

def list_images_direct(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if is_image_file(p)]


# -----------------------------
# Collect ALL images from dataset (for unselected tracking)
# -----------------------------

def collect_all_images_from_dataset(data_root: Path, parts_keep=("hand", "leaf", "flower", "fruit")) -> Dict[str, List[Path]]:
    """
    Collect ALL available images from the dataset without any limit.
    Returns: dict species_name -> list of ALL image paths
    """
    species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    all_class_imgs = {}

    for species_dir in species_dirs:
        species = species_dir.name
        all_imgs = []

        # Collect from all subfolders
        for part in parts_keep:
            part_dir = species_dir / part
            if part_dir.exists() and part_dir.is_dir():
                files = [p for p in part_dir.rglob("*") if is_image_file(p)]
                all_imgs.extend(files)

        # Collect available images (files in species root)
        all_imgs.extend(list_images_direct(species_dir))

        # Deduplicate and keep order
        seen = set()
        uniq = []
        for p in all_imgs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        all_class_imgs[species] = uniq

    return all_class_imgs


# -----------------------------
# pHash Data Leakage Detection
# -----------------------------

def compute_phash(img_path, hash_size=8):
    """Compute perceptual hash for an image."""
    if not PHASH_AVAILABLE:
        return None
    try:
        img = Image.open(img_path).convert('RGB')
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception as e:
        print(f"[WARNING] Could not compute pHash for {img_path}: {e}")
        return None


def compute_phash_for_paths(paths: List, hash_size=8) -> Dict:
    """Compute pHash for a list of image paths."""
    if not PHASH_AVAILABLE:
        return {}
    hash_map = {}
    for path in tqdm(paths, desc="Computing pHash"):
        h = compute_phash(path, hash_size=hash_size)
        if h is not None:
            hash_map[str(path)] = h
    return hash_map


def hamming_distance_int(int1, int2):
    """Compute hamming distance between two integers."""
    return (int1 ^ int2).bit_count()


def check_image_leakage_with_train(candidate_path, train_hashes: Dict, threshold=5) -> bool:
    """Check if a candidate image has leakage with any train image."""
    if not PHASH_AVAILABLE:
        return False
    candidate_hash = compute_phash(candidate_path)
    if candidate_hash is None:
        return False
    candidate_int = int(candidate_hash, 16)
    for train_path, train_hash in train_hashes.items():
        train_int = int(train_hash, 16)
        dist = hamming_distance_int(candidate_int, train_int)
        if dist <= threshold:
            return True
    return False


def check_data_leakage_phash(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                              out_dir: Path, hash_size=8, threshold=5) -> Dict:
    """Check for potential data leakage between train/val/test using pHash."""
    if not PHASH_AVAILABLE:
        print("[WARNING] imagehash not available. Skipping pHash leakage check.")
        return {'status': 'skipped', 'reason': 'imagehash not installed'}

    print("\n" + "=" * 80)
    print("[DATA LEAKAGE CHECK] Using pHash to detect duplicate images across splits")
    print("=" * 80)

    def collect_paths(df, split_name):
        paths = []
        for _, row in df.iterrows():
            paths.append((row["path"], split_name, row["species"]))
        return paths

    train_paths = collect_paths(df_train, 'train')
    val_paths = collect_paths(df_val, 'val')
    test_paths = collect_paths(df_test, 'test')

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
                dist = hamming_distance_int(int1, int2)
                if 0 < dist <= threshold:
                    record = {'path1': p1, 'split1': s1, 'class1': c1, 'path2': p2, 'split2': s2, 'class2': c2, 'distance': dist}
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
        print(f"\n  🚨 CROSS-SPLIT EXACT DUPLICATES: {len(exact_cross_split)} groups")
        print(f"  🚨 CROSS-SPLIT NEAR DUPLICATES: {len(near_cross_split)} pairs")
        print(f"  ⚠️  DATA LEAKAGE DETECTED! ⚠️")
    else:
        print(f"\n  ✅ NO CROSS-SPLIT DUPLICATES FOUND")
        print(f"  ✅ No data leakage detected between train/val/test splits")

    report_path = out_dir / 'data_leakage_check.csv'
    report_data = []
    for group in exact_cross_split:
        for item in group['items']:
            report_data.append({'type': 'exact_duplicate', 'hash': group['hash'], 'path': item[0], 'split': item[1], 'class': item[2], 'distance': 0, 'is_cross_split': True})
    for record in near_cross_split:
        report_data.append({'type': 'near_duplicate', 'path': record['path1'], 'split': record['split1'], 'class': record['class1'], 'distance': record['distance'], 'is_cross_split': True, 'paired_with': record['path2']})
        report_data.append({'type': 'near_duplicate', 'path': record['path2'], 'split': record['split2'], 'class': record['class2'], 'distance': record['distance'], 'is_cross_split': True, 'paired_with': record['path1']})

    if report_data:
        pd.DataFrame(report_data).to_csv(report_path, index=False)
        print(f"\n[INFO] Detailed leakage report saved to: {report_path}")

    print("\n" + "=" * 80)
    return {'status': 'completed', 'leakage_found': leakage_found, 'exact_cross_split': len(exact_cross_split), 'near_cross_split': len(near_cross_split), 'exact_cross_split_details': exact_cross_split, 'near_cross_split_details': near_cross_split, 'hash_map': hash_map}


def build_similarity_groups(df: pd.DataFrame, hash_size=8, threshold=5) -> List[List[str]]:
    """Build groups of similar images using Union-Find algorithm."""
    paths = df["path"].tolist()
    if not PHASH_AVAILABLE or len(paths) == 0:
        return [[p] for p in paths]

    hashes = {}
    for p in paths:
        h = compute_phash(p, hash_size=hash_size)
        if h is not None:
            hashes[p] = (h, int(h, 16))

    parent = {p: p for p in paths}
    rank = {p: 0 for p in paths}

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

    for i in range(len(paths)):
        p1 = paths[i]
        if p1 not in hashes:
            continue
        h1, int1 = hashes[p1]
        for j in range(i + 1, len(paths)):
            p2 = paths[j]
            if p2 not in hashes:
                continue
            h2, int2 = hashes[p2]
            if hamming_distance_int(int1, int2) <= threshold:
                union(p1, p2)

    groups_dict = {}
    for p in paths:
        root = find(p)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(p)
    return list(groups_dict.values())


def group_aware_split(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.2, hash_size=8, threshold=5, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by GROUPS to prevent data leakage."""
    print("\n" + "=" * 80)
    print("[GROUP-AWARE SPLIT] Building train/val/test splits by image groups")
    print("=" * 80)

    train_rows, val_rows, test_rows = [], [], []
    total_groups, total_images = 0, 0

    for species in tqdm(df["species"].unique(), desc="Processing classes"):
        df_species = df[df["species"] == species].copy()
        groups = build_similarity_groups(df_species, hash_size=hash_size, threshold=threshold)
        n_groups, n_images = len(groups), len(df_species)
        total_groups += n_groups
        total_images += n_images

        n_test_groups = max(1, int(round(n_groups * test_ratio)))
        n_val_groups = max(1, int(round(n_groups * val_ratio)))
        n_train_groups = n_groups - n_test_groups - n_val_groups
        if n_train_groups < 1:
            n_train_groups = 1
            n_val_groups = max(0, n_groups - n_train_groups - n_test_groups)

        rng = random.Random(hash(species) & 0xffffffff ^ seed)
        rng.shuffle(groups)

        test_paths = set(p for g in groups[:n_test_groups] for p in g)
        val_paths = set(p for g in groups[n_test_groups:n_test_groups + n_val_groups] for p in g)

        for _, row in df_species.iterrows():
            if row["path"] in test_paths:
                test_rows.append(row)
            elif row["path"] in val_paths:
                val_rows.append(row)
            else:
                train_rows.append(row)

    print(f"\n[INFO] Group-aware split: {total_images} images, {total_groups} groups")
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def handle_leakage_minor(df_train, df_val, df_test, all_class_imgs, leakage_result, out_dir, threshold=5):
    """Handle minor leakage (<5%): Move leaked images to train, replace from unselected."""
    print("\n[LEAKAGE FIX] Minor leakage (<5%). Moving leaked images to train...")

    leaked_from_val, leaked_from_test = {}, {}
    for group in leakage_result.get('exact_cross_split_details', []):
        for path, split, cls in group['items']:
            if split == 'val':
                leaked_from_val.setdefault(cls, []).append(path) if path not in leaked_from_val.get(cls, []) else None
            elif split == 'test':
                leaked_from_test.setdefault(cls, []).append(path) if path not in leaked_from_test.get(cls, []) else None

    for record in leakage_result.get('near_cross_split_details', []):
        for key in [('split1', 'class1', 'path1'), ('split2', 'class2', 'path2')]:
            split, cls, path = record[key[0]], record[key[1]], record[key[2]]
            if split == 'val':
                leaked_from_val.setdefault(cls, []).append(path) if path not in leaked_from_val.get(cls, []) else None
            elif split == 'test':
                leaked_from_test.setdefault(cls, []).append(path) if path not in leaked_from_test.get(cls, []) else None

    selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
    unselected_per_class = {cls: [str(p) for p in imgs if str(p) not in selected_images] for cls, imgs in all_class_imgs.items()}

    train_hashes = compute_phash_for_paths(df_train["path"].tolist())
    train_rows, val_rows, test_rows = df_train.to_dict('records'), df_val.to_dict('records'), df_test.to_dict('records')
    stats = {'val': 0, 'test': 0, 'failed': 0}

    for cls, leaked_paths in leaked_from_val.items():
        for leaked_path in leaked_paths:
            for i, row in enumerate(val_rows):
                if row["path"] == leaked_path:
                    leaked_row = val_rows.pop(i)
                    train_rows.append(leaked_row)
                    h = compute_phash(leaked_path)
                    if h: train_hashes[leaked_path] = h
                    for cand in unselected_per_class.get(cls, []):
                        if not check_image_leakage_with_train(cand, train_hashes, threshold):
                            new_row = leaked_row.copy()
                            new_row["path"] = cand
                            val_rows.append(new_row)
                            unselected_per_class[cls].remove(cand)
                            stats['val'] += 1
                            break
                    else:
                        stats['failed'] += 1
                    break

    for cls, leaked_paths in leaked_from_test.items():
        for leaked_path in leaked_paths:
            for i, row in enumerate(test_rows):
                if row["path"] == leaked_path:
                    leaked_row = test_rows.pop(i)
                    train_rows.append(leaked_row)
                    h = compute_phash(leaked_path)
                    if h: train_hashes[leaked_path] = h
                    for cand in unselected_per_class.get(cls, []):
                        if not check_image_leakage_with_train(cand, train_hashes, threshold):
                            new_row = leaked_row.copy()
                            new_row["path"] = cand
                            test_rows.append(new_row)
                            unselected_per_class[cls].remove(cand)
                            stats['test'] += 1
                            break
                    else:
                        stats['failed'] += 1
                    break

    print(f"[SUMMARY] Val replacements: {stats['val']}, Test: {stats['test']}, Failed: {stats['failed']}")
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), stats['failed'] == 0


def handle_data_leakage(df_train, df_val, df_test, df_all, all_class_imgs, out_dir, val_ratio=0.1, test_ratio=0.2, hash_size=8, threshold=5, max_iterations=3, seed=42):
    """Main function to detect and handle data leakage."""
    print("\n" + "=" * 80)
    print("[DATA LEAKAGE HANDLER] Starting leakage detection and fixing...")
    print("=" * 80)

    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iteration {iteration}/{max_iterations}]")
        leakage_result = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size, threshold)

        if not leakage_result.get('leakage_found', False):
            print("\n✅ No data leakage detected. Dataset is clean!")
            return df_train, df_val, df_test, leakage_result

        total_eval = len(df_val) + len(df_test)
        n_leaked = leakage_result.get('exact_cross_split', 0) + leakage_result.get('near_cross_split', 0)
        leakage_pct = (n_leaked / max(1, total_eval)) * 100

        print(f"\n[INFO] Leakage: {n_leaked} images ({leakage_pct:.2f}%)")

        if leakage_pct < 5.0:
            df_train, df_val, df_test, success = handle_leakage_minor(df_train, df_val, df_test, all_class_imgs, leakage_result, out_dir, threshold)
        else:
            print(f"\n[DECISION] Leakage >= 5% → Rebuilding with group-aware split")
            df_train, df_val, df_test = group_aware_split(df_all, val_ratio, test_ratio, hash_size, threshold, seed)

    final_result = check_data_leakage_phash(df_train, df_val, df_test, out_dir, hash_size, threshold)
    if final_result.get('leakage_found', False):
        print("\n⚠ WARNING: Some leakage still remains.")
    else:
        print("\n✅ All leakage resolved!")
    return df_train, df_val, df_test, final_result


def create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map):
    """Create dataset_unselected.csv containing images not in train/val/test."""
    selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())

    unselected_path = out_dir / 'dataset_unselected.csv'
    with open(unselected_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'species', 'label_id'])
        for species, all_imgs in all_class_imgs.items():
            label_id = label_map.get(species, -1)
            for img_path in all_imgs:
                if str(img_path) not in selected_images:
                    writer.writerow([str(img_path), species, label_id])

    print(f"\n[INFO] Created {unselected_path}")
    print(f"  - Total available: {total_available}, Selected: {len(selected_images)}, Unselected: {total_available - len(selected_images)}")

# -----------------------------
# Dataset curation
# -----------------------------

PartsKeep = ("hand", "leaf", "flower", "fruit")

def build_selection_for_species(
    species_dir: Path,
    parts_keep: Tuple[str, ...] = PartsKeep,
    per_class_cap: int = 100,
    seed: int = 42,
) -> List[Path]:
    rng = random.Random(seed)
    sub_images: List[Path] = []

    # Collect images from subfolders hand/leaf/flower/fruit
    for part in parts_keep:
        part_dir = species_dir / part
        if part_dir.exists() and part_dir.is_dir():
            imgs = [p for p in part_dir.rglob("*") if is_image_file(p)]
            sub_images.extend(imgs)

    # unique and shuffle
    sub_images = list(dict.fromkeys(sub_images))
    rng.shuffle(sub_images)

    # If enough for cap, cut to cap
    if len(sub_images) >= per_class_cap:
        return sub_images[:per_class_cap]

    # Else top up with "available" (images directly under species_dir)
    available_images = list_images_direct(species_dir)
    rng.shuffle(available_images)

    need = per_class_cap - len(sub_images)
    chosen = sub_images + available_images[:need]
    return chosen


def scan_dataset(
    data_root: Path,
    parts_keep: Tuple[str, ...] = PartsKeep,
    per_class_cap: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    species_names = [p.name for p in species_dirs]
    species_to_id = {name: i for i, name in enumerate(species_names)}

    rows = []
    for species_dir in tqdm(species_dirs, desc="Scanning species"):
        species = species_dir.name
        selected_paths = build_selection_for_species(
            species_dir, parts_keep=parts_keep, per_class_cap=per_class_cap, seed=seed
        )
        for img in selected_paths:
            # determine part/source metadata
            part_val = None
            for anc in img.parents:
                if anc == species_dir:
                    break
                if anc.name in parts_keep:
                    part_val = anc.name
                    break
            source = "sub" if part_val is not None else "available"
            rows.append({
                "path": str(img.resolve()),
                "species": species,
                "label_id": species_to_id[species],
                "source": source,
                "part": part_val if part_val is not None else "available",
            })
    return pd.DataFrame(rows)

# -----------------------------
# Torch Dataset
# -----------------------------

class PlantImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_id"])

# -----------------------------
# Transforms
# -----------------------------

def get_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfms, eval_tfms

# -----------------------------
# Model: Generic (via timm)
# -----------------------------

def build_model(model_name: str, num_classes: int):
    """Builds a model using timm."""
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model, f"timm:{model_name}"
    except Exception as e:
        raise RuntimeError(f"Could not construct {model_name} via timm: {e}")

# -----------------------------
# Class Weighting Helper
# -----------------------------

def compute_class_weights(df_train: pd.DataFrame, all_classes: List[str], device: torch.device):
    """Calculates class weights based on train set."""
    train_counts = df_train['species'].value_counts().to_dict()
    weights = []
    for c in all_classes:
        cnt = train_counts.get(c, 0)
        if cnt > 0:
            weights.append(1.0 / cnt)
        else:
            weights.append(0.0) # Will be fixed

    w = np.array(weights, dtype=np.float32)

    if (w > 0).sum() == 0:
        # All counts are 0? Fallback to equal weights
        w = np.ones_like(w)
    else:
        # Set 0-count classes to min weight (avoids 0 weight)
        nonzero_min = w[w>0].min()
        w[w==0] = nonzero_min

    # Normalize weights
    w = w / w.sum() * len(w)
    return torch.tensor(w, dtype=torch.float, device=device)

# -----------------------------
# Train / Eval loops
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use AMP (autocast)
        with autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    return avg_loss, acc, y_true, y_pred

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--out_dir", type=str, default="./output_resnet50")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="resnet50.a1_in1k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Target validation ratio (e.g., 0.10 for 10%)")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="Target test ratio (e.g., 0.20 for 20%)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--per_class_cap", type=int, default=100)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")

    args = parser.parse_args()

    assert 0.0 < args.test_ratio < 0.5, "test_ratio should be reasonable (e.g., 0.2)"
    assert 0.0 < args.val_ratio < 0.5,  "val_ratio should be reasonable (e.g., 0.1)"
    assert args.val_ratio + args.test_ratio < 1.0, "val_ratio + test_ratio must be < 1.0"

    set_seed(args.seed)

    # ---- Timestamp Start ----
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning dataset & selecting images (cap={args.per_class_cap})...")
    df = scan_dataset(data_root=data_root, per_class_cap=args.per_class_cap, seed=args.seed)
    assert len(df) > 0, f"No valid images found in {data_root}"
    df.to_csv(out_dir / "dataset_selected.csv", index=False)

    species_list = sorted(df["species"].unique().tolist())
    label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    num_classes = len(species_list)
    print(f"Selected images: {len(df)}; classes: {num_classes}")

    # ---- Collect ALL images for unselected tracking ----
    print("\nCollecting ALL available images from dataset...")
    all_class_imgs = collect_all_images_from_dataset(data_root)
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    print(f"Total available images: {total_available}")

    # ---- Split 70:10:20 ----
    print(f"Splitting train/val/test (target ≈ {1-args.test_ratio-args.val_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%})...")

    # 1. Split off Test (20%)
    df_trainval, df_test = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=df["species"],
    )

    # 2. Split Train/Val from the remainder
    # Target 10% val from 80% total = 0.10 / 0.80 = 0.125
    val_ratio_adj = args.val_ratio / (1.0 - args.test_ratio)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adj,
        random_state=args.seed,
        stratify=df_trainval["species"],
    )

    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        d.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} images")

    total_n = len(df)
    print(
        f"Actual ratios → train: {len(df_train)/total_n:.3f}, "
        f"val: {len(df_val)/total_n:.3f}, test: {len(df_test)/total_n:.3f}"
    )

    # ---- Create dataset_unselected.csv ----
    print("\nCreating dataset_unselected.csv...")
    create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

    # ---- Check and Handle Data Leakage using pHash ----
    print("\nChecking and handling data leakage using pHash...")
    df_train, df_val, df_test, leakage_result = handle_data_leakage(
        df_train, df_val, df_test,
        df_all=df,
        all_class_imgs=all_class_imgs,
        out_dir=out_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hash_size=8,
        threshold=5,
        max_iterations=3,
        seed=args.seed
    )

    # ---- Save updated CSVs after leakage fix ----
    print("\nSaving updated dataset splits after leakage fix...")
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        d.to_csv(out_dir / f"{name}.csv", index=False)

    # Update dataset_unselected.csv after leakage fix
    create_dataset_unselected_csv(all_class_imgs, df_train, df_val, df_test, out_dir, label_map)

    total_n = len(df_train) + len(df_val) + len(df_test)
    print(f"\n[INFO] Updated split sizes after leakage fix:")
    print(f"  - Train: {len(df_train)} images ({100*len(df_train)/max(1,total_n):.1f}%)")
    print(f"  - Val: {len(df_val)} images ({100*len(df_val)/max(1,total_n):.1f}%)")
    print(f"  - Test: {len(df_test)} images ({100*len(df_test)/max(1,total_n):.1f}%)")

    # ---- Datasets & Loaders ----
    train_tfms, eval_tfms = get_transforms(img_size=args.img_size)

    ds_train = PlantImageDataset(df_train, transform=train_tfms)
    ds_val   = PlantImageDataset(df_val,   transform=eval_tfms)
    ds_test  = PlantImageDataset(df_test,  transform=eval_tfms)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem
    )

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backend = build_model(model_name=args.model_name, num_classes=num_classes)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Use class weights
    weight_tensor = compute_class_weights(df_train, species_list, device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Use optimizer params
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"Start training on {device} | backend={backend} | model={args.model_name}")
    best_val_acc = 0.0
    best_ckpt = out_dir / "best_model.pth"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
        scheduler.step()
        print(f"Train | loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

        # --- Checkpointing ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "label_map": label_map,
                "backend": backend,
                "model_name": args.model_name,
            }
            torch.save(state, best_ckpt)
            print(f"✅ Saved best checkpoint to {best_ckpt} (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break
    # --- End Training Loop ---

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        # Handle DataParallel wrapper
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint (val_acc={ckpt.get('val_acc', -1):.4f})")
    else:
        print("Warning: No best checkpoint found. Using last epoch model.")

    # ---- Final Test ----
    print("\nRunning final evaluation on test set...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\nTest  | loss={test_loss:.4f}, acc={test_acc:.4f}")

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=[s for s in species_list],
        zero_division=0,
        output_dict=True,
    )
    rep_df = pd.DataFrame(report).transpose()
    rep_df.to_csv(out_dir / "test_classification_report.csv")

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "best_val_acc": float(best_val_acc),
            "splits": {
                "train": len(df_train),
                "val": len(df_val),
                "test": len(df_test),
            },
            "model": args.model_name,
        }, f, indent=2)

    print(f"Saved test report to {out_dir / 'test_classification_report.csv'}")

    # ---- Timestamp End ----
    end_time = datetime.now()
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    total_minutes = total_seconds / 60
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int(total_seconds % 60)

    print("\n" + "=" * 80)
    print(f"[END] Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DURATION] Total time: {hours}h {minutes}m {seconds}s ({total_minutes:.2f} minutes)")
    print("=" * 80)
    print("🎉🎉🎉Done!")

if __name__ == "__main__":
    main()
