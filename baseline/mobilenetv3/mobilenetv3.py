import os
import json
import argparse
import random
import csv
from pathlib import Path
from typing import List, Dict, Set, Tuple
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}


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


def compute_phash_for_paths(paths: List, hash_size=8) -> Dict:
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
            hash_map[str(path)] = h
    return hash_map


def hamming_distance_int(int1, int2):
    """Compute hamming distance between two integers."""
    return (int1 ^ int2).bit_count()


def check_image_leakage_with_train(candidate_path, train_hashes: Dict, threshold=5) -> bool:
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

    for train_path, train_hash in train_hashes.items():
        train_int = int(train_hash, 16)
        dist = hamming_distance_int(candidate_int, train_int)
        if dist <= threshold:
            return True  # Leakage found

    return False  # No leakage


def check_data_leakage_phash(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                              out_dir: Path, hash_size=8, threshold=5) -> Dict:
    """
    Check for potential data leakage between train/val/test using pHash.

    This function:
    1. Computes pHash for all images in train, val, test
    2. Finds exact duplicates (same hash)
    3. Finds near-duplicates (hamming distance <= threshold)
    4. Reports any cross-split duplicates as potential leakage

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
                dist = hamming_distance_int(int1, int2)

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
    report_path = out_dir / 'data_leakage_check.csv'
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
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(report_path, index=False)
        print(f"\n[INFO] Detailed leakage report saved to: {report_path}")

    # Print examples if leakage found
    if exact_cross_split:
        print(f"\n[EXAMPLES] Exact cross-split duplicates:")
        for i, group in enumerate(exact_cross_split[:5]):  # Show first 5
            print(f"  Group {i+1} (hash={group['hash'][:16]}...):")
            for item in group['items']:
                print(f"    - [{item[1]:5s}] {Path(item[0]).name} (class: {item[2]})")

    if near_cross_split:
        print(f"\n[EXAMPLES] Near cross-split duplicates (distance <= {threshold}):")
        for i, record in enumerate(near_cross_split[:5]):  # Show first 5
            print(f"  Pair {i+1} (distance={record['distance']}):")
            print(f"    - [{record['split1']:5s}] {Path(record['path1']).name} (class: {record['class1']})")
            print(f"    - [{record['split2']:5s}] {Path(record['path2']).name} (class: {record['class2']})")

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
        'report_path': str(report_path) if report_data else None,
        'hash_map': hash_map
    }


def build_similarity_groups(df: pd.DataFrame, hash_size=8, threshold=5) -> List[List[str]]:
    """
    Build groups of similar images using Union-Find algorithm.
    Images with pHash distance <= threshold are grouped together.

    Returns: list of groups, each group is a list of image paths
    """
    paths = df["path"].tolist()

    if not PHASH_AVAILABLE or len(paths) == 0:
        # No grouping possible, each image is its own group
        return [[p] for p in paths]

    # Compute hashes
    hashes = {}
    for p in paths:
        h = compute_phash(p, hash_size=hash_size)
        if h is not None:
            hashes[p] = (h, int(h, 16))

    # Union-Find
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

    # Compare all pairs and union similar ones
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

            dist = hamming_distance_int(int1, int2)
            if dist <= threshold:
                union(p1, p2)

    # Group by parent
    groups_dict = {}
    for p in paths:
        root = find(p)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(p)

    return list(groups_dict.values())


def group_aware_split(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.2,
                      hash_size=8, threshold=5, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by GROUPS instead of individual images.
    This ensures similar images always stay in the same split,
    preventing data leakage.

    Returns: df_train, df_val, df_test
    """
    print("\n" + "=" * 80)
    print("[GROUP-AWARE SPLIT] Building train/val/test splits by image groups")
    print("=" * 80)

    train_rows = []
    val_rows = []
    test_rows = []

    total_groups = 0
    total_images = 0

    for species in tqdm(df["species"].unique(), desc="Processing classes"):
        df_species = df[df["species"] == species].copy()

        # Build similarity groups for this class
        groups = build_similarity_groups(df_species, hash_size=hash_size, threshold=threshold)

        n_groups = len(groups)
        n_images = len(df_species)
        total_groups += n_groups
        total_images += n_images

        # Calculate split sizes (by number of groups)
        n_test_groups = max(1, int(round(n_groups * test_ratio)))
        n_val_groups = max(1, int(round(n_groups * val_ratio)))
        n_train_groups = n_groups - n_test_groups - n_val_groups

        # Ensure we have at least 1 group for train
        if n_train_groups < 1:
            n_train_groups = 1
            n_val_groups = max(0, n_groups - n_train_groups - n_test_groups)
            if n_val_groups < 0:
                n_test_groups = max(0, n_groups - n_train_groups)
                n_val_groups = 0

        # Shuffle groups deterministically
        rng = random.Random(hash(species) & 0xffffffff ^ seed)
        rng.shuffle(groups)

        # Assign groups to splits
        test_groups = groups[:n_test_groups]
        val_groups = groups[n_test_groups:n_test_groups + n_val_groups]
        train_groups = groups[n_test_groups + n_val_groups:]

        # Flatten groups to rows
        test_paths = set(p for group in test_groups for p in group)
        val_paths = set(p for group in val_groups for p in group)
        train_paths = set(p for group in train_groups for p in group)

        for _, row in df_species.iterrows():
            if row["path"] in test_paths:
                test_rows.append(row)
            elif row["path"] in val_paths:
                val_rows.append(row)
            else:
                train_rows.append(row)

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    # Statistics
    print(f"\n[INFO] Group-aware split completed:")
    print(f"  - Total classes: {df['species'].nunique()}")
    print(f"  - Total images: {total_images}")
    print(f"  - Total groups: {total_groups}")
    print(f"  - Average images per group: {total_images / max(1, total_groups):.2f}")
    print(f"\n[INFO] Split sizes:")
    print(f"  - Train: {len(df_train)} images ({100*len(df_train)/max(1,total_images):.1f}%)")
    print(f"  - Val: {len(df_val)} images ({100*len(df_val)/max(1,total_images):.1f}%)")
    print(f"  - Test: {len(df_test)} images ({100*len(df_test)/max(1,total_images):.1f}%)")

    return df_train, df_val, df_test


def handle_leakage_minor(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                         all_class_imgs: Dict[str, List[Path]], leakage_result: Dict,
                         out_dir: Path, threshold=5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """
    Handle minor leakage (<5%):
    1. Move leaked images from val/test to train
    2. Replace with non-leaked images from unselected pool
    3. Verify no new leakage

    Returns: updated df_train, df_val, df_test, success flag
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
    selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())

    # Get unselected images per class
    unselected_per_class = {}
    for cls, all_imgs in all_class_imgs.items():
        unselected_per_class[cls] = [str(p) for p in all_imgs if str(p) not in selected_images]

    # Compute hashes for all train images (for checking new candidates)
    print("\n[STEP 1] Computing pHash for all train images...")
    train_hashes = compute_phash_for_paths(df_train["path"].tolist())

    # Process each class
    replacement_stats = {'val': 0, 'test': 0, 'failed': 0}

    # Convert to lists for manipulation
    train_rows = df_train.to_dict('records')
    val_rows = df_val.to_dict('records')
    test_rows = df_test.to_dict('records')

    # Fix VAL leakage
    print("\n[STEP 2] Fixing VAL leakage...")
    for cls, leaked_paths in leaked_from_val.items():
        print(f"\n  Class '{cls}': {len(leaked_paths)} leaked images")

        for leaked_path in leaked_paths:
            # Find and move leaked image from val to train
            leaked_row = None
            for i, row in enumerate(val_rows):
                if row["path"] == leaked_path:
                    leaked_row = val_rows.pop(i)
                    break

            if leaked_row:
                train_rows.append(leaked_row)
                h = compute_phash(leaked_path)
                if h:
                    train_hashes[leaked_path] = h
                print(f"    → Moved to train: {Path(leaked_path).name}")

                # Find replacement from unselected
                found_replacement = False
                candidates = unselected_per_class.get(cls, [])

                for candidate in candidates:
                    if not check_image_leakage_with_train(candidate, train_hashes, threshold):
                        # No leakage - use this candidate
                        # Create new row with same structure
                        new_row = leaked_row.copy()
                        new_row["path"] = candidate
                        val_rows.append(new_row)
                        unselected_per_class[cls].remove(candidate)
                        selected_images.add(candidate)
                        replacement_stats['val'] += 1
                        found_replacement = True
                        print(f"    ← Replaced with: {Path(candidate).name}")
                        break

                if not found_replacement:
                    print(f"    ⚠ No suitable replacement found for val")
                    replacement_stats['failed'] += 1

    # Fix TEST leakage
    print("\n[STEP 3] Fixing TEST leakage...")
    for cls, leaked_paths in leaked_from_test.items():
        print(f"\n  Class '{cls}': {len(leaked_paths)} leaked images")

        for leaked_path in leaked_paths:
            # Find and move leaked image from test to train
            leaked_row = None
            for i, row in enumerate(test_rows):
                if row["path"] == leaked_path:
                    leaked_row = test_rows.pop(i)
                    break

            if leaked_row:
                train_rows.append(leaked_row)
                h = compute_phash(leaked_path)
                if h:
                    train_hashes[leaked_path] = h
                print(f"    → Moved to train: {Path(leaked_path).name}")

                # Find replacement from unselected
                found_replacement = False
                candidates = unselected_per_class.get(cls, [])

                for candidate in candidates:
                    if not check_image_leakage_with_train(candidate, train_hashes, threshold):
                        # No leakage - use this candidate
                        new_row = leaked_row.copy()
                        new_row["path"] = candidate
                        test_rows.append(new_row)
                        unselected_per_class[cls].remove(candidate)
                        selected_images.add(candidate)
                        replacement_stats['test'] += 1
                        found_replacement = True
                        print(f"    ← Replaced with: {Path(candidate).name}")
                        break

                if not found_replacement:
                    print(f"    ⚠ No suitable replacement found for test")
                    replacement_stats['failed'] += 1

    print(f"\n[SUMMARY] Replacement statistics:")
    print(f"  - Val replacements: {replacement_stats['val']}")
    print(f"  - Test replacements: {replacement_stats['test']}")
    print(f"  - Failed replacements: {replacement_stats['failed']}")

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows), replacement_stats['failed'] == 0


def handle_leakage_major(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.2,
                         hash_size=8, threshold=5, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Handle major leakage (>=5%):
    Rebuild train/val/test using group-aware splitting.

    Returns: df_train, df_val, df_test
    """
    print("\n" + "=" * 80)
    print("[LEAKAGE FIX] Major leakage detected (>=5%). Rebuilding with group-aware split...")
    print("=" * 80)

    df_train, df_val, df_test = group_aware_split(
        df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        hash_size=hash_size,
        threshold=threshold,
        seed=seed
    )

    return df_train, df_val, df_test


def handle_data_leakage(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                        df_all: pd.DataFrame, all_class_imgs: Dict[str, List[Path]],
                        out_dir: Path, val_ratio=0.1, test_ratio=0.2,
                        hash_size=8, threshold=5, max_iterations=3, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main function to detect and handle data leakage.

    Strategy:
    - If leakage < 5%: Move leaked images to train, replace from unselected pool
    - If leakage >= 5%: Rebuild using group-aware split

    Returns: final df_train, df_val, df_test, leakage_report
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
            df_train, df_val, df_test, out_dir,
            hash_size=hash_size, threshold=threshold
        )

        if not leakage_result.get('leakage_found', False):
            print("\n✅ No data leakage detected. Dataset is clean!")
            return df_train, df_val, df_test, leakage_result

        # Calculate leakage percentage
        total_val = len(df_val)
        total_test = len(df_test)
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
            df_train, df_val, df_test, success = handle_leakage_minor(
                df_train, df_val, df_test, all_class_imgs,
                leakage_result, out_dir, threshold=threshold
            )

            if not success:
                print("\n⚠ Some replacements failed. May need to use group-aware split.")
        else:
            # Major leakage: rebuild with group-aware split
            print(f"\n[DECISION] Leakage >= 5% → Applying major fix (group-aware split)")
            df_train, df_val, df_test = handle_leakage_major(
                df_all,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                hash_size=hash_size,
                threshold=threshold,
                seed=seed
            )

    # Final check
    print(f"\n{'─' * 40}")
    print(f"[Final Check after {max_iterations} iterations]")
    print(f"{'─' * 40}")

    final_result = check_data_leakage_phash(
        df_train, df_val, df_test, out_dir,
        hash_size=hash_size, threshold=threshold
    )

    if final_result.get('leakage_found', False):
        print("\n⚠ WARNING: Some leakage still remains after fixing attempts.")
        print("  Consider manual review or increasing threshold.")
    else:
        print("\n✅ All leakage has been successfully resolved!")

    return df_train, df_val, df_test, final_result


def create_dataset_unselected_csv(all_class_imgs: Dict[str, List[Path]],
                                   df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                                   out_dir: Path, label_map: Dict[str, int]):
    """
    Create dataset_unselected.csv containing images not selected in train/val/test.
    """
    # Collect all selected images
    selected_images = set(df_train["path"].tolist() + df_val["path"].tolist() + df_test["path"].tolist())

    # Count statistics
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = len(selected_images)
    total_unselected = total_available - total_selected

    # Create unselected CSV
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
    print(f"  - Total available: {total_available} images")
    print(f"  - Selected: {total_selected} images")
    print(f"  - Unselected: {total_unselected} images")


# -----------------------------
# Dataset curation
# -----------------------------

def build_selection_for_species(species_dir: Path, parts_keep=("hand","leaf","flower","fruit"), per_class_cap=100, seed=42):
    rng = random.Random(seed)
    sub_images: List[Path] = []

    for part in parts_keep:
        part_dir = species_dir / part
        if part_dir.exists() and part_dir.is_dir():
            imgs = [p for p in part_dir.rglob("*") if is_image_file(p)]
            sub_images.extend(imgs)

    # unique + shuffle
    sub_images = list(dict.fromkeys(sub_images))
    rng.shuffle(sub_images)

    if len(sub_images) >= per_class_cap:
        chosen = sub_images[:per_class_cap]
        return chosen

    available_images = list_images_direct(species_dir)
    rng.shuffle(available_images)
    need = per_class_cap - len(sub_images)
    chosen = sub_images + available_images[:need]
    return chosen


def scan_dataset(data_root: Path, parts_keep=("hand","leaf","flower","fruit"), per_class_cap=100, seed=42) -> pd.DataFrame:
    species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    species_names = [p.name for p in species_dirs]
    species_to_id = {name: i for i, name in enumerate(species_names)}

    rows = []
    for species_dir in species_dirs:
        species = species_dir.name
        selected_paths = build_selection_for_species(species_dir, parts_keep, per_class_cap, seed)
        for img in selected_paths:
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
# Torch Dataset & Transforms
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


def get_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15, fill=tuple(int(x*255) for x in mean)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfms, eval_tfms


# -----------------------------
# Model
# -----------------------------

def build_mobilenetv3_large(num_classes: int):
    try:
        import timm
        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        pass
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    tv_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = tv_model.classifier[3].in_features
    tv_model.classifier[3] = nn.Linear(in_features, num_classes)
    return tv_model, "torchvision"


# -----------------------------
# Train / Eval
# -----------------------------

from torch.amp import autocast, GradScaler


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and torch.cuda.is_available():
            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs_mnv3_cap100_70_10_20")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="validation ratio on the full dataset")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="test ratio on the full dataset")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    assert 0.0 < args.test_ratio < 0.5, "test_ratio should be reasonable (e.g. 0.2 for 20%)"
    assert 0.0 < args.val_ratio < 0.5,  "val_ratio should be reasonable (e.g. 0.1 for 10%)"
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

    print("\nScanning dataset & selecting images per class (cap=100)...")
    df = scan_dataset(data_root=data_root, per_class_cap=100, seed=args.seed)
    assert len(df) > 0, "No valid images found!"
    df.to_csv(out_dir / "dataset_selected.csv", index=False)

    species_list = sorted(df["species"].unique().tolist())
    label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"Selected images: {len(df)}; classes: {len(species_list)}")

    # ---- Collect ALL images for unselected tracking ----
    print("\nCollecting ALL available images from dataset...")
    all_class_imgs = collect_all_images_from_dataset(data_root)
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    print(f"Total available images: {total_available}")

    # ---- Split 70:10:20 ----
    print("\nSplitting train/val/test (target ≈ 70/10/20)...")
    df_trainval, df_test = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=df["species"],
    )
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
        f"Actual ratios → train: {len(df_train)/total_n:.3f}, val: {len(df_val)/total_n:.3f}, test: {len(df_test)/total_n:.3f}"
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
    num_classes = len(species_list)
    train_tfms, eval_tfms = get_transforms(img_size=args.img_size)

    ds_train = PlantImageDataset(df_train, transform=train_tfms)
    ds_val   = PlantImageDataset(df_val,   transform=eval_tfms)
    ds_test  = PlantImageDataset(df_test,  transform=eval_tfms)

    if args.use_weighted_sampler:
        class_counts = df_train["label_id"].value_counts().sort_index().values
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = df_train["label_id"].map(lambda x: class_weights[x]).values
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backend = build_mobilenetv3_large(num_classes=num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    print(f"Start training on {device} | backend={backend}")
    best_val_acc = 0.0
    best_ckpt = out_dir / "best_model.pt"
    patience = 7
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
        scheduler.step()
        print(f"Train | loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

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
            }
            torch.save(state, best_ckpt)
            print(f"✅ Saved best checkpoint to {best_ckpt} (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint (val_acc={ckpt.get('val_acc', -1):.4f})")

    # ---- Final Test ----
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
            }
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
