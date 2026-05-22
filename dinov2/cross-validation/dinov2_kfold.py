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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

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
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
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
            print(f"  {cls}: {len(uniq)} images")
    return all_class_imgs


def collect_images_per_class(data_root, max_per_class=100, verbose=True):
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
            selected = selected[:max_per_class]
        seen = set(); uniq = []
        for p in selected:
            if p not in seen: seen.add(p); uniq.append(p)
        selected_class_imgs[cls] = uniq
        if verbose:
            print(f"  {cls}: selected {len(uniq)} images")
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
    def __init__(self, model_name='dinov2_vitb14', n_classes=100,
                 pretrained=True, freeze_backbone=True, dropout=0.1):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        print(f"[MODEL] Loading DINOv2 backbone: {model_name}")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone(dummy)
        self.embed_dim = feat.shape[-1]
        print(f"[MODEL] DINOv2 embedding dim: {self.embed_dim}")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[MODEL] Backbone frozen (only classifier head is trainable)")
        else:
            print("[MODEL] Backbone unfrozen (full fine-tuning)")
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, n_classes)
        )

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits


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
            if h not in hash_to_paths: hash_to_paths[h] = []
            hash_to_paths[h].append((path, split, cls))

    print("\n[STEP 2] Finding EXACT duplicates...")
    exact_cross_split = []
    for h, items in hash_to_paths.items():
        if len(items) > 1:
            splits = set(item[1] for item in items)
            if len(splits) > 1:
                exact_cross_split.append({'hash': h, 'items': items, 'splits': list(splits)})

    print(f"\n[STEP 3] Finding NEAR duplicates (hamming distance <= {threshold})...")
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
                if 0 < dist <= threshold and s1 != s2:
                    near_cross_split.append({'path1': p1, 'split1': s1, 'class1': c1,
                                             'path2': p2, 'split2': s2, 'class2': c2, 'distance': dist})

    leakage_found = len(exact_cross_split) > 0 or len(near_cross_split) > 0
    if leakage_found:
        print(f"  🚨 CROSS-SPLIT EXACT: {len(exact_cross_split)}, NEAR: {len(near_cross_split)}")
    else:
        print("  ✅ No data leakage detected")

    # Save report
    report_data = []
    for group in exact_cross_split:
        for item in group['items']:
            report_data.append({'type': 'exact', 'path': item[0], 'split': item[1], 'class': item[2], 'distance': 0})
    for record in near_cross_split:
        report_data.append({'type': 'near', 'path': record['path1'], 'split': record['split1'],
                            'class': record['class1'], 'distance': record['distance']})
    if report_data:
        pd.DataFrame(report_data).to_csv(os.path.join(out_dir, 'data_leakage_check.csv'), index=False)

    return {
        'status': 'completed', 'leakage_found': leakage_found,
        'exact_cross_split': len(exact_cross_split), 'near_cross_split': len(near_cross_split),
        'exact_cross_split_details': exact_cross_split, 'near_cross_split_details': near_cross_split
    }


def handle_leakage_in_fold(train_map, val_map, test_map, fold_dir,
                            hash_size=8, threshold=5):
    leakage_result = check_data_leakage_phash(train_map, val_map, test_map, fold_dir,
                                               hash_size=hash_size, threshold=threshold)
    if not leakage_result.get('leakage_found', False):
        return train_map, val_map, test_map, leakage_result

    print("\n[LEAKAGE FIX] Moving leaked images to train...")
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

    moved_val, moved_test = 0, 0
    for cls, paths in leaked_from_val.items():
        for p in paths:
            if p in val_map.get(cls, []):
                val_map[cls].remove(p)
                train_map.setdefault(cls, []).append(p)
                moved_val += 1
    for cls, paths in leaked_from_test.items():
        for p in paths:
            if p in test_map.get(cls, []):
                test_map[cls].remove(p)
                train_map.setdefault(cls, []).append(p)
                moved_test += 1

    print(f"  Moved from val: {moved_val}, from test: {moved_test}")
    leakage_result['fixed'] = True
    return train_map, val_map, test_map, leakage_result


# ──────────────────────────────────────────────
# Save utilities
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
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, f'confusion_matrix_{phase}_normalized.csv'))


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
    pd.DataFrame(report_data).to_csv(os.path.join(out_dir, f'classification_report_{phase}.csv'), index=False)


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    if hasattr(model, 'freeze_backbone') and model.freeze_backbone:
        model.backbone.eval()
    total_loss, total_acc, n = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    epoch_start = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for imgs, labels, paths in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
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
    return {'loss': total_loss / n, 'acc': total_acc / n, 'time': time.time() - epoch_start}


def evaluate(model, dataloader, device, class_names, phase='val'):
    model.eval()
    y_true, y_pred, paths_all, all_probs = [], [], [], []
    eval_start = time.time()
    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc=f"[{phase.upper()}]"):
            imgs, labels = imgs.to(device), labels.to(device)
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
    cls_report = classification_report(y_true, y_pred, target_names=class_names,
                                        labels=range(len(class_names)), zero_division=0, output_dict=True)
    return {
        'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1,
        'accuracy': accuracy, 'per_precision': per_p, 'per_recall': per_r,
        'per_f1': per_f1, 'per_support': per_support,
        'y_true': y_true, 'y_pred': y_pred, 'paths': paths_all, 'probs': all_probs,
        'confusion_matrix': cm, 'classification_report': cls_report, 'time': eval_time
    }


# ──────────────────────────────────────────────
# Main K-Fold Cross-Validation
# ──────────────────────────────────────────────
def main(args):
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] DINOv2 K-Fold Cross-Validation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Number of folds: {args.n_folds}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Collect images
    print("\n[STEP 1] Collecting images...")
    class_to_imgs, all_class_imgs = collect_images_per_class(args.data_root, max_per_class=args.max_per_class, verbose=True)
    total_available = sum(len(imgs) for imgs in all_class_imgs.values())
    total_selected = sum(len(imgs) for imgs in class_to_imgs.values())
    print(f"\n  Total available: {total_available}, Selected: {total_selected}, Unselected: {total_available - total_selected}")

    classes = sorted(class_to_imgs.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Prepare samples for stratified splitting
    all_samples = []
    all_labels = []
    for cls in classes:
        for img_path in class_to_imgs[cls]:
            all_samples.append(img_path)
            all_labels.append(class_to_idx[cls])
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)
    print(f"[INFO] Total samples for K-Fold: {len(all_samples)}, Classes: {n_classes}")

    # Save dataset_unselected.csv (images not selected at all)
    selected_set = set(all_samples)
    with open(os.path.join(args.out_dir, 'dataset_unselected.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['class_name', 'class_idx', 'image_path'])
        for cls, imgs in all_class_imgs.items():
            for img in imgs:
                if img not in selected_set:
                    w.writerow([cls, class_to_idx[cls], img])

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

    # 2) K-Fold Cross-Validation
    print(f"\n[STEP 2] Starting {args.n_folds}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    all_fold_cms = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_samples, all_labels), 1):
        if fold < args.start_fold:
            print(f"\n[FOLD {fold}/{args.n_folds}] Skipped (start_fold={args.start_fold})")
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

        n_train = len(train_samples)
        n_val = int(n_train * args.val_split)
        indices = np.random.permutation(n_train)
        val_indices = indices[:n_val]
        train_indices_final = indices[n_val:]

        val_samples = train_samples[val_indices]
        val_labels = train_labels[val_indices]
        train_samples_final = train_samples[train_indices_final]
        train_labels_final = train_labels[train_indices_final]

        print(f"[FOLD {fold}] Train: {len(train_samples_final)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        # Build maps
        train_map = defaultdict(list)
        val_map = defaultdict(list)
        test_map = defaultdict(list)
        for path, label in zip(train_samples_final, train_labels_final):
            train_map[classes[label]].append(path)
        for path, label in zip(val_samples, val_labels):
            val_map[classes[label]].append(path)
        for path, label in zip(test_samples, test_labels):
            test_map[classes[label]].append(path)
        train_map = dict(train_map)
        val_map = dict(val_map)
        test_map = dict(test_map)

        # Create fold directory
        fold_dir = os.path.join(args.out_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # Save fold CSV splits
        for split_name, split_map in [('train', train_map), ('val', val_map), ('test', test_map)]:
            with open(os.path.join(fold_dir, f'{split_name}.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(['class_name', 'class_idx', 'image_path'])
                for cls, imgs in split_map.items():
                    for img in imgs: w.writerow([cls, class_to_idx[cls], img])

        # Check for data leakage
        if args.check_leakage:
            print(f"\n[FOLD {fold} - DATA LEAKAGE CHECK]")
            train_map, val_map, test_map, leakage_result = handle_leakage_in_fold(
                train_map, val_map, test_map, fold_dir,
                hash_size=args.phash_size, threshold=args.phash_threshold)
            train_count = sum(len(imgs) for imgs in train_map.values())
            val_count = sum(len(imgs) for imgs in val_map.values())
            test_count = sum(len(imgs) for imgs in test_map.values())
            print(f"[FOLD {fold}] After fix - Train: {train_count}, Val: {val_count}, Test: {test_count}")

        # Build datasets
        train_ds = BigPlantsDataset(train_map, class_to_idx, transform=train_tf)
        val_ds = BigPlantsDataset(val_map, class_to_idx, transform=val_tf)
        test_ds = BigPlantsDataset(test_map, class_to_idx, transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        # Create model (fresh for each fold)
        print(f"\n[FOLD {fold}] Creating DINOv2 model ({args.model_name})...")
        freeze_backbone = not args.unfreeze_backbone
        model = DINOv2Classifier(
            model_name=args.model_name, n_classes=n_classes,
            pretrained=True, freeze_backbone=freeze_backbone, dropout=args.dropout
        ).to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Training loop for this fold
        print(f"\n[FOLD {fold}] Training for {args.epochs} epochs...")
        best_macro = -1.0
        best_epoch = 0
        history = {'train_loss': [], 'train_acc': [], 'val_macro_f1': [], 'val_accuracy': []}

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = datetime.now()
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            val_metrics = evaluate(model, val_loader, device, classes, phase='val')

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['val_accuracy'].append(val_metrics['accuracy'])

            epoch_end_time = datetime.now()
            epoch_dur = (epoch_end_time - epoch_start_time).total_seconds()

            print(f"[FOLD {fold}] Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
                  f"Val Macro F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f} | "
                  f"Time: {epoch_dur:.1f}s")

            if val_metrics['macro_f1'] > best_macro:
                best_macro = val_metrics['macro_f1']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'macro_f1': best_macro, 'class_to_idx': class_to_idx,
                    'embed_dim': model.embed_dim, 'model_name': args.model_name,
                    'freeze_backbone': freeze_backbone, 'dropout': args.dropout,
                    'n_classes': n_classes, 'args': vars(args)
                }, os.path.join(fold_dir, 'best_model.pt'))
                print(f"  ★ New best model saved! (Macro F1: {best_macro:.4f})")

        # Save training history for this fold
        torch.save(history, os.path.join(fold_dir, 'training_history.pt'))

        # Load best model and evaluate on test
        print(f"\n[FOLD {fold}] Evaluating on test set with best model (epoch {best_epoch})...")
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pt'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate(model, test_loader, device, classes, phase='test')

        print(f"\n[FOLD {fold} TEST RESULTS]")
        print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
        print(f"  Micro F1:    {test_metrics['micro_f1']:.4f}")
        print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")

        save_classification_report(test_metrics['classification_report'], classes, fold_dir, phase='test')
        save_confusion_matrix(test_metrics['confusion_matrix'], classes, fold_dir, phase='test')

        fold_time = time.time() - fold_start

        fold_results.append({
            'fold': fold, 'best_epoch': best_epoch, 'best_val_macro_f1': best_macro,
            'test_macro_f1': test_metrics['macro_f1'], 'test_micro_f1': test_metrics['micro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'], 'test_accuracy': test_metrics['accuracy'],
            'fold_time': fold_time
        })
        all_fold_cms.append(test_metrics['confusion_matrix'])
        print(f"[FOLD {fold}] Completed in {fold_time/60:.2f} minutes")

    # 3) Aggregate results
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

    # Average confusion matrix
    if all_fold_cms:
        avg_cm = np.mean(all_fold_cms, axis=0)
        save_confusion_matrix(avg_cm.astype(int), classes, args.out_dir, phase='aggregated')

    # Summary
    summary = {
        'n_folds': args.n_folds,
        'model_name': args.model_name,
        'freeze_backbone': not args.unfreeze_backbone,
        'mean_test_macro_f1': df_results['test_macro_f1'].mean(),
        'std_test_macro_f1': df_results['test_macro_f1'].std(),
        'mean_test_accuracy': df_results['test_accuracy'].mean(),
        'std_test_accuracy': df_results['test_accuracy'].std(),
        'best_fold': int(df_results.loc[df_results['test_macro_f1'].idxmax(), 'fold']),
        'best_fold_macro_f1': df_results['test_macro_f1'].max()
    }
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 80)
    print(f"[END] K-Fold Cross-Validation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {total_duration/3600:.2f} hours ({total_duration:.0f} seconds)")
    print(f"[INFO] Best fold: {summary['best_fold']} with Macro F1: {summary['best_fold_macro_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINOv2 K-Fold Cross-Validation for BigPlants-100')

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs_kfold")
    parser.add_argument("--max_per_class", type=int, default=100)

    # K-Fold
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--start_fold", type=int, default=1,
                        help="Start from this fold (1-indexed). Use to resume after interruption.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                                 "dinov2_vitg14", "dinov2_vits14_reg", "dinov2_vitb14_reg",
                                 "dinov2_vitl14_reg", "dinov2_vitg14_reg"])
    parser.add_argument("--unfreeze_backbone", action='store_true', default=False)

    # Data Leakage Detection
    parser.add_argument("--check_leakage", action='store_true', default=True)
    parser.add_argument("--no_check_leakage", action='store_false', dest='check_leakage')
    parser.add_argument("--phash_size", type=int, default=8)
    parser.add_argument("--phash_threshold", type=int, default=5)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
