import os
import csv
import math
import time
import argparse
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.cluster import KMeans
from sklearn.metrics import (
	f1_score,
	precision_recall_fscore_support,
	confusion_matrix,
	classification_report,
)

try:
	from transformers import SegformerModel, SegformerConfig
except ImportError:
	SegformerModel = None
	SegformerConfig = None

try:
	import imagehash
	PHASH_AVAILABLE = True
except ImportError:
	PHASH_AVAILABLE = False
	print("[WARNING] imagehash not installed. pHash leakage check disabled.")
	print("          Install with: pip install imagehash")


# ============================================================================
# Dataset selection rules
# ============================================================================
PRIOR_ORG_ORDER = ["hand", "leaf", "flower", "fruit"]
SECOND_ORG_ORDER = ["seed", "root"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_img_file(p: str) -> bool:
	return p.lower().endswith(IMG_EXTS)


def dedup_keep_order(paths):
	seen = set()
	out = []
	for p in paths:
		if p not in seen:
			out.append(p)
			seen.add(p)
	return out


def collect_all_images(data_root, verbose=True):
	"""Collect all available images for each class."""
	classes = sorted([
		d for d in os.listdir(data_root)
		if os.path.isdir(os.path.join(data_root, d))
	])
	all_class_imgs = {}

	for cls in classes:
		cls_dir = os.path.join(data_root, cls)
		all_imgs = []

		for sub in PRIOR_ORG_ORDER + SECOND_ORG_ORDER:
			sub_dir = os.path.join(cls_dir, sub)
			if os.path.isdir(sub_dir):
				sub_imgs = [
					os.path.join(sub_dir, f)
					for f in sorted(os.listdir(sub_dir))
					if is_img_file(f)
				]
				all_imgs.extend(sub_imgs)

		# available images directly under class folder
		root_imgs = [
			os.path.join(cls_dir, f)
			for f in sorted(os.listdir(cls_dir))
			if is_img_file(f)
		]
		all_imgs.extend(root_imgs)

		all_class_imgs[cls] = dedup_keep_order(all_imgs)
		if verbose:
			print(f"[COLLECT-ALL] {cls:<35s}: {len(all_class_imgs[cls])} images")

	return all_class_imgs


def collect_images_per_class(data_root, max_per_class=100, verbose=True):
	"""
	Select up to max_per_class by priority:
	  hand, leaf, flower, fruit -> seed, root, stem -> available(root files)
	"""
	all_class_imgs = collect_all_images(data_root, verbose=False)
	classes = sorted(all_class_imgs.keys())
	selected_class_imgs = {}

	for cls in classes:
		cls_dir = os.path.join(data_root, cls)
		selected = []

		# first priority
		for sub in PRIOR_ORG_ORDER:
			sub_dir = os.path.join(cls_dir, sub)
			if not os.path.isdir(sub_dir):
				continue
			sub_imgs = [
				os.path.join(sub_dir, f)
				for f in sorted(os.listdir(sub_dir))
				if is_img_file(f)
			]
			selected.extend(sub_imgs)
			if len(selected) >= max_per_class:
				break

		# second priority
		if len(selected) < max_per_class:
			for sub in SECOND_ORG_ORDER:
				sub_dir = os.path.join(cls_dir, sub)
				if not os.path.isdir(sub_dir):
					continue
				sub_imgs = [
					os.path.join(sub_dir, f)
					for f in sorted(os.listdir(sub_dir))
					if is_img_file(f)
				]
				selected.extend(sub_imgs)
				if len(selected) >= max_per_class:
					break

		# available in class root
		if len(selected) < max_per_class:
			root_imgs = [
				os.path.join(cls_dir, f)
				for f in sorted(os.listdir(cls_dir))
				if is_img_file(f)
			]
			selected.extend(root_imgs)

		selected = dedup_keep_order(selected)[:max_per_class]
		selected_class_imgs[cls] = selected

		if verbose:
			print(
				f"[SELECT] {cls:<35s}: selected={len(selected):4d} | "
				f"available={len(all_class_imgs.get(cls, [])):4d}"
			)

	return selected_class_imgs, all_class_imgs


# ============================================================================
# Split / CSV helpers
# ============================================================================
def build_split_maps(class_to_imgs, val_split=0.1, test_split=0.2, seed=42):
	classes = sorted(list(class_to_imgs.keys()))
	class_to_idx = {c: i for i, c in enumerate(classes)}
	train_map, val_map, test_map = {}, {}, {}

	for c, imgs in class_to_imgs.items():
		imgs = list(imgs)
		rnd = random.Random((hash(c) ^ seed) & 0xFFFFFFFF)
		rnd.shuffle(imgs)

		n = len(imgs)
		n_test = int(math.ceil(test_split * n))
		n_val = int(math.ceil(val_split * n))

		if n_test + n_val >= n and n > 2:
			n_test = max(1, n_test - 1)
		if n_test + n_val >= n and n > 1:
			n_val = max(1, n_val - 1)

		test = imgs[:n_test]
		val = imgs[n_test:n_test + n_val]
		train = imgs[n_test + n_val:]

		if len(train) == 0 and len(imgs) > 0:
			train = [imgs[-1]]
			if test:
				test = test[:-1]
			elif val:
				val = val[:-1]

		train_map[c] = train
		val_map[c] = val
		test_map[c] = test

	return class_to_idx, train_map, val_map, test_map


def save_dataset_splits(out_dir, all_class_imgs, train_map, val_map, test_map, class_to_idx):
	os.makedirs(out_dir, exist_ok=True)

	selected_images = set()
	for split_map in [train_map, val_map, test_map]:
		for imgs in split_map.values():
			selected_images.update(imgs)

	# dataset_selected.csv
	selected_path = os.path.join(out_dir, "dataset_selected.csv")
	with open(selected_path, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["class_name", "class_idx", "image_path", "split"])
		for split_name, split_map in [("train", train_map), ("val", val_map), ("test", test_map)]:
			for cls, imgs in split_map.items():
				idx = class_to_idx[cls]
				for p in imgs:
					w.writerow([cls, idx, p, split_name])

	# dataset_unselected.csv
	unselected_path = os.path.join(out_dir, "dataset_unselected.csv")
	with open(unselected_path, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["class_name", "class_idx", "image_path"])
		for cls, all_imgs in all_class_imgs.items():
			idx = class_to_idx.get(cls, -1)
			for p in all_imgs:
				if p not in selected_images:
					w.writerow([cls, idx, p])

	# train/val/test csv
	for split_name, split_map in [("train", train_map), ("val", val_map), ("test", test_map)]:
		split_csv = os.path.join(out_dir, f"{split_name}.csv")
		with open(split_csv, "w", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			w.writerow(["class_name", "class_idx", "image_path"])
			for cls, imgs in split_map.items():
				idx = class_to_idx[cls]
				for p in imgs:
					w.writerow([cls, idx, p])

	total_available = sum(len(v) for v in all_class_imgs.values())
	total_selected = len(selected_images)
	print(f"[INFO] Saved split CSVs to {out_dir}")
	print(f"  - total available : {total_available}")
	print(f"  - total selected  : {total_selected}")
	print(f"  - total unselected: {total_available - total_selected}")


def save_confusion_matrix(cm, class_names, out_dir, phase="test"):
	os.makedirs(out_dir, exist_ok=True)

	cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
	cm_df.to_csv(os.path.join(out_dir, f"confusion_matrix_{phase}.csv"))

	if len(class_names) <= 50:
		plt.figure(figsize=(20, 18))
		sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
		plt.xlabel("Predicted")
		plt.ylabel("True")
		plt.title(f"Confusion Matrix - {phase.upper()}")
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f"confusion_matrix_{phase}.png"), dpi=150)
		plt.close()

	cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
	pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
		os.path.join(out_dir, f"confusion_matrix_{phase}_normalized.csv")
	)


def save_classification_report(cls_report, class_names, out_dir, phase="test"):
	os.makedirs(out_dir, exist_ok=True)
	rows = []
	for cls in class_names:
		if cls in cls_report:
			rows.append({
				"class_name": cls,
				"precision": cls_report[cls]["precision"],
				"recall": cls_report[cls]["recall"],
				"f1-score": cls_report[cls]["f1-score"],
				"support": cls_report[cls]["support"],
			})

	for avg_name in ["macro avg", "weighted avg"]:
		if avg_name in cls_report:
			rows.append({
				"class_name": avg_name,
				"precision": cls_report[avg_name]["precision"],
				"recall": cls_report[avg_name]["recall"],
				"f1-score": cls_report[avg_name]["f1-score"],
				"support": cls_report[avg_name]["support"],
			})

	pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"classification_report_{phase}.csv"), index=False)


# ============================================================================
# pHash leakage detection/fixing
# ============================================================================
def compute_phash(img_path, hash_size=8):
	if not PHASH_AVAILABLE:
		return None
	try:
		img = Image.open(img_path).convert("RGB")
		return str(imagehash.phash(img, hash_size=hash_size))
	except Exception:
		return None


def hamming_distance_hex(h1, h2):
	try:
		return (int(h1, 16) ^ int(h2, 16)).bit_count()
	except Exception:
		return 999999


def collect_split_items(split_map, split_name):
	items = []
	for cls, imgs in split_map.items():
		for p in imgs:
			items.append((p, split_name, cls))
	return items


def check_data_leakage_phash(train_map, val_map, test_map, out_dir, hash_size=8, threshold=5):
	if not PHASH_AVAILABLE:
		print("[WARNING] pHash unavailable. Skip leakage check.")
		return {"status": "skipped", "reason": "imagehash not installed", "leakage_found": False}

	print("\n" + "=" * 80)
	print("[DATA LEAKAGE CHECK] pHash cross-split duplicate search")
	print("=" * 80)

	items = (
		collect_split_items(train_map, "train")
		+ collect_split_items(val_map, "val")
		+ collect_split_items(test_map, "test")
	)
	print(f"[INFO] total images to check: {len(items)}")

	hash_map = {}
	hash_to_items = defaultdict(list)
	for p, sp, cls in tqdm(items, desc="Computing pHash"):
		h = compute_phash(p, hash_size=hash_size)
		if h is None:
			continue
		hash_map[p] = (h, sp, cls)
		hash_to_items[h].append((p, sp, cls))

	exact_cross = []
	exact_groups = 0
	for h, group in hash_to_items.items():
		if len(group) <= 1:
			continue
		exact_groups += 1
		splits = {x[1] for x in group}
		if len(splits) > 1:
			exact_cross.append({"hash": h, "items": group})

	# near duplicates by prefix buckets
	hash_bits = hash_size * hash_size
	prefix_bits = min(16, hash_bits)
	shift = hash_bits - prefix_bits
	buckets = defaultdict(list)

	for p, (h, sp, cls) in hash_map.items():
		try:
			prefix = int(h, 16) >> shift
			buckets[prefix].append((p, h, sp, cls))
		except Exception:
			continue

	near_pairs = 0
	near_cross = []
	checked_pairs = set()

	for bucket_items in tqdm(buckets.values(), desc="Checking near-duplicates"):
		n = len(bucket_items)
		for i in range(n):
			p1, h1, s1, c1 = bucket_items[i]
			for j in range(i + 1, n):
				p2, h2, s2, c2 = bucket_items[j]
				key = tuple(sorted([p1, p2]))
				if key in checked_pairs:
					continue
				checked_pairs.add(key)
				d = hamming_distance_hex(h1, h2)
				if d <= threshold:
					near_pairs += 1
					if s1 != s2:
						near_cross.append({
							"path1": p1,
							"path2": p2,
							"split1": s1,
							"split2": s2,
							"class1": c1,
							"class2": c2,
							"distance": d,
						})

	leakage_found = len(exact_cross) > 0 or len(near_cross) > 0
	print(f"[RESULT] exact_groups={exact_groups}, near_pairs={near_pairs}")
	print(f"[RESULT] cross_split_exact={len(exact_cross)}, cross_split_near={len(near_cross)}")
	print("[RESULT] leakage_found=", leakage_found)

	# Save detailed report
	report_rows = []
	for g in exact_cross:
		for p, sp, cls in g["items"]:
			report_rows.append({
				"type": "exact",
				"hash": g["hash"],
				"path": p,
				"split": sp,
				"class": cls,
				"distance": 0,
				"is_cross_split": True,
			})
	for r in near_cross:
		report_rows.append({
			"type": "near",
			"hash": "",
			"path": r["path1"],
			"split": r["split1"],
			"class": r["class1"],
			"distance": r["distance"],
			"is_cross_split": True,
			"paired_with": r["path2"],
			"paired_split": r["split2"],
		})
		report_rows.append({
			"type": "near",
			"hash": "",
			"path": r["path2"],
			"split": r["split2"],
			"class": r["class2"],
			"distance": r["distance"],
			"is_cross_split": True,
			"paired_with": r["path1"],
			"paired_split": r["split1"],
		})

	report_path = None
	if len(report_rows) > 0:
		report_path = os.path.join(out_dir, "data_leakage_check.csv")
		pd.DataFrame(report_rows).to_csv(report_path, index=False)
		print(f"[INFO] leakage report saved: {report_path}")

	return {
		"status": "completed",
		"leakage_found": leakage_found,
		"exact_duplicate_groups": exact_groups,
		"near_duplicate_pairs": near_pairs,
		"exact_cross_split": len(exact_cross),
		"near_cross_split": len(near_cross),
		"exact_cross_split_details": exact_cross,
		"near_cross_split_details": near_cross,
		"report_path": report_path,
	}


def compute_phash_for_paths(paths, hash_size=8):
	if not PHASH_AVAILABLE:
		return {}
	out = {}
	for p in tqdm(paths, desc="Computing pHash"):
		h = compute_phash(p, hash_size=hash_size)
		if h is not None:
			out[p] = h
	return out


def check_image_leakage_with_train(candidate_path, train_paths, train_hashes, threshold=5):
	if not PHASH_AVAILABLE:
		return False
	h = compute_phash(candidate_path)
	if h is None:
		return False
	h_int = int(h, 16)
	for p in train_paths:
		th = train_hashes.get(p)
		if th is None:
			continue
		if (h_int ^ int(th, 16)).bit_count() <= threshold:
			return True
	return False


def build_similarity_groups(imgs, hash_size=8, threshold=5):
	if not PHASH_AVAILABLE or len(imgs) == 0:
		return [[x] for x in imgs]

	hashes = {}
	for p in imgs:
		h = compute_phash(p, hash_size=hash_size)
		if h is not None:
			hashes[p] = int(h, 16)

	parent = {x: x for x in imgs}
	rank = {x: 0 for x in imgs}

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

	arr = list(imgs)
	for i in range(len(arr)):
		p1 = arr[i]
		if p1 not in hashes:
			continue
		h1 = hashes[p1]
		for j in range(i + 1, len(arr)):
			p2 = arr[j]
			if p2 not in hashes:
				continue
			h2 = hashes[p2]
			if (h1 ^ h2).bit_count() <= threshold:
				union(p1, p2)

	groups = defaultdict(list)
	for p in imgs:
		groups[find(p)].append(p)
	return list(groups.values())


def group_aware_split(class_to_imgs, val_split=0.1, test_split=0.2, hash_size=8, threshold=5, seed=42):
	train_map, val_map, test_map = {}, {}, {}
	for cls, imgs in tqdm(class_to_imgs.items(), desc="Group-aware split"):
		groups = build_similarity_groups(imgs, hash_size=hash_size, threshold=threshold)
		rnd = random.Random((hash(cls) ^ seed) & 0xFFFFFFFF)
		rnd.shuffle(groups)

		n = len(groups)
		n_test = max(1, int(round(n * test_split))) if n > 0 else 0
		n_val = max(1, int(round(n * val_split))) if n > 0 else 0
		n_train = n - n_test - n_val
		if n_train < 1 and n > 0:
			if n_test > 1:
				n_test -= 1
			elif n_val > 1:
				n_val -= 1

		test_groups = groups[:n_test]
		val_groups = groups[n_test:n_test + n_val]
		train_groups = groups[n_test + n_val:]

		train_map[cls] = [p for g in train_groups for p in g]
		val_map[cls] = [p for g in val_groups for p in g]
		test_map[cls] = [p for g in test_groups for p in g]

	return train_map, val_map, test_map


def handle_leakage_minor(train_map, val_map, test_map, all_class_imgs, leakage_result, threshold=5):
	leaked_from_val = defaultdict(set)
	leaked_from_test = defaultdict(set)

	for group in leakage_result.get("exact_cross_split_details", []):
		for p, split_name, cls in group["items"]:
			if split_name == "val":
				leaked_from_val[cls].add(p)
			elif split_name == "test":
				leaked_from_test[cls].add(p)

	for r in leakage_result.get("near_cross_split_details", []):
		for sk, ck, pk in [("split1", "class1", "path1"), ("split2", "class2", "path2")]:
			split_name = r[sk]
			cls = r[ck]
			p = r[pk]
			if split_name == "val":
				leaked_from_val[cls].add(p)
			elif split_name == "test":
				leaked_from_test[cls].add(p)

	selected = set()
	for m in [train_map, val_map, test_map]:
		for v in m.values():
			selected.update(v)

	unselected_per_class = {}
	for cls, all_imgs in all_class_imgs.items():
		unselected_per_class[cls] = [p for p in all_imgs if p not in selected]

	all_train_paths = [p for v in train_map.values() for p in v]
	train_hashes = compute_phash_for_paths(all_train_paths)
	stats = {"val": 0, "test": 0, "failed": 0}

	for split_name, leaked_dict, split_map in [
		("val", leaked_from_val, val_map),
		("test", leaked_from_test, test_map),
	]:
		for cls, leaked_paths in leaked_dict.items():
			for leaked_p in list(leaked_paths):
				if leaked_p in split_map.get(cls, []):
					split_map[cls].remove(leaked_p)
					train_map[cls].append(leaked_p)

				replaced = False
				candidates = unselected_per_class.get(cls, [])
				random.shuffle(candidates)
				for cand in candidates:
					if check_image_leakage_with_train(cand, train_map[cls], train_hashes, threshold=threshold):
						continue
					split_map[cls].append(cand)
					unselected_per_class[cls].remove(cand)
					stats[split_name] += 1
					replaced = True
					break
				if not replaced:
					stats["failed"] += 1

	return train_map, val_map, test_map, stats


def handle_data_leakage(
	train_map,
	val_map,
	test_map,
	class_to_imgs,
	all_class_imgs,
	out_dir,
	val_split=0.1,
	test_split=0.2,
	hash_size=8,
	threshold=5,
	max_iterations=3,
	seed=42,
):
	print("\n[DATA LEAKAGE HANDLER] Start...")
	for it in range(1, max_iterations + 1):
		print(f"\n--- Iteration {it}/{max_iterations} ---")
		res = check_data_leakage_phash(
			train_map, val_map, test_map, out_dir, hash_size=hash_size, threshold=threshold
		)
		if not res.get("leakage_found", False):
			print("[LEAKAGE] No leakage detected.")
			return train_map, val_map, test_map, res

		total_eval = sum(len(v) for v in val_map.values()) + sum(len(v) for v in test_map.values())
		n_leaked = res.get("exact_cross_split", 0) + res.get("near_cross_split", 0)
		leakage_pct = 100.0 * n_leaked / max(1, total_eval)
		print(f"[LEAKAGE] {n_leaked}/{total_eval} ({leakage_pct:.2f}%)")

		if leakage_pct < 5.0:
			train_map, val_map, test_map, stats = handle_leakage_minor(
				train_map, val_map, test_map, all_class_imgs, res, threshold=threshold
			)
			print(f"[LEAKAGE FIX - MINOR] {stats}")
		else:
			print("[LEAKAGE FIX - MAJOR] Rebuilding with group-aware split...")
			train_map, val_map, test_map = group_aware_split(
				class_to_imgs,
				val_split=val_split,
				test_split=test_split,
				hash_size=hash_size,
				threshold=threshold,
				seed=seed,
			)

	final_res = check_data_leakage_phash(
		train_map, val_map, test_map, out_dir, hash_size=hash_size, threshold=threshold
	)
	return train_map, val_map, test_map, final_res


# ============================================================================
# Dataset and pseudo-organ mining
# ============================================================================
class BigPlantsHybridDataset(Dataset):
	def __init__(self, class_to_imgs, class_to_idx, transform=None, pseudo_org=None, organ_dim=5):
		self.samples = []
		for cls, imgs in class_to_imgs.items():
			idx = class_to_idx[cls]
			for p in imgs:
				self.samples.append((p, idx, cls))
		self.transform = transform
		self.pseudo_org = pseudo_org or {}
		self.organ_dim = organ_dim

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, i):
		p, idx, _ = self.samples[i]
		img = Image.open(p).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		else:
			img = T.ToTensor()(img)

		if p in self.pseudo_org:
			prior = torch.tensor(self.pseudo_org[p], dtype=torch.float32)
		else:
			prior = torch.ones(self.organ_dim, dtype=torch.float32) / float(self.organ_dim)

		return img, idx, p, prior


class ResNet18FeatureExtractor(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)

	def forward(self, x):
		feat = self.backbone(x)
		if feat.dim() > 2:
			feat = feat.flatten(1)
		return feat


def extract_features_for_paths(paths, feature_extractor, device, batch_size=64, num_workers=4):
	transform = T.Compose([
		T.Resize((224, 224)),
		T.ToTensor(),
		T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
	])

	class PathDataset(Dataset):
		def __init__(self, pths):
			self.paths = pths

		def __len__(self):
			return len(self.paths)

		def __getitem__(self, i):
			p = self.paths[i]
			img = Image.open(p).convert("RGB")
			return transform(img), p

	if len(paths) == 0:
		return np.zeros((0, 1), dtype=np.float32), []

	ds = PathDataset(paths)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	feats = []
	out_paths = []
	feature_extractor.eval()

	with torch.no_grad():
		for imgs, pths in tqdm(loader, desc="Extracting features"):
			imgs = imgs.to(device)
			f = feature_extractor(imgs).detach().cpu().numpy()
			feats.append(f)
			out_paths.extend(list(pths))

	feats = np.vstack(feats) if len(feats) > 0 else np.zeros((0, 1), dtype=np.float32)
	return feats, out_paths


def generate_pseudo_organs_train_only(train_map, feature_extractor, device, n_clusters=5, batch_size=64):
	train_paths = [p for imgs in train_map.values() for p in imgs]
	print(f"[O1] Extracting TRAIN features only: {len(train_paths)} images")

	feats, paths = extract_features_for_paths(train_paths, feature_extractor, device, batch_size=batch_size)
	if feats.shape[0] == 0:
		return {}, None

	print(f"[O1] KMeans fit on TRAIN -> n_clusters={n_clusters}")
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats)
	labels = kmeans.labels_

	priors = {}
	for p, l in zip(paths, labels):
		v = np.zeros(n_clusters, dtype=np.float32)
		v[l] = 1.0
		priors[p] = v

	return priors, kmeans


def apply_kmeans_to_split(split_map, kmeans, feature_extractor, device, n_clusters=5, batch_size=64):
	all_paths = [p for imgs in split_map.values() for p in imgs]
	if len(all_paths) == 0 or kmeans is None:
		return {}

	print(f"[O1] Applying pre-fitted KMeans to split: {len(all_paths)} images")
	feats, paths = extract_features_for_paths(all_paths, feature_extractor, device, batch_size=batch_size)
	if feats.shape[0] == 0:
		return {}
	labels = kmeans.predict(feats)

	priors = {}
	for p, l in zip(paths, labels):
		v = np.zeros(n_clusters, dtype=np.float32)
		v[l] = 1.0
		priors[p] = v
	return priors


def build_dataloaders(train_map, val_map, test_map, class_to_idx, pseudo_org, organ_dim, batch_size, num_workers=4):
	train_tf = T.Compose([
		T.Resize((224, 224)),
		T.RandomResizedCrop(224, scale=(0.8, 1.0)),
		T.RandomHorizontalFlip(),
		T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
		T.ToTensor(),
		T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
	])
	val_tf = T.Compose([
		T.Resize((224, 224)),
		T.ToTensor(),
		T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
	])

	train_ds = BigPlantsHybridDataset(train_map, class_to_idx, transform=train_tf, pseudo_org=pseudo_org, organ_dim=organ_dim)
	val_ds = BigPlantsHybridDataset(val_map, class_to_idx, transform=val_tf, pseudo_org=pseudo_org, organ_dim=organ_dim)
	test_ds = BigPlantsHybridDataset(test_map, class_to_idx, transform=val_tf, pseudo_org=pseudo_org, organ_dim=organ_dim)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

	return train_loader, val_loader, test_loader


# ============================================================================
# Model blocks: Organ branch
# ============================================================================
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
		return self.linear(torch.cat([token, organ_prior], dim=-1))


class FFNExpert(nn.Module):
	def __init__(self, d_model, d_ff):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.GELU(),
			nn.Linear(d_ff, d_model),
		)

	def forward(self, x):
		return self.net(x)


class SwitchMoE(nn.Module):
	def __init__(self, d_model, organ_dim, n_experts=8, d_ff=2048, capacity_factor=1.25, top_k=1, entropy_threshold=1.5):
		super().__init__()
		self.n_experts = n_experts
		self.capacity_factor = capacity_factor
		self.top_k = top_k
		self.entropy_threshold = entropy_threshold
		self.router = Router(d_model, organ_dim, n_experts)
		self.experts = nn.ModuleList([FFNExpert(d_model, d_ff) for _ in range(n_experts)])
		self.register_buffer("expert_usage", torch.zeros(n_experts))

	def forward(self, tokens, organ_priors, training=True):
		# tokens: (B, T, D), organ_priors: (B, T, O)
		B, T, D = tokens.shape
		flat = tokens.reshape(B * T, D)
		flat_prior = organ_priors.reshape(B * T, -1)

		logits = self.router(flat, flat_prior)
		probs = F.softmax(logits, dim=-1)
		entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)

		# coarse-to-fine routing:
		# training -> allow top2 fallback
		# eval -> use configured top_k
		k = min(2, self.n_experts) if training else min(self.top_k, self.n_experts)
		topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
		topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)

		out = torch.zeros_like(flat)
		processed = torch.zeros(B * T, dtype=torch.bool, device=flat.device)
		capacity = int(math.ceil((B * T / self.n_experts) * self.capacity_factor))

		for e in range(self.n_experts):
			mask = (topk_idx == e).any(dim=-1)
			idx = mask.nonzero(as_tuple=True)[0]
			if idx.numel() == 0:
				continue

			# capacity limit
			if idx.numel() > capacity:
				ep = probs[idx, e]
				_, ord_idx = torch.sort(ep, descending=True)
				idx = idx[ord_idx[:capacity]]

			selected = flat[idx]
			e_out = self.experts[e](selected)

			w = torch.zeros(idx.shape[0], device=flat.device)
			for i, tid in enumerate(idx):
				m = (topk_idx[tid] == e)
				if m.any():
					pos = m.nonzero(as_tuple=True)[0][0]
					w[i] = topk_probs[tid, pos]

			out[idx] += e_out * w.unsqueeze(-1)
			processed[idx] = True

		out[~processed] = flat[~processed]
		out = out.reshape(B, T, D)

		batch_usage = probs.mean(dim=0)
		if training:
			self.expert_usage.mul_(0.9).add_(0.1 * batch_usage.detach())

		return out, probs.reshape(B, T, -1), entropy.reshape(B, T), batch_usage


class OrganAwareSwitchViTBranch(nn.Module):
	def __init__(
		self,
		vit_name="vit_base_patch16_224",
		n_classes=100,
		organ_dim=5,
		n_experts=8,
		d_ff_expert=1024,
		top_k=1,
		pretrained=True,
	):
		super().__init__()
		self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)

		with torch.no_grad():
			dummy = torch.randn(1, 3, 224, 224)
			feat = self.backbone.forward_features(dummy) if hasattr(self.backbone, "forward_features") else self.backbone(dummy)
			if isinstance(feat, torch.Tensor) and feat.dim() == 3:
				token_dim = feat.shape[-1]
			elif isinstance(feat, torch.Tensor) and feat.dim() == 2:
				token_dim = feat.shape[-1]
			else:
				token_dim = 768

		self.token_dim = token_dim
		self.organ_dim = organ_dim
		self.switch = SwitchMoE(
			d_model=token_dim,
			organ_dim=organ_dim,
			n_experts=n_experts,
			d_ff=d_ff_expert,
			top_k=top_k,
		)
		self.ln = nn.LayerNorm(token_dim)
		self.cls_head = nn.Linear(token_dim, n_classes)
		self.aux_head = OrganAuxHead(token_dim, organ_dim)

	def forward(self, x, organ_priors_image, training=True):
		tokens = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)

		if tokens.dim() == 2:
			cls = tokens
			patches = tokens.unsqueeze(1)
		else:
			cls = tokens[:, 0, :]
			patches = tokens[:, 1:, :]

		B, T, _ = patches.shape
		org_tokens = organ_priors_image.unsqueeze(1).expand(B, T, -1)

		patches_out, probs, entropy, batch_usage = self.switch(patches, org_tokens, training=training)
		pooled_patch = patches_out.mean(dim=1)

		fused_cls = self.ln(cls + pooled_patch)
		species_logits = self.cls_head(fused_cls)
		aux_org_logits = self.aux_head(cls)

		return species_logits, aux_org_logits, probs, entropy, batch_usage


# ============================================================================
# DINOv2 branch
# ============================================================================
class DINOv2Branch(nn.Module):
	def __init__(self, model_name="dinov2_vitb14", n_classes=100, dropout=0.1, freeze_backbone=True):
		super().__init__()
		self.freeze_backbone = freeze_backbone
		print(f"[MODEL] Loading DINOv2 backbone: {model_name}")
		self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

		with torch.no_grad():
			dummy = torch.randn(1, 3, 224, 224)
			feat = self.backbone(dummy)
			embed_dim = feat.shape[-1]

		if freeze_backbone:
			for p in self.backbone.parameters():
				p.requires_grad = False

		self.head = nn.Sequential(
			nn.LayerNorm(embed_dim),
			nn.Dropout(dropout),
			nn.Linear(embed_dim, n_classes),
		)

	def forward(self, x):
		if self.freeze_backbone:
			with torch.no_grad():
				feat = self.backbone(x)
		else:
			feat = self.backbone(x)
		return self.head(feat)


# ============================================================================
# SegFormer branch (classification view)
# ============================================================================
class SegFormerClassifierBranch(nn.Module):
	def __init__(self, model_name="nvidia/segformer-b1-finetuned-ade-512-512", n_classes=100, dropout=0.1, freeze_backbone=False):
		super().__init__()
		self.use_fallback = False

		if SegformerModel is None:
			print("[WARNING] transformers not installed. SegFormer branch fallback -> resnet18")
			self.use_fallback = True
			self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
			with torch.no_grad():
				dummy = torch.randn(1, 3, 224, 224)
				emb = self.backbone(dummy).shape[-1]
		else:
			try:
				self.backbone = SegformerModel.from_pretrained(model_name)
			except Exception as e:
				print(f"[WARNING] Cannot load SegFormer pretrained ({e}). Fallback to random config.")
				cfg = SegformerConfig()
				self.backbone = SegformerModel(cfg)

			emb = self.backbone.config.hidden_sizes[-1]

		if freeze_backbone:
			for p in self.backbone.parameters():
				p.requires_grad = False

		self.freeze_backbone = freeze_backbone
		self.head = nn.Sequential(
			nn.LayerNorm(emb),
			nn.Dropout(dropout),
			nn.Linear(emb, n_classes),
		)

	def forward(self, x):
		if self.use_fallback:
			if self.freeze_backbone:
				with torch.no_grad():
					f = self.backbone(x)
			else:
				f = self.backbone(x)
			if f.dim() > 2:
				f = f.flatten(1)
			return self.head(f)

		if self.freeze_backbone:
			with torch.no_grad():
				out = self.backbone(pixel_values=x)
		else:
			out = self.backbone(pixel_values=x)

		# SegFormer output can be:
		# - (B, C, H, W) for feature maps
		# - (B, N, C) for token-like outputs
		# - (B, C) already pooled
		hidden = out.last_hidden_state
		if hidden.dim() == 4:
			# (B, C, H, W) -> global average pool over spatial dims
			pooled = hidden.mean(dim=(2, 3))
		elif hidden.dim() == 3:
			# (B, N, C) -> average over tokens
			pooled = hidden.mean(dim=1)
		elif hidden.dim() == 2:
			pooled = hidden
		else:
			pooled = hidden.view(hidden.size(0), -1)
		return self.head(pooled)


# ============================================================================
# Hybrid fusion model
# ============================================================================
class HybridPlantModel(nn.Module):
	def __init__(
		self,
		n_classes,
		organ_dim=5,
		n_experts=8,
		d_ff_expert=1024,
		vit_name="vit_base_patch16_224",
		dino_model_name="dinov2_vitb14",
		segformer_model_name="nvidia/segformer-b1-finetuned-ade-512-512",
		dropout=0.1,
		top_k=1,
		freeze_dino=True,
		freeze_segformer=False,
	):
		super().__init__()
		self.organ_branch = OrganAwareSwitchViTBranch(
			vit_name=vit_name,
			n_classes=n_classes,
			organ_dim=organ_dim,
			n_experts=n_experts,
			d_ff_expert=d_ff_expert,
			top_k=top_k,
			pretrained=True,
		)
		self.dino_branch = DINOv2Branch(
			model_name=dino_model_name,
			n_classes=n_classes,
			dropout=dropout,
			freeze_backbone=freeze_dino,
		)
		self.segformer_branch = SegFormerClassifierBranch(
			model_name=segformer_model_name,
			n_classes=n_classes,
			dropout=dropout,
			freeze_backbone=freeze_segformer,
		)

		self.fusion = nn.Sequential(
			nn.LayerNorm(n_classes * 3),
			nn.Linear(n_classes * 3, n_classes),
		)

	def forward(self, x, organ_priors, training=True):
		organ_logits, aux_org_logits, router_probs, entropy, batch_usage = self.organ_branch(
			x, organ_priors, training=training
		)
		dino_logits = self.dino_branch(x)
		seg_logits = self.segformer_branch(x)

		fused = self.fusion(torch.cat([organ_logits, dino_logits, seg_logits], dim=-1))
		return {
			"logits": fused,
			"organ_logits": organ_logits,
			"dino_logits": dino_logits,
			"seg_logits": seg_logits,
			"aux_org_logits": aux_org_logits,
			"router_probs": router_probs,
			"entropy": entropy,
			"expert_usage_batch": batch_usage,
		}


# ============================================================================
# Training / Evaluation
# ============================================================================
def train_epoch(
	model,
	dataloader,
	optimizer,
	device,
	epoch,
	aux_weight=0.3,
	balance_weight=0.01,
	branch_weight=0.2,
):
	model.train()
	criterion = nn.CrossEntropyLoss()

	total_loss = 0.0
	total_acc = 0
	total_aux = 0.0
	total_balance = 0.0
	n = 0
	epoch_start = time.time()

	pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
	for imgs, labels, _, organ_priors in pbar:
		imgs = imgs.to(device)
		labels = labels.to(device)
		organ_priors = organ_priors.to(device)

		optimizer.zero_grad()
		out = model(imgs, organ_priors, training=True)

		main_loss = criterion(out["logits"], labels)
		organ_loss = criterion(out["organ_logits"], labels)
		dino_loss = criterion(out["dino_logits"], labels)
		seg_loss = criterion(out["seg_logits"], labels)
		branch_loss = (organ_loss + dino_loss + seg_loss) / 3.0

		# Auxiliary organ calibration against pseudo priors
		aux_loss = F.kl_div(
			F.log_softmax(out["aux_org_logits"], dim=-1),
			organ_priors,
			reduction="batchmean",
		)

		# Router load-balance penalty
		usage = out["expert_usage_batch"]
		uniform = torch.ones_like(usage) / float(usage.numel())
		balance_loss = ((usage - uniform) ** 2).mean()

		loss = main_loss + branch_weight * branch_loss + aux_weight * aux_loss + balance_weight * balance_loss
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		preds = out["logits"].argmax(dim=-1)
		bs = imgs.size(0)
		n += bs
		total_loss += loss.item() * bs
		total_acc += (preds == labels).sum().item()
		total_aux += aux_loss.item() * bs
		total_balance += balance_loss.item() * bs

		pbar.set_postfix({
			"loss": f"{loss.item():.4f}",
			"acc": f"{total_acc / max(1, n):.4f}",
		})

	return {
		"loss": total_loss / max(1, n),
		"acc": total_acc / max(1, n),
		"aux_loss": total_aux / max(1, n),
		"balance_loss": total_balance / max(1, n),
		"time": time.time() - epoch_start,
	}


@torch.no_grad()
def evaluate(model, dataloader, device, class_names, phase="val"):
	model.eval()
	y_true, y_pred, paths_all, all_probs = [], [], [], []
	eval_start = time.time()

	for imgs, labels, paths, organ_priors in tqdm(dataloader, desc=f"[{phase.upper()}]"):
		imgs = imgs.to(device)
		labels = labels.to(device)
		organ_priors = organ_priors.to(device)

		out = model(imgs, organ_priors, training=False)
		probs = F.softmax(out["logits"], dim=-1)
		preds = probs.argmax(dim=-1)

		y_true.extend(labels.cpu().tolist())
		y_pred.extend(preds.cpu().tolist())
		paths_all.extend(list(paths))
		all_probs.extend(probs.cpu().tolist())

	macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
	micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
	weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
	per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
		y_true, y_pred, labels=range(len(class_names)), zero_division=0
	)
	accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
	cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
	cls_report = classification_report(
		y_true,
		y_pred,
		target_names=class_names,
		labels=range(len(class_names)),
		zero_division=0,
		output_dict=True,
	)

	return {
		"macro_f1": macro_f1,
		"micro_f1": micro_f1,
		"weighted_f1": weighted_f1,
		"accuracy": accuracy,
		"per_precision": per_p,
		"per_recall": per_r,
		"per_f1": per_f1,
		"per_support": per_support,
		"y_true": y_true,
		"y_pred": y_pred,
		"paths": paths_all,
		"probs": all_probs,
		"confusion_matrix": cm,
		"classification_report": cls_report,
		"time": time.time() - eval_start,
	}


# ============================================================================
# Main
# ============================================================================
def main(args):
	os.makedirs(args.out_dir, exist_ok=True)

	start_time = datetime.now()
	print("=" * 90)
	print(f"[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print("=" * 90)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[INFO] device: {device}")

	# 1) Collect images by priority rules
	print("\n[STEP 1] Collecting images...")
	class_to_imgs, all_class_imgs = collect_images_per_class(
		args.data_root, max_per_class=args.max_per_class, verbose=True
	)
	total_available = sum(len(v) for v in all_class_imgs.values())
	total_selected = sum(len(v) for v in class_to_imgs.values())
	print(f"[STEP 1] Available={total_available}, Selected={total_selected}, Unselected={total_available - total_selected}")

	# 2) Split
	print("\n[STEP 2] Building train/val/test split...")
	class_to_idx, train_map, val_map, test_map = build_split_maps(
		class_to_imgs, val_split=args.val_split, test_split=args.test_split, seed=args.seed
	)
	classes = sorted(class_to_idx.keys())
	print(f"[INFO] classes={len(classes)}")
	print(f"[INFO] train={sum(len(v) for v in train_map.values())}, val={sum(len(v) for v in val_map.values())}, test={sum(len(v) for v in test_map.values())}")

	save_dataset_splits(args.out_dir, all_class_imgs, train_map, val_map, test_map, class_to_idx)

	# 2b) leakage check/fix
	print("\n[STEP 2b] Leakage check and fix...")
	train_map, val_map, test_map, leakage_result = handle_data_leakage(
		train_map,
		val_map,
		test_map,
		class_to_imgs,
		all_class_imgs,
		args.out_dir,
		val_split=args.val_split,
		test_split=args.test_split,
		hash_size=args.hash_size,
		threshold=args.hash_threshold,
		max_iterations=args.max_leakage_iterations,
		seed=args.seed,
	)

	# refresh class mapping/splits artifacts after leakage handling
	classes = sorted(train_map.keys())
	class_to_idx = {c: i for i, c in enumerate(classes)}
	save_dataset_splits(args.out_dir, all_class_imgs, train_map, val_map, test_map, class_to_idx)

	print("[INFO] Updated splits after leakage handling:")
	print(f"  train={sum(len(v) for v in train_map.values())}")
	print(f"  val  ={sum(len(v) for v in val_map.values())}")
	print(f"  test ={sum(len(v) for v in test_map.values())}")

	# 3) Pseudo-organ mining (TRAIN only)
	print("\n[STEP 3] Pseudo-organ mining (train-only clustering)...")
	feat_extractor = ResNet18FeatureExtractor().to(device)
	priors_train, kmeans = generate_pseudo_organs_train_only(
		train_map,
		feat_extractor,
		device,
		n_clusters=args.n_org_clusters,
		batch_size=args.cluster_bs,
	)
	priors_val = apply_kmeans_to_split(
		val_map,
		kmeans,
		feat_extractor,
		device,
		n_clusters=args.n_org_clusters,
		batch_size=args.cluster_bs,
	)
	priors_test = apply_kmeans_to_split(
		test_map,
		kmeans,
		feat_extractor,
		device,
		n_clusters=args.n_org_clusters,
		batch_size=args.cluster_bs,
	)
	pseudo_org = {}
	pseudo_org.update(priors_train)
	pseudo_org.update(priors_val)
	pseudo_org.update(priors_test)
	del feat_extractor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# 4) Build dataloaders
	print("\n[STEP 4] Building dataloaders...")
	train_loader, val_loader, test_loader = build_dataloaders(
		train_map,
		val_map,
		test_map,
		class_to_idx,
		pseudo_org,
		organ_dim=args.n_org_clusters,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	print(f"[INFO] train_ds={len(train_loader.dataset)}, val_ds={len(val_loader.dataset)}, test_ds={len(test_loader.dataset)}")

	# 5) Create hybrid model
	print("\n[STEP 5] Creating Hybrid model...")
	model = HybridPlantModel(
		n_classes=len(classes),
		organ_dim=args.n_org_clusters,
		n_experts=args.n_experts,
		d_ff_expert=args.d_ff_expert,
		vit_name=args.vit_name,
		dino_model_name=args.dino_model_name,
		segformer_model_name=args.segformer_model_name,
		dropout=args.dropout,
		top_k=args.top_k,
		freeze_dino=(not args.unfreeze_dino),
		freeze_segformer=(not args.unfreeze_segformer),
	).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"[INFO] total params    : {total_params:,}")
	print(f"[INFO] trainable params: {trainable_params:,}")

	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

	# 6) Train loop
	print("\n[STEP 6] Training...")
	print("=" * 90)

	best_macro = -1.0
	best_epoch = 0
	history = {
		"train_loss": [],
		"train_acc": [],
		"train_aux_loss": [],
		"train_balance_loss": [],
		"train_time": [],
		"val_macro_f1": [],
		"val_micro_f1": [],
		"val_weighted_f1": [],
		"val_accuracy": [],
		"val_time": [],
		"lr": [],
	}

	for epoch in range(1, args.epochs + 1):
		epoch_start = datetime.now()
		cur_lr = optimizer.param_groups[0]["lr"]
		print(
			f"\n[Epoch {epoch}/{args.epochs}] "
			f"start={epoch_start.strftime('%H:%M:%S')} | lr={cur_lr:.6g}"
		)

		train_metrics = train_epoch(
			model,
			train_loader,
			optimizer,
			device,
			epoch,
			aux_weight=args.aux_weight,
			balance_weight=args.balance_weight,
			branch_weight=args.branch_weight,
		)
		val_metrics = evaluate(model, val_loader, device, classes, phase="val")
		scheduler.step()

		epoch_end = datetime.now()
		epoch_dur = (epoch_end - epoch_start).total_seconds()

		history["train_loss"].append(train_metrics["loss"])
		history["train_acc"].append(train_metrics["acc"])
		history["train_aux_loss"].append(train_metrics["aux_loss"])
		history["train_balance_loss"].append(train_metrics["balance_loss"])
		history["train_time"].append(train_metrics["time"])
		history["val_macro_f1"].append(val_metrics["macro_f1"])
		history["val_micro_f1"].append(val_metrics["micro_f1"])
		history["val_weighted_f1"].append(val_metrics["weighted_f1"])
		history["val_accuracy"].append(val_metrics["accuracy"])
		history["val_time"].append(val_metrics["time"])
		history["lr"].append(cur_lr)

		print(
			f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, "
			f"aux={train_metrics['aux_loss']:.4f}, bal={train_metrics['balance_loss']:.6f}, "
			f"time={train_metrics['time']:.1f}s"
		)
		print(
			f"  Val  : macro_f1={val_metrics['macro_f1']:.4f}, acc={val_metrics['accuracy']:.4f}, "
			f"time={val_metrics['time']:.1f}s"
		)
		print(f"  Epoch duration: {epoch_dur:.1f}s | end={epoch_end.strftime('%H:%M:%S')}")

		if val_metrics["macro_f1"] > best_macro:
			best_macro = val_metrics["macro_f1"]
			best_epoch = epoch

			ckpt = {
				"epoch": epoch,
				"macro_f1": best_macro,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"class_to_idx": class_to_idx,
				"organ_dim": args.n_org_clusters,
				"n_experts": args.n_experts,
				"args": vars(args),
				"leakage_result": leakage_result,
				"kmeans_centers": None if kmeans is None else kmeans.cluster_centers_,
			}
			torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))
			print(f"  [BEST] Updated best model at epoch {epoch} (macro_f1={best_macro:.4f})")

	# save history
	torch.save(history, os.path.join(args.out_dir, "training_history.pt"))
	print(f"\n[INFO] training_history.pt saved to {args.out_dir}")

	# 7) Final test evaluation using best model
	print("\n[STEP 7] Final test evaluation (best model)...")
	best_ckpt = torch.load(os.path.join(args.out_dir, "best_model.pt"), map_location=device, weights_only=False)
	model.load_state_dict(best_ckpt["model_state_dict"])
	test_metrics = evaluate(model, test_loader, device, classes, phase="test")

	print("\n[TEST RESULTS]")
	print(f"  Macro F1    : {test_metrics['macro_f1']:.4f}")
	print(f"  Micro F1    : {test_metrics['micro_f1']:.4f}")
	print(f"  Weighted F1 : {test_metrics['weighted_f1']:.4f}")
	print(f"  Accuracy    : {test_metrics['accuracy']:.4f}")

	save_classification_report(test_metrics["classification_report"], classes, args.out_dir, phase="test")
	save_confusion_matrix(test_metrics["confusion_matrix"], classes, args.out_dir, phase="test")

	print("\n[Per-Class F1]")
	for i, cls in enumerate(classes):
		print(f"  {cls:<35s}: {test_metrics['per_f1'][i]:.4f}")

	end_time = datetime.now()
	total_sec = (end_time - start_time).total_seconds()

	print("\n" + "=" * 90)
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[INFO] total duration: {total_sec / 3600:.2f} hours ({total_sec:.0f} seconds)")
	print(f"[INFO] best epoch: {best_epoch} | best val macro_f1: {best_macro:.4f}")
	print("=" * 90)


def build_arg_parser():
	parser = argparse.ArgumentParser(description="Hybrid Organ-Aware V-MoE + DINOv2 + SegFormer for BigPlants-100")

	# data
	parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root containing 100 class folders")
	parser.add_argument("--out_dir", type=str, default="./outputs_hybrid", help="Output directory")
	parser.add_argument("--max_per_class", type=int, default=100, help="Max selected images per class")
	parser.add_argument("--val_split", type=float, default=0.1)
	parser.add_argument("--test_split", type=float, default=0.2)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)

	# leakage
	parser.add_argument("--hash_size", type=int, default=8)
	parser.add_argument("--hash_threshold", type=int, default=5)
	parser.add_argument("--max_leakage_iterations", type=int, default=3)

	# pseudo organ mining
	parser.add_argument("--n_org_clusters", type=int, default=5)
	parser.add_argument("--cluster_bs", type=int, default=64)

	# model
	parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224")
	parser.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
	parser.add_argument("--segformer_model_name", type=str, default="nvidia/segformer-b1-finetuned-ade-512-512")
	parser.add_argument("--n_experts", type=int, default=8)
	parser.add_argument("--d_ff_expert", type=int, default=1024)
	parser.add_argument("--top_k", type=int, default=1)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--unfreeze_dino", action="store_true", default=False)
	parser.add_argument("--unfreeze_segformer", action="store_true", default=False)

	# training
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)

	# loss weights
	parser.add_argument("--aux_weight", type=float, default=0.3)
	parser.add_argument("--balance_weight", type=float, default=0.01)
	parser.add_argument("--branch_weight", type=float, default=0.2)

	return parser


if __name__ == "__main__":
	parser = build_arg_parser()
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	main(args)
