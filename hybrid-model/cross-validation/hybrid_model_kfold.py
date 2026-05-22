#!/usr/bin/env python3
"""
Hybrid Model K-Fold Cross-Validation (BigPlants-100)

This script reuses components from hybrid_model.py and adds:
  - Stratified 5-fold CV (configurable)
  - Per-fold training/evaluation outputs
  - Per-fold best_model.pt / training_history.pt
  - Aggregated kfold_results.csv + summary
"""

import os
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from hybrid_model import (
	collect_images_per_class,
	save_dataset_splits,
	save_confusion_matrix,
	save_classification_report,
	check_data_leakage_phash,
	ResNet18FeatureExtractor,
	generate_pseudo_organs_train_only,
	apply_kmeans_to_split,
	build_dataloaders,
	HybridPlantModel,
	train_epoch,
	evaluate,
)


def build_map_from_items(items):
	out = {}
	for p, cls in items:
		out.setdefault(cls, []).append(p)
	return out


def split_train_val_per_class(trainval_map, val_split=0.1, seed=42, fold_id=1):
	train_map = {}
	val_map = {}
	for cls, imgs in trainval_map.items():
		imgs = list(imgs)
		rnd = random.Random((hash(cls) ^ seed ^ fold_id) & 0xFFFFFFFF)
		rnd.shuffle(imgs)
		n = len(imgs)
		n_val = int(np.ceil(val_split * n))
		if n > 1:
			n_val = max(1, min(n - 1, n_val))
		else:
			n_val = 0
		val_map[cls] = imgs[:n_val]
		train_map[cls] = imgs[n_val:]
	return train_map, val_map


def handle_leakage_in_fold(train_map, val_map, test_map, fold_dir, hash_size=8, threshold=5):
	"""Simple fold-safe leakage fixing: move leaked val/test samples to train (no replacement)."""
	res = check_data_leakage_phash(train_map, val_map, test_map, fold_dir, hash_size=hash_size, threshold=threshold)
	if not res.get("leakage_found", False):
		return train_map, val_map, test_map, res

	leaked_val = {}
	leaked_test = {}

	for g in res.get("exact_cross_split_details", []):
		for p, split_name, cls in g["items"]:
			if split_name == "val":
				leaked_val.setdefault(cls, set()).add(p)
			elif split_name == "test":
				leaked_test.setdefault(cls, set()).add(p)

	for r in res.get("near_cross_split_details", []):
		for sk, ck, pk in [("split1", "class1", "path1"), ("split2", "class2", "path2")]:
			split_name = r[sk]
			cls = r[ck]
			p = r[pk]
			if split_name == "val":
				leaked_val.setdefault(cls, set()).add(p)
			elif split_name == "test":
				leaked_test.setdefault(cls, set()).add(p)

	moved_val = 0
	for cls, leaked_paths in leaked_val.items():
		for p in list(leaked_paths):
			if p in val_map.get(cls, []):
				val_map[cls].remove(p)
				train_map[cls].append(p)
				moved_val += 1

	moved_test = 0
	for cls, leaked_paths in leaked_test.items():
		for p in list(leaked_paths):
			if p in test_map.get(cls, []):
				test_map[cls].remove(p)
				train_map[cls].append(p)
				moved_test += 1

	res["fixed_by_moving_to_train"] = True
	res["moved_from_val"] = moved_val
	res["moved_from_test"] = moved_test
	print(f"[LEAKAGE FIX] moved val->train: {moved_val}, test->train: {moved_test}")
	return train_map, val_map, test_map, res


def main(args):
	os.makedirs(args.out_dir, exist_ok=True)

	start_time = datetime.now()
	print("=" * 90)
	print(f"[START] K-Fold Hybrid run at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[INFO] n_folds={args.n_folds}")
	print("=" * 90)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[INFO] device: {device}")

	# 1) Collect selected dataset once
	class_to_imgs, all_class_imgs = collect_images_per_class(
		args.data_root, max_per_class=args.max_per_class, verbose=True
	)

	classes = sorted(class_to_imgs.keys())
	class_to_idx = {c: i for i, c in enumerate(classes)}

	all_items = []
	all_labels = []
	for cls in classes:
		for p in class_to_imgs[cls]:
			all_items.append((p, cls))
			all_labels.append(class_to_idx[cls])

	all_items = np.array(all_items, dtype=object)
	all_labels = np.array(all_labels)

	print(f"[INFO] total selected samples: {len(all_items)}")

	skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
	fold_results = []
	all_fold_cm = []

	for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
		if fold_idx < args.start_fold:
			continue

		fold_start = datetime.now()
		fold_dir = os.path.join(args.out_dir, f"fold_{fold_idx}")
		os.makedirs(fold_dir, exist_ok=True)

		print("\n" + "-" * 90)
		print(f"[FOLD {fold_idx}/{args.n_folds}] start={fold_start.strftime('%Y-%m-%d %H:%M:%S')}")
		print("-" * 90)

		trainval_items = all_items[trainval_idx].tolist()
		test_items = all_items[test_idx].tolist()

		trainval_map = build_map_from_items(trainval_items)
		test_map = build_map_from_items(test_items)
		for c in classes:
			trainval_map.setdefault(c, [])
			test_map.setdefault(c, [])

		train_map, val_map = split_train_val_per_class(
			trainval_map, val_split=args.val_split, seed=args.seed, fold_id=fold_idx
		)

		print(
			f"[FOLD {fold_idx}] split sizes: "
			f"train={sum(len(v) for v in train_map.values())}, "
			f"val={sum(len(v) for v in val_map.values())}, "
			f"test={sum(len(v) for v in test_map.values())}"
		)

		# Save initial split CSVs
		save_dataset_splits(fold_dir, all_class_imgs, train_map, val_map, test_map, class_to_idx)

		# Leakage check/fix per fold
		if args.check_leakage:
			train_map, val_map, test_map, leakage_result = handle_leakage_in_fold(
				train_map,
				val_map,
				test_map,
				fold_dir,
				hash_size=args.hash_size,
				threshold=args.hash_threshold,
			)
		else:
			leakage_result = {"status": "skipped", "leakage_found": False}

		# Save updated split CSVs
		save_dataset_splits(fold_dir, all_class_imgs, train_map, val_map, test_map, class_to_idx)

		# 2) Pseudo-organ mining (train only)
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

		# 3) DataLoaders
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

		# 4) Model
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

		optimizer = torch.optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=args.lr,
			weight_decay=args.weight_decay,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

		best_macro = -1.0
		best_epoch = 0

		# 5) Training loop
		for epoch in range(1, args.epochs + 1):
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
			val_metrics = evaluate(model, val_loader, device, classes, phase=f"fold{fold_idx}_val")
			scheduler.step()

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
			history["lr"].append(optimizer.param_groups[0]["lr"])

			print(
				f"[FOLD {fold_idx}][Epoch {epoch}/{args.epochs}] "
				f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['acc']:.4f}, "
				f"val_macro={val_metrics['macro_f1']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
			)

			if val_metrics["macro_f1"] > best_macro:
				best_macro = val_metrics["macro_f1"]
				best_epoch = epoch
				ckpt = {
					"fold": fold_idx,
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
				torch.save(ckpt, os.path.join(fold_dir, "best_model.pt"))

		torch.save(history, os.path.join(fold_dir, "training_history.pt"))

		# 6) Test with best model
		best_ckpt = torch.load(os.path.join(fold_dir, "best_model.pt"), map_location=device, weights_only=False)
		model.load_state_dict(best_ckpt["model_state_dict"])
		test_metrics = evaluate(model, test_loader, device, classes, phase=f"fold{fold_idx}_test")

		save_classification_report(test_metrics["classification_report"], classes, fold_dir, phase="test")
		save_confusion_matrix(test_metrics["confusion_matrix"], classes, fold_dir, phase="test")

		fold_time = (datetime.now() - fold_start).total_seconds()

		fold_result = {
			"fold": fold_idx,
			"best_epoch": best_epoch,
			"best_val_macro_f1": best_macro,
			"test_macro_f1": test_metrics["macro_f1"],
			"test_micro_f1": test_metrics["micro_f1"],
			"test_weighted_f1": test_metrics["weighted_f1"],
			"test_accuracy": test_metrics["accuracy"],
			"fold_time": fold_time,
		}
		fold_results.append(fold_result)
		all_fold_cm.append(test_metrics["confusion_matrix"])

		print(f"[FOLD {fold_idx}] TEST Macro F1={test_metrics['macro_f1']:.4f}, Acc={test_metrics['accuracy']:.4f}")

		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	# 7) Aggregate and save
	if len(fold_results) == 0:
		print("[WARNING] No fold was executed.")
		return

	df = pd.DataFrame(fold_results).sort_values("fold")
	df.to_csv(os.path.join(args.out_dir, "kfold_results.csv"), index=False)

	print("\n" + "=" * 90)
	print("[K-FOLD SUMMARY]")
	print("=" * 90)
	print(df.to_string(index=False))
	print("\nMean ± Std:")
	print(f"  Test Macro F1 : {df['test_macro_f1'].mean():.4f} ± {df['test_macro_f1'].std():.4f}")
	print(f"  Test Accuracy : {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")

	if len(all_fold_cm) > 0:
		avg_cm = np.mean(all_fold_cm, axis=0)
		save_confusion_matrix(avg_cm.astype(int), classes, args.out_dir, phase="kfold_avg_test")

	summary_path = os.path.join(args.out_dir, "summary.txt")
	with open(summary_path, "w", encoding="utf-8") as f:
		f.write("Hybrid K-Fold Summary\n")
		f.write("=" * 40 + "\n")
		f.write(f"n_folds: {args.n_folds}\n")
		f.write(f"mean_test_macro_f1: {df['test_macro_f1'].mean():.6f}\n")
		f.write(f"std_test_macro_f1: {df['test_macro_f1'].std():.6f}\n")
		f.write(f"mean_test_accuracy: {df['test_accuracy'].mean():.6f}\n")
		f.write(f"std_test_accuracy: {df['test_accuracy'].std():.6f}\n")
		best_idx = df['test_macro_f1'].idxmax()
		f.write(f"best_fold: {int(df.loc[best_idx, 'fold'])}\n")
		f.write(f"best_fold_macro_f1: {df.loc[best_idx, 'test_macro_f1']:.6f}\n")

	end_time = datetime.now()
	print("\n" + "=" * 90)
	print(f"[END] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"[INFO] total duration: {(end_time - start_time).total_seconds()/3600:.2f} hours")
	print("=" * 90)


def build_arg_parser():
	parser = argparse.ArgumentParser(description="Hybrid Model K-Fold Cross-Validation")

	# data
	parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing class folders")
	parser.add_argument("--out_dir", type=str, default="./outputs_hybrid_kfold", help="Output directory")
	parser.add_argument("--max_per_class", type=int, default=100)

	# cv
	parser.add_argument("--n_folds", type=int, default=5)
	parser.add_argument("--start_fold", type=int, default=1, help="Resume from this fold (1-indexed)")
	parser.add_argument("--val_split", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)

	# leakage
	parser.add_argument("--check_leakage", action="store_true", default=True)
	parser.add_argument("--no_check_leakage", action="store_false", dest="check_leakage")
	parser.add_argument("--hash_size", type=int, default=8)
	parser.add_argument("--hash_threshold", type=int, default=5)

	# pseudo organs
	parser.add_argument("--n_org_clusters", type=int, default=5)
	parser.add_argument("--cluster_bs", type=int, default=64)

	# model
	parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224")
	parser.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
	parser.add_argument("--segformer_model_name", type=str, default="nvidia/segformer-b1-finetuned-ade-512-512")
	parser.add_argument("--n_experts", type=int, default=8)
	parser.add_argument("--d_ff_expert", type=int, default=1024)
	parser.add_argument("--top_k", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--unfreeze_dino", action="store_true", default=False)
	parser.add_argument("--unfreeze_segformer", action="store_true", default=False)

	# training
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--num_workers", type=int, default=8)

	# losses
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

