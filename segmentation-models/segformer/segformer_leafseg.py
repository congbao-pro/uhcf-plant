"""
Leaf Disease Segmentation dataset structure:
	leaf-seg/
	├── images/   (*.jpg)
	├── masks/    (*.png)
	└── train.csv (imageid,maskid)
"""

import os
import csv
import math
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
	from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError:
	SegformerConfig = None
	SegformerForSemanticSegmentation = None

warnings.filterwarnings("ignore")


# ============================================================
# 1. JOINT TRANSFORMS
# ============================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class JointResize:
	def __init__(self, size):
		self.size = (size, size) if isinstance(size, int) else size

	def __call__(self, image, mask):
		image = image.resize(self.size, Image.BILINEAR)
		mask = mask.resize(self.size, Image.NEAREST)
		return image, mask


class JointRandomHFlip:
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image, mask):
		if random.random() < self.p:
			image = image.transpose(Image.FLIP_LEFT_RIGHT)
			mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
		return image, mask


class JointRandomVFlip:
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image, mask):
		if random.random() < self.p:
			image = image.transpose(Image.FLIP_TOP_BOTTOM)
			mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
		return image, mask


class JointRandomRotation:
	def __init__(self, degrees=15):
		self.degrees = degrees

	def __call__(self, image, mask):
		angle = random.uniform(-self.degrees, self.degrees)
		image = image.rotate(angle, Image.BILINEAR, fillcolor=0)
		mask = mask.rotate(angle, Image.NEAREST, fillcolor=0)
		return image, mask


class JointColorJitter:
	def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
		self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

	def __call__(self, image, mask):
		image = self.jitter(image)
		return image, mask


class JointToTensor:
	def __init__(self, value_mapping=None):
		self.normalize = transforms.Normalize(MEAN, STD)
		self.value_mapping = value_mapping or {}

	def __call__(self, image, mask):
		image = transforms.ToTensor()(image)
		image = self.normalize(image)

		mask_np = np.array(mask, dtype=np.int64)
		if self.value_mapping:
			mapped = np.zeros_like(mask_np)
			for old_v, new_v in self.value_mapping.items():
				mapped[mask_np == old_v] = new_v
			mask_np = mapped

		return image, torch.from_numpy(mask_np).long()


class JointCompose:
	def __init__(self, tfms):
		self.tfms = tfms

	def __call__(self, image, mask):
		for t in self.tfms:
			image, mask = t(image, mask)
		return image, mask


# ============================================================
# 2. DATASET
# ============================================================

def _read_pairs_from_csv(csv_path):
	pairs = []
	with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
		sample = f.read(4096)
		f.seek(0)
		try:
			dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
		except csv.Error:
			dialect = csv.excel
		reader = csv.DictReader(f, dialect=dialect)

		lower_map = {k.lower().strip(): k for k in reader.fieldnames or []}
		img_key = lower_map.get("imageid")
		mask_key = lower_map.get("maskid")
		if img_key is None or mask_key is None:
			raise ValueError("train.csv must contain columns: imageid, maskid")

		for row in reader:
			img_name = row[img_key].strip()
			mask_name = row[mask_key].strip()
			if img_name and mask_name:
				pairs.append((img_name, mask_name))
	return pairs


class LeafSegDataset(Dataset):
	def __init__(self, root_dir, joint_transform=None,
				 indices=None, value_mapping=None, verbose=True):
		self.root_dir = root_dir
		self.images_dir = os.path.join(root_dir, "images")
		self.masks_dir = os.path.join(root_dir, "masks")
		self.csv_path = os.path.join(root_dir, "train.csv")
		self.joint_transform = joint_transform

		self.pairs = _read_pairs_from_csv(self.csv_path)
		if indices is not None:
			self.pairs = [self.pairs[i] for i in indices]

		if value_mapping is None:
			self.num_classes, self.value_mapping, self.class_names = self._detect_classes()
		else:
			self.value_mapping = dict(sorted(value_mapping.items(), key=lambda x: x[1]))
			self.num_classes = len(set(self.value_mapping.values()))
			self.class_names = [f"class_{i}" for i in range(self.num_classes)]

		if verbose:
			print(f"  LeafSegDataset: {len(self.pairs)} samples | classes={self.num_classes}")

	def _detect_classes(self):
		unique_vals = set()
		n_sample = min(200, len(self.pairs))
		for i in range(n_sample):
			_, mask_name = self.pairs[i]
			mask_path = os.path.join(self.masks_dir, mask_name)
			if not os.path.isfile(mask_path):
				continue
			m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
			unique_vals.update(np.unique(m).tolist())

		sorted_vals = sorted(unique_vals)
		if not sorted_vals:
			return 2, {0: 0, 255: 1}, ["class_0", "class_1"]

		if sorted_vals == list(range(len(sorted_vals))):
			mapping = {v: v for v in sorted_vals}
		else:
			mapping = {v: i for i, v in enumerate(sorted_vals)}

		num_classes = len(mapping)
		class_names = [f"class_{i}" for i in range(num_classes)]
		return num_classes, mapping, class_names

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		img_name, mask_name = self.pairs[idx]
		img_path = os.path.join(self.images_dir, img_name)
		mask_path = os.path.join(self.masks_dir, mask_name)

		image = Image.open(img_path).convert("RGB")
		mask = Image.open(mask_path).convert("L")

		if self.joint_transform:
			image, mask = self.joint_transform(image, mask)
		else:
			image = transforms.ToTensor()(image)
			image = transforms.Normalize(MEAN, STD)(image)
			mask_np = np.array(mask, dtype=np.int64)
			mapped = np.zeros_like(mask_np)
			for old_v, new_v in self.value_mapping.items():
				mapped[mask_np == old_v] = new_v
			mask = torch.from_numpy(mapped).long()

		return image, mask


# ============================================================
# 3. MODEL
# ============================================================

def create_segformer_model(num_classes, model_name,
						   pretrained=True, img_size=512,
						   id2label=None, label2id=None):
	if SegformerForSemanticSegmentation is None:
		raise ImportError(
			"transformers is not installed. Please run: pip install transformers"
		)

	if pretrained:
		try:
			model = SegformerForSemanticSegmentation.from_pretrained(
				model_name,
				num_labels=num_classes,
				ignore_mismatched_sizes=True,
				id2label=id2label,
				label2id=label2id,
			)
			return model
		except Exception as e:
			print(f"  [Warn] Cannot load pretrained model '{model_name}': {e}")
			print("  [Warn] Fallback to randomly initialized SegFormer config.")

	config = SegformerConfig(
		num_labels=num_classes,
		id2label=id2label,
		label2id=label2id,
	)
	config.image_size = img_size
	return SegformerForSemanticSegmentation(config)


# ============================================================
# 4. METRICS
# ============================================================

def compute_metrics(pred, gt, num_classes):
	assert pred.shape == gt.shape

	per_class_iou = []
	per_class_dice = []
	valid_classes = []

	for c in range(num_classes):
		p = (pred == c)
		g = (gt == c)

		inter = np.logical_and(p, g).sum()
		union = np.logical_or(p, g).sum()
		p_sum = p.sum()
		g_sum = g.sum()

		if union == 0:
			continue

		iou = inter / (union + 1e-8)
		dice = (2.0 * inter) / (p_sum + g_sum + 1e-8)

		per_class_iou.append(float(iou))
		per_class_dice.append(float(dice))
		valid_classes.append(c)

	pixel_acc = float((pred == gt).sum() / pred.size) if pred.size > 0 else 0.0

	return {
		"mIoU": np.mean(per_class_iou) if per_class_iou else 0.0,
		"mean_Dice": np.mean(per_class_dice) if per_class_dice else 0.0,
		"pixel_accuracy": pixel_acc,
		"per_class_iou": dict(zip(valid_classes, per_class_iou)),
		"per_class_dice": dict(zip(valid_classes, per_class_dice)),
	}


def decode_logits(logits, out_h, out_w):
	logits_up = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
	return logits_up.argmax(dim=1)


# ============================================================
# 5. TRAIN / EVAL
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None):
	model.train()
	running_loss = 0.0
	total_pixels = 0
	correct = 0
	n_batches = 0

	pbar = tqdm(dataloader, desc=f"  Epoch {epoch:>3d} [Train]", ncols=120)
	for images, masks in pbar:
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		if scaler is not None:
			with torch.cuda.amp.autocast():
				outputs = model(pixel_values=images, labels=masks)
				loss = outputs.loss
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			outputs = model(pixel_values=images, labels=masks)
			loss = outputs.loss
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			pred = decode_logits(outputs.logits, masks.shape[-2], masks.shape[-1])
			correct += (pred == masks).sum().item()
			total_pixels += masks.numel()

		running_loss += float(loss.item())
		n_batches += 1

		pbar.set_postfix({
			"loss": f"{running_loss / n_batches:.4f}",
			"pAcc": f"{100.0 * correct / max(total_pixels, 1):.2f}%",
		})

	return {
		"loss": running_loss / max(n_batches, 1),
		"pixel_acc": 100.0 * correct / max(total_pixels, 1),
	}


@torch.no_grad()
def evaluate(model, dataloader, device):
	model.eval()
	running_loss = 0.0
	n_batches = 0

	all_pred = []
	all_gt = []
	num_classes = model.config.num_labels

	pbar = tqdm(dataloader, desc="  [Eval]", ncols=120)
	for images, masks in pbar:
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)

		outputs = model(pixel_values=images, labels=masks)
		loss = outputs.loss
		pred = decode_logits(outputs.logits, masks.shape[-2], masks.shape[-1])

		running_loss += float(loss.item())
		n_batches += 1

		all_pred.append(pred.cpu().numpy())
		all_gt.append(masks.cpu().numpy())

	if n_batches == 0:
		return {
			"loss": 0.0,
			"mIoU": 0.0,
			"mean_Dice": 0.0,
			"pixel_accuracy": 0.0,
			"per_class_iou": {},
			"per_class_dice": {},
		}

	pred_np = np.concatenate(all_pred, axis=0)
	gt_np = np.concatenate(all_gt, axis=0)
	metrics = compute_metrics(pred_np, gt_np, num_classes)
	metrics["loss"] = running_loss / n_batches
	return metrics


# ============================================================
# 6. VISUALIZATION
# ============================================================

PALETTE = np.array([
	[0, 0, 0],
	[255, 50, 50],
	[50, 255, 50],
	[50, 50, 255],
	[255, 255, 50],
	[255, 50, 255],
	[50, 255, 255],
	[255, 140, 0],
	[148, 0, 211],
	[0, 128, 128],
], dtype=np.uint8)


def colorize_mask(mask, palette=None):
	palette = PALETTE if palette is None else palette
	return palette[mask % len(palette)]


def denormalize_image(t):
	img = t.cpu().numpy().transpose(1, 2, 0)
	img = img * np.array(STD) + np.array(MEAN)
	img = np.clip(img, 0, 1)
	return (img * 255).astype(np.uint8)


def plot_training_curves(history, save_path):
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	epochs = range(1, len(history["train_loss"]) + 1)

	axes[0, 0].plot(epochs, history["train_loss"], "b-o", ms=3, lw=2, label="Train")
	axes[0, 0].plot(epochs, history["val_loss"], "r-o", ms=3, lw=2, label="Val")
	axes[0, 0].set_title("Total Loss")
	axes[0, 0].legend()
	axes[0, 0].grid(alpha=0.3)

	axes[0, 1].plot(epochs, history["val_miou"], "r-o", ms=3, lw=2, label="Val mIoU")
	axes[0, 1].set_title("mIoU (%)")
	axes[0, 1].legend()
	axes[0, 1].grid(alpha=0.3)

	axes[1, 0].plot(epochs, history["val_dice"], "r-o", ms=3, lw=2, label="Val Dice")
	axes[1, 0].set_title("Mean Dice (%)")
	axes[1, 0].legend()
	axes[1, 0].grid(alpha=0.3)

	axes[1, 1].plot(epochs, history["train_pacc"], "b-o", ms=3, lw=2, label="Train")
	axes[1, 1].plot(epochs, history["val_pacc"], "r-o", ms=3, lw=2, label="Val")
	axes[1, 1].set_title("Pixel Accuracy (%)")
	axes[1, 1].legend()
	axes[1, 1].grid(alpha=0.3)

	for ax in axes.flat:
		ax.set_xlabel("Epoch")

	fig.suptitle("SegFormer — Training Curves", fontsize=14, fontweight="bold")
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"  Curves saved -> {save_path}")


@torch.no_grad()
def plot_sample_predictions(model, dataset, device, save_path, n=6):
	model.eval()
	n = min(n, len(dataset))
	sample_indices = random.sample(range(len(dataset)), n)

	fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
	if n == 1:
		axes = np.expand_dims(axes, axis=0)

	for row, idx in enumerate(sample_indices):
		image, mask = dataset[idx]
		inp = image.unsqueeze(0).to(device)
		out = model(pixel_values=inp)
		pred = decode_logits(out.logits, mask.shape[-2], mask.shape[-1])[0].cpu().numpy()

		img_np = denormalize_image(image)
		gt_np = mask.cpu().numpy()

		axes[row, 0].imshow(img_np)
		axes[row, 0].set_title("Image")
		axes[row, 0].axis("off")

		axes[row, 1].imshow(colorize_mask(gt_np))
		axes[row, 1].set_title("Ground Truth")
		axes[row, 1].axis("off")

		axes[row, 2].imshow(colorize_mask(pred))
		axes[row, 2].set_title("Prediction")
		axes[row, 2].axis("off")

	fig.suptitle("Sample Predictions — SegFormer", fontsize=14, fontweight="bold")
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"  Sample predictions saved -> {save_path}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
	parser = argparse.ArgumentParser(
		description="SegFormer — Leaf Disease Segmentation",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	g = parser.add_argument_group("Data")
	g.add_argument("--data_dir", type=str, required=True,
				   help="Root of Leaf Disease Segmentation dataset (images/, masks/, train.csv)")
	g.add_argument("--output_dir", type=str, default="./output_segformer")
	g.add_argument("--img_size", type=int, default=512)
	g.add_argument("--train_ratio", type=float, default=0.7)
	g.add_argument("--val_ratio", type=float, default=0.15)

	g = parser.add_argument_group("Model")
	g.add_argument("--model_name", type=str, default="nvidia/segformer-b1-finetuned-ade-512-512")
	g.add_argument("--pretrained", action="store_true", default=True)
	g.add_argument("--no_pretrained", action="store_false", dest="pretrained")

	g = parser.add_argument_group("Training")
	g.add_argument("--epochs", type=int, default=50)
	g.add_argument("--batch_size", type=int, default=8)
	g.add_argument("--lr", type=float, default=6e-5)
	g.add_argument("--weight_decay", type=float, default=0.01)
	g.add_argument("--patience", type=int, default=10)
	g.add_argument("--num_workers", type=int, default=4)
	g.add_argument("--seed", type=int, default=42)

	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"\n{'='*65}")
	print("  SegFormer — Leaf Disease Segmentation")
	print(f"{'='*65}")
	print(f"  Device     : {device}" +
		  (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
	print(f"  Output dir : {args.output_dir}")

	print(f"\n  Loading dataset from: {args.data_dir}")
	all_pairs = _read_pairs_from_csv(os.path.join(args.data_dir, "train.csv"))
	total = len(all_pairs)

	indices = list(range(total))
	random.Random(args.seed).shuffle(indices)
	train_end = int(total * args.train_ratio)
	val_end = train_end + int(total * args.val_ratio)
	train_idx = indices[:train_end]
	val_idx = indices[train_end:val_end]
	test_idx = indices[val_end:]

	print(f"  Total: {total} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

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
							  indices=train_idx, value_mapping=value_mapping, verbose=False)
	val_ds = LeafSegDataset(args.data_dir, joint_transform=val_transform,
							indices=val_idx, value_mapping=value_mapping, verbose=False)
	test_ds = LeafSegDataset(args.data_dir, joint_transform=val_transform,
							 indices=test_idx, value_mapping=value_mapping, verbose=False)

	train_loader = DataLoader(
		train_ds, batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, pin_memory=True,
		persistent_workers=args.num_workers > 0,
	)
	val_loader = DataLoader(
		val_ds, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True,
		persistent_workers=args.num_workers > 0,
	)
	test_loader = DataLoader(
		test_ds, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True,
		persistent_workers=args.num_workers > 0,
	)

	id2label = {i: class_names[i] for i in range(num_classes)}
	label2id = {v: k for k, v in id2label.items()}

	model = create_segformer_model(
		num_classes=num_classes,
		model_name=args.model_name,
		pretrained=args.pretrained,
		img_size=args.img_size,
		id2label=id2label,
		label2id=label2id,
	).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"\n  Parameters: {total_params:,} total | {train_params:,} trainable")

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
	scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

	history = {
		"train_loss": [], "val_loss": [],
		"val_miou": [], "val_dice": [],
		"train_pacc": [], "val_pacc": [],
	}

	best_miou = 0.0
	patience_counter = 0

	print(f"\n{'='*65}")
	print(f"  Training for up to {args.epochs} epochs  (patience={args.patience})")
	print(f"{'='*65}\n")

	for epoch in range(1, args.epochs + 1):
		train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
		val_stats = evaluate(model, val_loader, device)
		scheduler.step()

		history["train_loss"].append(train_stats["loss"])
		history["val_loss"].append(val_stats["loss"])
		history["val_miou"].append(val_stats["mIoU"] * 100)
		history["val_dice"].append(val_stats["mean_Dice"] * 100)
		history["train_pacc"].append(train_stats["pixel_acc"])
		history["val_pacc"].append(val_stats["pixel_accuracy"] * 100)

		print(
			f"Epoch {epoch:03d} | "
			f"Train Loss: {train_stats['loss']:.4f} | "
			f"Val Loss: {val_stats['loss']:.4f} | "
			f"mIoU: {val_stats['mIoU'] * 100:.2f}% | "
			f"Dice: {val_stats['mean_Dice'] * 100:.2f}% | "
			f"Val PAcc: {val_stats['pixel_accuracy'] * 100:.2f}%"
		)

		if val_stats["mIoU"] > best_miou:
			best_miou = val_stats["mIoU"]
			patience_counter = 0

			best_ckpt = {
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"model_config": {
					"num_classes": num_classes,
					"img_size": args.img_size,
					"model_name": args.model_name,
					"pretrained": args.pretrained,
				},
				"value_mapping": value_mapping,
				"class_names": class_names,
			}
			torch.save(best_ckpt, os.path.join(args.output_dir, "best_model.pth"))
			print(f"  [Best] mIoU improved to {best_miou * 100:.2f}% -> saved best_model.pth")
		else:
			patience_counter += 1
			print(f"  [No Improve] patience {patience_counter}/{args.patience}")

		if patience_counter >= args.patience:
			print("\n  Early stopping triggered.")
			break

	last_ckpt = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"model_config": {
			"num_classes": num_classes,
			"img_size": args.img_size,
			"model_name": args.model_name,
			"pretrained": args.pretrained,
		},
		"value_mapping": value_mapping,
		"class_names": class_names,
	}
	torch.save(last_ckpt, os.path.join(args.output_dir, "last_model.pth"))

	plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))

	print(f"\n{'='*65}")
	print(f"  Evaluating on Test Set  ({len(test_idx)} images)")
	print(f"{'='*65}\n")

	best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device, weights_only=False)
	model.load_state_dict(best_ckpt["model_state_dict"])

	test_metrics = evaluate(model, test_loader, device)

	print(f"\n  {'='*50}")
	print("  TEST RESULTS  (SegFormer — Segmentation)")
	print(f"  {'='*50}")
	print(f"    mIoU           : {test_metrics['mIoU'] * 100:.2f}%")
	print(f"    Mean Dice      : {test_metrics['mean_Dice'] * 100:.2f}%")
	print(f"    Pixel Accuracy : {test_metrics['pixel_accuracy'] * 100:.2f}%")
	print(f"    Loss           : {test_metrics['loss']:.4f}")

	print("\n  Per-class IoU:")
	for c, iou in test_metrics["per_class_iou"].items():
		name = class_names[c] if c < len(class_names) else f"class_{c}"
		print(f"    {name:<20s}: {iou * 100:.2f}%")

	print("\n  Per-class Dice:")
	for c, d in test_metrics["per_class_dice"].items():
		name = class_names[c] if c < len(class_names) else f"class_{c}"
		print(f"    {name:<20s}: {d * 100:.2f}%")

	plot_sample_predictions(
		model, test_ds, device,
		os.path.join(args.output_dir, "sample_predictions.png"),
		n=6,
	)

	results_json = {
		"mIoU": float(test_metrics["mIoU"] * 100),
		"mean_Dice": float(test_metrics["mean_Dice"] * 100),
		"pixel_accuracy": float(test_metrics["pixel_accuracy"] * 100),
		"loss": float(test_metrics["loss"]),
		"best_val_miou": float(best_miou * 100),
		"epochs_trained": int(epoch),
		"per_class_iou": {str(k): float(v * 100) for k, v in test_metrics["per_class_iou"].items()},
		"per_class_dice": {str(k): float(v * 100) for k, v in test_metrics["per_class_dice"].items()},
		"num_classes": int(num_classes),
		"class_names": class_names,
	}

	with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
		import json
		json.dump(results_json, f, ensure_ascii=False, indent=2)

	print(f"\n  All results saved to: {args.output_dir}/")
	print("    - best_model.pth")
	print("    - last_model.pth")
	print("    - training_curves.png")
	print("    - sample_predictions.png")
	print("    - test_results.json")
	print("\n  Done!")


if __name__ == "__main__":
	main()
