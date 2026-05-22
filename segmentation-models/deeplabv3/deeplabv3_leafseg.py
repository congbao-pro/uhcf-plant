"""
Leaf Disease Segmentation dataset structure:
	leaf-seg/
	├── images/   (*.jpg)
	├── masks/    (*.png)
	└── train.csv (imageid,maskid)
"""

import os
import csv
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


# ============================================================
# 1. JOINT TRANSFORMS
# ============================================================

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
	def __init__(self, p=0.3):
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
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]

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
	def __init__(self, root_dir, joint_transform=None,
				 indices=None, value_mapping=None, verbose=True):
		self.root_dir = root_dir
		self.images_dir = os.path.join(root_dir, 'images')
		self.masks_dir = os.path.join(root_dir, 'masks')
		self.csv_path = os.path.join(root_dir, 'train.csv')
		self.joint_transform = joint_transform

		self.pairs = self._read_pairs()

		if indices is not None:
			self.pairs = [self.pairs[i] for i in indices]

		if value_mapping is None:
			self.num_classes, self.value_mapping, self.class_names = self._detect_classes()
		else:
			self.value_mapping = dict(value_mapping)
			inv = sorted(self.value_mapping.values())
			self.num_classes = (max(inv) + 1) if inv else 1
			self.class_names = [f'class_{i}' for i in range(self.num_classes)]

		if verbose:
			print(f"  Dataset loaded: {len(self.pairs)} samples | "
				  f"num_classes={self.num_classes} | mapping={self.value_mapping}")

	def _read_pairs(self):
		pairs = []
		with open(self.csv_path, 'r', encoding='utf-8-sig', newline='') as f:
			text = f.read(4096)
			f.seek(0)
			try:
				dialect = csv.Sniffer().sniff(text)
			except csv.Error:
				dialect = csv.excel_tab

			reader = csv.DictReader(f, dialect=dialect)
			for row in reader:
				imageid = row.get('imageid')
				maskid = row.get('maskid')
				if imageid and maskid:
					pairs.append((imageid.strip(), maskid.strip()))

		if not pairs:
			raise RuntimeError(f'No (imageid, maskid) pairs found in {self.csv_path}')
		return pairs

	def _detect_classes(self):
		unique_vals = set()
		n_sample = min(100, len(self.pairs))
		for i in range(n_sample):
			_, mask_name = self.pairs[i]
			mask_path = os.path.join(self.masks_dir, mask_name)
			if not os.path.isfile(mask_path):
				continue
			arr = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
			unique_vals.update(np.unique(arr).tolist())

		sorted_vals = sorted(unique_vals)
		if not sorted_vals:
			sorted_vals = [0, 1]

		if max(sorted_vals) >= len(sorted_vals):
			mapping = {v: i for i, v in enumerate(sorted_vals)}
		else:
			mapping = {v: int(v) for v in sorted_vals}

		num_classes = max(mapping.values()) + 1
		class_names = [f'class_{i}' for i in range(num_classes)]
		return num_classes, mapping, class_names

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		img_name, mask_name = self.pairs[idx]
		img_path = os.path.join(self.images_dir, img_name)
		mask_path = os.path.join(self.masks_dir, mask_name)

		image = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path).convert('L')

		if self.joint_transform:
			image, mask = self.joint_transform(image, mask)
		else:
			image = transforms.ToTensor()(image)
			image = transforms.Normalize(mean=JointToTensor.MEAN,
										 std=JointToTensor.STD)(image)
			mask_np = np.array(mask, dtype=np.int64)
			if self.value_mapping:
				mapped = np.zeros_like(mask_np)
				for old_v, new_v in self.value_mapping.items():
					mapped[mask_np == old_v] = new_v
				mask_np = mapped
			mask = torch.from_numpy(mask_np).long()

		return image, mask


# ============================================================
# 3. MODEL
# ============================================================

def build_deeplabv3_model(num_classes, pretrained_backbone=True):
	"""Build DeepLabV3-ResNet50 and replace classifier head."""
	try:
		weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained_backbone else None
		model = models.segmentation.deeplabv3_resnet50(weights=weights, aux_loss=True)
	except Exception:
		model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained_backbone, aux_loss=True)

	model.classifier = DeepLabHead(2048, num_classes)
	return model


# ============================================================
# 4. METRICS
# ============================================================

def compute_metrics(pred, gt, num_classes):
	"""Compute mIoU, mean Dice, Pixel Accuracy (numpy arrays)."""
	assert pred.shape == gt.shape

	per_class_iou = []
	per_class_dice = []
	valid_classes = []

	for c in range(num_classes):
		pred_c = (pred == c)
		gt_c = (gt == c)
		union = np.logical_or(pred_c, gt_c).sum()
		inter = np.logical_and(pred_c, gt_c).sum()

		if union == 0:
			continue

		iou = inter / union
		dice = (2 * inter) / (pred_c.sum() + gt_c.sum() + 1e-8)

		valid_classes.append(c)
		per_class_iou.append(float(iou))
		per_class_dice.append(float(dice))

	pixel_acc = float((pred == gt).sum() / pred.size) if pred.size > 0 else 0.0

	return {
		'mIoU': np.mean(per_class_iou) if per_class_iou else 0.0,
		'mean_Dice': np.mean(per_class_dice) if per_class_dice else 0.0,
		'pixel_accuracy': pixel_acc,
		'per_class_iou': dict(zip(valid_classes, per_class_iou)),
		'per_class_dice': dict(zip(valid_classes, per_class_dice)),
	}


# ============================================================
# 5. TRAIN / EVAL
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer,
					device, epoch, scaler=None):
	model.train()
	running_loss = 0.0
	total_pixels = 0
	correct = 0
	n_batches = 0

	pbar = tqdm(dataloader, desc=f'  Epoch {epoch:>3d} [Train]', ncols=120)
	for images, masks in pbar:
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		if scaler is not None:
			with torch.cuda.amp.autocast():
				logits = model(images)['out']
				loss = criterion(logits, masks)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			logits = model(images)['out']
			loss = criterion(logits, masks)
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			pred = logits.argmax(dim=1)
			correct += (pred == masks).sum().item()
			total_pixels += masks.numel()

		running_loss += loss.item()
		n_batches += 1
		pbar.set_postfix(loss=f'{running_loss / n_batches:.4f}',
						 pacc=f'{100.0 * correct / max(total_pixels, 1):.2f}%')

	return {
		'loss': running_loss / max(n_batches, 1),
		'pixel_acc': 100.0 * correct / max(total_pixels, 1),
	}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
	model.eval()
	running_loss = 0.0
	n_batches = 0

	all_pred = []
	all_gt = []

	pbar = tqdm(dataloader, desc='  [Eval]', ncols=120)
	for images, masks in pbar:
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)

		logits = model(images)['out']
		loss = criterion(logits, masks)

		pred = logits.argmax(dim=1).cpu().numpy()
		gt = masks.cpu().numpy()

		all_pred.append(pred)
		all_gt.append(gt)

		running_loss += loss.item()
		n_batches += 1

	if all_pred:
		all_pred = np.concatenate(all_pred, axis=0)
		all_gt = np.concatenate(all_gt, axis=0)
	else:
		all_pred = np.zeros((0, 1, 1), dtype=np.int64)
		all_gt = np.zeros((0, 1, 1), dtype=np.int64)

	metrics = compute_metrics(all_pred.reshape(-1), all_gt.reshape(-1), num_classes)
	metrics['loss'] = running_loss / max(n_batches, 1)
	return metrics


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_training_curves(history, save_path):
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	epochs = range(1, len(history['train_loss']) + 1)

	axes[0, 0].plot(epochs, history['train_loss'], 'b-o', ms=3, lw=2, label='Train')
	axes[0, 0].plot(epochs, history['val_loss'], 'r-o', ms=3, lw=2, label='Val')
	axes[0, 0].set_title('Loss')
	axes[0, 0].legend()
	axes[0, 0].grid(alpha=0.3)

	axes[0, 1].plot(epochs, history['val_miou'], 'r-o', ms=3, lw=2, label='Val mIoU')
	axes[0, 1].set_title('mIoU (%)')
	axes[0, 1].legend()
	axes[0, 1].grid(alpha=0.3)

	axes[1, 0].plot(epochs, history['val_dice'], 'r-o', ms=3, lw=2, label='Val Dice')
	axes[1, 0].set_title('Mean Dice (%)')
	axes[1, 0].legend()
	axes[1, 0].grid(alpha=0.3)

	axes[1, 1].plot(epochs, history['train_pacc'], 'b-o', ms=3, lw=2, label='Train')
	axes[1, 1].plot(epochs, history['val_pacc'], 'r-o', ms=3, lw=2, label='Val')
	axes[1, 1].set_title('Pixel Accuracy (%)')
	axes[1, 1].legend()
	axes[1, 1].grid(alpha=0.3)

	for ax in axes.flat:
		ax.set_xlabel('Epoch')

	fig.suptitle('DeepLabV3 — Training Curves', fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'  Curves saved -> {save_path}')


def plot_sample_predictions(model, dataset, device, num_classes, save_path, n=6):
	model.eval()
	n = min(n, len(dataset))
	sample_indices = random.sample(range(len(dataset)), n)

	palette = np.array([
		[0, 0, 0],
		[255, 50, 50],
		[50, 255, 50],
		[50, 50, 255],
		[255, 255, 50],
		[255, 50, 255],
		[50, 255, 255],
		[255, 140, 0],
	], dtype=np.uint8)

	fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
	if n == 1:
		axes = np.expand_dims(axes, axis=0)

	mean = np.array(JointToTensor.MEAN)
	std = np.array(JointToTensor.STD)

	for row, idx in enumerate(sample_indices):
		img_t, gt_t = dataset[idx]

		with torch.no_grad():
			logits = model(img_t.unsqueeze(0).to(device))['out']
			pred = logits.argmax(1)[0].cpu().numpy()

		img = img_t.permute(1, 2, 0).cpu().numpy()
		img = np.clip(img * std + mean, 0, 1)
		gt = gt_t.cpu().numpy()

		gt_color = palette[gt % len(palette)]
		pred_color = palette[pred % len(palette)]

		axes[row, 0].imshow(img)
		axes[row, 0].set_title('Image')
		axes[row, 0].axis('off')

		axes[row, 1].imshow(gt_color)
		axes[row, 1].set_title('GT')
		axes[row, 1].axis('off')

		axes[row, 2].imshow(pred_color)
		axes[row, 2].set_title('Pred')
		axes[row, 2].axis('off')

	fig.suptitle('Sample Predictions — DeepLabV3', fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'  Sample predictions saved -> {save_path}')


# ============================================================
# 7. MAIN
# ============================================================

def main():
	parser = argparse.ArgumentParser(
		description='DeepLabV3 — Leaf Disease Segmentation',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	g = parser.add_argument_group('Data')
	g.add_argument('--data_dir', type=str, required=True,
				   help='Root of Leaf Disease Segmentation dataset (contains images/, masks/, train.csv)')
	g.add_argument('--output_dir', type=str, default='./output_deeplabv3')
	g.add_argument('--img_size', type=int, default=512)
	g.add_argument('--train_ratio', type=float, default=0.7)
	g.add_argument('--val_ratio', type=float, default=0.15)

	g = parser.add_argument_group('Training')
	g.add_argument('--epochs', type=int, default=50)
	g.add_argument('--batch_size', type=int, default=4)
	g.add_argument('--lr', type=float, default=1e-4)
	g.add_argument('--weight_decay', type=float, default=0.01)
	g.add_argument('--patience', type=int, default=10)
	g.add_argument('--num_workers', type=int, default=4)
	g.add_argument('--seed', type=int, default=42)
	g.add_argument('--pretrained_backbone', action='store_true',
				   help='Use ImageNet-pretrained backbone')

	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"\n{'='*65}")
	print('  DeepLabV3 — Leaf Disease Segmentation')
	print(f"{'='*65}")
	print(f"  Device     : {device}" +
		  (f"  ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
	print(f'  Output dir : {args.output_dir}')

	print(f'\n  Loading dataset from: {args.data_dir}')
	csv_path = os.path.join(args.data_dir, 'train.csv')

	with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
		lines = [ln for ln in f.readlines() if ln.strip()]
	total = max(len(lines) - 1, 0)

	if total == 0:
		raise RuntimeError('No records found in train.csv')

	indices = list(range(total))
	random.Random(args.seed).shuffle(indices)
	train_end = int(total * args.train_ratio)
	val_end = train_end + int(total * args.val_ratio)
	train_idx = indices[:train_end]
	val_idx = indices[train_end:val_end]
	test_idx = indices[val_end:]

	print(f'  Total: {total} | Train: {len(train_idx)} | '
		  f'Val: {len(val_idx)} | Test: {len(test_idx)}')

	tmp_ds = LeafSegDataset(args.data_dir, verbose=False)
	num_classes = tmp_ds.num_classes
	value_mapping = tmp_ds.value_mapping
	class_names = tmp_ds.class_names
	del tmp_ds

	print(f'  Classes: {num_classes}   mapping: {value_mapping}')

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
	val_ds = LeafSegDataset(args.data_dir, joint_transform=val_transform,
							indices=val_idx, value_mapping=value_mapping,
							verbose=False)
	test_ds = LeafSegDataset(args.data_dir, joint_transform=val_transform,
							 indices=test_idx, value_mapping=value_mapping,
							 verbose=False)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
							  num_workers=args.num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
							num_workers=args.num_workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
							 num_workers=args.num_workers, pin_memory=True)

	model = build_deeplabv3_model(num_classes, args.pretrained_backbone).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'\n  Parameters: {total_params:,} total | {train_params:,} trainable')

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
								  weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=args.epochs, eta_min=1e-7)

	scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

	history = {
		'train_loss': [], 'val_loss': [],
		'val_miou': [], 'val_dice': [],
		'train_pacc': [], 'val_pacc': [],
	}
	best_miou = 0.0
	patience_counter = 0

	print(f"\n{'='*65}")
	print(f'  Training for up to {args.epochs} epochs  (patience={args.patience})')
	print(f"{'='*65}\n")

	for epoch in range(1, args.epochs + 1):
		train_stats = train_one_epoch(model, train_loader, criterion,
									  optimizer, device, epoch, scaler)
		val_stats = evaluate(model, val_loader, criterion, device, num_classes)
		scheduler.step()

		history['train_loss'].append(train_stats['loss'])
		history['val_loss'].append(val_stats['loss'])
		history['val_miou'].append(val_stats['mIoU'] * 100)
		history['val_dice'].append(val_stats['mean_Dice'] * 100)
		history['train_pacc'].append(train_stats['pixel_acc'])
		history['val_pacc'].append(val_stats['pixel_accuracy'] * 100)

		print(f'\n  Epoch {epoch:>3d}: '
			  f'train_loss={train_stats["loss"]:.4f} | '
			  f'val_loss={val_stats["loss"]:.4f} | '
			  f'mIoU={val_stats["mIoU"] * 100:.2f}% | '
			  f'Dice={val_stats["mean_Dice"] * 100:.2f}% | '
			  f'val_pAcc={val_stats["pixel_accuracy"] * 100:.2f}%')

		if val_stats['mIoU'] > best_miou:
			best_miou = float(val_stats['mIoU'])
			patience_counter = 0
			checkpoint = {
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'num_classes': num_classes,
				'value_mapping': value_mapping,
				'class_names': class_names,
				'best_val_miou': best_miou * 100.0,
				'model_config': {
					'arch': 'deeplabv3_resnet50',
					'num_classes': num_classes,
					'img_size': args.img_size,
					'pretrained_backbone': args.pretrained_backbone,
				},
			}
			torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
			print('  [*] Best model updated.')
		else:
			patience_counter += 1
			print(f'  [i] No improvement. patience={patience_counter}/{args.patience}')

		if patience_counter >= args.patience:
			print('\n  Early stopping triggered.')
			break

	last_ckpt = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'num_classes': num_classes,
		'value_mapping': value_mapping,
		'class_names': class_names,
		'model_config': {
			'arch': 'deeplabv3_resnet50',
			'num_classes': num_classes,
			'img_size': args.img_size,
			'pretrained_backbone': args.pretrained_backbone,
		},
	}
	torch.save(last_ckpt, os.path.join(args.output_dir, 'last_model.pth'))

	plot_training_curves(history, os.path.join(args.output_dir, 'training_curves.png'))

	print(f"\n{'='*65}")
	print(f'  Evaluating on Test Set  ({len(test_idx)} images)')
	print(f"{'='*65}\n")

	best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'),
						   map_location=device, weights_only=False)
	model.load_state_dict(best_ckpt['model_state_dict'])

	test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

	print(f"\n  {'='*50}")
	print('  TEST RESULTS  (DeepLabV3 — Leaf Disease Segmentation)')
	print(f"  {'='*50}")
	print(f"    mIoU           : {test_metrics['mIoU'] * 100:.2f}%")
	print(f"    Mean Dice      : {test_metrics['mean_Dice'] * 100:.2f}%")
	print(f"    Pixel Accuracy : {test_metrics['pixel_accuracy'] * 100:.2f}%")
	print(f"    Loss           : {test_metrics['loss']:.4f}")

	print('\n  Per-class IoU:')
	for c, iou in test_metrics['per_class_iou'].items():
		name = class_names[c] if c < len(class_names) else f'class_{c}'
		print(f'    {name:<20s}: {iou * 100:.2f}%')

	print('\n  Per-class Dice:')
	for c, d in test_metrics['per_class_dice'].items():
		name = class_names[c] if c < len(class_names) else f'class_{c}'
		print(f'    {name:<20s}: {d * 100:.2f}%')

	plot_sample_predictions(
		model, test_ds, device, num_classes,
		os.path.join(args.output_dir, 'sample_predictions.png'), n=6)

	results_json = {
		'mIoU': float(test_metrics['mIoU'] * 100),
		'mean_Dice': float(test_metrics['mean_Dice'] * 100),
		'pixel_accuracy': float(test_metrics['pixel_accuracy'] * 100),
		'loss': float(test_metrics['loss']),
		'best_val_miou': float(best_miou * 100),
		'epochs_trained': int(epoch),
		'per_class_iou': {str(k): float(v * 100) for k, v in test_metrics['per_class_iou'].items()},
		'per_class_dice': {str(k): float(v * 100) for k, v in test_metrics['per_class_dice'].items()},
		'num_classes': int(num_classes),
		'class_names': class_names,
	}

	import json
	with open(os.path.join(args.output_dir, 'test_results.json'),
			  'w', encoding='utf-8') as f:
		json.dump(results_json, f, ensure_ascii=False, indent=2)

	print(f'\n  All results saved to: {args.output_dir}/')
	print('    - best_model.pth')
	print('    - last_model.pth')
	print('    - training_curves.png')
	print('    - sample_predictions.png')
	print('    - test_results.json')
	print('\n  Done!')


if __name__ == '__main__':
	main()
