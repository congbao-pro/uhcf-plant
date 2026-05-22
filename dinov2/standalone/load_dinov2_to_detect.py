#!/usr/bin/env python3
"""
Load trained DINOv2 model (best_model.pt) and training history (training_history.pt) for:
1. Detect classes for unselected images (images not seen during train/val/test)
2. Measure inference time per image
3. Visualize training history with detailed plots

Usage:
    python load_dinov2_to_detect.py --model_path ./outputs/best_model.pt --history_path ./outputs/training_history.pt --unselected_csv ./outputs/dataset_unselected.csv --out_dir ./outputs/analysis
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ──────────────────────────────────────────────
# DINOv2 Classifier Model (same architecture as dinov2.py)
# ──────────────────────────────────────────────
class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + linear classification head."""

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

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
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
# Load Model
# ──────────────────────────────────────────────
def load_model(model_path, device):
    """Load trained DINOv2 model from checkpoint."""
    print(f"[INFO] Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[WARN] Failed default load with weights_only=False: {e}")
        print("[INFO] Retrying without explicit weights_only (fallback mode)...")
        checkpoint = torch.load(model_path, map_location=device)

    # Extract configuration
    args = checkpoint.get('args', {})
    class_to_idx = checkpoint.get('class_to_idx', {})
    model_name = checkpoint.get('model_name', args.get('model_name', 'dinov2_vitb14'))
    freeze_backbone = checkpoint.get('freeze_backbone', True)
    dropout = checkpoint.get('dropout', args.get('dropout', 0.1))
    n_classes = checkpoint.get('n_classes', len(class_to_idx))

    # Create model
    model = DINOv2Classifier(
        model_name=model_name,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        dropout=dropout
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")
    print(f"  - Model: {model_name}")
    print(f"  - Classes: {n_classes}")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Freeze backbone: {freeze_backbone}")
    print(f"  - Best Macro F1: {checkpoint.get('macro_f1', 'N/A')}")
    print(f"  - Best Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, class_to_idx


# ──────────────────────────────────────────────
# Detect Unselected Images
# ──────────────────────────────────────────────
def detect_unselected_images(model, unselected_csv, class_to_idx, device, out_dir):
    """Detect classes for unselected images and measure inference time."""
    print(f"\n[DETECTION] Processing unselected images from {unselected_csv}")

    if not os.path.exists(unselected_csv):
        print(f"[WARNING] Unselected CSV not found: {unselected_csv}")
        return

    df = pd.read_csv(unselected_csv)
    print(f"[INFO] Found {len(df)} unselected images")

    if len(df) == 0:
        print("[INFO] No unselected images to process")
        return

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    results = []
    inference_times = []

    print("[INFO] Starting inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detecting"):
        img_path = row['image_path']
        true_class = row['class_name']

        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Inference with time measurement
            start_time = time.time()
            with torch.no_grad():
                logits = model(img_tensor)
                pred_probs = F.softmax(logits, dim=-1)
                pred_idx = logits.argmax(dim=-1).item()
            inference_time = (time.time() - start_time) * 1000  # ms

            pred_class = idx_to_class[pred_idx]
            confidence = pred_probs[0, pred_idx].item()

            results.append({
                'image_path': img_path,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'correct': pred_class == true_class,
                'inference_time_ms': inference_time
            })
            inference_times.append(inference_time)

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(out_dir, 'unselected_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Saved predictions to {results_path}")

    # Statistics
    accuracy = results_df['correct'].mean()
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)

    print(f"\n[STATISTICS]")
    print(f"  Total images: {len(results_df)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Median inference time: {median_time:.2f} ms")
    print(f"  Std inference time: {std_time:.2f} ms")
    print(f"  Min time: {min(inference_times):.2f} ms")
    print(f"  Max time: {max(inference_times):.2f} ms")

    # Save statistics
    stats = {
        'total_images': len(results_df),
        'accuracy': accuracy,
        'avg_inference_time_ms': avg_time,
        'median_inference_time_ms': median_time,
        'std_inference_time_ms': std_time,
        'min_inference_time_ms': min(inference_times),
        'max_inference_time_ms': max(inference_times)
    }
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(out_dir, 'inference_statistics.csv')
    stats_df.to_csv(stats_path, index=False)

    # Plot inference time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(avg_time, color='r', linestyle='--', label=f'Mean: {avg_time:.2f} ms')
    plt.axvline(median_time, color='g', linestyle='--', label=f'Median: {median_time:.2f} ms')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('DINOv2 Inference Time Distribution for Unselected Images')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'inference_time_distribution.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved inference time plot")


# ──────────────────────────────────────────────
# Visualize Training History
# ──────────────────────────────────────────────
def visualize_training_history(history_path, out_dir):
    """Comprehensive visualization of training history."""
    print(f"\n[VISUALIZATION] Loading training history from {history_path}")
    try:
        history = torch.load(history_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[WARN] Failed loading history with weights_only=False: {e}")
        history = torch.load(history_path, map_location='cpu')

    print(f"[INFO] History contains {len(history['train_loss'])} epochs")
    print(f"[INFO] Available metrics: {list(history.keys())}")

    epochs = range(1, len(history['train_loss']) + 1)

    # Comprehensive plot (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('DINOv2 Training History - Comprehensive View', fontsize=16, fontweight='bold')

    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(alpha=0.3)

    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy'); ax.legend(); ax.grid(alpha=0.3)

    # 3. F1 Scores
    ax = axes[0, 2]
    if 'val_macro_f1' in history:
        ax.plot(epochs, history['val_macro_f1'], 'g-', label='Macro F1', linewidth=2)
    if 'val_micro_f1' in history:
        ax.plot(epochs, history['val_micro_f1'], 'b-', label='Micro F1', linewidth=2)
    if 'val_weighted_f1' in history:
        ax.plot(epochs, history['val_weighted_f1'], 'r-', label='Weighted F1', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Scores'); ax.legend(); ax.grid(alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 0]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule'); ax.grid(alpha=0.3)

    # 5. Time per epoch
    ax = axes[1, 1]
    if 'train_time' in history:
        ax.plot(epochs, history['train_time'], 'b-', label='Train Time', linewidth=2)
    if 'val_time' in history:
        ax.plot(epochs, history['val_time'], 'r-', label='Val Time', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Time (seconds)')
    ax.set_title('Training/Validation Time'); ax.legend(); ax.grid(alpha=0.3)

    # 6. Train vs Val Loss comparison
    ax = axes[1, 2]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss Curve'); ax.legend(); ax.grid(alpha=0.3)

    # 7. Overfitting Analysis
    ax = axes[2, 0]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
        gap = [t - v for t, v in zip(history['train_acc'], history['val_accuracy'])]
        ax2 = ax.twinx()
        ax2.plot(epochs, gap, 'g--', alpha=0.5, label='Gap')
        ax2.set_ylabel('Train-Val Gap', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Overfitting Analysis'); ax.legend(loc='upper left'); ax.grid(alpha=0.3)

    # 8. Best metrics summary
    ax = axes[2, 1]
    ax.axis('off')
    best_epoch = np.argmax(history['val_macro_f1']) + 1 if 'val_macro_f1' in history else len(list(epochs))
    summary_text = f"""
    Best Performance:

    Epoch: {best_epoch}
    Train Loss: {history['train_loss'][best_epoch-1]:.4f}
    Train Acc: {history['train_acc'][best_epoch-1]:.4f}
    Val Macro F1: {history['val_macro_f1'][best_epoch-1]:.4f}
    Val Accuracy: {history['val_accuracy'][best_epoch-1]:.4f}

    Final Performance:

    Epoch: {len(list(epochs))}
    Train Loss: {history['train_loss'][-1]:.4f}
    Train Acc: {history['train_acc'][-1]:.4f}
    Val Macro F1: {history['val_macro_f1'][-1]:.4f}
    Val Accuracy: {history['val_accuracy'][-1]:.4f}
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 9. Smoothed loss
    ax = axes[2, 2]
    epochs_list = list(epochs)
    if len(epochs_list) > 1:
        window = min(5, len(epochs_list) // 5)
        if window > 1:
            smoothed = np.convolve(history['train_loss'],
                                   np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(epochs_list) + 1), smoothed, 'b-',
                    label='Smoothed', linewidth=2)
        ax.plot(epochs_list, history['train_loss'], 'b-', alpha=0.3,
                label='Raw', linewidth=1)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Loss Smoothing'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history_comprehensive.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved comprehensive training history plot")

    # Detailed F1 plot
    plt.figure(figsize=(12, 6))
    if 'val_macro_f1' in history:
        plt.plot(epochs, history['val_macro_f1'], 'o-', label='Macro F1', linewidth=2)
    if 'val_micro_f1' in history:
        plt.plot(epochs, history['val_micro_f1'], 's-', label='Micro F1', linewidth=2)
    if 'val_weighted_f1' in history:
        plt.plot(epochs, history['val_weighted_f1'], '^-', label='Weighted F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12); plt.ylabel('F1 Score', fontsize=12)
    plt.title('DINOv2 Validation F1 Scores Across Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'f1_scores_detailed.png'), dpi=150)
    plt.close()

    # Export metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_df.insert(0, 'epoch', list(epochs))
    metrics_df.to_csv(os.path.join(out_dir, 'training_metrics.csv'), index=False)
    print(f"[INFO] Saved training metrics to CSV")

    print(f"\n[SUCCESS] All visualizations saved to {out_dir}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(args):
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model, class_to_idx = load_model(args.model_path, device)

    # Detect unselected images
    if args.unselected_csv and os.path.exists(args.unselected_csv):
        detect_unselected_images(model, args.unselected_csv, class_to_idx,
                                 device, args.out_dir)
    else:
        print(f"[WARNING] Skipping detection - unselected CSV not provided or not found")

    # Visualize training history
    if args.history_path and os.path.exists(args.history_path):
        visualize_training_history(args.history_path, args.out_dir)
    else:
        print(f"[WARNING] Skipping visualization - history file not found")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 80)
    print(f"[END] Analysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {duration:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load DINOv2 model and analyze results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--history_path', type=str, default=None,
                        help='Path to training_history.pt (optional)')
    parser.add_argument('--unselected_csv', type=str, default=None,
                        help='Path to dataset_unselected.csv')
    parser.add_argument('--out_dir', type=str, default='./outputs/analysis',
                        help='Output directory for analysis results')

    args = parser.parse_args()
    main(args)
