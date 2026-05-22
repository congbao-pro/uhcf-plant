#!/usr/bin/env python3
"""
Load best_model.pt and training_history.pt for:
1. Detect classes for unselected images
2. Measure inference time per image
3. Visualize training history with detailed plots

Usage:
    python load_model_to_detect.py --model_path ./outputs/best_model.pt --history_path ./outputs/training_history.pt --unselected_csv ./outputs/dataset_unselected.csv --out_dir ./outputs/analysis
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
import timm

# Import model classes from training script
import sys
sys.path.append(os.path.dirname(__file__))

# Model definitions (copied from organ_aware_switch_vit.py)
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
    def forward(self, x):
        return self.net(x)

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
        flat = tokens.reshape(B*T, D)
        flat_prior = organ_priors.reshape(B*T, -1)
        logits = self.router(flat, flat_prior)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs+1e-12).log()).sum(dim=-1)

        max_k = min(2, self.n_experts) if training else self.top_k
        topk_vals, topk_idx = torch.topk(probs, max_k, dim=-1)
        topk_probs = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)

        outputs = torch.zeros_like(flat, device=flat.device)
        import math
        capacity = math.ceil((B*T / self.n_experts) * self.capacity_factor)
        processed_mask = torch.zeros(B*T, dtype=torch.bool, device=flat.device)

        for e in range(self.n_experts):
            mask = (topk_idx == e).any(dim=-1)
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if idx.numel() > capacity:
                expert_probs = probs[idx, e]
                _, sorted_idx = torch.sort(expert_probs, descending=True)
                idx = idx[sorted_idx[:capacity]]

            selected = flat[idx]
            out_sel = self.experts[e](selected)

            weights = torch.zeros(idx.shape[0], device=flat.device)
            for i, token_idx in enumerate(idx):
                expert_mask = (topk_idx[token_idx] == e)
                if expert_mask.any():
                    k_pos = expert_mask.nonzero(as_tuple=True)[0][0]
                    weights[i] = topk_probs[token_idx, k_pos]

            outputs[idx] += out_sel * weights.unsqueeze(-1)
            processed_mask[idx] = True

        outputs[~processed_mask] = flat[~processed_mask]
        outputs = outputs.reshape(B, T, D)
        return outputs, probs.reshape(B, T, -1), entropy.reshape(B, T)

class OrganAwareSwitchViT(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', n_classes=100, organ_dim=5,
                 n_experts=8, d_ff_expert=1024, top_k=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(dummy)
            else:
                feat = self.backbone(dummy)

        if isinstance(feat, torch.Tensor) and feat.dim() == 3:
            _, NT, D = feat.shape
            self.token_dim = D
            self.num_tokens = NT
        else:
            self.token_dim = 768
            self.num_tokens = 197

        self.organ_dim = organ_dim
        self.switch = SwitchMoE(d_model=self.token_dim, organ_dim=self.organ_dim,
                                n_experts=n_experts, d_ff=d_ff_expert, top_k=top_k)
        self.cls_head = nn.Linear(self.token_dim, n_classes)
        self.aux_head = OrganAuxHead(self.token_dim, organ_dim)
        self.ln = nn.LayerNorm(self.token_dim)

    def forward(self, x, organ_priors_image, training=True, capacity_factor=None):
        if hasattr(self.backbone, 'forward_features'):
            tokens = self.backbone.forward_features(x)
        else:
            tokens = self.backbone(x)

        cls = tokens[:, 0:1, :]
        patches = tokens[:, 1:, :]
        B, T, D = patches.shape

        if isinstance(organ_priors_image, torch.Tensor):
            org_prior_tokens = organ_priors_image.unsqueeze(1).expand(B, T, -1)
        else:
            org_prior_tokens = torch.stack([torch.from_numpy(organ_priors_image[i]).float()
                                           for i in range(len(organ_priors_image))]).to(x.device)
            org_prior_tokens = org_prior_tokens.unsqueeze(1).expand(B, T, -1)

        outputs, probs, entropy = self.switch(patches, org_prior_tokens, training=training)
        tokens2 = torch.cat([cls, outputs], dim=1)
        cls_final = tokens2[:, 0, :]
        cls_final = self.ln(cls_final)
        logits = self.cls_head(cls_final)
        aux_org = self.aux_head(cls_final)
        return logits, aux_org, probs, entropy


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"[INFO] Loading model from {model_path}")
    # PyTorch >=2.6 mặc định weights_only=True gây lỗi Unpickling với dict checkpoint cũ.
    # Ép weights_only=False vì file này do ta tự tạo, tin cậy nguồn.
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[WARN] Failed default load with weights_only=False: {e}")
        print("[INFO] Retrying without explicit weights_only (fallback mode)...")
        checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    args = checkpoint.get('args', {})
    organ_dim = checkpoint.get('organ_dim', 5)
    n_experts = checkpoint.get('n_experts', 8)
    class_to_idx = checkpoint.get('class_to_idx', {})

    vit_name = args.get('vit_name', 'vit_base_patch16_224')
    d_ff_expert = args.get('d_ff_expert', 1024)
    top_k = args.get('top_k', 1)
    n_classes = len(class_to_idx)

    # Create model
    model = OrganAwareSwitchViT(
        vit_name=vit_name,
        n_classes=n_classes,
        organ_dim=organ_dim,
        n_experts=n_experts,
        d_ff_expert=d_ff_expert,
        top_k=top_k,
        pretrained=False
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")
    print(f"  - Classes: {n_classes}")
    print(f"  - Organ dim: {organ_dim}")
    print(f"  - Experts: {n_experts}")
    print(f"  - Best Macro F1: {checkpoint.get('macro_f1', 'N/A')}")

    return model, class_to_idx, organ_dim


def detect_unselected_images(model, unselected_csv, class_to_idx, organ_dim, device, out_dir):
    """Detect classes for unselected images and measure inference time"""
    print(f"\n[DETECTION] Processing unselected images from {unselected_csv}")

    if not os.path.exists(unselected_csv):
        print(f"[WARNING] Unselected CSV not found: {unselected_csv}")
        return

    # Load unselected images
    df = pd.read_csv(unselected_csv)
    print(f"[INFO] Found {len(df)} unselected images")

    if len(df) == 0:
        print("[INFO] No unselected images to process")
        return

    # Reverse class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Results
    results = []
    inference_times = []

    # Uniform organ prior
    organ_prior = torch.ones(1, organ_dim, device=device) * (1.0 / organ_dim)

    print("[INFO] Starting inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detecting"):
        img_path = row['image_path']
        true_class = row['class_name']

        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Inference with time measurement
            start_time = time.time()
            with torch.no_grad():
                logits, _, _, _ = model(img_tensor, organ_prior, training=False)
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
    plt.title('Inference Time Distribution for Unselected Images')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'inference_time_distribution.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved inference time plot")


def visualize_training_history(history_path, out_dir):
    """Comprehensive visualization of training history"""
    print(f"\n[VISUALIZATION] Loading training history from {history_path}")
    # Tương tự checkpoint, ép weights_only=False để đọc dict đầy đủ.
    try:
        history = torch.load(history_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[WARN] Failed loading history with weights_only=False: {e}")
        history = torch.load(history_path, map_location='cpu')

    print(f"[INFO] History contains {len(history['train_loss'])} epochs")
    print(f"[INFO] Available metrics: {list(history.keys())}")

    # Create comprehensive plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Training History - Comprehensive View', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. F1 scores
    ax = axes[0, 2]
    if 'val_macro_f1' in history:
        ax.plot(epochs, history['val_macro_f1'], 'g-', label='Macro F1', linewidth=2)
    if 'val_micro_f1' in history:
        ax.plot(epochs, history['val_micro_f1'], 'b-', label='Micro F1', linewidth=2)
    if 'val_weighted_f1' in history:
        ax.plot(epochs, history['val_weighted_f1'], 'r-', label='Weighted F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Loss components
    ax = axes[1, 0]
    if 'train_cls_loss' in history:
        ax.plot(epochs, history['train_cls_loss'], 'b-', label='Cls Loss', linewidth=2)
    if 'train_aux_loss' in history:
        ax.plot(epochs, history['train_aux_loss'], 'r-', label='Aux Loss', linewidth=2)
    if 'train_balance_loss' in history:
        ax.plot(epochs, history['train_balance_loss'], 'g-', label='Balance Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. Time per epoch
    ax = axes[1, 1]
    if 'train_time' in history:
        ax.plot(epochs, history['train_time'], 'b-', label='Train Time', linewidth=2)
    if 'val_time' in history:
        ax.plot(epochs, history['val_time'], 'r-', label='Val Time', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training/Validation Time')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. Capacity factor schedule
    ax = axes[1, 2]
    if 'capacity_factor' in history:
        ax.plot(epochs, history['capacity_factor'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Capacity Factor')
        ax.set_title('Capacity Factor Schedule (O7)')
        ax.grid(alpha=0.3)

    # 7. Train vs Val Accuracy comparison
    ax = axes[2, 0]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
        # Compute gap
        gap = [t - v for t, v in zip(history['train_acc'], history['val_accuracy'])]
        ax2 = ax.twinx()
        ax2.plot(epochs, gap, 'g--', alpha=0.5, label='Gap')
        ax2.set_ylabel('Train-Val Gap', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overfitting Analysis')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # 8. Best metrics summary
    ax = axes[2, 1]
    ax.axis('off')
    best_epoch = np.argmax(history['val_macro_f1']) + 1 if 'val_macro_f1' in history else len(epochs)
    summary_text = f"""
    Best Performance:

    Epoch: {best_epoch}
    Train Loss: {history['train_loss'][best_epoch-1]:.4f}
    Train Acc: {history['train_acc'][best_epoch-1]:.4f}
    Val Macro F1: {history['val_macro_f1'][best_epoch-1]:.4f}
    Val Accuracy: {history['val_accuracy'][best_epoch-1]:.4f}

    Final Performance:

    Epoch: {len(epochs)}
    Train Loss: {history['train_loss'][-1]:.4f}
    Train Acc: {history['train_acc'][-1]:.4f}
    Val Macro F1: {history['val_macro_f1'][-1]:.4f}
    Val Accuracy: {history['val_accuracy'][-1]:.4f}
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 9. Loss landscape
    ax = axes[2, 2]
    if len(epochs) > 1:
        # Smoothed loss
        window = min(5, len(epochs) // 5)
        if window > 1:
            smoothed = np.convolve(history['train_loss'],
                                  np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(epochs)+1), smoothed, 'b-',
                   label='Smoothed', linewidth=2)
        ax.plot(epochs, history['train_loss'], 'b-', alpha=0.3,
               label='Raw', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Smoothing')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history_comprehensive.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved comprehensive training history plot")

    # Individual detailed plots
    # Plot 1: Detailed F1 comparison
    plt.figure(figsize=(12, 6))
    if 'val_macro_f1' in history:
        plt.plot(epochs, history['val_macro_f1'], 'o-', label='Macro F1', linewidth=2)
    if 'val_micro_f1' in history:
        plt.plot(epochs, history['val_micro_f1'], 's-', label='Micro F1', linewidth=2)
    if 'val_weighted_f1' in history:
        plt.plot(epochs, history['val_weighted_f1'], '^-', label='Weighted F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Validation F1 Scores Across Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'f1_scores_detailed.png'), dpi=150)
    plt.close()

    # Plot 2: Loss breakdown
    if 'train_cls_loss' in history and 'train_aux_loss' in history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Stacked area
        ax1.fill_between(epochs, 0, history['train_cls_loss'],
                        alpha=0.3, label='Classification Loss')
        ax1.fill_between(epochs, history['train_cls_loss'],
                        [c+a for c, a in zip(history['train_cls_loss'], history['train_aux_loss'])],
                        alpha=0.3, label='Auxiliary Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Components (Stacked)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Individual lines
        ax2.plot(epochs, history['train_cls_loss'], 'b-', label='Cls Loss', linewidth=2)
        ax2.plot(epochs, history['train_aux_loss'], 'r-', label='Aux Loss', linewidth=2)
        if 'train_balance_loss' in history:
            ax2.plot(epochs, history['train_balance_loss'], 'g-', label='Balance Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components (Individual)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_components_detailed.png'), dpi=150)
        plt.close()

    # Export metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_df.insert(0, 'epoch', epochs)
    metrics_df.to_csv(os.path.join(out_dir, 'training_metrics.csv'), index=False)
    print(f"[INFO] Saved training metrics to CSV")

    print(f"\n[SUCCESS] All visualizations saved to {out_dir}")


def main(args):
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model, class_to_idx, organ_dim = load_model(args.model_path, device)

    # Detect unselected images
    if args.unselected_csv and os.path.exists(args.unselected_csv):
        detect_unselected_images(model, args.unselected_csv, class_to_idx,
                                organ_dim, device, args.out_dir)
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
    parser = argparse.ArgumentParser(description='Load model and analyze results')
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
