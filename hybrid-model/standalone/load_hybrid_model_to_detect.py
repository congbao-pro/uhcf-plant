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

## Import model classes from training script
from hybrid_model import (
    HybridPlantModel,
    ResNet18FeatureExtractor,
    PRIOR_ORG_ORDER,
    SECOND_ORG_ORDER,
    is_img_file
)


def load_model(model_path, device):
    """Load trained hybrid model from checkpoint"""
    print(f"[INFO] Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[WARN] Failed default load with weights_only=False: {e}")
        checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    args_ckpt = checkpoint.get('args', {})
    organ_dim = checkpoint.get('organ_dim', 5)
    n_experts = checkpoint.get('n_experts', 8)
    class_to_idx = checkpoint.get('class_to_idx', {})

    vit_name = args_ckpt.get('vit_name', 'vit_base_patch16_224')
    dino_model_name = args_ckpt.get('dino_model_name', "dinov2_vitb14")
    segformer_model_name = args_ckpt.get('segformer_model_name', "nvidia/segformer-b1-finetuned-ade-512-512")
    d_ff_expert = args_ckpt.get('d_ff_expert', 1024)
    top_k = args_ckpt.get('top_k', 1)
    dropout = args_ckpt.get('dropout', 0.1)
    n_classes = len(class_to_idx)

    # Create model
    model = HybridPlantModel(
        n_classes=n_classes,
        organ_dim=organ_dim,
        n_experts=n_experts,
        d_ff_expert=d_ff_expert,
        vit_name=vit_name,
        dino_model_name=dino_model_name,
        segformer_model_name=segformer_model_name,
        dropout=dropout,
        top_k=top_k
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Extract KMeans centers if available
    kmeans_centers = checkpoint.get('kmeans_centers', None)

    print(f"[INFO] Hybrid model loaded successfully")
    print(f"  - Classes: {n_classes}")
    print(f"  - Organ dim (clusters): {organ_dim}")
    print(f"  - Experts: {n_experts}")
    print(f"  - Branches: ViT ({vit_name}), DINOv2 ({dino_model_name}), SegFormer ({segformer_model_name})")
    print(f"  - Best Macro F1: {checkpoint.get('macro_f1', 'N/A')}")

    return model, class_to_idx, organ_dim, kmeans_centers


def detect_unselected_images(model, unselected_csv, class_to_idx, organ_dim, kmeans_centers, device, out_dir):
    """Detect classes for unselected images using hybrid model and pseudo-organ priors"""
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

    # Initialize feature extractor for pseudo-organ assignment
    feat_extractor = None
    if kmeans_centers is not None:
        print("[INFO] Initializing feature extractor for pseudo-organ assignment...")
        feat_extractor = ResNet18FeatureExtractor().to(device)
        feat_extractor.eval()
        kmeans_centers_torch = torch.from_numpy(kmeans_centers).to(device) # (N_clusters, Feat_dim)

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

    print("[INFO] Starting inference...")
    model.eval()

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detecting"):
            img_path = row['image_path']
            true_class = row['class_name']

            if not os.path.exists(img_path):
                continue

            try:
                # 1. Preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # 2. Compute organ prior
                if feat_extractor is not None:
                    feat = feat_extractor(img_tensor) # (1, Feat_dim)
                    # Compute distance to all centers
                    dist = torch.cdist(feat, kmeans_centers_torch) # (1, N_clusters)
                    cluster_idx = dist.argmin(dim=-1).item()
                    organ_prior = torch.zeros(1, organ_dim, device=device)
                    organ_prior[0, cluster_idx] = 1.0
                else:
                    # Fallback to uniform prior
                    organ_prior = torch.ones(1, organ_dim, device=device) / float(organ_dim)

                # 3. Hybrid Inference
                start_time = time.time()
                out = model(img_tensor, organ_prior, training=False)
                logits = out["logits"]
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

    if feat_extractor is not None:
        del feat_extractor

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
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'k-', label='Total Loss', linewidth=2, alpha=0.5)
    if 'train_aux_loss' in history:
        ax.plot(epochs, history['train_aux_loss'], 'r-', label='Aux Loss', linewidth=1.5)
    if 'train_balance_loss' in history:
        ax.plot(epochs, history['train_balance_loss'], 'g-', label='Balance Loss', linewidth=1.5)
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

    # 6. Learning Rate schedule
    ax = axes[1, 2]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.set_title('Learning Rate Schedule')
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
    if 'train_loss' in history:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['train_loss'], 'k-', label='Total Loss', linewidth=2)
        if 'train_aux_loss' in history:
            plt.plot(epochs, history['train_aux_loss'], 'r--', label='Aux Loss', linewidth=1.5)
        if 'train_balance_loss' in history:
            plt.plot(epochs, history['train_balance_loss'], 'g--', label='Balance Loss', linewidth=1.5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Breakdown')
        plt.legend()
        plt.grid(alpha=0.3)
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
    model, class_to_idx, organ_dim, kmeans_centers = load_model(args.model_path, device)

    # Detect unselected images
    if args.unselected_csv and os.path.exists(args.unselected_csv):
        detect_unselected_images(
            model,
            args.unselected_csv,
            class_to_idx,
            organ_dim,
            kmeans_centers,
            device,
            args.out_dir
        )
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
