import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ──────────────────────────────────────────────
# DINOv2 Classifier (same as dinov2_kfold.py)
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
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
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
        return self.classifier(features)


# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
def load_fold_model(fold_dir, device):
    """Load DINOv2 model from a specific fold directory."""
    model_path = os.path.join(fold_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"[WARNING] Model not found: {model_path}")
        return None, None, None

    print(f"[INFO] Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[WARN] Failed with weights_only=False: {e}")
        checkpoint = torch.load(model_path, map_location=device)

    class_to_idx = checkpoint.get('class_to_idx', {})
    model_name = checkpoint.get('model_name', 'dinov2_vitb14')
    freeze_backbone = checkpoint.get('freeze_backbone', True)
    dropout = checkpoint.get('dropout', 0.1)
    n_classes = checkpoint.get('n_classes', len(class_to_idx))

    model = DINOv2Classifier(
        model_name=model_name, n_classes=n_classes,
        pretrained=True, freeze_backbone=freeze_backbone, dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    best_epoch = checkpoint.get('epoch', 'N/A')
    best_f1 = checkpoint.get('macro_f1', 'N/A')
    print(f"  - Model: {model_name}, Best epoch: {best_epoch}, Macro F1: {best_f1}")

    return model, class_to_idx, checkpoint


# ──────────────────────────────────────────────
# Analyze K-Fold Results
# ──────────────────────────────────────────────
def analyze_kfold_results(kfold_dir, n_folds, out_dir):
    """Analyze aggregated K-Fold results."""
    print("\n[ANALYSIS] Analyzing K-Fold results...")

    results_path = os.path.join(kfold_dir, 'kfold_results.csv')
    if not os.path.exists(results_path):
        print(f"[WARNING] Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    print(f"\n[INFO] Loaded results for {len(df)} folds")
    print("\nPer-Fold Summary:")
    print(df.to_string(index=False))

    print("\n[STATISTICS]")
    print(f"  Mean Test Macro F1:    {df['test_macro_f1'].mean():.4f} ± {df['test_macro_f1'].std():.4f}")
    print(f"  Mean Test Micro F1:    {df['test_micro_f1'].mean():.4f} ± {df['test_micro_f1'].std():.4f}")
    print(f"  Mean Test Weighted F1: {df['test_weighted_f1'].mean():.4f} ± {df['test_weighted_f1'].std():.4f}")
    print(f"  Mean Test Accuracy:    {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
    print(f"  Best Fold: {df.loc[df['test_macro_f1'].idxmax(), 'fold']} "
          f"(Macro F1: {df['test_macro_f1'].max():.4f})")
    print(f"  Worst Fold: {df.loc[df['test_macro_f1'].idxmin(), 'fold']} "
          f"(Macro F1: {df['test_macro_f1'].min():.4f})")

    # Visualize metrics across folds
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DINOv2 K-Fold Cross-Validation Results', fontsize=16, fontweight='bold')

    # F1 Scores comparison
    ax = axes[0, 0]
    x = df['fold']
    width = 0.25
    ax.bar(x - width, df['test_macro_f1'], width, label='Macro F1', alpha=0.8)
    ax.bar(x, df['test_micro_f1'], width, label='Micro F1', alpha=0.8)
    ax.bar(x + width, df['test_weighted_f1'], width, label='Weighted F1', alpha=0.8)
    ax.axhline(df['test_macro_f1'].mean(), color='r', linestyle='--',
               label=f'Mean Macro: {df["test_macro_f1"].mean():.4f}')
    ax.set_xlabel('Fold'); ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores per Fold'); ax.set_xticks(x); ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Accuracy comparison
    ax = axes[0, 1]
    ax.bar(df['fold'], df['test_accuracy'], alpha=0.7, color='green')
    ax.axhline(df['test_accuracy'].mean(), color='r', linestyle='--',
               label=f'Mean: {df["test_accuracy"].mean():.4f}')
    ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy per Fold'); ax.set_xticks(df['fold']); ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Best epoch comparison
    ax = axes[1, 0]
    ax.bar(df['fold'], df['best_epoch'], alpha=0.7, color='purple')
    ax.axhline(df['best_epoch'].mean(), color='r', linestyle='--',
               label=f'Mean: {df["best_epoch"].mean():.1f}')
    ax.set_xlabel('Fold'); ax.set_ylabel('Best Epoch')
    ax.set_title('Convergence Speed'); ax.set_xticks(df['fold']); ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Training time
    if 'fold_time' in df.columns:
        ax = axes[1, 1]
        ax.bar(df['fold'], df['fold_time'] / 60, alpha=0.7, color='orange')
        ax.axhline((df['fold_time'] / 60).mean(), color='r', linestyle='--',
                   label=f'Mean: {(df["fold_time"] / 60).mean():.1f} min')
        ax.set_xlabel('Fold'); ax.set_ylabel('Time (minutes)')
        ax.set_title('Training Time per Fold'); ax.set_xticks(df['fold']); ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kfold_comparison.png'), dpi=150)
    plt.close()
    print(f"\n[INFO] Saved K-Fold comparison plot")

    # Box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bp = ax.boxplot([df['test_macro_f1']], labels=['Macro F1'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.scatter([1] * len(df), df['test_macro_f1'], alpha=0.6, color='red', s=100)
    for fold, val in zip(df['fold'], df['test_macro_f1']):
        ax.text(1.05, val, f'F{int(fold)}', fontsize=9)
    ax.set_ylabel('Macro F1'); ax.set_title('Macro F1 Distribution'); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    bp = ax.boxplot([df['test_accuracy']], labels=['Accuracy'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    ax.scatter([1] * len(df), df['test_accuracy'], alpha=0.6, color='red', s=100)
    for fold, val in zip(df['fold'], df['test_accuracy']):
        ax.text(1.05, val, f'F{int(fold)}', fontsize=9)
    ax.set_ylabel('Accuracy'); ax.set_title('Accuracy Distribution'); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kfold_distribution.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved distribution plot")


# ──────────────────────────────────────────────
# Compare Confusion Matrices
# ──────────────────────────────────────────────
def compare_confusion_matrices(kfold_dir, n_folds, out_dir):
    print("\n[CONFUSION MATRIX] Comparing across folds...")
    cms = []
    class_names = None
    for fold in range(1, n_folds + 1):
        cm_path = os.path.join(kfold_dir, f'fold_{fold}', 'confusion_matrix_test.csv')
        if os.path.exists(cm_path):
            cm_df = pd.read_csv(cm_path, index_col=0)
            cms.append(cm_df.values)
            if class_names is None:
                class_names = cm_df.index.tolist()
        else:
            print(f"[WARNING] CM not found for fold {fold}")

    if len(cms) == 0:
        print("[WARNING] No confusion matrices found"); return

    avg_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    pd.DataFrame(avg_cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, 'confusion_matrix_average.csv'))
    pd.DataFrame(std_cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, 'confusion_matrix_std.csv'))
    print(f"[INFO] Saved average and std confusion matrices")

    if len(class_names) <= 20:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Average CM'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
        sns.heatmap(std_cm, annot=True, fmt='.1f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Std Dev CM'); axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'confusion_matrix_comparison.png'), dpi=150)
        plt.close()


# ──────────────────────────────────────────────
# Per-Class Performance Analysis
# ──────────────────────────────────────────────
def analyze_per_class_performance(kfold_dir, n_folds, out_dir):
    print("\n[PER-CLASS] Analyzing per-class performance...")
    all_reports = []
    for fold in range(1, n_folds + 1):
        report_path = os.path.join(kfold_dir, f'fold_{fold}', 'classification_report_test.csv')
        if os.path.exists(report_path):
            df = pd.read_csv(report_path)
            df['fold'] = fold
            all_reports.append(df)

    if len(all_reports) == 0:
        print("[WARNING] No classification reports found"); return

    combined = pd.concat(all_reports, ignore_index=True)
    class_data = combined[~combined['class'].isin(['macro avg', 'weighted avg', 'accuracy'])]

    class_summary = class_data.groupby('class').agg({
        'precision': ['mean', 'std'], 'recall': ['mean', 'std'],
        'f1-score': ['mean', 'std'], 'support': 'mean'
    }).reset_index()
    class_summary.columns = ['class', 'precision_mean', 'precision_std',
                             'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'support_mean']
    class_summary = class_summary.sort_values('f1_mean', ascending=False)
    class_summary.to_csv(os.path.join(out_dir, 'per_class_summary.csv'), index=False)

    print("\n[TOP 5 BEST CLASSES]")
    print(class_summary.head(5)[['class', 'f1_mean', 'f1_std']].to_string(index=False))
    print("\n[TOP 5 WORST CLASSES]")
    print(class_summary.tail(5)[['class', 'f1_mean', 'f1_std']].to_string(index=False))

    top_n = min(10, len(class_summary))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    top = class_summary.head(top_n)
    ax.barh(range(top_n), top['f1_mean'], xerr=top['f1_std'], alpha=0.7, color='green')
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top['class'])
    ax.set_xlabel('F1 (mean ± std)'); ax.set_title(f'Top {top_n} Best Classes'); ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[1]
    bottom = class_summary.tail(top_n).iloc[::-1]
    ax.barh(range(top_n), bottom['f1_mean'], xerr=bottom['f1_std'], alpha=0.7, color='red')
    ax.set_yticks(range(top_n)); ax.set_yticklabels(bottom['class'])
    ax.set_xlabel('F1 (mean ± std)'); ax.set_title(f'Top {top_n} Worst Classes'); ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_performance.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved per-class performance plot")


# ──────────────────────────────────────────────
# Visualize Training Histories Across Folds
# ──────────────────────────────────────────────
def visualize_training_histories(kfold_dir, n_folds, out_dir):
    """Visualize and compare training histories across all folds."""
    print("\n[HISTORY] Visualizing training histories across folds...")
    histories = {}
    for fold in range(1, n_folds + 1):
        hist_path = os.path.join(kfold_dir, f'fold_{fold}', 'training_history.pt')
        if os.path.exists(hist_path):
            try:
                histories[fold] = torch.load(hist_path, map_location='cpu', weights_only=False)
            except:
                histories[fold] = torch.load(hist_path, map_location='cpu')

    if not histories:
        print("[WARNING] No training histories found"); return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DINOv2 Training History - All Folds', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # Train loss
    ax = axes[0, 0]
    for (fold, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], color=color, label=f'Fold {fold}', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss'); ax.legend(); ax.grid(alpha=0.3)

    # Train accuracy
    ax = axes[0, 1]
    for (fold, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist['train_acc']) + 1)
        ax.plot(epochs, hist['train_acc'], color=color, label=f'Fold {fold}', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Training Accuracy'); ax.legend(); ax.grid(alpha=0.3)

    # Val Macro F1
    ax = axes[1, 0]
    for (fold, hist), color in zip(histories.items(), colors):
        if 'val_macro_f1' in hist:
            epochs = range(1, len(hist['val_macro_f1']) + 1)
            ax.plot(epochs, hist['val_macro_f1'], color=color, label=f'Fold {fold}', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1'); ax.set_title('Val Macro F1'); ax.legend(); ax.grid(alpha=0.3)

    # Val Accuracy
    ax = axes[1, 1]
    for (fold, hist), color in zip(histories.items(), colors):
        if 'val_accuracy' in hist:
            epochs = range(1, len(hist['val_accuracy']) + 1)
            ax.plot(epochs, hist['val_accuracy'], color=color, label=f'Fold {fold}', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Val Accuracy'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history_all_folds.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved training history comparison plot")

    # Average curves with std band
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DINOv2 Average Training Curves (Mean ± Std)', fontsize=14, fontweight='bold')

    max_epochs = max(len(hist['train_loss']) for hist in histories.values())

    # Pad shorter histories
    def pad_histories(metric):
        all_vals = []
        for hist in histories.values():
            vals = hist.get(metric, [])
            if len(vals) < max_epochs:
                vals = vals + [vals[-1]] * (max_epochs - len(vals))
            all_vals.append(vals[:max_epochs])
        return np.array(all_vals)

    epochs = range(1, max_epochs + 1)

    # Loss
    ax = axes[0]
    loss_arr = pad_histories('train_loss')
    mean_loss = loss_arr.mean(axis=0)
    std_loss = loss_arr.std(axis=0)
    ax.plot(epochs, mean_loss, 'b-', linewidth=2, label='Mean')
    ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color='blue')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Average Training Loss'); ax.legend(); ax.grid(alpha=0.3)

    # Val Macro F1
    ax = axes[1]
    f1_arr = pad_histories('val_macro_f1')
    mean_f1 = f1_arr.mean(axis=0)
    std_f1 = f1_arr.std(axis=0)
    ax.plot(epochs, mean_f1, 'g-', linewidth=2, label='Mean')
    ax.fill_between(epochs, mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.2, color='green')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1'); ax.set_title('Average Val Macro F1'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history_average.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved average training curves plot")


# ──────────────────────────────────────────────
# Best Fold Details
# ──────────────────────────────────────────────
def visualize_best_fold_details(kfold_dir, n_folds, out_dir):
    print("\n[BEST FOLD] Analyzing best performing fold...")
    results_path = os.path.join(kfold_dir, 'kfold_results.csv')
    if not os.path.exists(results_path): return

    df = pd.read_csv(results_path)
    best_fold = int(df.loc[df['test_macro_f1'].idxmax(), 'fold'])
    best_f1 = df['test_macro_f1'].max()
    print(f"[INFO] Best fold: {best_fold} with Macro F1: {best_f1:.4f}")

    fold_dir = os.path.join(kfold_dir, f'fold_{best_fold}')

    # Normalized CM
    cm_path = os.path.join(fold_dir, 'confusion_matrix_test_normalized.csv')
    if os.path.exists(cm_path):
        cm_df = pd.read_csv(cm_path, index_col=0)
        if len(cm_df) <= 30:
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm_df.values, annot=False, cmap='Blues',
                       xticklabels=cm_df.columns, yticklabels=cm_df.index)
            plt.title(f'Best Fold ({best_fold}) - Normalized CM\nMacro F1: {best_f1:.4f}',
                     fontsize=14, fontweight='bold')
            plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'best_fold_{best_fold}_confusion_matrix.png'), dpi=150)
            plt.close()

    # Classification report
    report_path = os.path.join(fold_dir, 'classification_report_test.csv')
    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path)
        class_report = report_df[~report_df['class'].isin(['macro avg', 'weighted avg', 'accuracy'])]
        class_report = class_report.sort_values('f1-score', ascending=True)

        n = min(15, len(class_report))
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        ax = axes[0]
        top_classes = class_report.tail(n)
        ax.barh(range(n), top_classes['f1-score'], alpha=0.7, color='green')
        ax.set_yticks(range(n)); ax.set_yticklabels(top_classes['class'], fontsize=9)
        ax.set_xlabel('F1 Score'); ax.set_title(f'Best Fold ({best_fold}) - Top {n} Classes by F1')
        ax.grid(alpha=0.3, axis='x')

        ax = axes[1]
        ax.scatter(class_report['recall'], class_report['precision'],
                  s=class_report['support']*2, alpha=0.6, c=class_report['f1-score'], cmap='viridis')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'Best Fold ({best_fold}) - Precision vs Recall')
        ax.grid(alpha=0.3); ax.set_xlim([0, 1.05]); ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'best_fold_{best_fold}_details.png'), dpi=150)
        plt.close()
        print(f"[INFO] Saved best fold details")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(args):
    start_time = datetime.now()
    print("=" * 80)
    print(f"[START] DINOv2 K-Fold Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] K-Fold directory: {args.kfold_dir}")
    print(f"[INFO] Number of folds: {args.n_folds}")

    # 1. Analyze aggregated results
    analyze_kfold_results(args.kfold_dir, args.n_folds, args.out_dir)

    # 2. Compare confusion matrices
    compare_confusion_matrices(args.kfold_dir, args.n_folds, args.out_dir)

    # 3. Per-class analysis
    analyze_per_class_performance(args.kfold_dir, args.n_folds, args.out_dir)

    # 4. Visualize training histories
    visualize_training_histories(args.kfold_dir, args.n_folds, args.out_dir)

    # 5. Best fold details
    visualize_best_fold_details(args.kfold_dir, args.n_folds, args.out_dir)

    # 6. List models metadata
    print("\n[MODELS] Loading model metadata from each fold...")
    for fold in range(1, args.n_folds + 1):
        fold_dir = os.path.join(args.kfold_dir, f'fold_{fold}')
        model, class_to_idx, checkpoint = load_fold_model(fold_dir, device)
        if model is not None:
            print(f"  Fold {fold}: Classes={len(class_to_idx)}, Embed dim={model.embed_dim}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 80)
    print(f"[END] Analysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {duration:.2f} seconds")
    print(f"[INFO] Results saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze DINOv2 K-Fold Cross-Validation results')
    parser.add_argument('--kfold_dir', type=str, required=True,
                       help='Path to K-Fold output directory (e.g., ./outputs_kfold)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Output directory (default: kfold_dir/analysis)')

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(args.kfold_dir, 'analysis')

    main(args)
