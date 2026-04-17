#!/usr/bin/env python3
"""
Embedding-based training script for ClayTerratorch.

Trains a segmentation head on CLAY/Terratorch embeddings with LULC masks.
Uses Decadal LULC India (1985/1995/2005) class legend.
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lulc_legend import LULC_CLASS_MAP, LULC_VALID_CLASS_IDS, class_colors


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


class EmbeddingDataset(Dataset):
    """Dataset for loading embeddings and masks."""
    def __init__(self, npz_files):
        self.files = sorted(npz_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            embeddings = data['embeddings'].astype(np.float32)  # (257, 1024) or (256, 1024)
            mask = data['mask'].astype(np.int64)               # (H, W)

        # Convert CLAY embeddings to spatial format expected by decoder
        # CLAY returns: [CLS_token, patch1, patch2, ..., patch256] where patches are 16x16 grid
        if embeddings.shape[0] == 257:  # With CLS token
            patch_embeddings = embeddings[1:]  # Remove CLS token
        else:
            patch_embeddings = embeddings  # Assume already patch-only

        # Reshape from (256, 1024) to (16, 16, 1024) then to (1024, 16, 16)
        spatial_embeddings = patch_embeddings.reshape(16, 16, 1024)
        spatial_embeddings = spatial_embeddings.permute(2, 0, 1)  # (1024, 16, 16)

        return {
            "embeddings": spatial_embeddings,  # (1024, 16, 16)
            "mask": mask                       # (H, W) - will be resized to match
        }


class SimpleUNetHead(nn.Module):
    """Simple UNet-style decoder head for embedding-to-segmentation."""
    def __init__(self, in_channels=1024, num_classes=20):
        super().__init__()

        # Encoder blocks (process the embeddings)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder blocks
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, num_classes, 3, stride=2, padding=1, output_padding=1),
        )

        # Skip connections
        self.skip1 = nn.Conv2d(512, 512, 1)
        self.skip2 = nn.Conv2d(256, 256, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder with skip connections
        d1 = self.dec1(e3)
        d1 = d1 + self.skip2(e2)  # Skip connection

        d2 = self.dec2(d1)
        d2 = d2 + self.skip1(e1)  # Skip connection

        d3 = self.dec3(d2)

        return d3


class EmbeddingSegmentationModule(pl.LightningModule):
    def __init__(self, in_channels=1024, num_classes=20, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = SimpleUNetHead(in_channels=in_channels, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Metrics
        self.train_confmat = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=0)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        embeddings = batch["embeddings"]
        masks = batch["mask"]

        # Resize mask to match embedding spatial dimensions if needed
        if masks.shape[-2:] != embeddings.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks.float().unsqueeze(1),
                size=embeddings.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        logits = self(embeddings)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)
        self.train_confmat.update(preds, masks)

        # Calculate accuracy
        acc = (preds == masks).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Create visualization every 50 batches to avoid overhead
        if batch_idx % 50 == 0:
            self.create_visualization(embeddings, masks, logits, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = batch["embeddings"]
        masks = batch["mask"]

        # Resize mask to match embedding spatial dimensions if needed
        if masks.shape[-2:] != embeddings.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks.float().unsqueeze(1),
                size=embeddings.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        logits = self(embeddings)
        loss = self.criterion(logits, masks)

        preds = torch.argmax(logits, dim=1)
        self.val_confmat.update(preds, masks)

        # Calculate accuracy
        acc = (preds == masks).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Create visualization every 50 batches to avoid overhead
        if batch_idx % 50 == 0:
            self.create_visualization(embeddings, masks, logits, "val")

        return loss

    def on_validation_epoch_end(self):
        val_miou, iou_per_class = self._miou_from_confmat(self.val_confmat.compute())
        self.log("val_mIoU", val_miou, prog_bar=True)

        # Log per-class IoU
        for i, iou in enumerate(iou_per_class):
            if not torch.isnan(iou):
                class_name = LULC_CLASS_MAP.get(i, ("Unknown", "Unknown"))[1]
                self.log(f"val_iou_class_{i}_{class_name.replace(' ', '_')}", iou, prog_bar=False)

        self.val_confmat.reset()

        # Print epoch summary
        if self.trainer and self.trainer.is_global_zero:
            print(f"\nEpoch {self.current_epoch}: Val mIoU: {val_miou:.4f}")

            # Print per-class IoU for non-zero classes
            print("Per-class IoU:")
            for i, iou in enumerate(iou_per_class):
                if not torch.isnan(iou) and i != 0:  # Skip background class
                    class_name_i, class_name_ii = LULC_CLASS_MAP.get(i, ("Unknown", "Unknown"))
                    print(f"  Class {i:2d} ({class_name_i}: {class_name_ii}): {iou:.4f}")

        # Save final visualization and metadata at end of training
        if self.current_epoch == self.trainer.max_epochs - 1:  # Last epoch
            self.save_final_visualizations()

    def save_final_visualizations(self):
        """Save final visualizations and metadata after training completes."""
        try:
            # Create output directories
            viz_dir = Path(self.hparams.output_dir) / "final_visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Save class legend information
            legend_path = viz_dir / "lulc_legend.txt"
            with open(legend_path, 'w') as f:
                f.write("Decadal LULC India Class Legend (1985/1995/2005)\n")
                f.write("=" * 50 + "\n")
                f.write("Format: Class ID -> (Level-I Class, Level-II Class)\n\n")
                for class_id in sorted(LULC_CLASS_MAP.keys()):
                    level_i, level_ii = LULC_CLASS_MAP[class_id]
                    f.write(f"{class_id:2d}: ({level_i}, {level_ii})\n")

                f.write("\nClass Colors (hex):\n")
                for i, color in enumerate(class_colors):
                    if i in LULC_CLASS_MAP:
                        level_i, level_ii = LULC_CLASS_MAP[i]
                        f.write(f"{i:2d}: {color} - {level_ii}\n")

            # Create and save classification report-style summary
            report_path = viz_dir / "classification_info.txt"
            with open(report_path, 'w') as f:
                f.write("Model Training Summary\n")
                f.write("=" * 30 + "\n")
                f.write(f"Training completed at epoch: {self.current_epoch + 1}\n")
                f.write(f"Number of classes: {self.hparams.num_classes}\n")
                f.write(f"Ignore index (background): {self.hparams.ignore_index}\n")
                f.write(f"Embedding dimension: {self.hparams.in_channels}\n")
                f.write("\nClass Information:\n")
                for class_id in sorted(LULC_CLASS_MAP.keys()):
                    if class_id != self.hparams.ignore_index:  # Skip background
                        level_i, level_ii = LULC_CLASS_MAP[class_id]
                        f.write(f"Class {class_id:2d}: {level_i} - {level_ii}\n")
                        f.write(f"  Color: {class_colors[class_id]}\n")

            print(f"✅ Final visualizations and metadata saved to: {viz_dir}")

        except Exception as e:
            print(f"Warning: Could not save final visualizations: {e}")

    def _miou_from_confmat(self, confmat):
        confmat = confmat.to(torch.float64)
        tp = confmat.diag()
        fp = confmat.sum(dim=0) - tp
        fn = confmat.sum(dim=1) - tp
        denom = tp + fp + fn
        iou = torch.full_like(tp, float("nan"))
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        iou[valid] = torch.clamp(iou[valid], min=0.0, max=1.0)

        if 0 <= self.hparams.num_classes < len(iou):
            iou[self.hparams.num_classes] = float("nan")

        valid_iou = iou[torch.isfinite(iou)]
        if valid_iou.numel() == 0:
            return torch.tensor(0.0, device=confmat.device), iou
        return valid_iou.mean(), iou

    def create_visualization(self, embeddings, masks, logits, prefix="val"):
        """Create visualization plots for training monitoring."""
        try:
            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Convert to numpy for plotting
            embeddings_np = embeddings.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()

            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{prefix.capitalize()} Batch Visualization', fontsize=16)

            # Plot 1: Embedding visualization (show first channel)
            if embeddings_np.shape[1] > 0:
                embed_plot = axes[0, 0].imshow(embeddings_np[0, 0], cmap='viridis')
                axes[0, 0].set_title('Embedding Channel 0')
                plt.colorbar(embed_plot, ax=axes[0, 0])

            # Plot 2: Ground truth mask
            mask_plot = axes[0, 1].imshow(masks_np[0], cmap=plt.cm.tab20, vmin=0, vmax=19)
            axes[0, 1].set_title('Ground Truth Mask')
            plt.colorbar(mask_plot, ax=axes[0, 1], ticks=range(20))

            # Plot 3: Prediction mask
            pred_plot = axes[0, 2].imshow(preds_np[0], cmap=plt.cm.tab20, vmin=0, vmax=19)
            axes[0, 2].set_title('Prediction Mask')
            plt.colorbar(pred_plot, ax=axes[0, 2], ticks=range(20))

            # Plot 4: Overlay comparison
            overlay = axes[1, 0].imshow(masks_np[0], cmap=plt.cm.tab20, alpha=0.7, vmin=0, vmax=19)
            axes[1, 0].imshow(preds_np[0], cmap=plt.cm.tab20, alpha=0.7, vmin=0, vmax=19)
            axes[1, 0].set_title('GT vs Prediction Overlay')

            # Plot 5: Class prediction distribution
            unique, counts = np.unique(preds_np[0], return_counts=True)
            axes[1, 1].bar(unique, counts)
            axes[1, 1].set_title('Prediction Class Distribution')
            axes[1, 1].set_xlabel('Class ID')
            axes[1, 1].set_ylabel('Pixel Count')

            # Plot 6: Confusion matrix heatmap (sample)
            if hasattr(self, 'val_confmat') and self.val_confmat is not None:
                confmat = self.val_confmat.compute()
                if confmat is not None:
                    confmat_np = confmat.detach().cpu().numpy()
                    # Show only first 10 classes for clarity
                    confmat_subset = confmat_np[:10, :10]
                    im = axes[1, 2].imshow(confmat_subset, cmap='Blues')
                    axes[1, 2].set_title('Confusion Matrix (First 10 Classes)')
                    axes[1, 2].set_xlabel('Predicted')
                    axes[1, 2].set_ylabel('True')
                    plt.colorbar(im, ax=axes[1, 2])

            plt.tight_layout()

            # Save plot
            plot_dir = Path(self.hparams.output_dir) / "visualizations"
            plot_dir.mkdir(exist_ok=True)
            plot_path = plot_dir / f"{prefix}_epoch_{self.current_epoch}_batch_{plt.gcf().number}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            # Don't let visualization errors break training
            print(f"Warning: Visualization creation failed: {e}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mIoU"}
        }


def train_embedding_head(args):
    """Main training function."""
    # Set up paths
    project_root = Path(__file__).resolve().parents[1]

    # Data directories
    embedding_dir = _resolve(project_root, args.embedding_dir)

    # Collect all embedding files
    embedding_files = []
    for year_dir in embedding_dir.iterdir():
        if year_dir.is_dir():
            embedding_files.extend(list(year_dir.glob("emb_*.npz")))

    if not embedding_files:
        raise ValueError(f"No embedding files found in {embedding_dir}")

    print(f"📊 Found {len(embedding_files)} embedding files for training")

    # Split into train/val
    np.random.seed(42)
    np.random.shuffle(embedding_files)
    split_idx = int(0.8 * len(embedding_files))
    train_files = embedding_files[:split_idx]
    val_files = embedding_files[split_idx:]

    print(f"📈 Training set: {len(train_files)} files")
    print(f"📊 Validation set: {len(val_files)} files")

    # Create datasets
    train_dataset = EmbeddingDataset(train_files)
    val_dataset = EmbeddingDataset(val_files)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = EmbeddingSegmentationModule(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mIoU",
        mode="max",
        save_top_k=3,
        filename="embedding-head-best-{epoch:02d}-{val_mIoU:.4f}",
        save_last=True
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_mIoU",
        mode="max",
        patience=args.patience
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=args.output_dir,
        log_every_n_steps=10
    )

    # Train
    print("🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"🎉 Training complete! Best model saved to: {checkpoint_callback.best_model_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train segmentation head on embeddings.")
    p.add_argument("--embedding-dir", type=str, required=True,
                   help="Directory containing extracted embeddings (organized by year)")
    p.add_argument("--output-dir", type=str, default="outputs/embedding_training",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--in-channels", type=int, default=1024,
                   help="Number of input channels (embedding dimension)")
    p.add_argument("--num-classes", type=int, default=20,
                   help="Number of LULC classes")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for training")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of data loading workers")
    p.add_argument("--max-epochs", type=int, default=50,
                   help="Maximum number of training epochs")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    exit(train_embedding_head(args))