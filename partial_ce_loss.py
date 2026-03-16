#!/usr/bin/env python3
"""
Partial Cross Entropy Loss for Weakly Supervised Segmentation

This script implements a partial Cross Entropy loss for training segmentation models
with point annotations instead of full pixel-level masks.

Usage:
    python partial_ce_loss.py --experiment demo --epochs 20
    python partial_ce_loss.py --experiment density --epochs 30
    python partial_ce_loss.py --experiment comparison --epochs 30
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import copy
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# PARTIAL CROSS ENTROPY LOSS
# ============================================================================

class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross Entropy Loss for weakly supervised segmentation.

    Only computes loss on labeled pixels (where label_mask is True).
    Unlabeled pixels are ignored in the loss computation.

    Args:
        ignore_index: Index to ignore in the target (default: -1)
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing parameter
    """
    def __init__(self, ignore_index=-1, reduction='mean', label_smoothing=0.0):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, label_mask=None):
        """
        Args:
            pred: (B, C, H, W) - raw logits from the model
            target: (B, H, W) - ground truth labels (with -1 for unlabeled)
            label_mask: (B, H, W) - boolean mask indicating labeled pixels
                     If None, derived from target != ignore_index
        """
        B, C, H, W = pred.shape

        # Derive label_mask from target if not provided
        if label_mask is None:
            label_mask = (target != self.ignore_index)

        # Flatten tensors
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target_flat = target.view(-1)
        label_mask_flat = label_mask.view(-1)

        # Only consider labeled pixels
        labeled_indices = label_mask_flat.nonzero(as_tuple=True)[0]

        if len(labeled_indices) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_labeled = pred_flat[labeled_indices]
        target_labeled = target_flat[labeled_indices]

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            n_classes = pred.size(1)
            one_hot = F.one_hot(target_labeled, num_classes=n_classes).float()
            smoothed_labels = (1 - self.label_smoothing) * one_hot + \
                              self.label_smoothing / n_classes
            log_prob = F.log_softmax(pred_labeled, dim=-1)
            loss = -(smoothed_labels * log_prob).sum(dim=-1)
        else:
            loss = F.cross_entropy(pred_labeled, target_labeled, reduction='none')

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# POINT LABEL GENERATION
# ============================================================================

def generate_point_labels(mask, num_points_per_class=5, strategy='random'):
    """
    Generate point annotations from a full segmentation mask.

    Args:
        mask: (H, W) - full segmentation mask
        num_points_per_class: Number of points to sample per class
        strategy: 'random', 'grid', 'boundary'

    Returns:
        point_mask: (H, W) - mask with -1 for unlabeled, class label for labeled
    """
    H, W = mask.shape
    point_mask = torch.full((H, W), -1, dtype=torch.long)
    unique_classes = torch.unique(mask)
    unique_classes = unique_classes[unique_classes >= 0]

    for cls in unique_classes:
        cls_mask = (mask == cls)
        cls_pixels = torch.nonzero(cls_mask, as_tuple=False)

        if len(cls_pixels) == 0:
            continue

        if strategy == 'random':
            num_samples = min(num_points_per_class, len(cls_pixels))
            indices = torch.randperm(len(cls_pixels))[:num_samples]
            selected = cls_pixels[indices]
        elif strategy == 'boundary':
            cls_mask_np = cls_mask.cpu().numpy()
            eroded = binary_erosion(cls_mask_np)
            boundary = cls_mask_np & ~eroded
            boundary_pixels = torch.nonzero(torch.from_numpy(boundary), as_tuple=False)
            if len(boundary_pixels) > 0:
                num_samples = min(num_points_per_class, len(boundary_pixels))
                indices = torch.randperm(len(boundary_pixels))[:num_samples]
                selected = boundary_pixels[indices]
            else:
                selected = cls_pixels[:1]
        else:
            selected = cls_pixels[:min(num_points_per_class, len(cls_pixels))]

        for r, c in selected:
            point_mask[r, c] = cls.item()

    return point_mask


# ============================================================================
# SEGMENTATION MODEL (Lightweight U-Net)
# ============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetLite(nn.Module):
    """Lightweight U-Net for segmentation."""
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)

        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================================
# SYNTHETIC REMOTE SENSING DATASET
# ============================================================================

class SyntheticRemoteSensingDataset(Dataset):
    """Synthetic remote sensing dataset for segmentation."""
    def __init__(self, num_samples=500, img_size=256, num_classes=5, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

        self.class_colors = {
            0: [34, 139, 34],    # Forest - green
            1: [210, 180, 140],  # Urban - tan
            2: [65, 105, 225],   # Water - blue
            3: [240, 230, 140],  # Agricultural - yellow
            4: [139, 69, 19],    # Bare soil - brown
        }

    def __len__(self):
        return self.num_samples

    def generate_sample(self, idx):
        mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)

        num_seeds_per_class = random.randint(2, 5)
        seeds = []
        for cls in range(self.num_classes):
            for _ in range(num_seeds_per_class):
                seeds.append({
                    'class': cls,
                    'x': random.randint(0, self.img_size - 1),
                    'y': random.randint(0, self.img_size - 1)
                })

        seed_coords = np.array([[s['x'], s['y']] for s in seeds])
        seed_classes = np.array([s['class'] for s in seeds])

        y_coords, x_coords = np.meshgrid(np.arange(self.img_size), np.arange(self.img_size))
        pixels = np.column_stack([x_coords.ravel(), y_coords.ravel()])

        tree = cKDTree(seed_coords)
        _, indices = tree.query(pixels)
        mask = torch.from_numpy(seed_classes[indices].reshape(self.img_size, self.img_size)).long()

        # Add smoothness
        for _ in range(2):
            new_mask = mask.clone()
            for i in range(self.img_size):
                for j in range(self.img_size):
                    if random.random() < 0.1:
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < self.img_size and 0 <= nj < self.img_size:
                                    neighbors.append(mask[ni, nj].item())
                        if neighbors:
                            new_mask[i, j] = max(set(neighbors), key=neighbors.count)
            mask = new_mask

        # Generate image from mask
        image = torch.zeros(self.img_size, self.img_size, 3)
        for cls, color in self.class_colors.items():
            for c in range(3):
                base_color = color[c] / 255.0
                noise = torch.randn(self.img_size, self.img_size) * 0.05
                image[:, :, c] += (mask == cls).float() * (base_color + noise)

        return image.clamp(0, 1), mask

    def __getitem__(self, idx):
        image, mask = self.generate_sample(idx)
        if self.transform:
            image = self.transform(image)
        return image, mask


class PointLabelDataset(Dataset):
    """Wrapper that converts full masks to point labels."""
    def __init__(self, base_dataset, num_points_per_class=5, point_strategy='random'):
        self.base_dataset = base_dataset
        self.num_points_per_class = num_points_per_class
        self.point_strategy = point_strategy

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, full_mask = self.base_dataset[idx]
        point_mask = generate_point_labels(
            full_mask.squeeze(0) if full_mask.dim() > 2 else full_mask,
            num_points_per_class=self.num_points_per_class,
            strategy=self.point_strategy
        )
        return image, point_mask, full_mask


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, use_point_labels=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        if use_point_labels:
            images, point_masks, _ = batch
        else:
            images, point_masks = batch

        images = images.to(device)
        masks = point_masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if use_point_labels:
            label_mask = (masks != -1)
            loss = criterion(outputs, masks, label_mask)
        else:
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    correct = 0
    total = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, _, full_masks = batch
        else:
            images, full_masks = batch

        images = images.to(device)
        full_masks = full_masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, full_masks)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == full_masks).sum().item()
        total += full_masks.numel()

        for cls in range(num_classes):
            pred_cls = (preds == cls)
            mask_cls = (full_masks == cls)
            intersection[cls] += (pred_cls & mask_cls).sum().item()
            union[cls] += (pred_cls | mask_cls).sum().item()

    avg_loss = total_loss / len(dataloader)
    pixel_accuracy = correct / total
    iou_per_class = intersection / (union + 1e-8)
    miou = iou_per_class.mean().item()

    return {
        'loss': avg_loss,
        'pixel_accuracy': pixel_accuracy,
        'miou': miou,
        'iou_per_class': iou_per_class.cpu().numpy()
    }


def train_model(model, train_loader, val_loader, num_epochs, lr, device,
                use_point_labels=True, num_classes=5):
    """Train a model with partial CE loss."""
    if use_point_labels:
        criterion = PartialCrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_acc': []}
    best_miou = 0.0
    best_model = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, use_point_labels
        )

        val_metrics = validate(model, val_loader, nn.CrossEntropyLoss(), device, num_classes)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_miou'].append(val_metrics['miou'])
        history['val_acc'].append(val_metrics['pixel_accuracy'])

        scheduler.step()

        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            best_model = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val mIoU: {val_metrics['miou']:.4f}, "
              f"Val Acc: {val_metrics['pixel_accuracy']:.4f}")

    if best_model is not None:
        model.load_state_dict(best_model)

    return history, best_miou


def create_datasets(num_train=400, num_val=100, img_size=256):
    """Create train and validation datasets."""
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = SyntheticRemoteSensingDataset(
        num_samples=num_train, img_size=img_size, transform=transform
    )
    val_dataset = SyntheticRemoteSensingDataset(
        num_samples=num_val, img_size=img_size, transform=transform
    )

    return train_dataset, val_dataset


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_point_density(num_epochs=20):
    """
    Experiment: How does the number of point annotations per class affect performance?

    Hypothesis: More point annotations should improve performance, but with diminishing returns.
    """
    print("="*60)
    print("EXPERIMENT 1: Effect of Point Annotation Density")
    print("="*60)

    points_per_class_list = [1, 3, 5, 10, 20]
    batch_size = 16
    lr = 0.001
    img_size = 128
    num_classes = 5

    results = []

    for num_points in points_per_class_list:
        print(f"\n--- Training with {num_points} points per class ---")

        train_base, val_base = create_datasets(num_train=200, num_val=50, img_size=img_size)

        train_dataset = PointLabelDataset(
            train_base, num_points_per_class=num_points, point_strategy='random'
        )
        val_dataset = PointLabelDataset(
            val_base, num_points_per_class=num_points, point_strategy='random'
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = UNetLite(in_channels=3, num_classes=num_classes).to(device)

        history, best_miou = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, lr=lr, device=device,
            use_point_labels=True, num_classes=num_classes
        )

        val_metrics = validate(model, val_loader, nn.CrossEntropyLoss(), device, num_classes)

        results.append({
            'num_points': num_points,
            'miou': val_metrics['miou'],
            'accuracy': val_metrics['pixel_accuracy'],
            'history': history
        })

        print(f"Final mIoU: {val_metrics['miou']:.4f}, Accuracy: {val_metrics['pixel_accuracy']:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Points/Class | mIoU    | Accuracy")
    print("-" * 40)
    for r in results:
        print(f"{r['num_points']:12} | {r['miou']:.4f} | {r['accuracy']:.4f}")
    print("="*60)

    return results


def experiment_supervision_comparison(num_epochs=30):
    """
    Experiment: Compare partial CE loss (point supervision) vs. full supervision.
    """
    print("="*60)
    print("EXPERIMENT: Point vs. Full Supervision Comparison")
    print("="*60)

    batch_size = 16
    lr = 0.001
    img_size = 128
    num_classes = 5
    num_points = 5

    results = {}
    train_base, val_base = create_datasets(num_train=200, num_val=50, img_size=img_size)

    # Point Supervision
    print("\n--- Point Supervision (5 points per class) ---")
    train_point = PointLabelDataset(train_base, num_points_per_class=num_points)
    val_point = PointLabelDataset(val_base, num_points_per_class=num_points)

    train_loader_p = DataLoader(train_point, batch_size=batch_size, shuffle=True)
    val_loader_p = DataLoader(val_point, batch_size=batch_size, shuffle=False)

    model_point = UNetLite(in_channels=3, num_classes=num_classes).to(device)

    history_point, _ = train_model(
        model_point, train_loader_p, val_loader_p,
        num_epochs=num_epochs, lr=lr, device=device,
        use_point_labels=True, num_classes=num_classes
    )

    metrics_point = validate(model_point, val_loader_p, nn.CrossEntropyLoss(), device, num_classes)
    results['point'] = {'miou': metrics_point['miou'], 'accuracy': metrics_point['pixel_accuracy']}

    print(f"Point Supervision - mIoU: {metrics_point['miou']:.4f}")

    # Full Supervision
    print("\n--- Full Supervision ---")
    train_loader_f = DataLoader(train_base, batch_size=batch_size, shuffle=True)
    val_loader_f = DataLoader(val_base, batch_size=batch_size, shuffle=False)

    model_full = UNetLite(in_channels=3, num_classes=num_classes).to(device)

    history_full, _ = train_model(
        model_full, train_loader_f, val_loader_f,
        num_epochs=num_epochs, lr=lr, device=device,
        use_point_labels=False, num_classes=num_classes
    )

    metrics_full = validate(model_full, val_loader_f, nn.CrossEntropyLoss(), device, num_classes)
    results['full'] = {'miou': metrics_full['miou'], 'accuracy': metrics_full['pixel_accuracy']}

    print(f"Full Supervision - mIoU: {metrics_full['miou']:.4f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Point Supervision mIoU: {results['point']['miou']:.4f}")
    print(f"Full Supervision mIoU:  {results['full']['miou']:.4f}")
    print(f"Performance Gap: {(results['full']['miou'] - results['point']['miou']):.4f}")
    print("="*60)

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train segmentation with Partial CE Loss')
    parser.add_argument('--experiment', type=str, default='demo',
                        choices=['demo', 'density', 'comparison'],
                        help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--img-size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes')

    args = parser.parse_args()

    print(f"Using device: {device}")

    if args.experiment == 'demo':
        print("\nRunning quick demo...")
        train_base, val_base = create_datasets(num_train=100, num_val=20, img_size=args.img_size)

        train_dataset = PointLabelDataset(train_base, num_points_per_class=5)
        val_dataset = PointLabelDataset(val_base, num_points_per_class=5)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = UNetLite(in_channels=3, num_classes=args.num_classes).to(device)

        history, best_miou = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs, lr=args.lr, device=device,
            use_point_labels=True, num_classes=args.num_classes
        )

        val_metrics = validate(model, val_loader, nn.CrossEntropyLoss(), device, args.num_classes)

        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Validation Loss:   {val_metrics['loss']:.4f}")
        print(f"Pixel Accuracy:    {val_metrics['pixel_accuracy']:.4f}")
        print(f"Mean IoU:          {val_metrics['miou']:.4f}")
        print("="*60)

    elif args.experiment == 'density':
        results = experiment_point_density(num_epochs=args.epochs)

    elif args.experiment == 'comparison':
        results = experiment_supervision_comparison(num_epochs=args.epochs)
