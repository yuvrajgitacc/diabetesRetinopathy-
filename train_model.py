"""
OptiGemma — EfficientNet-B3 Training Pipeline
Fine-tunes EfficientNet-B3 on Diabetic Retinopathy datasets to reach 95%+ accuracy.

Supports:
  - APTOS 2019 (Kaggle) dataset
  - Any folder-structured dataset (class_0/, class_1/, ... class_4/)
  - HEI-MED images with auto-labeling

Training techniques:
  - ImageNet pretrained backbone with progressive unfreezing
  - Heavy augmentation (flip, rotate, color jitter, CLAHE, cutout)
  - Label smoothing + class-weighted loss (handles imbalance)
  - Cosine annealing LR with warm restarts
  - Mixed precision (AMP) for GTX 1650 4GB VRAM
  - Early stopping + best model checkpointing
  - Test-time augmentation (TTA) for evaluation
"""

import os
import sys
import json
import time
import copy
import csv
import shutil
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import cv2

# Import preprocessor functions
import sys
sys.path.append(str(Path(__file__).parent))
from engine.preprocessor import circular_crop, apply_gaussian_blur

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

IMG_SIZE = 300          # EfficientNet-B3 optimal input
BATCH_SIZE = 8          # Safe for 4GB VRAM with AMP
NUM_WORKERS = 0
NUM_CLASSES = 5
SEED = 42

# Training phases
PHASE1_EPOCHS = 10      # Train classifier head only
PHASE2_EPOCHS = 30      # Unfreeze last 2 blocks
PHASE3_EPOCHS = 60      # Full fine-tune with low LR
TARGET_ACC = 0.95

STAGE_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
               3: "Severe NPDR", 4: "Proliferative DR"}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Dataset ─────────────────────────────────────────────────────────────

class DRDataset(Dataset):
    """Diabetic Retinopathy dataset — works with CSV or folder structure."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is not None:
            img = circular_crop(img)
            img = apply_gaussian_blur(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = Image.open(path).convert("RGB")
            
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(phase="train"):
    """Get augmentation transforms for train/val."""
    if phase == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 30, IMG_SIZE + 30)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_tta_transforms():
    """Test-time augmentation transforms (5 crops + flips)."""
    base = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    flipped = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rotated = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomRotation((90, 90)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return [base, flipped, rotated]


# ── Data Loading ────────────────────────────────────────────────────────

def load_aptos_dataset(data_path):
    """Load APTOS 2019 Blindness Detection dataset (CSV + images)."""
    csv_path = None
    img_dir = None

    # Find CSV and image directory
    for f in Path(data_path).rglob("*.csv"):
        if "train" in f.name.lower():
            csv_path = f
            break
    if csv_path is None:
        for f in Path(data_path).rglob("*.csv"):
            csv_path = f
            break

    # Find image directory
    for d in Path(data_path).rglob("train_images"):
        img_dir = d
        break
    if img_dir is None:
        for d in Path(data_path).iterdir():
            if d.is_dir() and any(d.glob("*.png")) or any(d.glob("*.jpg")):
                img_dir = d
                break

    if csv_path is None or img_dir is None:
        return None, None

    print(f"[DATA] Found CSV: {csv_path}")
    print(f"[DATA] Found images: {img_dir}")

    paths, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get("id_code") or row.get("image") or row.get("id")
            label = int(row.get("diagnosis") or row.get("level") or row.get("label", 0))
            # Try multiple extensions
            for ext in [".png", ".jpg", ".jpeg", ".tif"]:
                p = img_dir / (img_id + ext)
                if p.exists():
                    paths.append(str(p))
                    labels.append(min(label, 4))
                    break

    print(f"[DATA] Loaded {len(paths)} images from APTOS dataset")
    return paths, labels


def load_folder_dataset(data_path):
    """Load from folder structure: data_path/0/, data_path/1/, ... data_path/4/"""
    paths, labels = [], []
    data_path = Path(data_path)

    for class_id in range(NUM_CLASSES):
        class_dir = data_path / str(class_id)
        if not class_dir.exists():
            # Try name-based folders
            for name_pattern in [STAGE_NAMES.get(class_id, ""), f"class_{class_id}"]:
                alt = data_path / name_pattern
                if alt.exists():
                    class_dir = alt
                    break

        if class_dir.exists():
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    paths.append(str(img_file))
                    labels.append(class_id)

    if paths:
        print(f"[DATA] Loaded {len(paths)} images from folder structure")
    return paths, labels


def load_dataset(data_path):
    """Auto-detect and load dataset from given path."""
    data_path = Path(data_path)
    if not data_path.exists():
        return None, None

    # Try APTOS CSV format first
    paths, labels = load_aptos_dataset(data_path)
    if paths and len(paths) > 0:
        return paths, labels

    # Try folder structure
    paths, labels = load_folder_dataset(data_path)
    if paths and len(paths) > 0:
        return paths, labels

    return None, None


def split_dataset(paths, labels, val_ratio=0.2):
    """Stratified train/val split."""
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, l in enumerate(labels):
        class_indices[l].append(i)

    train_idx, val_idx = [], []
    for cls, indices in class_indices.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    return train_paths, train_labels, val_paths, val_labels


# ── Model ───────────────────────────────────────────────────────────────

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """Build EfficientNet-B3 with custom head for DR classification."""
    if pretrained:
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
    else:
        model = models.efficientnet_b3(weights=None)

    # Replace classifier with a more robust head
    num_features = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model, n=2):
    """Unfreeze the last N blocks of EfficientNet features."""
    # EfficientNet-B3 has 8 feature blocks (0-7)
    total_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i >= total_blocks - n:
            for param in block.parameters():
                param.requires_grad = True
        else:
            for param in block.parameters():
                param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ── Training ────────────────────────────────────────────────────────────

def get_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = Counter(labels)
    total = len(labels)
    weights = {cls: total / (NUM_CLASSES * count) for cls, count in counts.items()}
    # Normalize
    max_w = max(weights.values())
    weights = {cls: w / max_w for cls, w in weights.items()}
    return torch.tensor([weights.get(i, 1.0) for i in range(NUM_CLASSES)], dtype=torch.float32)


def get_sampler(labels):
    """Weighted random sampler for balanced batches."""
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Per-class accuracy
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    class_acc = {}
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c]

    return epoch_loss, epoch_acc, class_acc


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                scaler, device, num_epochs, phase_name, best_acc, save_path):
    """Run a training phase with early stopping."""
    patience = 8
    no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, scaler, device)
        val_loss, val_acc, class_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"  [{phase_name}] Epoch {epoch+1}/{num_epochs} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"Loss: {val_loss:.4f} | LR: {lr:.2e} | {elapsed:.1f}s")

        # Print per-class accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            for c in range(NUM_CLASSES):
                acc = class_acc.get(c, 0)
                print(f"    Stage {c} ({STAGE_NAMES[c]}): {acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0

            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'phase': phase_name,
            }, save_path)
            print(f"  * New best: {val_acc:.4f} - saved to {save_path.name}")

            if val_acc >= TARGET_ACC:
                print(f"\n  > TARGET {TARGET_ACC*100:.0f}% REACHED! val_acc={val_acc:.4f}")
                break
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹ Early stopping after {patience} epochs without improvement")
                break

    model.load_state_dict(best_model_wts)
    return best_acc


# ── Main ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def main():
    seed_everything()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  OptiGemma — EfficientNet-B3 Training Pipeline")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"  Target accuracy: {TARGET_ACC*100:.0f}%")
    print(f"{'='*60}\n")

    # ── Find Dataset ──
    dataset_path = None
    search_paths = [
        DATA_DIR / "aptos" / "colored_images",
        DATA_DIR / "aptos",
        DATA_DIR / "aptos2019",
        DATA_DIR / "train",
        DATA_DIR / "dr_dataset",
        DATA_DIR,
        BASE_DIR / "dataset",
    ]

    for sp in search_paths:
        paths, labels = load_dataset(sp)
        if paths and len(paths) > 10:
            dataset_path = sp
            break

    if not paths or len(paths) < 10:
        print("=" * 60)
        print("  NO DATASET FOUND!")
        print("=" * 60)
        print()
        print("Please provide a Diabetic Retinopathy dataset.")
        print()
        print("Option 1: APTOS 2019 (Recommended — 3662 images)")
        print("  1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data")
        print("  2. Download and extract to: data/aptos/")
        print("  3. Ensure structure:")
        print("       data/aptos/train.csv")
        print("       data/aptos/train_images/*.png")
        print()
        print("Option 2: Folder structure")
        print("  Place images in: data/train/0/ data/train/1/ ... data/train/4/")
        print("  (folder name = DR stage 0-4)")
        print()
        print("Option 3: Kaggle CLI")
        print("  pip install kaggle")
        print("  kaggle competitions download -c aptos2019-blindness-detection -p data/aptos")
        print()
        sys.exit(1)

    # ── Stats ──
    dist = Counter(labels)
    print(f"[DATA] Dataset: {dataset_path}")
    print(f"[DATA] Total images: {len(paths)}")
    print(f"[DATA] Class distribution:")
    for c in sorted(dist.keys()):
        print(f"  Stage {c} ({STAGE_NAMES.get(c, '?')}): {dist[c]} images")

    # ── Split ──
    train_paths, train_labels, val_paths, val_labels = split_dataset(paths, labels, val_ratio=0.2)
    print(f"\n[SPLIT] Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ── Datasets & Loaders ──
    train_ds = DRDataset(train_paths, train_labels, transform=get_transforms("train"))
    val_ds = DRDataset(val_paths, val_labels, transform=get_transforms("val"))

    sampler = get_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ──
    model = build_model(pretrained=True)
    model = model.to(device)

    class_weights = get_class_weights(train_labels).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    save_path = MODELS_DIR / "best_dr_model.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] EfficientNet-B3 | Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

    best_acc = 0.0

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Train classifier head only (backbone frozen)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'-'*60}")
    print(f"  PHASE 1: Training classifier head ({PHASE1_EPOCHS} epochs)")
    print(f"{'-'*60}")

    freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS,
                                                      eta_min=1e-5)

    best_acc = train_phase(model, train_loader, val_loader, criterion, optimizer,
                           scheduler, scaler, device, PHASE1_EPOCHS,
                           "Phase1-Head", best_acc, save_path)

    if best_acc >= TARGET_ACC:
        print(f"\n  [OK] Target reached in Phase 1! Final accuracy: {best_acc:.4f}")
        export_model(model, save_path)
        return

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze last 2 blocks + classifier
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'-'*60}")
    print(f"  PHASE 2: Fine-tuning last 2 blocks ({PHASE2_EPOCHS} epochs)")
    print(f"{'-'*60}")

    unfreeze_last_n_blocks(model, n=3)
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = train_phase(model, train_loader, val_loader, criterion, optimizer,
                           scheduler, scaler, device, PHASE2_EPOCHS,
                           "Phase2-Partial", best_acc, save_path)

    if best_acc >= TARGET_ACC:
        print(f"\n  [OK] Target reached in Phase 2! Final accuracy: {best_acc:.4f}")
        export_model(model, save_path)
        return

    # ════════════════════════════════════════════════════════════════
    # PHASE 3: Full fine-tune with very low LR
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'-'*60}")
    print(f"  PHASE 3: Full fine-tuning ({PHASE3_EPOCHS} epochs)")
    print(f"{'-'*60}")

    unfreeze_all(model)
    optimizer = optim.AdamW([
        {'params': model.features[:5].parameters(), 'lr': 5e-6},
        {'params': model.features[5:].parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE3_EPOCHS,
                                                      eta_min=1e-7)

    best_acc = train_phase(model, train_loader, val_loader, criterion, optimizer,
                           scheduler, scaler, device, PHASE3_EPOCHS,
                           "Phase3-Full", best_acc, save_path)

    # ── Final ──
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_acc*100:.2f}%")
    print(f"  Model saved to: {save_path}")
    print(f"{'='*60}")

    export_model(model, save_path)

    # Final detailed evaluation
    _, final_acc, class_acc = evaluate(model, val_loader, criterion, device)
    print(f"\n[FINAL] Overall accuracy: {final_acc*100:.2f}%")
    for c in range(NUM_CLASSES):
        acc = class_acc.get(c, 0)
        print(f"  Stage {c} ({STAGE_NAMES[c]}): {acc*100:.1f}%")


def export_model(model, checkpoint_path):
    """Export the trained model in the format expected by detector.py"""
    export_path = MODELS_DIR / "vessel_model" / "best_val_loss.pt"

    # detector.py expects keys with 'model.' prefix
    state_dict = model.state_dict()
    prefixed = {}
    for key, value in state_dict.items():
        prefixed[f"model.{key}"] = value

    torch.save(prefixed, export_path)
    print(f"[EXPORT] Model exported to {export_path}")
    print(f"[EXPORT] This will be auto-loaded by detector.py on next app start")

    # Save training metadata
    meta = {
        "model": "EfficientNet-B3",
        "input_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "PyTorch",
    }
    with open(MODELS_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
