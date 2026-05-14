# ============================================================================
# 🔬 OptiGemma — EfficientNet-B3 Training on Kaggle
# Dataset: APTOS 2019 Blindness Detection
# Platform: Kaggle Notebooks (T4 GPU, 9hr session)
# Target: 85-90%+ accuracy with Transfer Learning
#
# HOW TO USE:
# 1. Go to kaggle.com → Notebooks → New Notebook
# 2. Add Competition Data: "APTOS 2019 Blindness Detection"
# 3. Set Accelerator: GPU T4 x2 (or P100)
# 4. Copy-paste each CELL into separate Kaggle cells
# 5. Run all → Download model from Output panel
# ============================================================================

# %% [markdown]
# # 🔬 OptiGemma — DR Detection Model Training
# **EfficientNet-B3 + Transfer Learning + Advanced Techniques**

# %% ============ CELL 1: GPU Check & Dependencies ============
import torch
import os
print(f"🔥 GPU: {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
if torch.cuda.is_available():
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Kaggle has most deps pre-installed, just need timm & albumentations
os.system('pip install -q timm albumentations')
print("✅ Dependencies ready!")

# %% ============ CELL 2: Load Dataset (already on Kaggle) ============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Auto-detect dataset path (works whether added as competition or dataset)
OUTPUT_DIR = '/kaggle/working'
DATA_DIR = None

# Search for train.csv in /kaggle/input/
print("🔍 Searching for dataset...")
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train.csv' in files:
        DATA_DIR = root
        break

if DATA_DIR is None:
    # List what's available so user can debug
    print("❌ train.csv not found! Available inputs:")
    for item in os.listdir('/kaggle/input'):
        sub = os.path.join('/kaggle/input', item)
        print(f"   📁 {item}/")
        if os.path.isdir(sub):
            for f in os.listdir(sub)[:10]:
                print(f"      - {f}")
    raise FileNotFoundError("Add APTOS 2019 dataset: Notebook → Add Data → Search 'aptos2019'")

# Auto-detect image directory
IMG_DIR = os.path.join(DATA_DIR, 'train_images')
if not os.path.exists(IMG_DIR):
    # Maybe images are in a subfolder
    for d in os.listdir(DATA_DIR):
        candidate = os.path.join(DATA_DIR, d)
        if os.path.isdir(candidate) and 'train' in d.lower():
            IMG_DIR = candidate
            break

print(f"✅ Dataset found at: {DATA_DIR}")
print(f"   Images at: {IMG_DIR}")
print(f"   Image count: {len(os.listdir(IMG_DIR))}")

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
print(f"📦 Total samples: {len(df)}")
print(f"\n📊 Class Distribution:")
print(df['diagnosis'].value_counts().sort_index())

labels = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'PDR (4)']
colors = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#dc2626']
ax = df['diagnosis'].value_counts().sort_index().plot(kind='bar', color=colors, figsize=(10, 5))
ax.set_xticklabels(labels, rotation=45)
ax.set_title('APTOS 2019 — DR Stage Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150)
plt.show()

# %% ============ CELL 3: Dataset & DataLoader ============
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 300  # EfficientNet-B3

def crop_image_from_gray(img, tol=7):
    """Ben Graham's preprocessing — crop black borders."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, row['diagnosis']

# Heavy augmentations for training
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=IMG_SIZE//12, max_width=IMG_SIZE//12, fill_value=0, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# TTA transforms (5 versions per image during eval)
tta_transforms = [
    val_transform,
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0),
               A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.VerticalFlip(p=1.0),
               A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.RandomRotate90(p=1.0),
               A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
               A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
]

# 80/20 stratified split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
print(f"📦 Train: {len(train_df)} | Val: {len(val_df)}")

train_dataset = APTOSDataset(train_df, IMG_DIR, train_transform)
val_dataset = APTOSDataset(val_df, IMG_DIR, val_transform)

# Class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.arange(5), y=train_df['diagnosis'].values)
class_weights = torch.FloatTensor(cw).cuda()
print(f"⚖️ Class weights: {class_weights}")

# AMP allows batch_size=32 on T4
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"✅ DataLoaders ready! (batch={BATCH_SIZE})")

# %% ============ CELL 4: Model — EfficientNet-B3 ============
import timm
import torch.nn as nn

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name='efficientnet_b3'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 {model_name} — {total:,} params")

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freeze everything except classifier head."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        t = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  ❄️ Backbone frozen. Trainable: {t:,}")

    def unfreeze_last_blocks(self, n_blocks=2):
        """Unfreeze last N blocks + classifier."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        # Unfreeze last N blocks
        blocks = list(self.model.blocks.children()) if hasattr(self.model, 'blocks') else []
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        t = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  🔓 Last {n_blocks} blocks unfrozen. Trainable: {t:,}")

    def unfreeze_all(self):
        """Unfreeze entire network."""
        for param in self.model.parameters():
            param.requires_grad = True
        t = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  🔓 ALL unfrozen. Trainable: {t:,}")

model = DRClassifier(num_classes=5).cuda()
model.freeze_backbone()  # Phase 1: only classifier head

# %% ============ CELL 5: Training Loop (Progressive Unfreezing + AMP) ============
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# PyTorch 2.x compatible AMP imports
try:
    from torch.amp import GradScaler, autocast
    AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_DEVICE = None
from sklearn.metrics import cohen_kappa_score
import time

EPOCHS = 25
PATIENCE = 7

# Label smoothing for better generalization
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# AMP scaler for mixed precision
scaler = GradScaler('cuda') if AMP_DEVICE else GradScaler()

# Optimizer — will be rebuilt on unfreeze
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_val_acc = 0
best_kappa = 0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_kappa': []}

print("🚀 Training with Progressive Unfreezing + AMP + Label Smoothing")
print("=" * 70)

for epoch in range(EPOCHS):
    start = time.time()

    # === PROGRESSIVE UNFREEZING ===
    if epoch == 5:
        print("\n🔓 PHASE 2: Unfreezing last 2 blocks...")
        model.unfreeze_last_blocks(2)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif epoch == 15:
        print("\n🔓 PHASE 3: Unfreezing ALL layers (fine LR)...")
        model.unfreeze_all()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # === TRAIN ===
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        with autocast('cuda') if AMP_DEVICE else autocast():  # Mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # === VALIDATE ===
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            with autocast('cuda') if AMP_DEVICE else autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic') * 100
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_kappa'].append(val_kappa)

    elapsed = time.time() - start
    phase = "HEAD" if epoch < 5 else ("PARTIAL" if epoch < 15 else "FULL")
    print(f"E{epoch+1:02d}/{EPOCHS} [{phase}] {elapsed:.0f}s — "
          f"TrL:{train_loss:.4f} | VL:{val_loss:.4f} | Acc:{val_acc:.1f}% | κ:{val_kappa:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_kappa = val_kappa
        patience_counter = 0
        torch.save(model.model.state_dict(), os.path.join(OUTPUT_DIR, 'optigemma_effnetb3_best.pt'))
        print(f"  💾 BEST! Saved (Acc:{val_acc:.1f}%, κ:{val_kappa:.1f}%)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  ⏹️ Early stop at epoch {epoch+1}")
            break

print("=" * 70)
print(f"🏆 BEST — Accuracy: {best_val_acc:.1f}% | Kappa: {best_kappa:.1f}%")

# %% ============ CELL 6: TTA (Test Time Augmentation) ============
print("🔄 Running TTA (5 augmented passes)...")
model.model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'optigemma_effnetb3_best.pt'), weights_only=True))
model.eval()

tta_all_probs = []
all_labels_tta = []

with torch.no_grad():
    for idx in range(len(val_df)):
        row = val_df.iloc[idx]
        img_path = os.path.join(IMG_DIR, f"{row['id_code']}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        avg_probs = np.zeros(5)
        for tfm in tta_transforms:
            aug = tfm(image=image)['image'].unsqueeze(0).cuda()
            with autocast('cuda') if AMP_DEVICE else autocast():
                out = model(aug)
            probs = torch.softmax(out, 1).cpu().numpy()[0]
            avg_probs += probs
        avg_probs /= len(tta_transforms)
        tta_all_probs.append(avg_probs)
        all_labels_tta.append(row['diagnosis'])

        if (idx + 1) % 100 == 0:
            print(f"  TTA: {idx+1}/{len(val_df)}")

tta_preds = np.argmax(tta_all_probs, axis=1)
tta_acc = np.mean(tta_preds == np.array(all_labels_tta)) * 100
tta_kappa = cohen_kappa_score(all_labels_tta, tta_preds, weights='quadratic') * 100
print(f"\n🏆 TTA Results — Accuracy: {tta_acc:.1f}% | Kappa: {tta_kappa:.1f}%")
print(f"   (vs non-TTA: Acc {best_val_acc:.1f}% | κ {best_kappa:.1f}%)")

# %% ============ CELL 7: Confusion Matrix & Metrics ============
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

labels_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
print("📊 Classification Report (with TTA):")
print(classification_report(all_labels_tta, tta_preds, target_names=labels_names))

cm = confusion_matrix(all_labels_tta, tta_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_names, yticklabels=labels_names)
plt.title(f'OptiGemma — Confusion Matrix (Acc: {tta_acc:.1f}%, κ: {tta_kappa:.1f}%)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].axvline(x=5, color='r', ls='--', alpha=0.5); axes[0].axvline(x=15, color='r', ls='--', alpha=0.5)
axes[1].plot(history['val_acc']); axes[1].set_title('Val Accuracy (%)'); axes[1].axvline(x=5, color='r', ls='--', alpha=0.5); axes[1].axvline(x=15, color='r', ls='--', alpha=0.5)
axes[2].plot(history['val_kappa']); axes[2].set_title('Cohen Kappa (%)'); axes[2].axvline(x=5, color='r', ls='--', alpha=0.5); axes[2].axvline(x=15, color='r', ls='--', alpha=0.5)
for ax in axes: ax.set_xlabel('Epoch')
plt.suptitle('Red lines = unfreeze phases')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
plt.show()

# %% ============ CELL 8: Save & Summary ============
model_path = os.path.join(OUTPUT_DIR, 'optigemma_effnetb3_best.pt')
model_size = os.path.getsize(model_path) / 1e6
print(f"""
{'='*60}
🏆 TRAINING COMPLETE!
{'='*60}
📦 Model: optigemma_effnetb3_best.pt ({model_size:.1f} MB)
📊 Accuracy: {tta_acc:.1f}% (with TTA)
📊 Kappa: {tta_kappa:.1f}% (with TTA)
{'='*60}

📥 HOW TO DOWNLOAD:
   1. Click "Save Version" (top right)
   2. Select "Save & Run All"
   3. After completion → go to Output tab
   4. Download: optigemma_effnetb3_best.pt

🔧 HOW TO INTEGRATE:
   1. Copy file to: models/vessel_model/best_val_loss.pt
   2. Restart: python app.py
   3. Done! 🚀
{'='*60}
""")
