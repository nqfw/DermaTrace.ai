"""
retrain_ham10000.py
====================
Improved HAM10000 retraining script v2.

Key improvements over `train.py`:
  - Resumes from existing `melanoma_finetuned.pth` (warm start)
  - Covers BOTH HAM10000 parts (part_1 + part_2)
  - Stronger class weights to reduce the nv -> mel false alarm rate
  - Learning rate scheduler (StepLR) to avoid plateau
  - Early stopping on val loss to prevent overfitting

Usage:
    python core/retrain_ham10000.py
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model
from core.dullrazor import apply_dullrazor

cv2.setLogLevel(0)

# ─── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHTS_IN  = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
WEIGHTS_OUT = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"  # Overwrite best
CKPT_PATH   = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\ham_retrain_latest.pth"
CSV_PATH    = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_metadata.csv"
IMG_DIRS    = [
    r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_images_part_1",
    r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_images_part_2",
]

CLASS_TO_IDX = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
EPOCHS       = 15
BATCH_SIZE   = 16
LR           = 5e-5          # Slightly higher than Fitz since this is a bigger dataset shift
VAL_SPLIT    = 0.1           # 10% of data used for validation
PATIENCE     = 5             # Early stop after 5 epochs of no val improvement

# ─── Dataset ───────────────────────────────────────────────────────────────────
class HAMDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records   # list of (img_path, label_str)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, label_str = self.records[idx]
        label_idx = CLASS_TO_IDX[label_str]

        img = cv2.imread(img_path)
        if img is None:
            # Return a blank tensor on corrupt image
            return torch.zeros(3, 224, 224), torch.tensor(label_idx, dtype=torch.long)

        img = cv2.resize(img, (224, 224))           # Full-frame resize (no crop for HAM10000)
        img = apply_dullrazor(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        if self.transform:
            return self.transform(pil_img), torch.tensor(label_idx, dtype=torch.long)
        return transforms.ToTensor()(pil_img), torch.tensor(label_idx, dtype=torch.long)


# ─── Main Training Function ────────────────────────────────────────────────────
def train():
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"\n[retrain_ham10000] Starting on {DEVICE}")

    # 1. Build records from CSV + image dirs
    df = pd.read_csv(CSV_PATH)
    img_lookup = {}
    for d in IMG_DIRS:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.jpg'):
                    img_lookup[os.path.splitext(f)[0]] = os.path.join(d, f)

    records = []
    for _, row in df.iterrows():
        img_id = row['image_id']
        if img_id in img_lookup:
            records.append((img_lookup[img_id], row['dx']))

    print(f"Found {len(records)} valid image-label pairs across both HAM10000 parts.")

    # 2. Train/Val split — split records FIRST, then build separate datasets
    import random
    random.shuffle(records)
    val_size   = int(len(records) * VAL_SPLIT)
    val_records   = records[:val_size]
    train_records = records[val_size:]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = HAMDataset(train_records, transform=train_transform)
    val_ds   = HAMDataset(val_records,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_records)} | Val: {len(val_records)} images")

    # 3. Model (Warm start from current weights)
    model, _ = get_resnet50_model(num_classes=7)
    if os.path.exists(WEIGHTS_IN):
        print(f"RESUMING from {WEIGHTS_IN}")
        model.load_state_dict(torch.load(WEIGHTS_IN, map_location=DEVICE))
    else:
        print("WARNING: No base weights found, starting from ImageNet defaults.")

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    # 4. Loss: Heavy penalty on misclassifying nv as mel (false alarm) AND
    #    mel as nv (missed diagnosis).
    # Class weights: nv=0.5 (penalise it being over-predicted), mel=3.0 (critical)
    # bcc=2.0, akiec=1.5 (pre-cancerous), rest=1.0
    class_weights = torch.tensor(
        [0.4,  # nv    — most common class, reduce over-prediction
         3.0,  # mel   — most dangerous, penalise misses
         1.0,  # bkl
         2.5,  # bcc   — cancerous
         1.5,  # akiec — pre-cancerous
         1.2,  # vasc
         1.0], # df
        dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 5. Training loop with validation and early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

    for epoch in range(EPOCHS):
        # --- Train Phase ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pred': IDX_TO_CLASS[preds[0].item()],
                'true': IDX_TO_CLASS[labels[0].item()],
            })

        train_loss = running_loss / len(train_loader)
        scheduler.step()

        # --- Val Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        # Save intermediate checkpoint
        torch.save(model.state_dict(), CKPT_PATH)

        # Save best model + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  --> Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  --> No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nRetraining Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Best weights saved to: {WEIGHTS_OUT}")


if __name__ == "__main__":
    train()
