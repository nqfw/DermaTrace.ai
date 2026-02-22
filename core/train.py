import os
import cv2
cv2.setLogLevel(0)  # Suppress OpenCV warnings for missing files in partial datasets
import torch
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dullrazor import apply_dullrazor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None, limit=None):
        self.df = df
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)
            
        self.img_dir = img_dir
        self.transform = transform
        
        # Mapping 7 skin lesions to integers
        self.diagnosis_map = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
            'akiec': 4, 'vasc': 5, 'df': 6
        }

    def __len__(self):
        return len(self.df)

    def center_crop_image(self, img, crop_size=224):
        """Replicates evaluating center crop identically."""
        h, w = img.shape[:2]
        if h <= crop_size or w <= crop_size:
            return cv2.resize(img, (crop_size, crop_size))
            
        start_y = h//2 - crop_size//2
        start_x = w//2 - crop_size//2
        return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

    def __getitem__(self, idx):
        # Gracefully handle missing images in our partial dataset
        while True:
            # 1. Grab ID and construct path
            img_id = self.df.loc[idx, 'image_id']
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            
            # 2. Load Image
            image = cv2.imread(img_path)
            if image is not None:
                break
            
            # If missing, just grab the next random index so training doesn't crash
            idx = (idx + 1) % len(self.df)
            
        # Soft resize for base speed before cropping
        image = cv2.resize(image, (400, 300))
        
        # Apply Custom Center Crop
        image = self.center_crop_image(image)
        
        # Apply DullRazor
        image = apply_dullrazor(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. Grab Label
        label_str = self.df.loc[idx, 'dx']
        label = self.diagnosis_map[label_str]
        
        # 5. Apply Albumentations/PyTorch Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)


def train_model(epochs=3, batch_size=1, experimental_limit=1):
    # Setup Device for MAXIMUM GPU POWER
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training on Device: {device} ---")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True # Optimizes GPU hardware algorithms
        
    base_dir = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset"
    metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")
    img_dir = os.path.join(base_dir, "HAM10000_images_part_1")
    
    # 1. Load Metadata
    df = pd.read_csv(metadata_path)
    
    # 2. Standard ResNet50 IMAGENET1K_V2 augmentation requirements
    train_transform = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 3. Load Dataset & Dataloader
    print(f"Generating Dataset (Limit: {experimental_limit} images)...")
    train_dataset = HAM10000Dataset(df=df, img_dir=img_dir, transform=train_transform, limit=experimental_limit)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 4. Load Model
    print("Loading Pretrained ResNet50...")
    model, _ = get_resnet50_model(num_classes=7)
    
    # 5. Unfreeze Architecture
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # 5. Setup Weighted Loss function
    # 'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
    # We softly penalize the model for missing 'mel' (1) and 'bcc' (3)
    class_weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Lower learning rate since network is fully unfrozen
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 6. Training Loop Setup
    os.makedirs(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models", exist_ok=True)
    best_loss = float('inf')
    
    # Reverse mapping for console output
    rev_map = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Manual TQDM Progress Bar
        progress_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Decode the first prediction in the batch for debugging
            _, preds = torch.max(outputs, 1)
            pred_dx = rev_map[preds[0].item()]
            actual_dx = rev_map[labels[0].item()]
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'pred': pred_dx, 'true': actual_dx})
            progress_bar.update(1)
            
            # Dynamic mid-epoch Checkpointing (Save progress in case of crash)
            if batch_idx > 0 and batch_idx % 500 == 0:
                ckpt_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\latest_checkpoint.pth"
                torch.save({'epoch': epoch, 'batch': batch_idx, 'model': model.state_dict()}, ckpt_path)
                
        progress_bar.close()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
        
        # Checkpoint Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model to {save_path}")

    print("\nTraining Complete!")

if __name__ == "__main__":
    # Test on a small but representative subset before full 10k training
    train_model(epochs=10, batch_size=16, experimental_limit=5000)
