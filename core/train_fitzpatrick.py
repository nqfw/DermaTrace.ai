import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model
from dullrazor import apply_dullrazor

# Suppress minor OpenCV parsing warnings
cv2.setLogLevel(0)

class FitzpatrickDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the Fitzpatrick sorted images.
    Applies center cropping and DullRazor *during* the __getitem__ DataLoader loop.
    While this is slightly slower than offline processing, it guarantees the 
    training code is mathematically identical to the evaluation code.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        
        # HAM10000 0-6 index mapping
        self.class_to_idx = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
            'akiec': 4, 'vasc': 5, 'df': 6
        }
        
        # Load all valid Fitzpatrick images from all tone folders (1-6)
        for tone in ['1', '2', '3', '4', '5', '6']:
            tone_dir = os.path.join(data_dir, tone)
            if not os.path.exists(tone_dir): continue
            
            for file in os.listdir(tone_dir):
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(tone_dir, file))

    def __len__(self):
        return len(self.images)

    def center_crop_image(self, img, crop_size=224):
        """Replicates evaluating center crop identically."""
        h, w = img.shape[:2]
        if h <= crop_size or w <= crop_size:
            return cv2.resize(img, (crop_size, crop_size))
            
        start_y = h//2 - crop_size//2
        start_x = w//2 - crop_size//2
        return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Extract disease string prefix (e.g. "mel_0a7d...")
        filename = os.path.basename(img_path)
        disease_label = filename.split('_')[0]
        label_idx = self.class_to_idx[disease_label]
        
        # 1. Load via OpenCV
        orig_img = cv2.imread(img_path)
        # Soft resize for base speed before cropping
        orig_img = cv2.resize(orig_img, (400, 300))
        
        # 2. Apply Custom Center Crop (User Requested)
        cropped_img = self.center_crop_image(orig_img)
        
        # 3. Apply DullRazor (User Requested)
        clean_img = apply_dullrazor(cropped_img)
        
        # 4. PyTorch Prep
        clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(clean_img_rgb)
        
        if self.transform:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)
            
        return tensor_img, torch.tensor(label_idx, dtype=torch.long)


def train_fitzpatrick():
    # 1. Hardware Optimization (User Requested)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Firing up GPU Acceleration on {device} ---")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # Optimizes underlying hardware algorithms to prevent throttling
    
    # Paths
    data_dir = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\fitz_ham10000_subset"
    base_ham10000_weights = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
    save_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\fitzpatrick_weights.pth"
    
    # 2. PyTorch Transform (Needs Data Augmentations so it doesn't just memorize the small Fitzpatrick dataset)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Standard skin augmentations
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Very minor jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Dataloader
    dataset = FitzpatrickDataset(data_dir=data_dir, transform=train_transform)
    print(f"Detected {len(dataset)} combined Fitzpatrick training images.")
    # Keeping batch size relatively small (16) because CPU has to do DullRazor on the fly, 
    # preventing the dataloader queue from timing out.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # 4. Initialize Model with Pre-Existing HAM10000 Weights (User Requested)
    model, _ = get_resnet50_model(num_classes=7)
    if os.path.exists(base_ham10000_weights):
        print(f"Loading Base HAM10000 64% accuracy weights from {base_ham10000_weights}")
        model.load_state_dict(torch.load(base_ham10000_weights, map_location=device))
    else:
        print("WARNING: Base HAM10000 weights not found, starting from pure ImageNet.")
        
    # 5. Unfreeze Architecture (User Requested)
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # 6. Setup Optimizer
    # We softly penalize the model for missing 'mel' (1) and 'bcc' (3)
    class_weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Extremely low learning rate (1e-5) because we are fine-tuning a model that is ALREADY fine-tuned.
    # A standard learning rate would blow out the existing HAM10000 logic and cause 'catastrophic forgetting'.
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    epochs = 15 # Because learning rate is tiny, we need more epochs
    
    # 7. Training Loop
    best_loss = float('inf')
    rev_map = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    
    print("\nStarting Fitzpatrick Retraining Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Predict
            _, preds = torch.max(outputs, 1)
            pred_dx = rev_map[preds[0].item()]
            actual_dx = rev_map[labels[0].item()]
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'pred': pred_dx, 'true': actual_dx})
            progress_bar.update(1)
            
        progress_bar.close()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
        
        # Checkpoint Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best Fitzpatrick weights to {save_path}")

    print("\nFitzpatrick Fine-Tuning Complete!")

if __name__ == "__main__":
    train_fitzpatrick()
