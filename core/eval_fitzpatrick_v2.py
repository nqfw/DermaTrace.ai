"""
eval_fitzpatrick_v2.py
======================
Evaluates the NEWLY RETRAINED Fitzpatrick model (0.49 loss) on:
  - Top-1/Top-2 Accuracy + Weighted F1 on first 500 Fitzpatrick images
  - Grad-CAM heatmaps saved for the first 20 images

Usage:
    python core/eval_fitzpatrick_v2.py
"""

import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model
from core.dullrazor import apply_dullrazor
from core.gradcam_engine import generate_cam

# ─── Config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "fitzpatrick_weights.pth")
DATA_DIR     = os.path.join(BASE_DIR, "data", "fitz_ham10000_subset")
HEATMAP_DIR  = os.path.join(BASE_DIR, "output", "fitz_heatmaps_v2")
os.makedirs(HEATMAP_DIR, exist_ok=True)
EVAL_LIMIT   = 500
HEATMAP_LIMIT = 20

CLASS_TO_IDX = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Helpers ───────────────────────────────────────────────────────────────────
def center_crop(img, size=224):
    h, w = img.shape[:2]
    if h <= size or w <= size:
        return cv2.resize(img, (size, size))
    y = h // 2 - size // 2
    x = w // 2 - size // 2
    return img[y:y+size, x:x+size]

def preprocess(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

# ─── Load Model ────────────────────────────────────────────────────────────────
print(f"\n[DermaTrace.ai] Fitzpatrick V2 Evaluation")
print(f"   Device : {DEVICE}")
print(f"   Weights: {WEIGHTS_PATH}")

if not os.path.exists(WEIGHTS_PATH):
    print("ERROR: fitzpatrick_weights.pth not found. Please check path.")
    sys.exit(1)

model, _ = get_resnet50_model(num_classes=7)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
target_layer = model.layer4[-1]
print(f"   Model loaded successfully.\n")

# ─── Gather images from all tone folders ──────────────────────────────────────
all_images = []
for tone in ['1', '2', '3', '4', '5', '6']:
    tone_dir = os.path.join(DATA_DIR, tone)
    if not os.path.exists(tone_dir):
        continue
    for f in os.listdir(tone_dir):
        if f.endswith('.jpg'):
            all_images.append(os.path.join(tone_dir, f))

# First 500 in stable order
all_images = sorted(all_images)[:EVAL_LIMIT]
print(f"Evaluating first {len(all_images)} images across all tone folders.\n")

# ─── Evaluation Loop ──────────────────────────────────────────────────────────
correct_top1 = 0
correct_top2 = 0
all_true, all_pred = [], []
heatmap_count = 0

pbar = tqdm(enumerate(all_images), total=len(all_images), desc="Evaluating", unit="img")

for i, img_path in pbar:
    filename = os.path.basename(img_path)
    true_label = filename.split('_')[0]
    
    if true_label not in CLASS_TO_IDX:
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Preprocessing: resize → center crop → dullrazor
    img = cv2.resize(img, (400, 300))
    cropped = center_crop(img)
    cleaned = apply_dullrazor(cropped)
    
    input_tensor = preprocess(cleaned)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs   = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_preds = torch.topk(probs, 2, dim=1)

    p1 = IDX_TO_CLASS[top_preds[0][0].item()]
    p2 = IDX_TO_CLASS[top_preds[0][1].item()]

    all_true.append(true_label)
    all_pred.append(p1)

    match = "FAIL"
    if p1 == true_label:
        correct_top1 += 1
        correct_top2 += 1
        match = "TOP1"
    elif p2 == true_label:
        correct_top2 += 1
        match = "TOP2"

    pbar.set_postfix({'match': match, 'pred': p1, 'true': true_label})

    # ─── Grad-CAM for first 20 ────────────────────────────────────────────────
    if heatmap_count < HEATMAP_LIMIT:
        heatmap = generate_cam(model, target_layer, input_tensor, cleaned)
        
        # Side-by-side: original | cleaned | heatmap
        orig_display   = cv2.resize(cropped, (224, 224))
        clean_display  = cv2.resize(cleaned, (224, 224))
        hmap_display   = cv2.resize(heatmap, (224, 224))
        combined       = np.hstack([orig_display, clean_display, hmap_display])

        # Label on top
        label_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, f"TRUE: {true_label.upper()}  |  PRED-1: {p1.upper()}  |  PRED-2: {p2.upper()}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if p1 == true_label else (0, 100, 255), 2)
        
        final_composite = np.vstack([label_bar, combined])
        
        out_path = os.path.join(HEATMAP_DIR, f"{i:03d}_{true_label}_pred{p1}.jpg")
        cv2.imwrite(out_path, final_composite)
        heatmap_count += 1

# ─── Final Report ─────────────────────────────────────────────────────────────
total = len(all_true)
print(f"\n{'='*60}")
print(f"  DermaTrace.ai -- Fitzpatrick V2 Evaluation Report")
print(f"{'='*60}")
print(f"  Total Valid Images Evaluated : {total}")
print(f"  Top-1 Accuracy              : {(correct_top1/total)*100:.2f}%")
print(f"  Top-2 Accuracy              : {(correct_top2/total)*100:.2f}%")
print(f"  Weighted F1 Score           : {f1_score(all_true, all_pred, average='weighted', zero_division=0):.4f}")
print(f"\n  Heatmaps saved to           : {HEATMAP_DIR}")
print(f"{'='*60}")
print("\n  Per-class Report:")
print(classification_report(all_true, all_pred, zero_division=0))
