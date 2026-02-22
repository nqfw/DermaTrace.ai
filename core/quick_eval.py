"""
quick_eval.py
=============
Runs two targeted accuracy checks:
  1. HAM10000 model (melanoma_finetuned.pth) on last 50 HAM10000 images (CSV lookup for ground truth)
  2. Fitzpatrick model (fitzpatrick_weights.pth) on last 50 images of Tone 5 (filename ground truth)

Usage:
    python core/quick_eval.py
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import glob
import torchvision.transforms as transforms
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model
from core.dullrazor import apply_dullrazor

# ─── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HAM_WEIGHTS  = os.path.join(BASE_DIR, "models", "melanoma_finetuned.pth")
FITZ_WEIGHTS = os.path.join(BASE_DIR, "models", "fitzpatrick_weights.pth")
HAM_IMG_DIRS = [
    os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_images_part_2"),
]
HAM_CSV      = os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_metadata.csv")
FITZ_TONE5   = os.path.join(BASE_DIR, "data", "fitz_ham10000_subset", "5")

CLASS_TO_IDX = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(weights_path):
    model, _ = get_resnet50_model(num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def preprocess(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

def run_eval(label, images, model, get_true_label_fn, crop=False):
    correct_top1 = correct_top2 = 0
    all_true, all_pred = [], []

    for img_path in images:
        true_label = get_true_label_fn(img_path)
        if true_label is None or true_label not in CLASS_TO_IDX:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (400, 300))
        if crop:
            h, w = img.shape[:2]
            y, x = h//2 - 112, w//2 - 112
            img = img[y:y+224, x:x+224]
        else:
            img = cv2.resize(img, (224, 224))

        cleaned = apply_dullrazor(img)
        input_tensor = preprocess(cleaned)

        with torch.no_grad():
            out = model(input_tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            top_probs, top_preds = torch.topk(probs, 2, dim=1)

        p1 = IDX_TO_CLASS[top_preds[0][0].item()]
        p2 = IDX_TO_CLASS[top_preds[0][1].item()]
        conf1 = top_probs[0][0].item() * 100
        conf2 = top_probs[0][1].item() * 100

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

        img_id = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  [{match:4s}] {img_id:30s} | True: {true_label:6s} | Pred1: {p1:6s}({conf1:.0f}%) | Pred2: {p2:6s}({conf2:.0f}%)")

    total = len(all_true)
    if total == 0:
        print("  No valid images evaluated.")
        return

    top1 = (correct_top1 / total) * 100
    top2 = (correct_top2 / total) * 100
    f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)

    print(f"\n  --- {label} Summary ---")
    print(f"  Images Evaluated : {total}")
    print(f"  Top-1 Accuracy   : {top1:.2f}%")
    print(f"  Top-2 Accuracy   : {top2:.2f}%")
    print(f"  Weighted F1      : {f1:.4f}")
    print()


# ─── 1. HAM10000: Last 50 images, CSV ground truth ────────────────────────────
print("=" * 65)
print("  TEST 1: HAM10000 Model | Last 50 Images | melanoma_finetuned.pth")
print("=" * 65)

ham_model = load_model(HAM_WEIGHTS)
df_meta = pd.read_csv(HAM_CSV)

all_ham = []
for d in HAM_IMG_DIRS:
    all_ham.extend(sorted(glob.glob(os.path.join(d, "*.jpg"))))

last50_ham = sorted(all_ham)[-50:]

def get_ham_label(img_path):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    match = df_meta.loc[df_meta['image_id'] == img_id, 'dx']
    return match.values[0] if not match.empty else None

run_eval("HAM10000 (Last 50)", last50_ham, ham_model, get_ham_label, crop=False)


# ─── 2. Fitzpatrick Tone-5: Last 50 images, filename ground truth ─────────────
print("=" * 65)
print("  TEST 2: Fitzpatrick Model | Tone-5 Last 50 | fitzpatrick_weights.pth")
print("=" * 65)

fitz_model = load_model(FITZ_WEIGHTS)

all_tone5 = sorted(glob.glob(os.path.join(FITZ_TONE5, "*.jpg")))
last50_tone5 = all_tone5[-50:]

def get_fitz_label(img_path):
    prefix = os.path.basename(img_path).split('_')[0].lower()
    return prefix if prefix in CLASS_TO_IDX else None

run_eval("Fitzpatrick Tone-5 (Last 50)", last50_tone5, fitz_model, get_fitz_label, crop=True)

print("Evaluation complete.")
