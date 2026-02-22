"""
eval_ham_last300.py
====================
Evaluates `melanoma_finetuned.pth` on the last 300 sorted HAM10000 images.
Ground truth is pulled from HAM10000_metadata.csv by image_id.
"""
import os, sys, cv2, glob, torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model
from core.dullrazor import apply_dullrazor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(BASE_DIR, "models", "melanoma_finetuned.pth")
CSV     = os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_metadata.csv")
DIRS    = [
    os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_images_part_2"),
]
CLASS_TO_IDX = {'nv':0,'mel':1,'bkl':2,'bcc':3,'akiec':4,'vasc':5,'df':6}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load model
model, _ = get_resnet50_model(num_classes=7)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"[eval_ham_last300] Device: {DEVICE} | Weights: {WEIGHTS}\n")

# Load CSV
df = pd.read_csv(CSV)

# Gather and slice last 300
all_imgs = sorted(
    glob.glob(os.path.join(DIRS[0], "*.jpg")) +
    glob.glob(os.path.join(DIRS[1], "*.jpg"))
)
images = all_imgs[-300:]
print(f"Evaluating last {len(images)} images...\n")

c1 = c2 = 0
all_t, all_p = [], []

for img_path in images:
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    row = df.loc[df['image_id'] == img_id, 'dx']
    if row.empty:
        continue

    true = row.values[0]
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    img = apply_dullrazor(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out   = model(tensor)
        prob  = torch.nn.functional.softmax(out, dim=1)
        tp, ti = torch.topk(prob, 2, dim=1)

    p1 = IDX_TO_CLASS[ti[0][0].item()]
    p2 = IDX_TO_CLASS[ti[0][1].item()]
    c1t = tp[0][0].item() * 100
    c2t = tp[0][1].item() * 100

    all_t.append(true)
    all_p.append(p1)

    tag = "FAIL"
    if p1 == true:
        c1 += 1; c2 += 1; tag = "TOP1"
    elif p2 == true:
        c2 += 1; tag = "TOP2"

    print(f"[{tag:4s}] {img_id:32s} | True:{true:6s} | P1:{p1:6s}({c1t:.0f}%) | P2:{p2:6s}({c2t:.0f}%)")

n = len(all_t)
avg = 'weighted'
print("\n" + "="*60)
print("  HAM10000 Evaluation -- Last 300 Images")
print("="*60)
print(f"  Valid images evaluated : {n}")
print(f"  Top-1 Accuracy         : {c1/n*100:.2f}%")
print(f"  Top-2 Accuracy         : {c2/n*100:.2f}%")
print(f"  Weighted F1 Score      : {f1_score(all_t, all_p, average=avg, zero_division=0):.4f}")
print("\n  Per-class Report:")
print(classification_report(all_t, all_p, zero_division=0))
