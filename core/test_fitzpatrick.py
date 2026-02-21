import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import cv2
import numpy as np
import random

sys.path.append(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models")
from model import get_resnet50_model
sys.path.append(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\core")
from dullrazor import apply_dullrazor
from gradcam_engine import generate_cam

sys.path.append(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\research\skintone")
from skintone import estimate_skin_tone

# 1. Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
DATA_DIR = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\fitz_ham10000_subset"

# Testing ALL tones this time, with 200 random samples total
TONES_TO_TEST = ['1', '2', '3', '4', '5', '6']
TOTAL_TEST_IMAGES = 200

# 2. HAM10000 Class Mapping 
idx_to_class = {
    0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 
    4: 'akiec', 5: 'vasc', 6: 'df'
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def center_crop_image(img, crop_size=224):
    h, w = img.shape[:2]
    if h <= crop_size or w <= crop_size:
        return cv2.resize(img, (crop_size, crop_size))
        
    start_y = h//2 - crop_size//2
    start_x = w//2 - crop_size//2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

def evaluate_tone_bias():
    print(f"--- Running 200-Image Fitzpatrick Evaluation + MST Audit on {DEVICE} ---")
    
    # Load Fine-Tuned Model
    model, _ = get_resnet50_model(num_classes=7)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    else:
        print(f"ERROR: Cannot find finetuned weights at {MODEL_WEIGHTS}")
        return
        
    model = model.to(DEVICE)
    model.eval()
    
    # Gather ALL images from ALL tone folders
    all_images = []
    for tone in TONES_TO_TEST:
        tone_dir = os.path.join(DATA_DIR, tone)
        if os.path.exists(tone_dir):
            images = [f for f in os.listdir(tone_dir) if f.endswith('.jpg')]
            for img in images:
                all_images.append({'name': img, 'tone': tone, 'path': os.path.join(tone_dir, img)})
                
    # Randomly shuffle and slice 200
    random.shuffle(all_images)
    test_set = all_images[:TOTAL_TEST_IMAGES]
    
    # Dictionary to track results
    results = {
        'total_tested': 0, 'total_correct_top1': 0, 'total_correct_top2': 0,
        'mst_exact_match': 0, 'mst_close_match': 0 # Close = within 2 points
    }
    
    all_true = []
    all_pred = []
    
    print(f"Began testing {len(test_set)} mixed images...")
    
    for i, img_data in enumerate(test_set):
        img_path = img_data['path']
        actual_tone = int(img_data['tone'])
        actual_label = img_data['name'].split('_')[0]
        
        try:
            orig_img = cv2.imread(img_path)
            if orig_img is None: continue
            
            # 1. TEST MST SCORE PREDICTION ON RAW IMAGE
            # The MST function returns 1-10. We map it down to fit 1-6 standard Fitzpatrick.
            raw_mst = estimate_skin_tone(orig_img)
            # Map 1-10 back down to 1-6 scale roughly
            pred_tone = max(1, min(6, int((raw_mst / 10.0) * 6)))
            
            mst_diff = abs(actual_tone - pred_tone)
            mst_status = "PERFECT" if mst_diff == 0 else "CLOSE" if mst_diff <= 2 else "FLOP"
            
            if mst_status == "PERFECT": results['mst_exact_match'] += 1
            if mst_status in ["PERFECT", "CLOSE"]: results['mst_close_match'] += 1
            
            # 2. TEST CNN DISEASE MODEL ON CLEANED IMAGE
            cropped_img = center_crop_image(orig_img)
            clean_img = apply_dullrazor(cropped_img)
            
            clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(clean_img_rgb).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_preds = torch.topk(probs, 2, dim=1)
                
            pred1_label = idx_to_class[top_preds[0][0].item()]
            pred2_label = idx_to_class[top_preds[0][1].item()]
            
            all_true.append(actual_label)
            all_pred.append(pred1_label)
            
            status_str = "[FAIL]"
            if pred1_label == actual_label:
                results['total_correct_top1'] += 1
                results['total_correct_top2'] += 1
                status_str = "[TOP1 WIN]"
            elif pred2_label == actual_label:
                results['total_correct_top2'] += 1
                status_str = "[TOP2 WIN]"
                
            results['total_tested'] += 1
            
            # Print Every 10th image to save terminal space, OR if MST flops massively
            if i % 10 == 0 or mst_status == "FLOP":
                print(f"{i:3d}/200 | AI: {status_str:10s} (Top1:{pred1_label.upper():5s} Top2:{pred2_label.upper():5s} A:{actual_label.upper():5s}) | Tone: {mst_status:7s} (P:{pred_tone} A:{actual_tone})")
            
        except Exception as e:
            print(f"  Error processing {img_data['name']}: {e}")

    # Final Report
    print("\n" + "="*40)
    print("MIXED 200-IMAGE EVALUATION REPORT")
    print("="*40)
    if results['total_tested'] > 0:
        from sklearn.metrics import f1_score
        t = results['total_tested']
        cnn_acc_top1 = (results['total_correct_top1'] / t) * 100
        cnn_acc_top2 = (results['total_correct_top2'] / t) * 100
        f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
        
        mst_acc = (results['mst_exact_match'] / t) * 100
        mst_close = (results['mst_close_match'] / t) * 100
        
        print(f"CNN Disease Accuracy (Top-1): {cnn_acc_top1:.1f}% ({results['total_correct_top1']}/{t})")
        print(f"CNN Disease Accuracy (Top-2): {cnn_acc_top2:.1f}% ({results['total_correct_top2']}/{t})")
        print(f"CNN Disease Weighted F1-Score: {f1:.4f}")
        print(f"MST Tone Prediction Accuracy: {mst_acc:.1f}% Exact | {mst_close:.1f}% Close (+/- 2)")
    print("="*40)

if __name__ == "__main__":
    evaluate_tone_bias()
