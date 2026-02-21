import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import sys
import os

from gradcam_engine import generate_cam
from dullrazor import apply_dullrazor
from skin import process_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_resnet50_model

def run_gradcam_demo(image_path, model=None, true_label=None):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        return None, None
        
    img = cv2.resize(img, (400, 300))
    
    # 2. Clean image (DullRazor + Ruler Removal)
    clean_img = apply_dullrazor(img)
    
    # 3. Check for skin (Optional safety check)
    is_skin, pct, _, has_lesion = process_image(clean_img)
        
    if not has_lesion:
        return "healthy", true_label

    # 4. Prepare for PyTorch Model
    clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    # Standard ImageNet normalization for IMAGENET1K_V2 weights
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(clean_img_rgb).unsqueeze(0) # 1x3x224x224
    
    # 5. Load Finetuned Model
    if model is None:
        model, _ = get_resnet50_model(num_classes=7)
        weights_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
        model.eval()
    
    # Run Inference to get the predicted disease
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_preds = torch.topk(probs, 2, dim=1)
        
    rev_map = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    top2_diagnoses = [rev_map[top_preds[0][0].item()], rev_map[top_preds[0][1].item()]]
    
    return top2_diagnoses, true_label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/demo_gradcam.py <path_to_directory>")
        sys.exit(1)
        
    path = sys.argv[1]
    
    if os.path.isdir(path):
        import glob
        import random
        # Pre-load model to save time during batch testing
        print("Pre-loading model for batch Evaluation...")
        test_model, _ = get_resnet50_model(num_classes=7)
        weights_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
        if os.path.exists(weights_path):
            test_model.load_state_dict(torch.load(weights_path))
        test_model.eval()
        
        # Load dataset metadata to get ground truth labels
        import pandas as pd
        csv_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_metadata.csv"
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

        images = glob.glob(os.path.join(path, "*.jpg"))
        
        # Test exactly the first 500 images
        sample_size = min(500, len(images))
        print(f"\n--- Model Sequential Test (First {sample_size} images) ---")
        
        from sklearn.metrics import f1_score
        
        correct_top1 = 0
        correct_top2 = 0
        all_true = []
        all_pred = []
        
        # Taking the first 500 instead of a random sample
        for img_path in images[:sample_size]:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            true_label = "unknown"
            if df is not None:
                matches = df.loc[df['image_id'] == img_id, 'dx'].values
                if len(matches) > 0:
                    true_label = matches[0]
            
            # Run without visual output
            top2_pred, true = run_gradcam_demo(img_path, model=test_model, true_label=true_label)
            
            if top2_pred is None: continue
            
            if top2_pred == "healthy":
                pred1 = "healthy"
                pred2 = "healthy"
            else:
                pred1 = top2_pred[0]
                pred2 = top2_pred[1]
                
            all_true.append(true)
            all_pred.append(pred1)
            
            match_status = "[FAIL]"
            if pred1 == true: 
                correct_top1 += 1
                correct_top2 += 1
                match_status = "[TOP-1 MATCH]"
            elif pred2 == true:
                correct_top2 += 1
                match_status = "[TOP-2 MATCH]"
            
            print(f"{match_status:13s} | Top1: {str(pred1).upper():5s} | Top2: {str(pred2).upper():5s} | Actual: {str(true).upper():5s} | {img_id}")
            
        if len(all_true) > 0:
            acc_top1 = (correct_top1 / len(all_true)) * 100
            acc_top2 = (correct_top2 / len(all_true)) * 100
            f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
            
            print(f"\nEvaluation Complete ({len(all_true)} valid images):")
            print(f"Top-1 Accuracy: {acc_top1:.1f}%")
            print(f"Top-2 Accuracy: {acc_top2:.1f}%")
            print(f"Weighted F1-Score: {f1:.4f}")
        else:
            print("No valid images processed.")
    else:
        print("Please provide a directory path.")
