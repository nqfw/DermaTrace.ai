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
        _, preds = torch.max(outputs, 1)
        
    rev_map = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    predicted_diagnosis = rev_map[preds[0].item()]
    
    return predicted_diagnosis, true_label

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
        
        # Test exactly 20 images
        sample_size = min(20, len(images))
        print(f"\n--- Model Blind Test ({sample_size} random images) ---")
        
        correct = 0
        for img_path in random.sample(images, sample_size):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            true_label = "unknown"
            if df is not None:
                matches = df.loc[df['image_id'] == img_id, 'dx'].values
                if len(matches) > 0:
                    true_label = matches[0]
            
            # Run without visual output
            pred, true = run_gradcam_demo(img_path, model=test_model, true_label=true_label)
            
            if pred is None: continue
            
            # Terminal printout
            match_status = "[MATCH]" if pred == true else "[FAIL]"
            if pred == true: correct += 1
            
            print(f"{match_status} | Predicted: {str(pred).upper():5s} | Actual: {str(true).upper():5s} | {img_id}")
            
        acc = (correct / sample_size) * 100
        print(f"\nEvaluation Complete: {acc:.1f}% Accuracy on this subset.")
    else:
        print("Please provide a directory path.")
