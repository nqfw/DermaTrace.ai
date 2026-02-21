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
        print(f"Error: Could not load image at {image_path}")
        return
        
    img = cv2.resize(img, (400, 300))
    
    # 2. Clean image (DullRazor + Ruler Removal)
    print(f"Pre-processing Image {os.path.basename(image_path)}...")
    clean_img = apply_dullrazor(img)
    
    # 3. Check for skin (Optional safety check)
    is_skin, pct, _, has_lesion = process_image(clean_img)
    if not is_skin:
        print(f"Warning: Low skin coverage ({pct:.1f}%). Generating heatmap anyway...")
        
    if not has_lesion:
        print(f"Pipeline Intercepted: Image {os.path.basename(image_path)} is healthy skin. No lesion detected.")
        # If no lesion, we bypass PyTorch entirely and just show a blank heatmap with a message
        blank_heatmap = np.zeros((300, 400, 3), dtype=np.uint8)
        combined = np.hstack((img, clean_img, blank_heatmap))
        text = "Pipeline: HEALTHY TISSUE (No Mole Detected)"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Trust Layout (Original | Cleaned | Grad-CAM)", combined)
        cv2.waitKey(0)
        return

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
        print("Loading Finetuned ResNet50...")
        model, _ = get_resnet50_model(num_classes=7)
        
        weights_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            print("Successfully loaded custom Hackathon weights!")
        else:
            print("Warning: Custom weights not found, falling back to base ImageNet weights.")
            
        model.eval()
    
    # Run Inference to get the predicted disease
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        
    rev_map = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    predicted_diagnosis = rev_map[preds[0].item()]
    
    # 6. Generate Grad-CAM Match
    # For ResNet50, the last conv layer is layer4[-1]
    target_layer = model.layer4[-1]
    
    print("Inferring Heatmap...")
    # Generate the heatmap. We pass the 224x224 BGR original so the cam size matches exactly!
    clean_img_224 = cv2.resize(clean_img, (224, 224))
    clean_img_float = np.float32(clean_img_224) / 255.0
    heatmap_bgr = generate_cam(model, target_layer, input_tensor, clean_img_float)
    
    # Resize the heatmap back up to 400x300 so it matches the other images in our display loop
    heatmap_bgr = cv2.resize(heatmap_bgr, (400, 300))
    
    # 7. Create Side-by-Side Visualization
    # [Original | Cleaned | Grad-CAM Heatmap]
    combined = np.hstack((img, clean_img, heatmap_bgr))
    
    # Add text overlay with results
    true_str = f" | True: {str(true_label).upper()}" if true_label else ""
    text = f"Pred: {predicted_diagnosis.upper()}{true_str} | Skin Cover: {pct:.1f}%"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 8. Show Image
    print("Done! Displaying results. Close the window or press any key for the next image...")
    cv2.imshow("Trust Layout (Original | Cleaned | Grad-CAM)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/demo_gradcam.py <path_to_image_or_directory>")
        sys.exit(1)
        
    path = sys.argv[1]
    
    if os.path.isdir(path):
        import glob
        import random
        # Pre-load model to save time during batch testing
        print("Pre-loading model for batch test...")
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
        # Test 10 images
        sample_size = min(10, len(images))
        print(f"Testing on {sample_size} random images from {path}...")
        for img_path in random.sample(images, sample_size):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            true_label = None
            if df is not None:
                matches = df.loc[df['image_id'] == img_id, 'dx'].values
                if len(matches) > 0:
                    true_label = matches[0]
            
            run_gradcam_demo(img_path, model=test_model, true_label=true_label)
    else:
        run_gradcam_demo(path)
