import cv2
import numpy as np
import os
from pathlib import Path

def estimate_skin_tone(image):
    """
    Monk Skin Tone estimation (1 = light, 10 = dark)
    Based on LAB brightness channel
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    brightness = np.mean(l_channel)

    # Map brightness to MST scale (Higher L = MST 1, Lower L = MST 10)
    # Tuned for dermoscopic images
    mst_score = int(np.interp(brightness, [40, 220], [10, 1]))
    return max(1, min(10, mst_score))

def score_dataset():
    # 1. Dynamically find the HACKATHON root folder
    # Goes up 2 levels from research/skintone/skintone.py to HACKATHON/
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2] 
    
    # 2. Point to the data folder
    base_path = project_root / "data" / "HAM10000 dataset"
    
    print(f"Project root: {project_root}")
    print(f"Searching in: {base_path}\n")

    if not base_path.exists():
        print(f"ERROR: Could not find data at {base_path}")
        return

    # 3. Find images and prevent double-counting
    raw_paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg"]:
        raw_paths.extend(list(base_path.rglob(ext)))

    # Use set() to remove duplicates if images exist in multiple subfolders
    image_paths = sorted(list(set(raw_paths)))

    print(f"Total Unique Images Found: {len(image_paths)}")
    
    if not image_paths:
        return

    # 4. Process and Display
    print("-" * 30)
    for img_path in image_paths[:1000]:  # Show first 1000 results
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        score = estimate_skin_tone(img)
        print(f"MST [{score}] -> {img_path.name}")
    
    print("-" * 30)
    print(f"Successfully analyzed {min(len(image_paths), 15)} samples.")

if __name__ == "__main__":
    score_dataset()