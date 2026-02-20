import cv2
import numpy as np
import os
import glob

def process_image(img, threshold=15.0, max_texture=300):
    """
    Uses YCrCb color space + Texture constraint to detect skin.
    Color alone cannot separate orange cats from skin, so we check
    how 'smooth' the image is. Skin is smooth, fur is textured.
    """
    # 1. Standard YCrCb skin color detection
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin_ycbcr = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin_ycbcr = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycbcr, lower_skin_ycbcr, upper_skin_ycbcr)

    # Skin coverage
    skin_pixels = cv2.countNonZero(mask)
    pct = (skin_pixels / (img.shape[0] * img.shape[1])) * 100
    
    # Texture score (Variance of Laplacian)
    # Fur has high variance (lots of edges), skin has low variance (smooth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Is it skin? Must meet coverage % AND be relatively smooth
    is_skin = bool(pct >= threshold and texture_score < max_texture)
    
    return is_skin, pct, mask

# --- DYNAMIC PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

input_folder = os.path.join(ROOT_DIR, "data", "HAM10000 dataset", "HAM10000_images_part_1")
verified_folder = os.path.join(ROOT_DIR, "data", "HAM10000_verified")
unverified_folder = os.path.join(ROOT_DIR, "data", "HAM10000_unverified")

# Create folders
for folder in [verified_folder, unverified_folder]:
    os.makedirs(folder, exist_ok=True)

# Fetch first 100 images (Git-safe slice)
all_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))[:100]

print(f"--- RUNNING HIGH-YIELD AUDIT ---")
print(f"Input: {input_folder}")

stats = {"passed": 0, "failed": 0}

for f_path in all_files:
    filename = os.path.basename(f_path)
    img = cv2.imread(f_path)
    if img is None: continue
    
    # Standardize size for HAM10000
    img = cv2.resize(img, (600, 450))
    is_skin, pct, mask = process_image(img)
    
    if is_skin: 
        save_path = os.path.join(verified_folder, filename)
        stats["passed"] += 1
        tag = "[VERIFIED]"
    else:
        save_path = os.path.join(unverified_folder, filename)
        stats["failed"] += 1
        tag = "[FLAGGED] "
    
    cv2.imwrite(save_path, img)
    print(f"{tag} {filename} | Coverage: {pct:.1f}%")

print("-" * 50)
print(f"PIPELINE COMPLETE.")
print(f"Total: {len(all_files)} | Passed: {stats['passed']} | Failed: {stats['failed']}")
print(f"Pass Rate: {(stats['passed']/len(all_files))*100:.1f}%")
print("-" * 50)