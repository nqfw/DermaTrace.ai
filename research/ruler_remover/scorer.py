# research/mst_scorer.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

PROCESSED_DIR = Path("../data/processed")
OUTPUT_CSV = Path("../data/mst_scores.csv")

def extract_skin_region(image):
    """
    Simple skin detection using HSV range
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad skin color range (works for multiple tones)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    skin = cv2.bitwise_and(image, image, mask=mask)

    return skin, mask

def get_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3)
    pixels = pixels[pixels.sum(axis=1) > 0]  # remove black pixels

    if len(pixels) < 10:
        return [0, 0, 0]

    kmeans = KMeans(n_clusters=k, n_init=5)
    kmeans.fit(pixels)
    dominant = kmeans.cluster_centers_[0]

    return dominant

def rgb_to_mst(rgb):
    """
    Approximate MST scoring based on brightness
    (Research approximation, not official Google MST)
    """
    brightness = np.mean(rgb)

    if brightness > 200:
        return 1
    elif brightness > 180:
        return 3
    elif brightness > 150:
        return 5
    elif brightness > 120:
        return 7
    else:
        return 9

def score_dataset():
    print("🎨 Calculating Monk Skin Tone (MST) scores...")

    records = []
    image_files = list(PROCESSED_DIR.glob("*.jpg")) + list(PROCESSED_DIR.glob("*.png"))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        skin, mask = extract_skin_region(img)
        dominant_color = get_dominant_color(skin)

        mst_score = rgb_to_mst(dominant_color)

        records.append({
            "image": img_path.name,
            "mst_score": int(mst_score),
            "dominant_rgb": dominant_color.tolist()
        })

        print(f"✅ {img_path.name} → MST: {mst_score}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"📊 MST scores saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    score_dataset()