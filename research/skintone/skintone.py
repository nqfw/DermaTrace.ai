# research/mst_scorer.py
import cv2
import numpy as np
from pathlib import Path

def estimate_skin_tone(image):
    """
    Monk Skin Tone estimation (1 = light, 10 = dark)
    Based on LAB brightness channel
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    brightness = np.mean(l_channel)

    # Map brightness to MST scale (tuned for dermoscopy)
    mst_score = int(np.interp(brightness, [40, 220], [10, 1]))
    mst_score = max(1, min(10, mst_score))

    return mst_score


def get_all_image_paths():
    """
    Load images from BOTH HAM10000 folders
    """
    base_path = Path(r"C:\Users\dipak\OneDrive\Desktop\skin-cancer\data")

    folder1 = base_path / "HAM10000_images_part_1"
    folder2 = base_path / "HAM10000_images_part_2"

    images = list(folder1.glob("*.jpg")) + list(folder2.glob("*.jpg"))
    return images


def score_dataset():
    image_paths = get_all_image_paths()

    print(f"Total Images Found: {len(image_paths)}")
    print("Starting Monk Skin Tone Scoring...\n")

    results = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        score = estimate_skin_tone(img)
        results.append((img_path.name, score))

        print(f"{img_path.name} -> MST Score: {score}")

    print("\nScoring Completed!")
    return results


if __name__ == "__main__":
    score_dataset()