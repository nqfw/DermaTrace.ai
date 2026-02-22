import cv2
import numpy as np
import os
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIRS = [
    os.path.join(BASE_DIR, "data", "HAM10000 dataset", "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "data", "raw"),
    os.path.join(BASE_DIR, "test images")
]

RULER_OUTPUT_DIR = os.path.join(BASE_DIR, "research", "ruler_remover", "rulers_found")
NON_RULER_OUTPUT_DIR = os.path.join(BASE_DIR, "research", "ruler_remover", "no_rulers")

results = []

for d in TARGET_DIRS:
    if not os.path.exists(d): continue
    for f in glob.glob(os.path.join(d, "*.jpg")):
        img = cv2.imread(f)
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([140, 255, 255]))
        c = cv2.countNonZero(mask)
        if c > 1000:
            results.append((c, f))

# Sort by pixel count descending
results.sort(reverse=True)
for count, filepath in results[:10]:
    print(f"{count} pixels -> {filepath}")
