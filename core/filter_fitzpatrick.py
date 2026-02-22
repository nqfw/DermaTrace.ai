import os
import shutil
import pandas as pd
import requests

# 1. Configuration
FITZPATRICK_CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "fitzpatrick17k.csv")
SOURCE_DIR = os.path.join(DATA_DIR, "organized_fitzpatrick")
DEST_DIR = os.path.join(DATA_DIR, "fitz_ham10000_subset")

# HAM10000 7 condition strings for matching
# The fitzpatrick label column uses full names, so we map them to the 7 diagnoses
# nv: Melanocytic nevi
# mel: Melanoma
# bkl: Benign keratosis-like lesions
# bcc: Basal cell carcinoma
# akiec: Actinic keratoses and intraepithelial carcinoma / Bowen's disease
# vasc: Vascular lesions
# df: Dermatofibroma

TARGET_CONDITIONS = {
    "melanoma": "mel",
    "basal cell carcinoma": "bcc",
    "squamous cell carcinoma": "akiec", # Close enough for akiec category
    "actinic keratosis": "akiec",
    "seborrheic keratosis": "bkl",
    "dermatofibroma": "df",
    "nevus": "nv", # melanocytic nevi
    "hemangioma": "vasc", # vascular
    "pyogenic granuloma": "vasc" # vascular
}

def download_csv():
    if not os.path.exists(CSV_PATH):
        print(f"Downloading {FITZPATRICK_CSV_URL}...")
        response = requests.get(FITZPATRICK_CSV_URL)
        with open(CSV_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("fitzpatrick17k.csv already exists.")

def filter_and_copy():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Check for required columns
    if 'md5hash' not in df.columns or 'label' not in df.columns or 'fitzpatrick_scale' not in df.columns:
        print("Error: Required columns missing from CSV.")
        print(df.columns)
        return

    # Filter for standard 1-6 skin tones
    df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4, 5, 6])]
    
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Create the 6 subfolders
    for tone in range(1, 7):
        os.makedirs(os.path.join(DEST_DIR, str(tone)), exist_ok=True)
    
    # Process
    matched_count = 0
    missing_count = 0
    
    # Lowercase string labels for matching
    df['label_lower'] = df['label'].astype(str).str.lower()
    
    for index, row in df.iterrows():
        hash_id = row['md5hash']
        label = row['label_lower']
        tone = int(row['fitzpatrick_scale'])
        
        # Check if the label is in our target map
        mapped_ham_label = None
        for target_key, ham_val in TARGET_CONDITIONS.items():
            if target_key in label:
                mapped_ham_label = ham_val
                break
                
        if not mapped_ham_label:
            continue
            
        # Source file path
        filename = f"{hash_id}.jpg"
        src_path = os.path.join(SOURCE_DIR, str(tone), filename)
        
        if os.path.exists(src_path):
            # Target path
            # Give it the new HAM10000 label suffix to make training easy later
            dest_path = os.path.join(DEST_DIR, str(tone), f"{mapped_ham_label}_{filename}")
            shutil.copy2(src_path, dest_path)
            matched_count += 1
        else:
            missing_count += 1
            
    print(f"\nFiltering Complete!")
    print(f"Successfully copied {matched_count} images matching HAM10000 conditions.")
    if missing_count > 0:
        print(f"Skipped {missing_count} images that were in CSV but missing from your folder.")

if __name__ == "__main__":
    download_csv()
    filter_and_copy()
