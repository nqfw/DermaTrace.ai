import os
import json
import time
import shutil
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\output\audit_uploads"
LOG_FILE   = os.path.join(OUTPUT_DIR, "audit_log.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Initialization ───────────────────────────────────────────────────────────
def _init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            json.dump([], f)

_init_log()

# ─── Logger Function ──────────────────────────────────────────────────────────
def log_inference(image_bytes, filename, true_label, pred1_label, pred2_label, conf1, mst_score, model_name):
    """
    Saves the uploaded image bytes to disk and appends an inference record to the JSON log.
    
    Args:
        image_bytes: Raw bytes from Streamlit's UploadedFile
        filename: Original string filename
        true_label: Known actual disease label (or 'UNKNOWN')
        pred1_label: Top-1 predicted label
        pred2_label: Top-2 predicted label
        conf1: Confidence % for Top-1 (float)
        mst_score: Monk Skin Tone scale prediction (1-10 string/int)
        model_name: The name of the model used (e.g. 'Clinical (HAM10000)')
    """
    try:
        # 1. Create unique filename to prevent overwrite
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{os.path.basename(filename)}"
        save_path = os.path.join(OUTPUT_DIR, safe_filename)
        
        # 2. Save pure image bytes
        with open(save_path, "wb") as f:
            f.write(image_bytes)
            
        # 3. Build Record
        record = {
            "timestamp": datetime.now().isoformat(),
            "filepath": save_path,
            "original_filename": filename,
            "true_label": true_label.lower() if true_label else "unknown",
            "pred1": pred1_label.lower(),
            "pred2": pred2_label.lower(),
            "confidence": round(float(conf1), 2),
            "mst_tone": str(mst_score),
            "model_used": model_name
        }
        
        # 4. Append to JSON Log
        # Read existing
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
                
        # Append new
        logs.append(record)
        
        # Write back
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)
            
        return save_path
        
    except Exception as e:
        print(f"Audit Logger Error: {e}")
        return None
