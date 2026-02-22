"""
launch_audit_hub.py
===================
Standalone script to launch a live FiftyOne Audit session based on
images uploaded and processed by the DermaTrace.ai Streamlit dashboard.

Usage:
    python research/launch_audit_hub.py
"""

import os
import sys
import json
import time
import fiftyone as fo

# ─── Config ───────────────────────────────────────────────────────────────────
LOG_FILE = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\output\audit_uploads\audit_log.json"
DATASET_NAME = "DermaTrace_Live_Audit"

# ─── Main Logic ───────────────────────────────────────────────────────────────
def load_and_sync_dataset():
    if not os.path.exists(LOG_FILE):
        print(f"Waiting for user uploads... Log file not found at {LOG_FILE}")
        return None

    try:
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        return None
        
    if not logs:
        return None

    # Load or Create Dataset
    if DATASET_NAME in fo.list_datasets():
        dataset = fo.load_dataset(DATASET_NAME)
        dataset.clear() # Clear out old to prevent duplicates during sync
    else:
        dataset = fo.Dataset(DATASET_NAME)
        dataset.persistent = True

    # Build Samples
    samples = []
    for record in logs:
        filepath = record.get('filepath')
        if not filepath or not os.path.exists(filepath):
            continue
            
        sample = fo.Sample(filepath=filepath)
        
        # Add Ground Truth (Actual)
        true_label = record.get('true_label', 'unknown').upper()
        sample["ground_truth"] = fo.Classification(label=true_label)
        
        # Add Predictions
        pred1 = record.get('pred1', 'unknown').upper()
        conf = record.get('confidence', 0.0) / 100.0 # FiftyOne prefers 0.0-1.0 confidence
        sample["predictions"] = fo.Classification(label=pred1, confidence=conf)
        
        # Add Metadata tags
        sample["mst_tone"] = record.get('mst_tone', 'unknown')
        sample["model_used"] = record.get('model_used', 'unknown')
        sample["upload_time"] = record.get('timestamp', '')
        
        # Flag False Positives / False Negatives dynamically
        if true_label != "UNKNOWN":
            sample["is_correct"] = (true_label == pred1)
            
        samples.append(sample)

    dataset.add_samples(samples)
    
    # Compute basic evaluation if possible
    if len(dataset) > 0 and dataset.has_sample_field("is_correct"):
        try:
            dataset.evaluate_classifications(
                "predictions",
                gt_field="ground_truth",
                eval_key="eval"
            )
        except Exception:
            pass # Fails cleanly if only "UNKNOWN" labels exist
            
    return dataset

# ─── Launch Server ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 🏥 DermaTrace.ai -- Live Audit Hub Launcher")
    print("="*50)
    
    dataset = load_and_sync_dataset()
    
    if dataset is None:
        print("No uploads detected yet. Creating empty placeholder dataset.")
        if DATASET_NAME in fo.list_datasets():
            dataset = fo.load_dataset(DATASET_NAME)
        else:
            dataset = fo.Dataset(DATASET_NAME)
    else:
        print(f"Loaded {len(dataset)} user uploads into FiftyOne.")

    # Launch the app natively bound to all interfaces
    print("\nLaunching FiftyOne on port 5151...")
    print("NOTE: Close this terminal window to shut down the Audit Hub.")
    
    # By binding to 0.0.0.0, the port can be forwarded by VS Code or accessed via external IP
    session = fo.launch_app(dataset, port=5151, address="0.0.0.0")
    
    # Keep the script alive so the server runs
    try:

        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down Audit Hub.")
        sys.exit(0)
