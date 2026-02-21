# research/fiftyone_curation.py
import fiftyone as fo
import fiftyone.types as fot
from pathlib import Path
import pandas as pd

DATASET_DIR = "../data/processed"
MST_CSV = "../data/mst_scores.csv"

def load_dataset():
    print("📦 Loading dataset into FiftyOne...")

    dataset = fo.Dataset.from_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fot.ImageDirectory
    )

    return dataset

def attach_mst_labels(dataset):
    try:
        df = pd.read_csv(MST_CSV)
        mst_dict = dict(zip(df["image"], df["mst_score"]))

        for sample in dataset:
            filename = Path(sample.filepath).name
            if filename in mst_dict:
                sample["mst_score"] = int(mst_dict[filename])
                sample.save()

        print("🎯 MST labels attached successfully!")
    except Exception as e:
        print("⚠️ Could not attach MST labels:", e)

def audit_dataset():
    dataset = load_dataset()
    attach_mst_labels(dataset)

    print("🚀 Launching FiftyOne App (Dataset Viewer)...")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    audit_dataset()