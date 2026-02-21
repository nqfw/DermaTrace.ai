# research/fiftyone_curation.py
import fiftyone as fo
from pathlib import Path

def load_ham10000_dataset():
    """
    Load HAM10000 images from both part_1 and part_2 folders
    (Matches your folder structure exactly)
    """

    base_path = Path(r"C:\Users\dipak\OneDrive\Desktop\skin-cancer\data")

    folder1 = base_path / "HAM10000_images_part_1"
    folder2 = base_path / "HAM10000_images_part_2"

    # Collect all image paths
    image_paths = list(folder1.glob("*.jpg")) + list(folder2.glob("*.jpg"))

    print(f"Total Images Found: {len(image_paths)}")

    # Create FiftyOne dataset
    dataset = fo.Dataset(name="ham10000_dataset", overwrite=True)

    samples = []
    for img_path in image_paths:
        sample = fo.Sample(filepath=str(img_path))
        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


def audit_dataset():
    print("Loading HAM10000 dataset into FiftyOne...\n")
    
    dataset = load_ham10000_dataset()

    print("Dataset Loaded Successfully!")
    print("Total Samples:", len(dataset))

    # Compute metadata (size, resolution, etc.)
    dataset.compute_metadata()
    print("Metadata computed!")

    # Launch FiftyOne App (visual inspection UI)
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    audit_dataset()