"""
inspect_yolo_dataset.py

Counts how many bounding box instances exist per class across all YOLO
.txt label files in a dataset. Run this FIRST on any new dataset before
trusting it - this is exactly the EDA step that revealed the class
imbalance problem in prior face-mask-detection projects using similar data.

Usage:
    python inspect_yolo_dataset.py
"""

from pathlib import Path
from collections import Counter

# This dataset uses a Roboflow-style train/valid/test split, each with
# its own images/ and labels/ subfolders. We check all three.
SPLIT_DIRS = ["train/labels", "valid/labels", "test/labels"]

# Class order confirmed from this dataset's data.yaml - note this is
# DIFFERENT from the andrewmvd dataset's order, so double-check this
# every time you use a new dataset rather than assuming.
CLASS_NAMES = ["Mask", "Mask Incorrect", "No Mask"]


def inspect_labels(labels_dir):
    label_files = list(Path(labels_dir).rglob("*.txt"))
    print(f"Found {len(label_files)} label files in '{labels_dir}'")

    class_counts = Counter()
    empty_files = 0

    for label_file in label_files:
        with open(label_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

    print(f"\nEmpty label files (no objects): {empty_files}")
    print("\nInstance counts per class:")
    total = sum(class_counts.values())
    for class_id in sorted(class_counts.keys()):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_id_{class_id}"
        count = class_counts[class_id]
        pct = (count / total * 100) if total else 0
        print(f"  {name}: {count} instances ({pct:.1f}%)")

    print(f"\nTotal instances: {total}")


if __name__ == "__main__":
    for split_dir in SPLIT_DIRS:
        if Path(split_dir).exists():
            print(f"\n{'='*50}")
            print(f"Split: {split_dir}")
            print('='*50)
            inspect_labels(split_dir)
        else:
            print(f"\n(skipping '{split_dir}' - not found)")
