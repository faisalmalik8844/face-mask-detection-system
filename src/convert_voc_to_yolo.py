"""
convert_voc_to_yolo.py

Converts the Kaggle Face Mask Detection dataset's PASCAL VOC XML
annotations into YOLO's normalized .txt format.

Run this once, locally, after downloading the dataset from Kaggle.

Usage:
    python convert_voc_to_yolo.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# TODO: update these paths to match where you extracted the Kaggle dataset
ANNOTATIONS_DIR = "annotations"   # folder containing the .xml files
OUTPUT_LABELS_DIR = "labels"      # where the converted .txt files will go

# Must match the exact class names used in the dataset's XML files
CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def convert_bbox(size, box):
    """Converts a VOC bounding box (xmin, ymin, xmax, ymax) into YOLO's
    normalized (x_center, y_center, width, height) format, each 0-1."""
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def convert_annotation(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_NAMES:
            print(f"Warning: unknown class '{class_name}' in {xml_path}, skipping")
            continue
        class_id = CLASS_NAMES.index(class_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center, y_center, width, height = convert_bbox((img_w, img_h), (xmin, ymin, xmax, ymax))
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    Path(OUTPUT_LABELS_DIR).mkdir(exist_ok=True)

    xml_files = list(Path(ANNOTATIONS_DIR).glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")

    for xml_file in xml_files:
        output_path = Path(OUTPUT_LABELS_DIR) / (xml_file.stem + ".txt")
        convert_annotation(xml_file, output_path)

    print(f"Conversion complete. YOLO labels saved to '{OUTPUT_LABELS_DIR}/'")
