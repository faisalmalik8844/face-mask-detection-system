"""
evaluate.py

Runs the trained model against the test split locally and prints the
per-class table (Precision, Recall, mAP50, mAP50-95), plus saves a
confusion matrix to reports/figures/.

Requires the full dataset (data/raw/train, valid, test + data.yaml) to
be present locally - this is NOT needed just to run the webcam demo.

Run with:
    python src/evaluate.py
"""

from ultralytics import YOLO

MODEL_PATH = "../models/face_mask_yolo26n_best.pt"
DATA_YAML = "../data/raw/data.yaml"


def main():
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_YAML, split="test", project="../reports", name="test_eval")

    print("\n=== Per-class results ===")
    for i, name in model.names.items():
        print(f"{name}: P={metrics.box.p[i]:.3f}  R={metrics.box.r[i]:.3f}  "
              f"mAP50={metrics.box.ap50[i]:.3f}  mAP50-95={metrics.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()
