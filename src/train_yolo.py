"""
train_yolo.py

Trains YOLO26n on the 3-class face mask dataset (Mask / Mask Incorrect / No Mask).

Run from the FaceMaskDetection folder (same level as data.yaml):
    python train_yolo.py

Step 1: leave EPOCHS = 2 and run once. This is a smoke test: it proves the
        dataset paths work and tells you how long one epoch takes on your machine.
Step 2: change EPOCHS to 50 for the real baseline run.
"""

import torch
from ultralytics import YOLO

EPOCHS = 2        # change to 50 after the smoke test works
IMG_SIZE = 640
BATCH = 16        # if you get an out-of-memory error, try 8


def main():
    if torch.cuda.is_available():
        device = 0
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Training on CPU (slower, but works fine for a nano model)")

    # Starts from pretrained weights, same idea as transfer learning
    # with MobileNetV2 in the currency project.
    model = YOLO("yolo26n.pt")

    model.train(
        data="data.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        workers=2,
        project="runs",
        name="baseline",
    )

    # Final check on the held-out TEST split (not used during training).
    # Ultralytics prints a per-class table: Precision, Recall, mAP50, mAP50-95.
    print("\n=== Evaluation on TEST split ===")
    model.val(data="data.yaml", split="test")


# On Windows this guard is required. Without it, YOLO's data loader workers
# re-run this file and crash with a confusing "multiprocessing" error.
if __name__ == "__main__":
    main()
