"""
webcam_detect.py

Real-time face mask detection from your webcam.
Run this directly - it's a plain script, not a Streamlit app, since
Streamlit's camera widget only takes single snapshots, not a live feed.

Usage:
    python webcam_detect.py
Press 'q' to quit.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_mask_yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.4  # boxes below this confidence are hidden


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))
    print("Class names:", model.names)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check it isn't in use by another app.")
        return

    print("Webcam started. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, stopping.")
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()  # draws boxes + class + confidence automatically

        cv2.imshow("Face Mask Detection - press q to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
