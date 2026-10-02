"""
webcam_detect.py

Real-time face mask detection using the trained YOLO26n model.
Draws bounding boxes + labels directly on the live webcam feed.

Run with:
    python src/webcam_detect.py
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "../models/face_mask_yolo26n_best.pt"
CONFIDENCE_THRESHOLD = 0.4  # boxes below this confidence are hidden


def main():
    model = YOLO(MODEL_PATH)
    print("Classes:", model.names)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera index/permissions.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()  # draws boxes + labels automatically

        cv2.imshow("Face Mask Detection (press q to quit)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
