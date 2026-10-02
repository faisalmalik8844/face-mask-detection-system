# Face Mask Detection (Real-Time)

A real-time object detection system that identifies whether people in a
webcam feed are wearing a mask correctly, not wearing one, or wearing one
incorrectly — built with YOLO26n.

## Problem Statement

Unlike a simple image classifier, this task requires finding *where* faces
are in a frame before judging each one, since a single frame can contain
multiple people. YOLO handles both steps (detection + classification) in
a single pass, which is why it was chosen over a two-stage
face-detector-plus-classifier pipeline.

## Dataset

- Source: Roboflow-exported "Face Mask Dataset" (3 classes: Mask, Mask
  Incorrect, No Mask), already in YOLO format
- Train: 1,742 images / 5,964 box instances
- Validation: 499 images / 1,662 box instances
- Test: 248 images / 896 box instances
- **Class imbalance:** Mask (80%), No Mask (15%), Mask Incorrect (4.7%) —
  present across all three splits consistently

## Model & Training

- Model: YOLO26n (nano), pretrained weights fine-tuned via transfer learning
- 50 epochs, image size 640, batch size 16, trained on a Colab T4 GPU

### Training Results (honest account)

| Metric | Value |
|---|---|
| Test mAP50 (all classes) | 0.639 |
| Test mAP50 — Mask | 0.725 |
| Test mAP50 — Mask Incorrect | 0.618 |
| Test mAP50 — No Mask | 0.575 |

See `reports/figures/training_history.png` for full epoch-by-epoch curves.

**Key finding:** validation mAP plateaued by epoch 15 (~0.64) and never
meaningfully improved through epoch 50, peaking at 0.687 (epoch 32).
Classification loss shows mild overfitting onset around epoch 36 — training
loss kept falling while validation loss flattened and crept back up
slightly. This means further training epochs would not help; the model
reached its practical ceiling for this resolution/architecture/data
combination.

**Contrary to the initial hypothesis:** performance did not track class
size as expected. `Mask` (80% of data) scored highest as expected, but
`Mask Incorrect` (4.7% of data) actually scored higher than `No Mask`
(15% of data) — though with only 36 test instances for Mask Incorrect,
this comparison carries real uncertainty and shouldn't be over-interpreted.

**Safety-relevant observation:** recall on `No Mask` (0.598) means roughly
40% of unmasked faces are missed or misclassified. Given this is the
highest-stakes error direction for this use case, this is flagged as the
primary limitation rather than the originally-suspected class-imbalance
issue.

## Repository Structure

```
FaceMaskDetection/
├── data/
│   ├── raw/              # Original dataset (not committed - see .gitignore)
│   └── processed/
├── notebooks/
│   └── train_yolo_colab.ipynb    # Full training notebook (Colab)
├── src/
│   ├── convert_voc_to_yolo.py    # (not used - dataset came pre-converted)
│   ├── inspect_yolo_dataset.py   # Class balance EDA script
│   └── train_yolo.py             # Local training script (alternative to Colab)
├── models/
│   └── face_mask_yolo26n.pt      # Trained weights
├── app/
│   ├── streamlit_app.py          # Browser-based live detection app
│   └── webcam_detect.py          # Desktop webcam detection script
├── reports/
│   └── figures/
│       └── training_history.png  # mAP/loss curves across 50 epochs
├── tests/
├── requirements.txt
└── README.md
```

## How to Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Open the displayed local URL (usually `http://localhost:8501`) and allow
camera access when your browser prompts you. Use the sidebar to adjust the
detection confidence threshold. Click **Enable alert sound** before starting
the camera to allow the browser to play a 3-second beep when `No Mask` or
`Mask Incorrect` is detected. If a flagged detection continues, the alert
repeats no more often than once every six seconds. The browser camera app requires HTTPS when
accessed remotely; webcam permissions are not passed through from the host
machine to a remote browser session.

The app defaults to **Capture a snapshot**, which uses the browser camera to
capture one frame and avoids WebRTC network restrictions. Choose **Live webcam
stream** for continuous video; some hosted/mobile networks require TURN settings
below for live streaming to work.

To run the standalone desktop webcam script instead:

```bash
python app/webcam_detect.py
```
Press `q` to quit its video window.

### Deploy to Streamlit Community Cloud

The app can be deployed from a GitHub repository:

1. Push this project to a GitHub repository, including `app/streamlit_app.py`,
  `app/requirements.txt`, and the model weights in `models/`.
2. In Streamlit Community Cloud, create an app from that repository and set
  the main file to `app/streamlit_app.py`.
  In Advanced settings, choose Python 3.12 (rather than 3.14).
3. Deploy and open the generated HTTPS URL, then grant browser camera access.

If the hosted camera shows a WebRTC connection timeout or a black video, add
TURN relay credentials in the app's **Manage app → Settings → Secrets**. Use
the values provided by your TURN service:

```toml
TURN_SERVER_URLS = "turn:your-turn-host:3478?transport=udp,turns:your-turn-host:5349?transport=tcp"
TURN_USERNAME = "your-turn-username"
TURN_CREDENTIAL = "your-turn-credential"
```

Keep real credentials private; do not commit them to GitHub. STUN is included
as a best-effort fallback, but some cellular, school, office, and restrictive
Wi-Fi networks require TURN to relay the video stream.

The browser's camera is streamed to the server for inference. Public hosting
means camera frames are processed by the deployed app's server; use only with
consent. The included public STUN server helps establish WebRTC connections,
but restrictive networks may still need TURN configuration.

## Limitations

- Recall on `No Mask` (~60%) means real-world use would miss a meaningful
  fraction of unmasked faces — not reliable enough for enforcement, only
  suitable as an assistive/informational tool
- `Mask Incorrect` evaluation rests on a small test set (36 instances) and
  should be treated as a rough estimate, not a precise number
- Model plateaued at 640px input resolution; group photos with many small
  faces are a likely source of missed detections, untested directly

## Future Improvements

- [ ] Test `imgsz=960` to check whether higher resolution helps with small/
      distant faces (direct test of the current leading hypothesis)
- [ ] Try `yolo26s` (small variant) for more model capacity
- [ ] Manually review a sample of training labels for annotation quality
- [ ] Add more `Mask Incorrect` examples from a supplementary dataset
- [ ] Deploy as a `streamlit-webrtc` app for browser-based live detection

## Author

Malik Faisal Mukhtar — ZYNVEX-CERT-1130
