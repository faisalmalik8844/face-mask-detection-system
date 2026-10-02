"""Browser-based live face-mask detection with Streamlit and WebRTC."""

import time
from pathlib import Path
from threading import Lock

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = (
    ROOT_DIR / "models" / "face_mask_yolo26n_best.pt",
    ROOT_DIR / "models" / "face_mask_yolo26n.pt",
)


class AlertSignal:
    """Thread-safe alert event counter with a repeat cooldown."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._event_id = 0
        self._last_alert_at = 0.0

    def notify(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_alert_at >= 6.0:
                self._event_id += 1
                self._last_alert_at = now

    def current_event_id(self) -> int:
        with self._lock:
            return self._event_id


st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
)

ALERT_SOUND_COMPONENT = st.components.v2.component(
        "face_mask_alert_sound",
        html="""
        <div class="sound-control">
            <button type="button">Enable alert sound</button>
            <span role="status">Sound is off. Enable it before starting detection.</span>
        </div>
        <style>
            .sound-control {display:flex; align-items:center; gap:.75rem; flex-wrap:wrap;}
            button {border:0; border-radius:8px; padding:.55rem .9rem; cursor:pointer;
                            background:#176b87; color:white; font:inherit;}
            span {font-size:.9rem;}
        </style>
        """,
        js="""
        export default function (component) {
            const { data, parentElement } = component
            const button = parentElement.querySelector("button")
            const status = parentElement.querySelector("[role=status]")
            if (!button || !status) return

            if (parentElement.__maskLastAlertId === undefined) {
                parentElement.__maskLastAlertId = Number(data?.event_id ?? 0)
            } else {
                const previous = Number(parentElement.__maskLastAlertId)
                const current = Number(data?.event_id ?? 0)
                if (current > previous && parentElement.__maskAlertAudioEnabled) {
                    const audio = parentElement.__maskAudioContext
                    if (audio?.state === "running") {
                        const oscillator = audio.createOscillator()
                        const gain = audio.createGain()
                        const start = audio.currentTime + 0.03
                        oscillator.type = "sine"
                        oscillator.frequency.value = 880
                        oscillator.connect(gain)
                        gain.connect(audio.destination)
                        gain.gain.setValueAtTime(0, start)
                        for (let offset = 0; offset < 3; offset += 0.5) {
                            gain.gain.setValueAtTime(0.16, start + offset)
                            gain.gain.setValueAtTime(0, start + offset + 0.22)
                        }
                        oscillator.start(start)
                        oscillator.stop(start + 3)
                    }
                }
                parentElement.__maskLastAlertId = current
            }

            button.textContent = parentElement.__maskAlertAudioEnabled
                ? "Mute alert sound"
                : "Enable alert sound"
            status.textContent = parentElement.__maskAlertAudioEnabled
                ? "Sound enabled. Alerts beep for 3 seconds."
                : "Sound is off. Enable it before starting detection."

            button.onclick = async () => {
                if (parentElement.__maskAlertAudioEnabled) {
                    parentElement.__maskAlertAudioEnabled = false
                    await parentElement.__maskAudioContext?.suspend()
                } else {
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext
                    if (!AudioContextClass) {
                        status.textContent = "This browser does not support audio alerts."
                        return
                    }
                    parentElement.__maskAudioContext ??= new AudioContextClass()
                    await parentElement.__maskAudioContext.resume()
                    parentElement.__maskAlertAudioEnabled =
                        parentElement.__maskAudioContext.state === "running"
                }
                button.textContent = parentElement.__maskAlertAudioEnabled
                    ? "Mute alert sound"
                    : "Enable alert sound"
                status.textContent = parentElement.__maskAlertAudioEnabled
                    ? "Sound enabled. Alerts beep for 3 seconds."
                    : "Sound is off. Enable it before starting detection."
            }
        }
        """,
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.5rem;}
      .hero {padding: 1.6rem 1.8rem; border-radius: 18px;
             background: linear-gradient(120deg, #102a43, #176b87); color: white;
             margin-bottom: 1.4rem;}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2.25rem;}
      .hero p {margin: 0; opacity: .88; font-size: 1.05rem;}
      [data-testid="stMetric"] {background: #f4f8fb; padding: 1rem;
                                  border: 1px solid #e1eaf0; border-radius: 12px;}
    </style>
    <div class="hero">
      <h1>😷 Face Mask Detection</h1>
      <p>Live browser-camera detection for Mask, Mask Incorrect, and No Mask.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the detection model…")
def load_model(model_path: str) -> YOLO:
    """Load and cache YOLO weights across Streamlit reruns."""
    return YOLO(model_path)


def has_alert_detection(prediction) -> bool:
    """Return true if a result contains an unmasked or incorrectly masked face."""
    if prediction.boxes is None or len(prediction.boxes) == 0:
        return False

    class_ids = prediction.boxes.cls.int().tolist()
    labels = [
        str(prediction.names.get(class_id, "")).lower().replace("_", " ").strip()
        for class_id in class_ids
    ]
    return any(label in {"no mask", "mask incorrect", "incorrect mask"} for label in labels)


class MaskDetectionProcessor(VideoProcessorBase):
    """Run YOLO on each frame received from the browser webcam."""

    def __init__(self, model: YOLO, confidence: float, alert_signal: AlertSignal) -> None:
        self.model = model
        self.confidence = confidence
        self.alert_signal = alert_signal
        self.model_lock = Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with self.model_lock:
            prediction = self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
            )[0]
        if has_alert_detection(prediction):
            self.alert_signal.notify()
        annotated = prediction.plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Predictions below this confidence are hidden.",
    )
    st.caption("Allow camera access in your browser when prompted.")
    st.caption("The alert beeps for 3 seconds, then waits 6 seconds before repeating.")

if "alert_signal" not in st.session_state:
    st.session_state.alert_signal = AlertSignal()
alert_signal = st.session_state.alert_signal

model_path = next((path for path in MODEL_CANDIDATES if path.is_file()), None)
if model_path is None:
    st.error(
        "No model weights found. Add face_mask_yolo26n_best.pt or "
        "face_mask_yolo26n.pt to the project's models folder."
    )
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as error:
    st.error(f"Could not load the YOLO model: {error}")
    st.stop()


@st.fragment(run_every="1s")
def alert_sound_control() -> None:
    """Poll alert events and show the sound control above the camera."""
    st.subheader("Alert sound")
    ALERT_SOUND_COMPONENT(
        key="mask-alert-sound",
        data={"event_id": alert_signal.current_event_id()},
    )


st.subheader("Live camera")
st.write("Tap **START** below and allow camera access when your browser asks.")
st.caption("If the stream does not connect, check camera permission or try another network.")
stream_context = webrtc_streamer(
    key="face-mask-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: MaskDetectionProcessor(model, confidence, alert_signal),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
        ]
    },
    async_processing=True,
)
if stream_context.video_processor is not None:
    with stream_context.video_processor.model_lock:
        stream_context.video_processor.confidence = confidence

if stream_context.state.playing:
    st.success("Camera is running. Tap **STOP** to end the live detection.")
else:
    st.caption("Check browser camera permission if the camera cannot start.")

alert_sound_control()

st.subheader("About this model")
st.metric("Confidence threshold", f"{confidence:.0%}")
st.markdown(
    """
    The model detects three classes:
    - **Mask**
    - **Mask Incorrect**
    - **No Mask**

    This is an informational demo, not a reliable system for health,
    safety, or enforcement decisions.
    """
)
st.caption(f"Weights: `{model_path.relative_to(ROOT_DIR).as_posix()}`")

st.info(
    "For local use, open this app at `http://localhost:8501`. Remote hosting "
    "requires HTTPS for browser camera access and may require WebRTC/STUN/TURN "
    "network configuration."
)