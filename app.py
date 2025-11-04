import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import time
import threading
import logging
import tempfile
from collections import defaultdict

import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Pedestrian Tracker", layout="wide")

# -------------------------------
# Helper Class
# -------------------------------
class PedestrianTracker:
    def __init__(self, model_path='best (2).pt', conf_threshold=0.3, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)

    def process_video(self, input_path, output_path, show_trails=True):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                tracker="bytetrack.yaml"
            )
            annotated_frame = self.visualize_tracks(frame, results[0], show_trails)
            out.write(annotated_frame)

        cap.release()
        out.release()
        return output_path

    def process_image(self, image, show_trails=True):
        results = self.model.track(
            image,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            tracker="bytetrack.yaml"
        )
        return self.visualize_tracks(image, results[0], show_trails)

    def visualize_tracks(self, frame, result, show_trails=True):
        annotated = frame.copy()
        if getattr(result, "boxes", None) is None or getattr(result.boxes, "id", None) is None:
            return annotated

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            color = self._get_color(track_id)
            label = f"ID:{track_id} ({conf:.2f})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            if show_trails and len(self.track_history[track_id]) > 1:
                pts = np.array(self.track_history[track_id], np.int32)
                cv2.polylines(annotated, [pts], False, color, 2)

        return annotated

    def _get_color(self, track_id):
        rng = np.random.RandomState(track_id)
        return tuple(int(x) for x in rng.randint(0, 255, 3))

# -------------------------------
# Load model once (cached)
# -------------------------------
@st.cache_resource
def load_model(model_path="best (2).pt"):
    logging.info("Loading model...")
    return PedestrianTracker(model_path=model_path)

tracker = load_model()

# -------------------------------
# WebRTC Transformer (robust)
# -------------------------------
class WebcamTransformer(VideoTransformerBase):
    def __init__(self, tracker, show_trails=True, process_every_n=2):
        self.tracker = tracker
        self.show_trails = show_trails
        self.process_every_n = process_every_n
        self.frame_count = 0
        self.lock = threading.Lock()

    def transform(self, frame):
        # Ensure we always return a valid ndarray to avoid component crashes
        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception as e:
            logging.exception("Failed to convert frame to ndarray")
            # return a black frame fallback with common resolution
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Throttle processing to reduce CPU/GPU load
        self.frame_count += 1
        if (self.frame_count % self.process_every_n) != 0:
            # draw a small indicator and return original frame
            cv2.putText(img, "skip", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            return img

        # Run detection + tracking safely
        try:
            with self.lock:
                results = self.tracker.model.track(
                    img,
                    persist=True,
                    conf=self.tracker.conf_threshold,
                    iou=self.tracker.iou_threshold,
                    classes=[0],
                    tracker="bytetrack.yaml"
                )
                annotated = self.tracker.visualize_tracks(img, results[0], self.show_trails)
            return annotated
        except Exception as e:
            logging.exception("Error during model.track")
            return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚶 Pedestrian Tracking System using YOLOv8 + ByteTrack")
st.write("Choose input type below 👇")

option = st.radio("Select input type", ["Webcam Live", "Upload Video", "Upload Image"])
show_trails = st.checkbox("Show Tracking Trails", value=True)

# -------------------------------
# Option: Webcam Live
# -------------------------------
if option == "Webcam Live":
    st.info("Webcam mode uses browser-based capture and works on Streamlit Cloud ✅")
    webrtc_streamer(
        key="webcam",
        video_transformer_factory=lambda: WebcamTransformer(tracker, show_trails, process_every_n=2),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# -------------------------------
# Option: Upload Video
# -------------------------------
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        start_btn = st.button("Start Processing ▶️")
        if start_btn:
            with st.spinner("Processing video... please wait ⏳"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
                tracker.process_video(tfile.name, out_path, show_trails=show_trails)

                st.success("✅ Processing complete!")
                st.video(out_path)

                with open(out_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video 🎥",
                        data=f,
                        file_name="pedestrian_tracked.mp4",
                        mime="video/mp4"
                    )

# -------------------------------
# Option: Upload Image
# -------------------------------
elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        start_btn = st.button("Start Detection 🖼️")
        if start_btn:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result_img = tracker.process_image(img, show_trails=show_trails)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)

st.markdown("---")
st.caption("Developed by Akhilesh Singh | YOLOv8 + ByteTrack 🚀")
