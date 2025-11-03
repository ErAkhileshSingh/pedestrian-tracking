import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

st.set_page_config(page_title="Pedestrian Tracker", layout="wide")

# -------------------------------
# Helper Class
# -------------------------------
class PedestrianTracker:
    def __init__(self, model_path='best (2).pt', conf_threshold=0.3, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(lambda: [])

    def process_video(self, input_path, output_path, show_trails=True):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

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

    def visualize_tracks(self, frame, result, show_trails=True):
        annotated = frame.copy()
        if result.boxes is None or result.boxes.id is None:
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
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚶 Pedestrian Tracking System using YOLOv8 + ByteTrack")
st.write("Upload a video and track pedestrians with unique IDs in real-time!")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
show_trails = st.checkbox("Show Tracking Trails", value=True)

# Load YOLO model only once
@st.cache_resource
def load_model():
    return PedestrianTracker(model_path="best (2).pt")

if uploaded_file is not None:
    with st.spinner("Processing video... please wait ⏳"):
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Output temp file
        out_path = os.path.join(tempfile.gettempdir(), "output.mp4")

        # Run tracker
        tracker = load_model()
        tracker.process_video(tfile.name, out_path, show_trails=show_trails)

        st.success("✅ Processing complete!")
        st.video(out_path)

        # Option to download result
        with open(out_path, "rb") as f:
            st.download_button(
                label="Download Processed Video 🎥",
                data=f,
                file_name="pedestrian_tracked.mp4",
                mime="video/mp4"
            )

st.markdown("---")
st.caption("Developed by Akhilesh Singh | YOLOv8 + ByteTrack 🚀")
