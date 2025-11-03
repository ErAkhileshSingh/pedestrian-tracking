import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile

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

    def process_webcam(self, show_trails=True):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
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
            annotated = self.visualize_tracks(frame, results[0], show_trails)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            if not st.session_state["running"]:
                break
        cap.release()

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
st.write("Choose input type below 👇")

option = st.radio("Select input type", ["Webcam Live", "Upload Video", "Upload Image"])
show_trails = st.checkbox("Show Tracking Trails", value=True)

@st.cache_resource
def load_model():
    return PedestrianTracker(model_path="best (2).pt")

tracker = load_model()

# Initialize session state for webcam
if "running" not in st.session_state:
    st.session_state["running"] = False

# -------------------------------
# Option: Webcam Live
# -------------------------------
if option == "Webcam Live":
    start_btn = st.button("Start Webcam Tracking 🎥")
    stop_btn = st.button("Stop Webcam")

    if start_btn:
        st.session_state["running"] = True
        tracker.process_webcam(show_trails=show_trails)
    elif stop_btn:
        st.session_state["running"] = False

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
            img = cv2.imdecode(file_bytes, 1)
            result_img = tracker.process_image(img, show_trails=show_trails)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)

st.markdown("---")
st.caption("Developed by Akhilesh Singh | YOLOv8 + ByteTrack 🚀")
