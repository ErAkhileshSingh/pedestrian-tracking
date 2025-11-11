import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "CustomTrainedModel.pt"           # change to your trained model
INPUT_PATH = "input/video.mp4"   # or input/image.jpg
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


# -----------------------------
# TRACKER CLASS
# -----------------------------
class PedestrianTracker:
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)

    def process_video(self, input_path, output_path, show_trails=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video: {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[INFO] Processing video: {input_path}")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],  # class 0 = person
                tracker="bytetrack.yaml"
            )

            annotated_frame = self.visualize_tracks(frame, results[0], show_trails)
            out.write(annotated_frame)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"[DONE] Tracked video saved at: {output_path}")

    def process_image(self, input_path, output_path, show_trails=True):
        image = cv2.imread(input_path)
        if image is None:
            print(f"[ERROR] Unable to read image: {input_path}")
            return

        results = self.model.track(
            image,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            tracker="bytetrack.yaml"
        )
        annotated = self.visualize_tracks(image, results[0], show_trails)
        cv2.imwrite(output_path, annotated)
        print(f"[DONE] Tracked image saved at: {output_path}")

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

            # Draw trail
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


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    tracker = PedestrianTracker(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)

    ext = os.path.splitext(INPUT_PATH)[-1].lower()
    output_file = os.path.join(OUTPUT_DIR, f"tracked_output{ext}")

    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        tracker.process_video(INPUT_PATH, output_file, show_trails=True)
    elif ext in [".jpg", ".jpeg", ".png"]:
        tracker.process_image(INPUT_PATH, output_file, show_trails=True)
    else:
        print("[ERROR] Unsupported file format. Use video (.mp4, .avi) or image (.jpg, .png).")
