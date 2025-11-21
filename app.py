import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from ultralytics import YOLO


# -----------------------------
# CONFIGURATION
# -----------------------------
# MODEL_PATH = r"C:\Users\91941\Downloads\best (3).pt"  
MODEL_PATH = "CustomTrainedModel.pt"  # Use the model in current directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


# -----------------------------
# FILE PICKER
# -----------------------------
def choose_file(file_types):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=file_types)
    return file_path


# -----------------------------
# MAIN INPUT SELECTION
# -----------------------------
def choose_input_source():
    print("\nSelect Input Type:")
    print("1. Image")
    print("2. Video")
    print("3. Webcam")
    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        print("\nChoose Image File...")
        img_file = choose_file([("Images", "*.jpg *.jpeg *.png")])
        return "image", img_file

    elif choice == "2":
        print("\nChoose Video File...")
        vid_file = choose_file([("Videos", "*.mp4 *.avi *.mkv *.mov")])
        return "video", vid_file

    elif choice == "3":
        print("\nSelect Webcam:")
        print("0 = Laptop Camera")
        print("1 = External USB Camera")
        print("2 = Other Camera (if connected)")
        cam_index = int(input("Enter camera index: "))
        return "webcam", cam_index

    else:
        print("[ERROR] Invalid choice")
        exit()


# -----------------------------
# TRACKER CLASS (same as yours)
# -----------------------------
class PedestrianTracker:
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)
        
        # Print model info for debugging
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Model names: {self.model.names}")

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
                frame, persist=True, conf=self.conf_threshold,
                iou=self.iou_threshold, tracker="bytetrack.yaml"
            )

            annotated_frame = self.visualize_tracks(frame, results[0], show_trails)
            out.write(annotated_frame)

            frame_count += 1
            if frame_count % 10 == 0:
                detected = len(results[0].boxes) if results[0].boxes else 0
                print(f"Processed {frame_count} frames... (detected {detected} objects in last frame)")

        cap.release()
        out.release()
        print(f"[DONE] Tracked video saved: {output_path}")

    def process_webcam(self, cam_index):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("[ERROR] Unable to open webcam.")
            return

        print(f"[INFO] Using webcam index: {cam_index}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame, persist=True, conf=self.conf_threshold,
                iou=self.iou_threshold, tracker="bytetrack.yaml"
            )

            annotated = self.visualize_tracks(frame, results[0], show_trails=True)
            detected = len(results[0].boxes) if results[0].boxes else 0
            print(f"Detected: {detected} objects")
            cv2.imshow("Webcam Tracking", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, input_path, output_path, show_trails=True):
        image = cv2.imread(input_path)
        if image is None:
            print(f"[ERROR] Unable to read image: {input_path}")
            return

        results = self.model.track(
            image, persist=True, conf=self.conf_threshold,
            iou=self.iou_threshold, tracker="bytetrack.yaml"
        )

        detected = len(results[0].boxes) if results[0].boxes else 0
        print(f"[INFO] Detected {detected} objects in image")
        
        annotated = self.visualize_tracks(image, results[0], show_trails)
        cv2.imwrite(output_path, annotated)
        print(f"[DONE] Tracked image saved at: {output_path}")

    def visualize_tracks(self, frame, result, show_trails=False):
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


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":

    tracker = PedestrianTracker(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)

    input_type, input_value = choose_input_source()

    # Handle Webcam
    if input_type == "webcam":
        tracker.process_webcam(input_value)

    # Handle Image
    elif input_type == "image":
        out_path = os.path.join(OUTPUT_DIR, "tracked_image.jpg")
        tracker.process_image(input_value, out_path)

    # Handle Video
    elif input_type == "video":
        ext = os.path.splitext(input_value)[-1]
        out_path = os.path.join(OUTPUT_DIR, f"tracked_video{ext}")
        tracker.process_video(input_value, out_path)
