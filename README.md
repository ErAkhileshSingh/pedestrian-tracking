
# 🧍‍♂️ Pedestrian Tracking with YOLOv8n

An AI-driven **real-time pedestrian detection and tracking system** built using **YOLOv8n** on a **custom dataset**.  
This project was trained and tested on **Google Colab** with the goal of achieving **high accuracy and fast inference** for real-world pedestrian tracking applications — such as vehicle-mounted or CCTV camera systems.

---

## 🎯 Objective

Follow this guide to train and deploy a YOLOv8n model for **pedestrian detection and tracking** using a **custom dataset**.  
The main objectives are:
- Build a lightweight, real-time pedestrian detection system.  
- Train YOLOv8n on a merged custom dataset created from multiple sources.  
- Compare its accuracy and inference speed with other YOLOv8 variants (YOLOv8s, YOLOv8m).  
- Achieve smooth, real-time tracking on live camera feeds.

---

## 📦 Features

✅ Custom dataset creation (merged from multiple pedestrian datasets)  
✅ Model training on Google Colab with YOLOv8  
✅ Real-time pedestrian detection & tracking  
✅ Live camera inference support  
✅ Model benchmarking across YOLOv8 variants  
✅ Accuracy and FPS evaluation

---

## 🧩 Project Structure

```

pedestrian-tracking/
│
├── app.py                  # Main application for detection & tracking
├── CustomTrainedModel.pt                 # Trained YOLOv8n model weights
├── requirements.txt        # Dependencies for local setup
├── packages.txt            # Extra packages for Google Colab setup
├── data.yaml               # Dataset configuration file
└── README.md               # Project documentation

````

---

## ⚙️ Requirements

Make sure you have the following installed:
```bash
Python >= 3.8
torch >= 2.2.0
torchvision >= 0.15.2
opencv-python == 4.8.1.78
pandas == 2.2.2
numpy == 1.26.4
matplotlib == 3.8.1
ultralytics == 8.3.0
````

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Train the Model (Google Colab)

1. **Clone the repository**

   ```bash
   !git clone https://github.com/ErAkhileshSingh/pedestrian-tracking.git
   %cd pedestrian-tracking
   ```

2. **Install dependencies**

   ```bash
   !pip install -r requirements.txt
   ```

3. **Upload your dataset**

   * Ensure your dataset is in YOLO format (`images/train`, `images/val`, `labels/train`, `labels/val`).
   * Update `data.yaml` with correct paths and class names.

4. **Train the model**

   ```bash
   !yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
   ```

5. **View training results**

   * Training metrics (loss, mAP, precision, recall) are saved in the `runs/` directory.

6. **Export your best weights**

   * The trained model will be saved as `runs/detect/train/weights/best.pt`.

---

## 🎥 Run Real-Time Detection & Tracking

Once trained, run the model locally or on Colab:

```bash
python app.py --weights best.pt --source 0
```

* `--source 0` → Use laptop/webcam
* Replace `0` with a video path for offline testing

Pedestrians will be detected and tracked with bounding boxes and unique IDs.

---

## 📊 Model Benchmarking

We benchmarked YOLOv8n, YOLOv8s, and YOLOv8m models using:

* **mAP@50**: Mean Average Precision for detection accuracy
* **FPS**: Frames per second (speed)
* **Latency**: Model inference time per frame

> The YOLOv8n model achieved the best trade-off between **speed and accuracy** for real-time pedestrian tracking.

---

## 🧠 Technical Workflow

1. **Dataset Creation** — Merged multiple pedestrian datasets into one consistent YOLO-format dataset.
2. **Model Training** — Fine-tuned YOLOv8n on the new dataset in Google Colab using GPU runtime.
3. **Evaluation** — Compared trained model with other YOLOv8 variants on validation data.
4. **Real-Time Tracking** — Implemented live inference and object tracking using OpenCV.

---

## 📁 Example Dataset Structure

```
datasets/
│
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
```

---

## 📸 Future Improvements

* Integrate DeepSORT or ByteTrack for smoother multi-object tracking
* Add pedestrian re-identification (Re-ID) module
* Optimize model for edge deployment (Jetson Nano, Raspberry Pi, etc.)
* Build a Streamlit web interface for live video uploads

---

## 👨‍💻 Author

**Akhilesh Singh**

---

