# ▶️ Execution Guide — AI Road Damage Detection

---

## 🚀 Run Dashboard (Main App)

```bash
# Activate venv first
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Launch
streamlit run app.py
```

Browser opens at **http://localhost:8501**

---

## 🖼️ Detect Damage in Image (CLI)

```bash
# Basic
python predict.py --source path/to/road.jpg

# With save
python predict.py --source path/to/road.jpg --save

# High confidence only
python predict.py --source road.jpg --conf 0.50

# With trained YOLO model
python predict.py --source road.jpg --model models/best_road_damage.pt
```

**Output saved to:** `outputs/detections/road_detected.jpg`

---

## 🎬 Detect Damage in Video (CLI)

```bash
# Basic
python predict.py --source road_video.mp4

# Save annotated video
python predict.py --source road_video.mp4 --save

# Faster processing (skip every 3 frames)
python predict.py --source road_video.mp4 --skip 3 --save

# No display window (server mode)
python predict.py --source road_video.mp4 --save --no-display
```

---

## 📹 Real-Time Camera Detection

```bash
# Default camera (index 0)
python predict.py --source 0

# Second camera
python predict.py --source 1
```

Press **Q** to quit.

---

## 🏋️ Train YOLOv8 Model

```bash
# Step 1: Generate synthetic training data (demo)
python preprocessing.py

# Step 2: Train (nano model, fast)
python train.py --model yolov8n.pt --epochs 50 --batch 16

# Step 3: Train (medium model, better accuracy)
python train.py --model yolov8m.pt --epochs 100 --batch 8

# Step 4: Train with GPU
python train.py --model yolov8m.pt --epochs 100 --device cuda --batch 32

# Resume interrupted training
python train.py --resume

# Hyperparameter search
python train.py --model yolov8n.pt --epochs 10 --batch 16
```

Best model saved to: **`models/best_road_damage.pt`**

---

## 📊 Generate Damage Analysis Report

```python
from damage_analysis import DamageAnalyzer

analyzer = DamageAnalyzer()

# Analyse detections
detections = [
    {"label":"Pothole","conf":0.87,"x1":100,"y1":300,"x2":200,"y2":380},
    {"label":"Crack",  "conf":0.72,"x1":250,"y1":400,"x2":420,"y2":420},
]
report = analyzer.analyse(detections, image_shape=(640,640,3), source="road.jpg")

print(report.to_json())
print("Severity:", report.severity_level)
print("Recommendations:")
for r in report.recommendations:
    print(" ", r)
```

---

## 🗺️ Segmentation & Heatmap

```python
import cv2
from segmentation import RoadSegmenter

seg = RoadSegmenter()
img = cv2.imread("road.jpg")

# Segment road area
mask    = seg.segment_road(img)
overlay = seg.overlay_mask(img, mask)

# Generate damage heatmap
detections = [{"label":"Pothole","conf":0.85,"x1":100,"y1":300,"x2":220,"y2":390}]
heatmap = seg.generate_damage_heatmap(img, detections)

cv2.imwrite("road_mask.jpg", overlay)
cv2.imwrite("road_heatmap.jpg", heatmap)
```

---

## 🔬 Preprocessing Pipeline

```python
from preprocessing import ImagePreprocessor, DatasetBuilder
import cv2

# Preprocess one image
proc = ImagePreprocessor()
img  = cv2.imread("road.jpg")
enhanced, meta = proc.preprocess(img)
print("Output shape:", meta["output_shape"])

# Generate synthetic dataset
db = DatasetBuilder("./data")
db.generate_synthetic({"train":60,"valid":15,"test":10})
stats = db.validate()
print("Dataset:", stats)
```

---

## 📦 Full Workflow (End to End)

```bash
# 1. Setup
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate demo dataset
python preprocessing.py

# 3. (Optional) Train YOLO model
python train.py --epochs 30 --model yolov8n.pt

# 4. Run CLI detection on test image
python predict.py --source data/test/images/road_test_0000.jpg --save

# 5. Launch full dashboard
streamlit run app.py
```

---

## ⚙️ Configuration

Edit these values in `app.py` sidebar or pass as CLI args:

| Parameter | Default | Effect |
|---|---|---|
| `--conf` | 0.30 | Confidence threshold |
| `--model` | None | Path to trained .pt file |
| `--skip` | 1 | Frame skip for video |
| `--save` | True | Save results to disk |
| `--output` | outputs/detections | Output directory |

---

## 📂 Where Files Are Saved

```
outputs/
├── detections/
│   ├── road_detected.jpg      ← Annotated image
│   ├── road_result.json       ← Detection JSON
│   └── road_video_detected.mp4
└── reports/
    └── report_1234567890.json ← Damage analysis report
```

---

## 🛑 Stop the App

```bash
# In terminal running streamlit
Ctrl + C

# Kill port 8501 (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Kill port 8501 (Mac/Linux)
lsof -ti:8501 | xargs kill
```