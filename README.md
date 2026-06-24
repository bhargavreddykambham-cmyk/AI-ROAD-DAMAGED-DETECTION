# 🛣️ AI Road Damage Detection System

> **Automated detection of Potholes, Cracks, Patches & Road Damage**  
> Powered by OpenCV Computer Vision + Streamlit Dashboard

---

## 📸 Screenshots

| Original Image | Detection Output |
|---|---|
| Raw road image | Annotated with bounding boxes + severity panel |

Detection classes:
- 🔴 **Pothole** — Dark concave depressions
- 🟠 **Crack** — Longitudinal/transverse cracks  
- 🟡 **Patch** — Previously repaired areas
- 🟣 **Road Damage** — General surface distress

---

## 🚀 Quick Start (5 minutes)

### Option A — Run Locally

```bash
# 1. Clone project
git clone <your-repo-url>
cd road_damage_ai

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run dashboard
streamlit run app.py
```

Open **http://localhost:8501** ✅

---

### Option B — Docker

```bash
docker build -t road-damage-ai .
docker run -p 8501:8501 road-damage-ai
```

Open **http://localhost:8501** ✅

---

## 📁 Project Structure

```
road_damage_ai/
│
├── app.py                    ← Main Streamlit dashboard (5 pages)
├── predict.py                ← CLI inference engine
├── train.py                  ← YOLOv8 training script
├── preprocessing.py          ← Image preprocessing pipeline
├── segmentation.py           ← Road segmentation & heatmaps
├── damage_analysis.py        ← Damage scoring & reports
│
├── data.yaml                 ← YOLO dataset config
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Docker deployment
│
├── .streamlit/
│   └── config.toml           ← Dark theme config
│
├── data/
│   ├── train/
│   │   ├── images/           ← Training images (.jpg/.png)
│   │   └── labels/           ← YOLO labels (.txt)
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── models/
│   └── best_road_damage.pt   ← Trained YOLO model (after training)
│
├── outputs/
│   ├── detections/           ← Saved annotated images/videos
│   ├── reports/              ← JSON damage reports
│   └── runs/                 ← YOLOv8 training runs
│
└── logs/
    ├── training.log
    └── predict.log
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | Overview, damage classes, severity scale |
| 🖼️ Image Detection | Upload image → detect → side-by-side view + charts |
| 🎬 Video Detection | Upload video → frame-by-frame analysis + timeline chart |
| 📊 Analytics | Pie chart, bar chart, line trend, road health gauge |
| 📋 About & Guide | Step-by-step user guide + FAQ |

---

## ⚙️ Detection Settings

| Setting | Default | Description |
|---|---|---|
| Confidence Threshold | 0.30 | Lower = more detections, higher = fewer but certain |
| Frame Skip (video) | 2 | Process every N frames — increase for faster video |

---

## 🏋️ Train Your Own Model

If you have road damage data, you can train a YOLO model for higher accuracy:

### 1. Prepare dataset
```bash
# Generate synthetic demo dataset
python preprocessing.py

# OR put your own images in:
# data/train/images/*.jpg
# data/train/labels/*.txt  (YOLO format: class cx cy w h)
```

### 2. Train
```bash
# Fast demo training (10 epochs)
python train.py --epochs 10 --model yolov8n.pt

# Full training
python train.py --epochs 100 --model yolov8m.pt --batch 16

# With GPU
python train.py --epochs 100 --device cuda --batch 32
```

### 3. Use trained model in app
Edit `app.py` — the app auto-loads `models/best_road_damage.pt` if present.

---

## 🔍 CLI Inference

```bash
# Single image
python predict.py --source road.jpg

# Video
python predict.py --source road_video.mp4 --save

# Camera (webcam)
python predict.py --source 0

# Folder of images
python predict.py --source ./test_images/ --save

# With custom model
python predict.py --source road.jpg --model models/best_road_damage.pt

# Higher confidence (fewer false positives)
python predict.py --source road.jpg --conf 0.50
```

---

## 📊 Output Files

After detection, these files are saved in `outputs/detections/`:

| File | Description |
|---|---|
| `*_detected.jpg` | Annotated image with bounding boxes |
| `*_result.json`  | Detection data (class, confidence, bbox, counts) |
| `*_detected.mp4` | Annotated video |

---

## 🗂️ Dataset Sources

For training a high-accuracy model, use these public datasets:

| Dataset | Classes | Link |
|---|---|---|
| **RDD2022** | Longitudinal/Transverse Crack, Alligator Crack, Pothole | Crowdsensing Road Damage Dataset |
| **IDD** | Road damage categories | Indian Driving Dataset |
| **Pothole-600** | Pothole | Research datasets |
| **CrackForest** | Various crack types | Research datasets |

---

## 🔧 Why No Vehicles Detected?

Previous versions used COCO-pretrained YOLO which detects 80 classes including motorcycles, cars, etc.

**This version uses OpenCV-only detection tuned specifically for road surfaces:**
- Pothole detection via dark-region thresholding + blob analysis
- Crack detection via Canny edges + elongated contour filtering
- Patch detection via colour difference in HSV space

If you train on a road-damage dataset (RDD2022), YOLO will also detect only road damage classes.

---

## 📦 Requirements

```
streamlit==1.28.0
opencv-python-headless==4.8.0.76
Pillow==10.0.0
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0

# Optional (for YOLO training/inference)
# ultralytics==8.0.196
```

---

## 🐳 Docker Commands

```bash
# Build
docker build -t road-damage-ai .

# Run
docker run -p 8501:8501 road-damage-ai

# With volume mount (for custom models)
docker run -p 8501:8501 -v $(pwd)/models:/app/models road-damage-ai

# Background
docker run -d -p 8501:8501 --name road-ai road-damage-ai
```

---

## 🔮 Future Enhancements

- [ ] GPS coordinate logging for each detection
- [ ] Export PDF maintenance report
- [ ] Road health scoring over time (MLflow tracking)
- [ ] Mobile app (Streamlit + camera)
- [ ] Integration with Google Maps / GIS
- [ ] Automatic work order generation
- [ ] Comparison before/after repair
- [ ] Multi-camera support

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/pothole-gps`
3. Commit changes: `git commit -m "Add GPS tagging"`
4. Push and open Pull Request

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👤 Credits

AI Road Damage Detection System  
Powered by **OpenCV** & **Streamlit**  
Model training via **Ultralytics YOLOv8**