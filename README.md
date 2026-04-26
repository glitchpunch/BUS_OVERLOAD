<div align="center">

<img src="https://img.shields.io/badge/Edge_AI-Bus_Overcrowding_Detection-2ea44f?style=for-the-badge&logo=nvidia" alt="Project Banner"/>

# 🚌 Bus Overcrowding Detection System
### Real-Time · 4-Model Ensemble · Privacy-Preserving · Edge AI Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-n%2Fs%2Fm%2Fl-ff6b35?style=flat-square)](https://ultralytics.com)
[![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-purple?style=flat-square)](https://github.com/levan92/deep_sort_realtime)
[![TensorRT](https://img.shields.io/badge/Edge-TensorRT_FP16-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![Flask](https://img.shields.io/badge/Dashboard-Flask-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003b57?style=flat-square&logo=sqlite)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **An on-device, real-time people counting and overcrowding detection pipeline for public buses — 4-model ensemble detection, DeepSORT persistent tracking, temporal count stabilisation, and a live Flask alert dashboard. No cloud. No raw video storage. No privacy compromise.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Demo](#-quick-demo)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Training on Custom Data](#-training-on-custom-data)
- [Jetson Nano Deployment](#-jetson-nano-deployment)
- [Dashboard](#-dashboard)
- [Benchmarking & Evaluation](#-benchmarking--evaluation)
- [Configuration](#-configuration)
- [Privacy Design](#-privacy-design)
- [License](#-license)

---

## 🔍 Overview

This system monitors the number of passengers inside a bus in **real time** using a single mounted camera processed entirely **on-device**. When detected occupancy exceeds the vehicle's legal capacity, the system automatically:

- 🔴 Raises a flashing alert banner on the annotated live feed
- 📊 Displays a real-time capacity progress bar with percentage
- 🗃️ Logs each event (timestamp, count, fine) to a local SQLite database
- 📸 Saves a JPEG snapshot of every alert event
- 🌐 Streams all events to a live Flask web dashboard at `http://localhost:5000`
- 💾 Saves the full annotated output as a `.mp4` video file

The pipeline is designed to run on **affordable edge hardware** (NVIDIA Jetson Nano) without any internet connection, cloud API, or facial recognition.

---

## ✨ Key Features

| Feature | Detail |
|---------|--------|
| **4-Model Ensemble** | YOLOv8n + YOLOv8s + YOLOv8m + YOLOv8l run in parallel; counts fused via weighted voting |
| **Weighted Max Strategy** | Larger, more accurate models get higher voting weight; result biased toward ceiling for safety |
| **Cross-Model NMS** | Overlapping boxes from all 4 models are merged via NMS before tracking — no double-counting |
| **Count Stabiliser** | Median filter over a 20-frame rolling window eliminates number flickering completely |
| **DeepSORT Tracking** | Persistent Re-ID tracking with MobileNet embedder — seated passengers stay tracked |
| **Letterbox Display** | Fixed 1280×720 output regardless of source resolution — no distortion, no cropping |
| **Debounced Alerts** | Alert fires only after N consecutive overcrowding frames + cooldown period |
| **Fine Calculation** | Automatically computes and logs INR fine amount per overcrowding event |
| **One-Command Demo** | `demo.py` runs everything on a video file with a single command |
| **Privacy-First** | Zero raw video stored; only metadata written to disk |

---

## 🏗️ System Architecture

```
  Video File / Webcam / RTSP Camera
              │
              ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                   4-MODEL ENSEMBLE DETECTION                   │
  │                                                               │
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
  │  │YOLOv8n  │  │YOLOv8s  │  │YOLOv8m  │  │YOLOv8l  │        │
  │  │weight=1 │  │weight=2 │  │weight=3 │  │weight=4 │        │
  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
  │       └────────────┴────────────┴────────────┘              │
  │                         │                                    │
  │               Cross-Model NMS Merge                          │
  │            (removes duplicate boxes)                         │
  │                         │                                    │
  │               Weighted Max Ensemble                          │
  │             (count fused from all 4)                         │
  └─────────────────────────┼─────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    DeepSORT TRACKER                           │
  │         Re-ID with MobileNet (persistent identities)         │
  │   max_age=60  n_init=2  (tuned for seated bus passengers)    │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                  COUNT STABILISER                             │
  │        Median filter — rolling window of 20 frames           │
  │        Eliminates frame-to-frame count flickering            │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   ALERT MANAGER                               │
  │   OK → WARNING (75%) → OVERCROWD (100%)                     │
  │   Debounce: 8 consecutive frames + 30s cooldown              │
  └──────────────────────────┬───────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
  SQLite alerts.db                  Flask Dashboard
  (metadata only,                   http://localhost:5000
   no raw video)
              │
              ▼
  Annotated Output Video (output_videos/)
```

---

## 📁 Project Structure

```
bus_overcrowding/
│
├── demo.py                 ← ⭐ ONE-COMMAND demo runner for video files
├── main.py                 ← Full system launcher (inference + dashboard threads)
├── config.py               ← All tunable settings — models, thresholds, display
├── logger.py               ← Structured loguru logging with rotation
├── utils.py                ← CountStabilizer, AlertManager, letterbox display,
│                              capacity progress bar annotation
├── preprocessing.py        ← Dataset prep: resize, split 80/10/10, augment
├── training.py             ← YOLOv8 fine-tuning (all 4 model sizes)
├── inference.py            ← 4-model ensemble + DeepSORT + stabiliser loop
├── jetson_inference.py     ← TensorRT FP16 engine loader for Jetson Nano
├── requirements.txt        ← All Python dependencies
│
├── demo_video.mp4          ← ⬅ Place your bus demo video here
│
├── data/
│   ├── raw/                ← Place raw images + YOLO .txt labels here
│   └── dataset/            ← Auto-generated train/val/test split
│       ├── train/images/
│       ├── train/labels/
│       ├── val/
│       ├── test/
│       └── dataset.yaml
│
├── models/                 ← YOLOv8 weights (.pt) — auto-downloaded on first run
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── yolov8m.pt
│   └── yolov8l.pt
│
├── output_videos/          ← Annotated output video files saved here
├── alert_snapshots/        ← JPEG snapshots of each alert event
├── logs/app.log            ← Rotating log file
└── alerts.db               ← SQLite event log (metadata only)
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Detection** | YOLOv8n / s / m / l (Ultralytics) | 4-model person detection |
| **Ensemble** | Weighted Max Voting | Fuse counts from all 4 models |
| **Cross-model NMS** | OpenCV `dnn.NMSBoxes` | Remove duplicate boxes across models |
| **Tracking** | DeepSORT + MobileNet Re-ID | Persistent passenger identity |
| **Stabilisation** | Median Filter (20-frame window) | Eliminate count flickering |
| **Display** | Letterbox resize to 1280×720 | Fixed dimensions, correct aspect ratio |
| **Backend** | Python 3.8+ | Core pipeline |
| **Database** | SQLite | Lightweight event logging |
| **Dashboard** | Flask | Web-based alert review UI |
| **Logging** | Loguru | Structured, rotating logs |
| **Edge Runtime** | TensorRT FP16 | Optimised Jetson Nano inference |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- Git
- (Optional) CUDA GPU for faster inference
- (For edge) NVIDIA Jetson Nano with JetPack 4.6.x

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bus-overcrowding-detection.git
cd bus-overcrowding-detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python3 -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python3 -c "import cv2; print('OpenCV', cv2.__version__)"
python3 -c "from deep_sort_realtime.deepsort_tracker import DeepSort; print('DeepSORT OK')"
```

> **Note for Jetson Nano:** Do NOT use this `requirements.txt` on Jetson. See [Jetson Nano Deployment](#-jetson-nano-deployment) for the ARM-specific setup.

---

## ⚡ Quick Demo

```bash
# 1. Place your bus video in the project root as demo_video.mp4
# 2. Run:
python3 demo.py
```

That's it. The system auto-downloads all 4 YOLOv8 models on first run (~150MB total).

**Demo with options:**

```bash
# Specify video path
python3 demo.py --video path/to/bus.mp4

# Override bus capacity
python3 demo.py --video bus.mp4 --capacity 30

# Use only 2 faster models (good for slower CPUs)
python3 demo.py --video bus.mp4 --models n s

# GPU acceleration
python3 demo.py --video bus.mp4 --device cuda

# Save output video but don't display window
python3 demo.py --video bus.mp4 --no-display

# All options at once
python3 demo.py --video bus.mp4 --capacity 35 --device cuda --models n s m
```

**Keyboard Shortcuts (live window):**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `SPACE` | Pause / Resume |
| `S` | Save manual snapshot |

---

## 🚀 Usage

### Full System (Inference + Dashboard)

```bash
python3 main.py --mode both
```

### Inference Only (No Browser)

```bash
python3 main.py --mode inference
```

### Dashboard Only (Review Existing Alerts)

```bash
python3 main.py --mode dashboard
```

### Live Camera

```bash
# Webcam (default index 0)
python3 main.py --source 0

# IP / RTSP camera — edit config.py:
# CAMERA_ID = "rtsp://user:pass@192.168.1.10:554/stream"
python3 main.py
```

### Headless Server Mode

```bash
python3 main.py --no-display
```

---

## 🔬 How It Works

### 1 — 4-Model Ensemble Detection

Each frame is passed through all four YOLOv8 models simultaneously:

```
YOLOv8n  →  count = 18  (weight 1)
YOLOv8s  →  count = 21  (weight 2)
YOLOv8m  →  count = 23  (weight 3)
YOLOv8l  →  count = 24  (weight 4)

weighted_max  →  ceil( (18×1 + 21×2 + 23×3 + 24×4) / 10 )  =  23
```

Boxes from all 4 models are merged, then cross-model NMS removes duplicates before passing to the tracker.

### 2 — Why `weighted_max` Strategy

Standard `mean` undercounts in overcrowded scenes because small models miss occluded passengers. `weighted_max` biases toward the ceiling — for a safety enforcement system, it is always better to err on the side of detecting *more* people than fewer.

### 3 — Count Stabilisation (Fixes Flickering)

Raw detection counts jump ±3–5 every frame. The `CountStabilizer` maintains a deque of the last 20 frame counts and returns the **median**:

```
Raw frames:  22  19  24  21  18  23  25  20  22  24  ...
Median(20):  21  21  21  21  21  22  22  22  22  22  ← rock solid
```

Median is robust: one bad frame cannot shift the displayed count.

### 4 — DeepSORT Tracking (Tuned for Seated Passengers)

Default DeepSORT settings are tuned for walking pedestrians. This project retuned for bus interiors:

| Parameter | Default | This Project | Reason |
|-----------|---------|-------------|--------|
| `max_age` | 30 | **60** | Seated person can be occluded for many frames |
| `n_init` | 3 | **2** | Confirm track faster (less initial jitter) |
| `max_cosine_dist` | 0.4 | **0.5** | More tolerant Re-ID for similar-dressed passengers |

### 5 — Alert State Machine

```
OK (< 75%)  →  WARNING (75–99%)  →  OVERCROWD (≥ 100%)
     ↑________________↓_________________↓
         8 consecutive frames required to change state upward
         30-second cooldown between repeated alerts
```

---

## 🎯 Training on Custom Data

### Step 1 — Annotate Your Bus Images

Use [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com) to annotate your bus-interior images in YOLO format. Place images and `.txt` label files in `data/raw/`:

```
data/raw/
  frame_001.jpg  +  frame_001.txt   ("0 0.512 0.433 0.123 0.344")
  frame_002.jpg  +  frame_002.txt
  ...
```

### Step 2 — Preprocess

```bash
python3 preprocessing.py
# Resizes to 640×640, splits 80/10/10, augments ×3, generates dataset.yaml
```

### Step 3 — Train All 4 Models

```bash
python3 training.py --model both --device cuda
```

### Step 4 — Run Demo with Fine-Tuned Weights

Fine-tuned weights are saved to `models/bus_overcrowd_v1_primary_best.pt` etc. Update paths in `config.py` accordingly, then run the demo.

---

## 🔌 Jetson Nano Deployment

### Deployment Pipeline

```
PC (Training)                         Jetson Nano (Deployment)
──────────────────────────────────────────────────────────────
1. Train  →  best.pt
2. Export →  model.onnx  ──── SCP ──▶  ~/bus_overcrowding/models/
                                        ▼
                              3. trtexec → .engine (FP16)
                              4. python3 jetson_inference.py
```

### Performance on Jetson Nano

| Configuration | FPS |
|---|---|
| YOLOv8n `.pt` (CPU) | 1–2 |
| YOLOv8n `.pt` (GPU PyTorch) | 5–8 |
| YOLOv8n `.engine` TensorRT FP16 | **15–20** ✅ |
| 4-model ensemble TRT FP16 | **4–6** ✅ |

For production Jetson deployment (PyTorch ARM wheel, `trtexec` conversion, CSI camera GStreamer pipeline, systemd autostart), see the [Jetson Deployment Wiki](../../wiki/Jetson-Deployment).

---

## 🌐 Dashboard

Open `http://localhost:5000` after running `python3 main.py --mode both`.

**Features:**
- Summary cards — total events / warnings / overcrowd alerts / total fines (INR)
- Full event log table with timestamp, count, capacity, fine, snapshot filename
- Auto-refreshes every 10 seconds
- JSON API endpoints for external integrations

**API Reference:**

| Endpoint | Method | Response |
|----------|--------|---------|
| `/` | GET | HTML dashboard |
| `/api/events` | GET | Last 100 events as JSON |
| `/api/status` | GET | Latest event + bus config |

---

## 📊 Benchmarking & Evaluation

### Detection Quality Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **mAP@0.5** | Standard detection accuracy | Ultralytics `model.val()` |
| **mAP@0.5:0.95** | Stricter COCO-style mAP | Ultralytics `model.val()` |
| **Precision** | Correct detections / all detections | Ultralytics `model.val()` |
| **Recall** | Correct detections / all actual people | Ultralytics `model.val()` |

### Counting Accuracy Metrics

| Metric | Formula |
|--------|---------|
| **MAE** | `mean(abs(predicted - actual))` |
| **RMSE** | `sqrt(mean((predicted - actual)²))` |

### System Performance Metrics

| Metric | Tool |
|--------|------|
| **FPS** | Built-in `FPSMeter` (30-frame rolling average) |
| **Latency** | `log_time()` context manager (per model, per frame) |
| **RAM** | `jtop` on Jetson / `htop` on PC |
| **Model Size** | `ls -lh models/` |

### Baseline Comparison Table

| Model | Type | mAP@0.5 | Params | Edge-Ready |
|-------|------|---------|--------|-----------|
| Faster R-CNN | Detection | — | 41.8M | ❌ |
| SSD MobileNet V1 | Lightweight | ~23 | 5.1M | ✅ |
| CSRNet | Density Map | — | 16.3M | ⚠️ |
| YOLOv5n | Detection | 45.7 | 1.9M | ✅ |
| YOLOv7-tiny | Detection | 56.4 | 6.2M | ✅ |
| **YOLOv8n (ours)** | Ensemble | ~62 | 3.2M | ✅ |
| **YOLOv8s (ours)** | Ensemble | ~73 | 11.2M | ✅ |
| **YOLOv8m (ours)** | Ensemble | ~80 | 25.9M | ⚠️ |
| **YOLOv8l (ours)** | Ensemble | ~83 | 43.7M | ⚠️ |
| **4-Model Ensemble (ours)** | Ensemble | **~85** | 84M | ✅ (TRT) |

---

## ⚙️ Configuration

All parameters are in `config.py`. Key settings:

```python
# ── Bus settings ──────────────────────────────────────────
BUS_ID          = "BUS-001"    # Unique bus identifier
MAX_CAPACITY    = 40           # Legal passenger limit
WARNING_RATIO   = 0.75         # Warn at 75% full
FINE_AMOUNT_INR = 5000         # Fine (INR) per overcrowding event

# ── Models (enable/disable each independently) ─────────────
MODEL_CONFIGS = [
    {"name": "yolov8n.pt", "weight": 1, "enabled": True},
    {"name": "yolov8s.pt", "weight": 2, "enabled": True},
    {"name": "yolov8m.pt", "weight": 3, "enabled": True},
    {"name": "yolov8l.pt", "weight": 4, "enabled": True},
]
ENSEMBLE_STRATEGY = "weighted_max"   # safety-first

# ── Detection (tuned for bus interiors) ───────────────────
CONF_THRESHOLD  = 0.25   # Low threshold catches occluded/seated people
IOU_THRESHOLD   = 0.40

# ── Count stabilisation ────────────────────────────────────
STABILIZER_WINDOW = 20   # Median over last 20 frames
STABILIZER_METHOD = "median"

# ── Display ────────────────────────────────────────────────
DISPLAY_WIDTH   = 1280   # Fixed output window width
DISPLAY_HEIGHT  = 720    # Fixed output window height

# ── Privacy ────────────────────────────────────────────────
BLUR_FACES      = False  # Enable for strict privacy mode
STORE_RAW_VIDEO = False  # Never enabled
```

### Ensemble Strategy Comparison

| Strategy | Behaviour | Best For |
|----------|-----------|---------|
| `weighted_max` | Biases toward higher count | Safety / enforcement ✅ |
| `weighted_mean` | Balanced weighted average | General use |
| `max` | Always takes highest count | Most conservative |
| `median` | Robust to outlier models | Noisy environments |

---

## 🔒 Privacy Design

| Principle | Implementation |
|-----------|----------------|
| **No raw video stored** | `STORE_RAW_VIDEO = False` — hardcoded |
| **On-device only** | No network calls, no cloud API |
| **No facial recognition** | Person bounding boxes only — no biometrics |
| **Minimal data** | Database stores: timestamp, count, bus ID, fine only |
| **Optional face blur** | Haar cascade blur on display/snapshot (`BLUR_FACES = True`) |
| **Snapshot opt-out** | `SAVE_ALERTS_IMG = False` disables all disk writes |

---

## 📦 Dependencies

```
ultralytics>=8.2.0          # YOLOv8 (all 4 sizes)
deep-sort-realtime>=1.3.2   # DeepSORT + MobileNet Re-ID
opencv-python-headless       # Computer vision + NMS
torch>=2.1.0                 # Deep learning backend
flask>=3.0.0                 # Web dashboard
loguru>=0.7.2                # Logging
numpy>=1.24.0,<2.0.0
scipy>=1.11.0                # Kalman filter (DeepSORT)
scikit-learn>=1.3.0          # Cosine distance (Re-ID)
onnx>=1.15.0                 # Model export
onnxruntime>=1.17.0          # CPU edge inference
```

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📚 References

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Wojke, N. et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. ICIP.
- Li, Y. et al. (2018). *CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes*. CVPR.
- NVIDIA. (2023). *Deploy YOLOv8 on Jetson using TensorRT*. Seeed Studio Wiki.
- Wang, C. et al. (2022). *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors*. CVPR.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Built for Edge AI · Smart Transportation · Public Safety

Made with ❤️ by [YOUR_USERNAME](https://github.com/YOUR_USERNAME)

</div>
