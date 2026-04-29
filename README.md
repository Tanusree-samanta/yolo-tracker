# Real-Time Object Detection & Tracking
### YOLOv8 · SORT · (optional) DeepSORT · OpenCV

A production-quality, modular Python pipeline that detects and tracks objects
in real time from a webcam or video file.

---

## Feature Overview

| Feature | Details |
|---|---|
| Detector | YOLOv8n (nano – fast) via Ultralytics |
| Tracker  | SORT (built-in, no extra repo) or DeepSORT (optional) |
| Display  | Bounding boxes, unique track IDs, class labels, confidence, FPS, live object counts |
| Controls | Pause, Screenshot, dynamic confidence adjustment |
| Output   | Real-time window + optional MP4 save |

---

## Project Structure

```
object_detection_tracking/
├── main.py          ← Entry point & main loop
├── detector.py      ← YOLOv8 detection wrapper
├── tracker.py       ← SORT & DeepSORT implementations
├── utils.py         ← Drawing helpers, FPS counter, object counter
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

---

## Installation

### Step 1 — Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Or using conda
conda create -n tracker python=3.10 -y
conda activate tracker
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — frame capture and display
- `ultralytics`   — YOLOv8 model + inference
- `filterpy`      — Kalman filter (used by SORT)
- `scipy`         — Hungarian algorithm for data association
- `numpy`         — array operations

---

### Step 3 — Download the YOLOv8 model

The model is downloaded **automatically** on the first run by Ultralytics.

To download it manually in advance:

```bash
# Option A – via Python
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Option B – direct download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Available model sizes (trade-off speed vs. accuracy):

| Model       | Size   | Speed  | Accuracy |
|-------------|--------|--------|----------|
| yolov8n.pt  | 6 MB   | ★★★★★ | ★★☆☆☆   |
| yolov8s.pt  | 22 MB  | ★★★★☆ | ★★★☆☆   |
| yolov8m.pt  | 52 MB  | ★★★☆☆ | ★★★★☆   |
| yolov8l.pt  | 87 MB  | ★★☆☆☆ | ★★★★★   |
| yolov8x.pt  | 131 MB | ★☆☆☆☆ | ★★★★★   |

---

### Step 4 (Optional) — Install DeepSORT

```bash
pip install deep-sort-realtime
```

---

## Running the System

### Basic — webcam (default camera)

```bash
python main.py
```

### Video file

```bash
python main.py --source path/to/video.mp4
```

### Specific webcam index

```bash
python main.py --source 1
```

### Track only people and cars

```bash
python main.py --classes person car
```

### Track only people, save output to file

```bash
python main.py --classes person --output tracked_output.mp4
```

### Use DeepSORT instead of SORT

```bash
python main.py --tracker deepsort
```

### Use a larger model for better accuracy

```bash
python main.py --model yolov8m.pt --conf 0.5
```

### Resize input for faster processing

```bash
python main.py --width 640
```

### Run on GPU (CUDA)

```bash
python main.py --device cuda
```

### Run headless (no window, just save output)

```bash
python main.py --source video.mp4 --output out.mp4 --no-display
```

---

## All Command-Line Options

```
usage: main.py [-h] [--source SOURCE] [--model MODEL] [--conf CONF]
               [--iou IOU] [--device DEVICE] [--classes CLASS [CLASS ...]]
               [--tracker {sort,deepsort}] [--max-age MAX_AGE]
               [--min-hits MIN_HITS] [--tracker-iou TRACKER_IOU]
               [--no-display] [--output FILE] [--width WIDTH]

options:
  --source SOURCE            Video source (webcam index or file path) [default: 0]
  --model MODEL              YOLOv8 weights file                      [default: yolov8n.pt]
  --conf CONF                Detection confidence threshold            [default: 0.4]
  --iou IOU                  YOLO NMS IOU threshold                   [default: 0.45]
  --device DEVICE            Inference device (cpu/cuda/mps)          [default: auto]
  --classes CLASS [...]      Filter to specific COCO class names      [default: ALL]
  --tracker {sort,deepsort}  Tracking algorithm                       [default: sort]
  --max-age MAX_AGE          Track death age (frames)                 [default: 30]
  --min-hits MIN_HITS        Min hits before confirming (SORT only)   [default: 3]
  --tracker-iou TRACKER_IOU  SORT association IOU threshold           [default: 0.3]
  --no-display               Disable live preview window
  --output FILE              Save annotated video to FILE
  --width WIDTH              Resize input frame to this width
```

---

## Keyboard Controls (in the live window)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `P`          | Pause / Resume |
| `S`          | Save screenshot (timestamped JPEG) |
| `+` / `=`    | Increase confidence threshold by 5% |
| `-`          | Decrease confidence threshold by 5% |

---

## COCO Class Names Reference

The model can detect 80 COCO classes.  Common ones:

```
person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe, backpack, umbrella, handbag, suitcase,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
orange, pizza, donut, cake, chair, sofa, pottedplant, bed, diningtable,
toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase,
scissors, teddy bear, hair drier, toothbrush
```

---

## Architecture Overview

```
┌──────────────┐     frame      ┌──────────────────┐
│  VideoCapture│ ─────────────► │  ObjectDetector   │
│  (OpenCV)    │                │  (YOLOv8)         │
└──────────────┘                │                   │
                                │  → [x1,y1,x2,y2,  │
                                │     conf, cls_id] │
                                └─────────┬─────────┘
                                          │ detections (N×6)
                                          ▼
                                ┌──────────────────┐
                                │  SORTTracker /   │
                                │  DeepSORTTracker │
                                │                  │
                                │  Kalman predict  │
                                │  Hungarian match │
                                │  Track update    │
                                └─────────┬────────┘
                                          │ tracks [{id, bbox, conf, cls}]
                                          ▼
                                ┌──────────────────┐
                                │  utils.py        │
                                │  draw_track()    │
                                │  draw_fps()      │
                                │  draw_object_    │
                                │    count()       │
                                └─────────┬────────┘
                                          │ annotated frame
                                          ▼
                                ┌──────────────────┐
                                │  cv2.imshow()    │  ← live window
                                │  VideoWriter     │  ← optional file save
                                └──────────────────┘
```

---

## Troubleshooting

**Camera not found**
```bash
# List available cameras
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"
```

**CUDA out of memory**
```bash
# Use a smaller model or run on CPU
python main.py --model yolov8n.pt --device cpu
```

**Low FPS on CPU**
```bash
# Resize frames to reduce computation
python main.py --width 480
# Or use a nano model
python main.py --model yolov8n.pt
```

**DeepSORT import error**
```bash
pip install deep-sort-realtime
```

**ModuleNotFoundError: filterpy**
```bash
pip install filterpy
```

---

## License

MIT — free to use, modify, and distribute.
#   y o l o - t r a c k e r  
 