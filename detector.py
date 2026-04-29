"""
detector.py
-----------
YOLOv8-based object detection module.
Wraps Ultralytics YOLO to produce bounding boxes in the format
expected by the SORT tracker: [x1, y1, x2, y2, confidence, class_id].
"""
 
from __future__ import annotations
 
import numpy as np
from ultralytics import YOLO
 
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
 
 
def _auto_device() -> str:
    """Pick the best available device — cuda > mps > cpu."""
    if not _TORCH_AVAILABLE:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
 
 
class ObjectDetector:
    """
    Wraps YOLOv8 for frame-level object detection.
 
    Args
    ----
    model_path     : Path to YOLO weights file (e.g. 'yolov8n.pt').
                     Ultralytics will download it on first run if missing.
    conf_threshold : Minimum confidence to keep a detection (0.0–1.0).
    iou_threshold  : NMS IOU threshold passed to YOLO (0.0–1.0).
    target_classes : Optional list/tuple of class *names* to keep
                     (e.g. ['person', 'car']). None = keep all classes.
    device         : 'cpu', 'cuda', 'mps', or None for auto-detect.
    imgsz          : Inference image size (square). 640 is YOLO's native.
                     Smaller = faster but less accurate.
    """
 
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.40,
        target_classes: list | tuple | None = None,
        device: str | None = None,
        imgsz: int = 640,
    ):
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = device or _auto_device()
 
        # ── FP16 only works on CUDA. On CPU/MPS we MUST use FP32. ──
        # (This was the original "yolov8s/m/l hangs" bug.)
        self.use_half = self.device == "cuda"
 
        # Load YOLOv8 model (downloads automatically if not present)
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            # Fall back silently to CPU if the requested device is unavailable
            self.device = "cpu"
            self.use_half = False
            self.model.to("cpu")
 
        # COCO class names dictionary  {0: 'person', 1: 'bicycle', ...}
        self.class_names: dict[int, str] = self.model.names
 
        # Build the set of class IDs we care about (None = all classes).
        # We pass these to YOLO directly so it can skip non-matching classes
        # at the kernel level — much faster than post-filtering.
        self.target_class_ids: list[int] | None = None
        if target_classes:
            wanted = {str(c).strip().lower() for c in target_classes if str(c).strip()}
            ids = [
                cid
                for cid, name in self.class_names.items()
                if name.lower() in wanted
            ]
            if not ids:
                # Don't crash — just track everything if user typed an unknown class.
                self.target_class_ids = None
            else:
                self.target_class_ids = ids
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame (as returned by OpenCV).
 
        Returns
        -------
        np.ndarray of shape (N, 6) where each row is:
            [x1, y1, x2, y2, confidence, class_id]
        Returns an empty array with shape (0, 6) when nothing is detected.
        """
        # Build kwargs — only pass `classes=` if the user has restricted them,
        # otherwise YOLO returns ALL 80 COCO classes.
        kwargs = dict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            agnostic_nms=True,
            half=self.use_half,   # ← FALSE on CPU — fixes the hang
            device=self.device,
            verbose=False,
        )
        if self.target_class_ids is not None:
            kwargs["classes"] = self.target_class_ids
 
        results = self.model.predict(**kwargs)[0]
 
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
 
        # Vectorised extraction (much faster than per-box loops)
        try:
            xyxy = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = results.boxes.cls.cpu().numpy().reshape(-1, 1)
            detections = np.hstack([xyxy, conf, cls]).astype(np.float32)
        except Exception:
            # Fallback per-box (handles edge cases where tensors aren't on a known device)
            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append([x1, y1, x2, y2, confidence, class_id])
            if not detections:
                return np.empty((0, 6), dtype=np.float32)
            detections = np.array(detections, dtype=np.float32)
 
        return detections
 
    def get_class_name(self, class_id: int) -> str:
        """Return the human-readable name for a COCO class ID."""
        return self.class_names.get(int(class_id), f"cls_{class_id}")