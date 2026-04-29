"""
tracker.py
----------
Object tracking module.
 
Implements:
  • SORT  (Simple Online and Realtime Tracking)
      - Kalman filter per object  (via filterpy)
      - Hungarian algorithm for data association  (via scipy)
      - Fully self-contained, no external SORT repo needed
 
  • DeepSORT  (optional – requires `deep_sort_realtime` package)
      pip install deep-sort-realtime
 
Select which tracker to use via the `TrackerFactory.create()` helper or
instantiate SORTTracker / DeepSORTTracker directly.
"""
 
from __future__ import annotations
 
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
 
# -----------------------------------------------------------------------
# IOU helpers
# -----------------------------------------------------------------------
 
def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IOU between two sets of bounding boxes.
 
    Parameters
    ----------
    bb_test : (N, 4) array  [x1 y1 x2 y2]
    bb_gt   : (M, 4) array  [x1 y1 x2 y2]
 
    Returns
    -------
    iou_matrix : (N, M) float array
    """
    bb_gt = np.expand_dims(bb_gt, 0)       # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1)   # (N, 1, 4)
 
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
 
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
 
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (
        bb_test[..., 3] - bb_test[..., 1]
    )
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (
        bb_gt[..., 3] - bb_gt[..., 1]
    )
    union = area_test + area_gt - intersection
 
    return np.where(union > 0, intersection / union, 0.0)
 
 
# -----------------------------------------------------------------------
# Bounding box ↔ Kalman state conversions
# -----------------------------------------------------------------------
 
def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    [x1, y1, x2, y2]  →  [cx, cy, area, aspect_ratio]  (column vector)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # area
    r = w / (h + 1e-6) # aspect ratio (guard divide-by-zero)
    return np.array([cx, cy, s, r], dtype=np.float32).reshape((4, 1))
 
 
def _z_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """
    Kalman state  [cx, cy, area, aspect_ratio, ...]  →  [x1, y1, x2, y2(, score)]
    """
    x = x.flatten()
    w = np.sqrt(np.maximum(x[2] * x[3], 0.0))
    h = np.maximum(x[2] / (w + 1e-6), 0.0)
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    return np.array([x1, y1, x2, y2, score], dtype=np.float32)
 
 
# -----------------------------------------------------------------------
# Single-object Kalman tracker
# -----------------------------------------------------------------------
 
class _KalmanBoxTracker:
    """
    Represents a single tracked object using a Kalman filter.
 
    State vector:  [cx, cy, s, r,  Δcx, Δcy, Δs]
    Measurement:   [cx, cy, s, r]
    """
 
    _id_counter: int = 0
 
    def __init__(self, bbox: np.ndarray, class_id: int, confidence: float):
        # ── Kalman filter setup ──────────────────────────────────────────
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
 
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
 
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
 
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.05
        self.kf.Q[4:, 4:] *= 0.05
 
        self.kf.x[:4] = _bbox_to_z(bbox)
 
        _KalmanBoxTracker._id_counter += 1
        self.id: int = _KalmanBoxTracker._id_counter
        self.class_id: int = class_id
        self.confidence: float = confidence
 
        self.age: int = 0
        self.hits: int = 0
        self.hit_streak: int = 0
        self.time_since_update: int = 0
 
    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return _z_to_bbox(self.kf.x)
 
    def update(self, bbox: np.ndarray, confidence: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self.kf.update(_bbox_to_z(bbox))
 
    def get_state(self) -> np.ndarray:
        return _z_to_bbox(self.kf.x)
 
 
# -----------------------------------------------------------------------
# SORT tracker
# -----------------------------------------------------------------------
 
class SORTTracker:
    """
    SORT – Simple Online and Realtime Tracking.
 
    Reference: Bewley et al. (2016)  https://arxiv.org/abs/1602.00763
 
    Args
    ----
    max_age       : Maximum frames to keep a track alive without a match.
    min_hits      : Minimum consecutive matched frames before a track is
                    considered confirmed and returned.
    iou_threshold : Minimum IOU required to associate a detection with a track.
    reset_ids     : If True (default), reset the global track ID counter
                    each time a new SORTTracker is instantiated. This keeps
                    IDs starting from 1 for every new processing run.
    """
 
    def __init__(
        self,
        max_age: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        reset_ids: bool = True,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
 
        self.trackers: list[_KalmanBoxTracker] = []
        self.frame_count: int = 0
 
        # Reset the global ID counter so each new run starts at ID 1.
        if reset_ids:
            _KalmanBoxTracker._id_counter = 0
 
    # ── Internal helpers ─────────────────────────────────────────────────
 
    def _associate(
        self,
        detections: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(predictions) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections), dtype=int),
                np.empty(0, dtype=int),
            )
 
        iou_matrix = _iou_batch(detections, predictions)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
 
        matches, unmatched_dets, unmatched_trks = [], [], []
 
        matched_dets = set(row_ind)
        for d in range(len(detections)):
            if d not in matched_dets:
                unmatched_dets.append(d)
 
        matched_trks = set(col_ind)
        for t in range(len(predictions)):
            if t not in matched_trks:
                unmatched_trks.append(t)
 
        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] < self.iou_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matches.append([d, t])
 
        return (
            np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int),
            np.array(unmatched_dets, dtype=int),
            np.array(unmatched_trks, dtype=int),
        )
 
    # ── Public API ───────────────────────────────────────────────────────
 
    def update(self, detections: np.ndarray) -> list[dict]:
        self.frame_count += 1
 
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
            else:
                predicted_boxes.append(pred)
 
        for i in sorted(to_del, reverse=True):
            self.trackers.pop(i)
 
        pred_array = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))
 
        if len(detections) > 0:
            det_boxes = detections[:, :4]
        else:
            det_boxes = np.empty((0, 4))
 
        matches, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, pred_array
        )
 
        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(
                detections[d_idx, :4],
                float(detections[d_idx, 4]),
            )
            self.trackers[t_idx].class_id = int(detections[d_idx, 5])
 
        for d_idx in unmatched_dets:
            new_trk = _KalmanBoxTracker(
                bbox=detections[d_idx, :4],
                class_id=int(detections[d_idx, 5]),
                confidence=float(detections[d_idx, 4]),
            )
            self.trackers.append(new_trk)
 
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]
 
        active_tracks = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                bbox = trk.get_state().tolist()
                active_tracks.append(
                    {
                        "track_id": trk.id,
                        "bbox": [int(v) for v in bbox],
                        "confidence": round(trk.confidence, 2),
                        "class_id": trk.class_id,
                    }
                )
 
        return active_tracks
 
    def reset(self) -> None:
        """Reset tracker state (e.g. when switching video sources)."""
        self.trackers.clear()
        self.frame_count = 0
        _KalmanBoxTracker._id_counter = 0
 
 
# -----------------------------------------------------------------------
# DeepSORT wrapper  (optional – requires deep_sort_realtime)
# -----------------------------------------------------------------------
 
class DeepSORTTracker:
    def __init__(
        self,
        max_age: int = 30,
        embedder: str | None = "mobilenet",
        half: bool = False,
    ):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "deep_sort_realtime is not installed.\n"
                "Install it with:  pip install deep-sort-realtime"
            ) from exc
 
        self._ds = DeepSort(
            max_age=max_age,
            embedder=embedder,
            half=half,
            bgr=True,
        )
 
    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[dict]:
        if len(detections) == 0:
            return []
 
        ds_input = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], float(conf), str(int(cls_id))))
 
        raw_tracks = self._ds.update_tracks(ds_input, frame=frame)
 
        active_tracks = []
        for trk in raw_tracks:
            if not trk.is_confirmed():
                continue
            l, t, r, b = map(int, trk.to_ltrb())
            active_tracks.append(
                {
                    "track_id": int(trk.track_id),
                    "bbox": [l, t, r, b],
                    "confidence": round(float(trk.det_conf or 0.0), 2),
                    "class_id": int(trk.det_class) if trk.det_class is not None else -1,
                }
            )
 
        return active_tracks
 
    def reset(self) -> None:
        try:
            self._ds.tracker.tracks.clear()
        except Exception:
            pass
 
 
# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------
 
class TrackerFactory:
    @staticmethod
    def create(
        tracker_type: str = "sort",
        **kwargs,
    ) -> SORTTracker | DeepSORTTracker:
        tracker_type = tracker_type.lower()
        if tracker_type == "sort":
            allowed = {"max_age", "min_hits", "iou_threshold", "reset_ids"}
            kw = {k: v for k, v in kwargs.items() if k in allowed}
            return SORTTracker(**kw)
        elif tracker_type in ("deepsort", "deep_sort"):
            allowed = {"max_age", "embedder", "half"}
            kw = {k: v for k, v in kwargs.items() if k in allowed}
            return DeepSORTTracker(**kw)
        else:
            raise ValueError(
                f"Unknown tracker type '{tracker_type}'. Choose 'sort' or 'deepsort'."
            )