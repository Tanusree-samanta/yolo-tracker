"""
main.py
-------
Real-Time Object Detection & Tracking
======================================
Entry point for the detection + tracking pipeline.

Quick start
-----------
  # Webcam (default)
  python main.py

  # Video file
  python main.py --source path/to/video.mp4

  # Track only people and cars
  python main.py --classes person car

  # Use DeepSORT instead of SORT
  python main.py --tracker deepsort

Keyboard controls while running
--------------------------------
  Q  or  ESC  →  Quit
  P           →  Pause / Resume
  S           →  Save current frame as screenshot
  +           →  Increase confidence threshold (step 0.05)
  -           →  Decrease confidence threshold (step 0.05)
"""

from __future__ import annotations

import argparse
import sys
import time
import datetime
import cv2
import numpy as np

from detector import ObjectDetector
from tracker import TrackerFactory
from utils import (
    FPSCounter,
    ObjectCounter,
    draw_track,
    draw_fps,
    draw_object_count,
    draw_info_bar,
    format_label,
    open_video_source,
)


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-Time Object Detection & Tracking (YOLOv8 + SORT/DeepSORT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Source ──────────────────────────────────────────────────────────
    p.add_argument(
        "--source",
        default="0",
        help=(
            "Video source.  Integer index for webcam (e.g. 0, 1) "
            "or path to a video file (e.g. video.mp4)."
        ),
    )

    # ── Model ───────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLOv8 weights file.  Downloads automatically if not found.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold (0.0 – 1.0).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU threshold for YOLO (0.0 – 1.0).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Inference device: 'cpu', 'cuda', 'mps', etc.  Auto-selected if omitted.",
    )

    # ── Class filter ────────────────────────────────────────────────────
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help=(
            "Only track these COCO class names.  "
            "E.g.: --classes person car bicycle.  "
            "Omit to track all classes."
        ),
    )

    # ── Tracker ─────────────────────────────────────────────────────────
    p.add_argument(
        "--tracker",
        choices=["sort", "deepsort"],
        default="sort",
        help="Tracking algorithm to use.",
    )
    p.add_argument(
        "--max-age",
        type=int,
        default=50,
        dest="max_age",
        help="Max frames to keep a track alive without a matched detection.",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=1,
        dest="min_hits",
        help="(SORT only) Min consecutive matched frames before a track is confirmed.",
    )
    p.add_argument(
        "--tracker-iou",
        type=float,
        default=0.2,
        dest="tracker_iou",
        help="(SORT only) Min IOU to associate a detection with an existing track.",
    )

    # ── Display ─────────────────────────────────────────────────────────
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live preview window (useful for headless servers).",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save annotated output to a video file (e.g. output.mp4).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize frame to this width before processing (keeps aspect ratio).",
    )

    return p


# -----------------------------------------------------------------------
# Resize helper
# -----------------------------------------------------------------------

def maybe_resize(frame: np.ndarray, target_width: int | None) -> np.ndarray:
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ── Parse source (int for webcam, str for file) ───────────────────
    try:
        source = int(args.source)
    except ValueError:
        source = args.source   # file path

    print(f"\n{'='*60}")
    print("  Real-Time Object Detection & Tracking")
    print(f"{'='*60}")
    print(f"  Source   : {args.source}")
    print(f"  Model    : {args.model}")
    print(f"  Tracker  : {args.tracker.upper()}")
    print(f"  Conf     : {args.conf}")
    print(f"  Classes  : {args.classes or 'ALL'}")
    print(f"{'='*60}\n")

    # ── Detector ──────────────────────────────────────────────────────
    print("[1/3] Loading YOLOv8 detector …")
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        target_classes=args.classes,
        device=args.device,
    )
    print("      ✓ Detector ready.\n")

    # ── Tracker ───────────────────────────────────────────────────────
    print(f"[2/3] Initialising {args.tracker.upper()} tracker …")
    tracker = TrackerFactory.create(
        tracker_type=args.tracker,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.tracker_iou,
    )
    print("      ✓ Tracker ready.\n")

    # ── Video source ──────────────────────────────────────────────────
    print("[3/3] Opening video source …")
    cap = open_video_source(source)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"      ✓ Source opened  ({src_w}×{src_h} @ {src_fps:.1f} FPS)\n")

    # ── Output writer (optional) ──────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = args.width or src_w
        out_h = int(src_h * (out_w / src_w)) if args.width else src_h
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (out_w, out_h))
        print(f"  Output video → {args.output}  ({out_w}×{out_h})")

    # ── Helpers ───────────────────────────────────────────────────────
    fps_counter = FPSCounter(window=30)
    obj_counter = ObjectCounter()

    conf_threshold = args.conf   # mutable (keyboard +/-)
    paused = False

    print("  Controls:  Q/ESC=Quit  |  P=Pause  |  S=Screenshot  |  +/-=Confidence\n")
    print("  Starting … (press Q or ESC in the window to exit)\n")

    window_name = "Object Detection & Tracking"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0

    try:
        while True:
            # ── Keyboard input ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):          # Q / ESC → quit
                print("\n  [Q] Quit requested.")
                break
            elif key in (ord("p"), ord("P")):             # P → pause/resume
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"  [P] {status}")
            elif key in (ord("s"), ord("S")):             # S → screenshot
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{ts}.jpg"
                if "annotated" in dir():                  # type: ignore[name-defined]
                    cv2.imwrite(filename, annotated)      # type: ignore[name-defined]
                    print(f"  [S] Screenshot saved → {filename}")
            elif key == ord("+") or key == ord("="):      # + → increase conf
                conf_threshold = min(conf_threshold + 0.05, 0.95)
                detector.conf_threshold = conf_threshold
                print(f"  [+] Confidence threshold → {conf_threshold:.2f}")
            elif key == ord("-"):                         # - → decrease conf
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                detector.conf_threshold = conf_threshold
                print(f"  [-] Confidence threshold → {conf_threshold:.2f}")

            if paused:
                time.sleep(0.05)
                continue

            # ── Read frame ─────────────────────────────────────────────
            ok, frame = cap.read()
            if not ok:
                # End of file – loop back for video files, stop for webcams
                if isinstance(source, str):
                    print("\n  End of video file – restarting …")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker.reset()
                    continue
                else:
                    print("\n  Webcam frame grab failed – exiting.")
                    break

            frame = maybe_resize(frame, args.width)
            frame_idx += 1

            # ── Detect ─────────────────────────────────────────────────
            detections = detector.detect(frame)   # (N, 6): x1y1x2y2 conf cls

            # ── Track ──────────────────────────────────────────────────
            # DeepSORT needs the raw frame for its Re-ID embedder
            if args.tracker == "deepsort":
                tracks = tracker.update(detections, frame)
            else:
                tracks = tracker.update(detections)

            # ── Update counters ────────────────────────────────────────
            fps_counter.tick()
            obj_counter.update(tracks, detector.class_names)

            # ── Annotate frame ─────────────────────────────────────────
            annotated = frame.copy()

            for trk in tracks:
                class_name = detector.get_class_name(trk["class_id"])
                label = format_label(class_name, trk["track_id"], trk["confidence"])
                draw_track(annotated, trk, label)

            # Overlays
            draw_fps(annotated, fps_counter.fps)
            draw_object_count(annotated, obj_counter.counts)

            # Info bar at bottom
            mode_str = f"Tracker: {args.tracker.upper()}"
            conf_str = f"Conf: {conf_threshold:.0%}"
            cls_str  = f"Classes: {', '.join(args.classes) if args.classes else 'ALL'}"
            info_text = f"  {mode_str}  |  {conf_str}  |  {cls_str}  |  Frame: {frame_idx}"
            draw_info_bar(annotated, info_text)

            # ── Display ────────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow(window_name, annotated)

            # ── Write output ───────────────────────────────────────────
            if writer is not None:
                writer.write(annotated)

    except KeyboardInterrupt:
        print("\n  [Ctrl-C] Interrupted.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n  Output saved → {args.output}")
        cv2.destroyAllWindows()
        print("  Resources released.  Goodbye!\n")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
