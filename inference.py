"""
Usage:
    python inference.py --source bus_video.mp4          # demo video
    python inference.py --source 0                      # webcam
    python inference.py --source 0 --no-display         # headless
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

import config
from logger import logger, log_time
from utils import (
    AlertManager, CountStabilizer, FPSMeter,
    annotate_frame, blur_faces, ensemble_count,
    init_db, letterbox_to_display,
)

class YOLODetector:
    """
    Thin wrapper around one Ultralytics YOLO model.
    Auto-downloads weights if not present locally.
    """

    def __init__(self, model_cfg: dict, device: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Run: pip install ultralytics")

        path = model_cfg["path"]
        name = model_cfg["name"]

        if path.exists():
            logger.info(f"[{model_cfg['label']}] Loading: {path}")
            self.model = YOLO(str(path))
        else:
            logger.info(f"[{model_cfg['label']}] Downloading {name}…")
            self.model = YOLO(name)          # downloads to Ultralytics cache

        self.label  = model_cfg["label"]
        self.weight = model_cfg["weight"]
        self.device = device
        self.model.to(device)
        logger.info(f"[{self.label}] Ready on {device}")

    def detect(self, frame: np.ndarray) -> Tuple[List[List[float]], int]:
        """
        Returns (detections, count) where detections = [[x1,y1,x2,y2,conf], ...]
        """
        results = self.model.predict(
            source  = frame,
            conf    = config.CONF_THRESHOLD,
            iou     = config.IOU_THRESHOLD,
            classes = [config.PERSON_CLASS_ID],
            imgsz   = config.IMG_SIZE,
            device  = self.device,
            verbose = False,
        )
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                dets.append([x1, y1, x2, y2, conf])
        return dets, len(dets)

class EnsembleDetector:
    """
    Loads up to 4 YOLOv8 detectors (n / s / m / l) and runs them
    in sequence per frame.  Returns:
      - merged detections (union of all boxes, NMS'd)
      - per-model counts for HUD breakdown
      - final ensemble count
    """

    def __init__(self, device: str):
        self.detectors: List[YOLODetector] = []
        for cfg in config.MODEL_CONFIGS:
            if cfg["enabled"]:
                self.detectors.append(YOLODetector(cfg, device))

        if not self.detectors:
            raise RuntimeError("No models enabled in MODEL_CONFIGS")

        logger.info(f"Ensemble: {len(self.detectors)} models loaded "
                    f"({[d.label for d in self.detectors]})")

    def run(self, frame: np.ndarray) -> Tuple[List, Dict[str, int], int, Dict[str, float]]:
        """
        Returns
        -------
        all_dets     : merged + NMS'd detection list  [[x1,y1,x2,y2,conf],…]
        model_counts : {label: count} per model
        raw_ensemble : ensemble count BEFORE stabilisation
        latencies    : {label: elapsed_seconds} per model
        """
        all_dets: List[List[float]] = []
        model_counts: Dict[str, int] = {}
        latencies: Dict[str, float] = {}
        counts:  List[int] = []
        weights: List[int] = []

        for det in self.detectors:
            t0 = time.time()
            dets, count = det.detect(frame)
            latencies[det.label] = time.time() - t0
            model_counts[det.label] = count
            counts.append(count)
            weights.append(det.weight)
            all_dets.extend(dets)

        # Merge overlapping boxes from different models with NMS
        all_dets = _nms_merge(all_dets, iou_threshold=config.IOU_THRESHOLD)

        # Ensemble count strategy
        raw_ensemble = ensemble_count(counts, weights, config.ENSEMBLE_STRATEGY)

        return all_dets, model_counts, raw_ensemble, latencies

    @property
    def primary_dets_source(self):
        """Return the heaviest model's detections for tracking."""
        return self.detectors[-1] if self.detectors else None


def _nms_merge(
    dets: List[List[float]],
    iou_threshold: float = 0.40,
) -> List[List[float]]:
    """
    Apply NMS across the merged box pool from all models.
    This prevents counting the same person multiple times when
    two models both detect them.
    """
    if not dets:
        return []

    boxes  = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float32)
    scores = np.array([d[4] for d in dets], dtype=np.float32)

    # OpenCV NMS expects [x, y, w, h]
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] -= boxes[:, 0]
    boxes_xywh[:, 3] -= boxes[:, 1]

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(),
        score_threshold=config.CONF_THRESHOLD,
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    kept = [dets[i] for i in indices.flatten()]
    return kept

class DeepSORTTracker:
    def __init__(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age             = config.DEEPSORT_MAX_AGE,
                n_init              = config.DEEPSORT_N_INIT,
                max_iou_distance    = config.DEEPSORT_MAX_IOU_DIST,
                max_cosine_distance = config.DEEPSORT_MAX_COSINE_DIST,
                nn_budget           = config.DEEPSORT_NN_BUDGET,
                embedder            = "mobilenet",
                half                = False,
                bgr                 = True,
            )
            self._mode = "deepsort"
            logger.info("DeepSORT tracker ready")
        except ImportError:
            logger.warning("deep-sort-realtime not found — using simple IoU fallback")
            self.tracker = None
            self._mode   = "simple"
            self._tracks = []
            self._next_id = 1

    def update(self, dets, frame):
        if self._mode == "deepsort":
            return self._deepsort_update(dets, frame)
        return self._simple_update(dets)

    def _deepsort_update(self, dets, frame):
        ds_in = []
        for x1, y1, x2, y2, conf in dets:
            ds_in.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
        return self.tracker.update_tracks(ds_in, frame=frame)

    def _simple_update(self, dets):
        new_tracks = []
        used = set()
        for det in dets:
            x1, y1, x2, y2, conf = det
            best_iou, best_idx = 0.0, None
            for i, prev in enumerate(self._tracks):
                if i in used:
                    continue
                iou = _iou([x1, y1, x2, y2], prev["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou > 0.3 and best_idx is not None:
                tid = self._tracks[best_idx]["id"]
                used.add(best_idx)
            else:
                tid = self._next_id
                self._next_id += 1
            new_tracks.append(_SimpleTrack(tid, [x1, y1, x2, y2]))
        self._tracks = [{"id": t.track_id, "box": t.to_tlbr()} for t in new_tracks]
        return new_tracks


class _SimpleTrack:
    def __init__(self, tid, box):
        self.track_id = tid
        self._box     = box
    def is_confirmed(self):  return True
    def to_tlbr(self):       return self._box


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)

def open_source(source) -> cv2.VideoCapture:
    """Open camera / video file and return a configured VideoCapture."""
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # For video files, let OpenCV report the native resolution
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    return cap


def get_video_writer(cap: cv2.VideoCapture, out_path: Path) -> Optional[cv2.VideoWriter]:
    """Create a VideoWriter that matches the capture dimensions."""
    if not config.SAVE_OUTPUT_VIDEO:
        return None
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = config.OUTPUT_VIDEO_FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
    logger.info(f"Output video: {out_path}  ({config.DISPLAY_WIDTH}x{config.DISPLAY_HEIGHT} @ {fps}fps)")
    return writer

def run_inference(
    source=config.CAMERA_ID,
    device: str   = config.DEVICE,
    show:   bool  = config.SHOW_LIVE_FEED,
    conn          = None,
    out_path: Optional[Path] = None,
):
    """
    Main inference loop.  Works on live webcam, video files, and RTSP streams.

    Parameters
    ----------
    source   : int | str — camera index, video file path, or RTSP URL
    device   : compute device
    show     : display annotated window
    conn     : SQLite connection (created if None)
    out_path : path to save annotated output video (None = auto-name)
    """
    if conn is None:
        conn = init_db()

    # ── Load models ───────────────────────────────────────────────────
    ensemble  = EnsembleDetector(device)
    tracker   = DeepSORTTracker()
    stabilizer = CountStabilizer()
    alert_mgr  = AlertManager(conn)
    fps_meter  = FPSMeter()

    # ── Open source ───────────────────────────────────────────────────
    cap = open_source(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source: {source}  res={src_w}x{src_h}  frames={total_frames}")

    # ── Output video writer ───────────────────────────────────────────
    if out_path is None and config.SAVE_OUTPUT_VIDEO:
        ts       = time.strftime("%Y%m%d_%H%M%S")
        out_path = config.OUTPUT_VIDEO_DIR / f"annotated_{ts}.mp4"
    writer = get_video_writer(cap, out_path) if out_path else None

    frame_idx  = 0
    proc_count = 0

    logger.info(
        f"Inference started | device={device} "
        f"models={len(ensemble.detectors)} "
        f"conf={config.CONF_THRESHOLD} "
        f"strategy={config.ENSEMBLE_STRATEGY}"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            frame_idx += 1
            if frame_idx % config.SKIP_FRAMES != 0:
                continue
            proc_count += 1

            # Privacy: blur faces
            if config.BLUR_FACES:
                frame = blur_faces(frame)

            # ── Ensemble detection ─────────────────────────────────────
            all_dets, model_counts, raw_ensemble, latencies = ensemble.run(frame)

            # Record per-model latency for the FPS panel
            for lbl, elapsed in latencies.items():
                fps_meter.record(lbl, elapsed)

            # ── DeepSORT tracking ──────────────────────────────────────
            t_ds = time.time()
            tracks = tracker.update(all_dets, frame)
            fps_meter.record("DeepSORT", time.time() - t_ds)

            # Prefer confirmed track count (more stable than raw NMS count)
            confirmed = sum(1 for t in tracks if _simple_is_confirmed(t))
            raw_count = confirmed if confirmed > 0 else raw_ensemble

            # ── Temporal stabilisation (FIXES FLICKERING) ──────────────
            stable_count = stabilizer.update(raw_count)

            # ── Alert logic ────────────────────────────────────────────
            status = alert_mgr.update(stable_count, frame)

            # ── FPS (tick AFTER all processing) ───────────────────────
            fps = fps_meter.tick()

            # ── Annotate ───────────────────────────────────────────────
            annotated = annotate_frame(
                frame,
                tracks,
                stable_count,
                status,
                model_counts = model_counts,
                raw_count    = raw_count,
            )

            # ── Letterbox to fixed display size ───────────────────────
            display = letterbox_to_display(annotated)

            # ── Draw FPS panel on DISPLAY frame (AFTER letterbox) ─────
            # This is the fix: drawing BEFORE letterbox caused it to disappear
            fps_meter.draw_overlay(display, fps)

            if show:
                cv2.imshow("Bus Overcrowding Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("User quit.")
                    break
                if key == ord("s"):
                    _manual_snap(display, stable_count, status)
                if key == ord(" "):      # spacebar = pause
                    cv2.waitKey(0)

            if writer:
                writer.write(display)

            # ── Console heartbeat ──────────────────────────────────────
            if proc_count % 30 == 0:
                pct = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
                logger.info(
                    f"frame={pct} "
                    f"stable={stable_count} raw={raw_count} "
                    f"status={status} fps={fps:.1f} "
                    f"models={model_counts}"
                )

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
            logger.info(f"Output video saved: {out_path}")
        if show:
            cv2.destroyAllWindows()
        # Save FPS log CSV for report graphs
        if fps_meter._fps_log:
            fps_meter.save_fps_log(str(config.LOGS_DIR / "fps_log.csv"))
        # Print final FPS summary
        if len(fps_meter._fps_log) > 5:
            all_fps = [f for _, f in fps_meter._fps_log]
            logger.info(
                f"FPS SUMMARY | avg={sum(all_fps)/len(all_fps):.1f} "
                f"min={min(all_fps):.1f} max={max(all_fps):.1f} "
                f"frames_processed={proc_count}"
            )
        logger.info("Done.")


def _simple_is_confirmed(track) -> bool:
    if hasattr(track, "is_confirmed"):
        return track.is_confirmed()
    return True


def _manual_snap(frame, count, status):
    import datetime
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.ALERTS_IMG_DIR / f"manual_{status}_{count}_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    logger.info(f"Snapshot: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bus Overcrowding — Inference v2.0")
    parser.add_argument("--source",     default=str(config.CAMERA_ID),
                        help="Video file path, camera index, or RTSP URL")
    parser.add_argument("--device",     default=config.DEVICE,
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--out",        default=None,
                        help="Output video path (auto-named if omitted)")
    args = parser.parse_args()

    run_inference(
        source   = args.source,
        device   = args.device,
        show     = not args.no_display,
        out_path = Path(args.out) if args.out else None,
    )
