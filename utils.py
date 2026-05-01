import sqlite3
import time
import datetime
import collections
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import config
from logger import logger


class CountStabilizer:
    """
    Smooths the raw per-frame person count to eliminate flickering.

    Problem: YOLO detects 23 people on frame 10, 19 on frame 11, 24 on frame 12.
             The displayed number jumps every frame — looks broken.
    Solution: Keep a rolling window of the last N counts and return the median.
              Median is robust: a single bad frame can't shift the displayed number.

    Usage:
        stabilizer = CountStabilizer()
        stable_count = stabilizer.update(raw_count)
    """

    def __init__(
        self,
        window : int = config.STABILIZER_WINDOW,
        method : str = config.STABILIZER_METHOD,
        alpha  : float = config.EWM_ALPHA,
    ):
        self.window  = window
        self.method  = method
        self.alpha   = alpha
        self._buffer : collections.deque = collections.deque(maxlen=window)
        self._ewm    : float = 0.0
        self._initialized = False

    def update(self, raw_count: int) -> int:
        """Feed a new raw count; returns the stabilised count."""
        self._buffer.append(raw_count)

        if self.method == "median":
            return int(np.median(list(self._buffer)))

        if self.method == "mean":
            return int(round(np.mean(list(self._buffer))))

        if self.method == "ewm":
            if not self._initialized:
                self._ewm = float(raw_count)
                self._initialized = True
            else:
                self._ewm = self.alpha * raw_count + (1 - self.alpha) * self._ewm
            return int(round(self._ewm))

        # fallback
        return raw_count

    def reset(self):
        self._buffer.clear()
        self._initialized = False

def ensemble_count(
    counts  : List[int],
    weights : Optional[List[int]] = None,
    strategy: str = config.ENSEMBLE_STRATEGY,
) -> int:
    """
    Combine person counts from 2–4 models into one reliable count.

    Strategies
    ----------
    weighted_max  : weighted average then round up — safety-first
    weighted_mean : straight weighted average
    max           : take the maximum count (most conservative)
    median        : median of all counts (robust to outliers)
    """
    if not counts:
        return 0

    w = weights if weights else [1] * len(counts)

    if strategy == "max":
        return max(counts)

    if strategy == "median":
        return int(np.median(counts))

    if strategy in ("weighted_mean", "weighted_max"):
        total_w   = sum(w)
        w_avg     = sum(c * wt for c, wt in zip(counts, w)) / total_w
        if strategy == "weighted_max":
            return int(np.ceil(w_avg))
        return int(round(w_avg))

    return round(sum(counts) / len(counts))

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            bus_id        TEXT    NOT NULL,
            timestamp     TEXT    NOT NULL,
            event_type    TEXT    NOT NULL,
            person_count  INTEGER NOT NULL,
            max_capacity  INTEGER NOT NULL,
            fine_inr      INTEGER DEFAULT 0,
            snapshot_path TEXT,
            notes         TEXT
        );
    """)
    conn.commit()
    logger.info(f"Database ready: {config.DB_PATH}")
    return conn


def log_event(conn, event_type, person_count, snapshot_path=None, notes=None):
    fine = config.FINE_AMOUNT_INR if event_type == "OVERCROWD" else 0
    ts   = datetime.datetime.now().isoformat(timespec="seconds")
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO events (bus_id,timestamp,event_type,person_count,"
        "max_capacity,fine_inr,snapshot_path,notes) VALUES (?,?,?,?,?,?,?,?)",
        (config.BUS_ID, ts, event_type, person_count,
         config.MAX_CAPACITY, fine, snapshot_path, notes),
    )
    conn.commit()
    logger.info(f"[DB] {event_type} | count={person_count} fine=INR{fine}")
    return cur.lastrowid


def fetch_recent_events(conn, limit=50):
    cur = conn.cursor()
    cur.execute(
        "SELECT id,bus_id,timestamp,event_type,person_count,"
        "max_capacity,fine_inr,snapshot_path,notes "
        "FROM events ORDER BY id DESC LIMIT ?", (limit,)
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

class AlertManager:
    def __init__(self, conn):
        self.conn              = conn
        self._consec_warn      = 0
        self._consec_overcrowd = 0
        self._last_alert_time  = 0.0

    def update(self, person_count: int, frame=None) -> str:
        now = time.time()
        if person_count >= config.OVERCROWD_THRESHOLD:
            self._consec_overcrowd += 1
            self._consec_warn       = 0
            status = "OVERCROWD"
            if (self._consec_overcrowd >= config.CONSECUTIVE_FRAMES_ALERT
                    and now - self._last_alert_time > config.ALERT_COOLDOWN_SEC):
                snap = _save_snapshot(frame, status) if frame is not None else None
                log_event(self.conn, "OVERCROWD", person_count, snap)
                self._last_alert_time = now

        elif person_count >= config.WARNING_THRESHOLD:
            self._consec_warn      += 1
            self._consec_overcrowd  = 0
            status = "WARNING"
            if (self._consec_warn >= config.CONSECUTIVE_FRAMES_ALERT
                    and now - self._last_alert_time > config.ALERT_COOLDOWN_SEC):
                snap = _save_snapshot(frame, status) if frame is not None else None
                log_event(self.conn, "WARNING", person_count, snap)
                self._last_alert_time = now
        else:
            self._consec_warn      = 0
            self._consec_overcrowd = 0
            status = "OK"
        return status


def _save_snapshot(frame: np.ndarray, status: str) -> Optional[str]:
    if not config.SAVE_ALERTS_IMG or frame is None:
        return None
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.ALERTS_IMG_DIR / f"{status}_{ts}_{config.BUS_ID}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


# ─────────────────────────────────────────────
# FRAME DISPLAY — letterbox to fixed size
# ─────────────────────────────────────────────

def letterbox_to_display(frame: np.ndarray) -> np.ndarray:
    """
    Resize frame to DISPLAY_WIDTH x DISPLAY_HEIGHT with letterboxing
    (black bars) so the aspect ratio is never distorted.
    """
    h, w      = frame.shape[:2]
    target_w  = config.DISPLAY_WIDTH
    target_h  = config.DISPLAY_HEIGHT
    scale     = min(target_w / w, target_h / h)
    new_w     = int(w * scale)
    new_h     = int(h * scale)
    resized   = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas    = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top   = (target_h - new_h) // 2
    pad_left  = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


# ─────────────────────────────────────────────
# FRAME ANNOTATION  v2.0
# ─────────────────────────────────────────────

def annotate_frame(
    frame        : np.ndarray,
    tracks       : List,
    person_count : int,
    status       : str,
    model_counts : Optional[Dict[str, int]] = None,
    raw_count    : Optional[int] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, capacity progress bar,
    model breakdown panel, and alert banner onto a copy of `frame`.
    """
    out   = frame.copy()
    color = _status_color(status)
    h, w  = out.shape[:2]

    # ── 1. Bounding boxes + track IDs ─────────────────────────────────
    for track in tracks:
        if not _is_confirmed(track):
            continue
        x1, y1, x2, y2 = [int(v) for v in _get_box(track)]
        tid = track.track_id

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"#{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    # ── 2. Left panel — occupancy HUD ─────────────────────────────────
    panel_w = 300
    panel_h = 200 if (config.SHOW_MODEL_BREAKDOWN and model_counts) else 155
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Bus ID + status
    cv2.putText(out, config.BUS_ID, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, status, (12, 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Person count (big)
    count_str = str(person_count)
    (cw, ch), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3)
    cv2.putText(out, count_str, (12, 100),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, color, 3, cv2.LINE_AA)
    cv2.putText(out, f"/ {config.MAX_CAPACITY}", (12 + cw + 4, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(out, "PASSENGERS", (12, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    # ── 3. Capacity progress bar ───────────────────────────────────────
    bar_x, bar_y = 12, 130
    bar_w, bar_h = panel_w - 24, 14
    fill_ratio   = min(person_count / config.MAX_CAPACITY, 1.0)
    fill_w       = int(bar_w * fill_ratio)

    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    if fill_w > 0:
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    # Threshold tick mark
    warn_x = bar_x + int(bar_w * config.WARNING_RATIO)
    cv2.line(out, (warn_x, bar_y - 2), (warn_x, bar_y + bar_h + 2), (0, 165, 255), 2)

    pct_text = f"{int(fill_ratio * 100)}%"
    cv2.putText(out, pct_text, (bar_x + bar_w + 6, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Fine amount (if overcrowded)
    if status == "OVERCROWD":
        cv2.putText(out, f"Fine: INR {config.FINE_AMOUNT_INR:,}", (12, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1, cv2.LINE_AA)

    # ── 4. Model breakdown (mini panel) ───────────────────────────────
    if config.SHOW_MODEL_BREAKDOWN and model_counts:
        y_off = 162 if status != "OVERCROWD" else 170
        cv2.putText(out, "MODEL VOTES:", (12, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
        y_off += 15
        for m_label, m_count in model_counts.items():
            cv2.putText(out, f"  {m_label}: {m_count}", (12, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 180, 120), 1, cv2.LINE_AA)
            y_off += 13

    # ── 5. Raw vs stable count indicator ──────────────────────────────
    if raw_count is not None and raw_count != person_count:
        cv2.putText(out, f"raw:{raw_count} stable:{person_count}",
                    (12, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    # ── 6. Alert banner (bottom centre) ───────────────────────────────
    if status != "OK":
        banner = ("!! OVERCROWDING DETECTED — FINE APPLICABLE !!"
                  if status == "OVERCROWD"
                  else "! CAPACITY WARNING — APPROACHING LIMIT !")
        font   = cv2.FONT_HERSHEY_DUPLEX
        scale  = 0.72
        thick  = 2
        (bw, bh), _ = cv2.getTextSize(banner, font, scale, thick)
        bx = (w - bw) // 2
        by = h - 22

        # Flashing effect: alternate opacity based on time
        alpha = 0.85 if (int(time.time() * 2) % 2 == 0) else 0.60
        bar_overlay = out.copy()
        cv2.rectangle(bar_overlay, (bx - 14, by - bh - 10),
                      (bx + bw + 14, by + 10), (0, 0, 0), -1)
        cv2.addWeighted(bar_overlay, alpha, out, 1 - alpha, 0, out)
        cv2.putText(out, banner, (bx, by), font, scale, color, thick, cv2.LINE_AA)

    return out


def _status_color(status: str) -> Tuple[int, int, int]:
    return {
        "OK"       : config.COLOR_OK,
        "WARNING"  : config.COLOR_WARN,
        "OVERCROWD": config.COLOR_ALERT,
    }.get(status, config.COLOR_OK)


def _is_confirmed(track) -> bool:
    if hasattr(track, "is_confirmed"):
        return track.is_confirmed()
    return True


def _get_box(track):
    if hasattr(track, "to_tlbr"):
        return track.to_tlbr()
    return getattr(track, "_box", [0, 0, 0, 0])


# ─────────────────────────────────────────────
# FACE BLURRING
# ─────────────────────────────────────────────

def blur_faces(frame: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    out   = frame.copy()
    for (x, y, fw, fh) in faces:
        out[y:y+fh, x:x+fw] = cv2.GaussianBlur(out[y:y+fh, x:x+fw], (51, 51), 30)
    return out


# ─────────────────────────────────────────────
# FPS METER
# ─────────────────────────────────────────────

class FPSMeter:
    """
    Rolling-average FPS meter with per-component latency tracking.

    Usage:
        meter = FPSMeter()
        fps   = meter.tick()               # call once per processed frame
        meter.record("yolov8n", 0.045)     # record a model latency (seconds)
        meter.draw_overlay(display, fps)   # draw on the DISPLAY frame
    """

    def __init__(self, window: int = 60):
        self._times: collections.deque = collections.deque(maxlen=window)
        self._latencies: dict = collections.defaultdict(
            lambda: collections.deque(maxlen=30)
        )
        self._fps_log: list = []


    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)

        if len(self._times) < 2:
            return 20.0  

        elapsed = self._times[-1] - self._times[0]
        fps = (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0   

        self._fps_log.append((now, fps))
        return fps

    def record(self, label: str, elapsed_sec: float):
        """Record latency for a named component (e.g. 'yolov8n')."""
        self._latencies[label].append(elapsed_sec)

    def avg_latency(self, label: str) -> float:
        buf = self._latencies.get(label)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def draw_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw a dark semi-transparent FPS + latency panel on the
        BOTTOM-RIGHT of the frame.
        IMPORTANT: call this on the DISPLAY frame (after letterboxing),
        NOT on the raw annotated frame — otherwise letterbox erases it.
        """
        h, w = frame.shape[:2]

        lines = [f"FPS:  {fps:5.1f}"]
        for label, buf in self._latencies.items():
            if buf:
                ms = sum(buf) / len(buf) * 1000
                lines.append(f"{label}: {ms:5.1f} ms")

        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale  = 0.52
        thick  = 1
        pad    = 8
        line_h = 20
        max_w  = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
        box_w  = max_w + pad * 2
        box_h  = len(lines) * line_h + pad * 2

        bx = w - box_w - 10
        by = h - box_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (80, 200, 160), 1)

        for i, line in enumerate(lines):
            tx = bx + pad
            ty = by + pad + (i + 1) * line_h - 4
            if i == 0:
                color = (50, 220, 50) if fps >= 15 else \
                        (0, 165, 255) if fps >= 8  else \
                        (0, 80, 255)
            else:
                color = (180, 220, 180)
            cv2.putText(frame, line, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

        return frame

    def draw(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Backward-compat alias — prefer draw_overlay() on the display frame."""
        return self.draw_overlay(frame, fps)

    def save_fps_log(self, path: str = "fps_log.csv"):
        """Save full FPS history to CSV — use for your report graphs."""
        import csv
        with open(path, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["timestamp_sec", "fps"])
            t0 = self._fps_log[0][0] if self._fps_log else 0
            for t, fps_val in self._fps_log:
                w_csv.writerow([round(t - t0, 3), round(fps_val, 2)])
        logger.info(f"FPS log saved: {path}")
