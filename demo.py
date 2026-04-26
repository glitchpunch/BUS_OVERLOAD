#!/usr/bin/env python3
# ============================================================
# demo.py — One-Command Demo Runner for Bus Demo Video
# ============================================================
"""
The fastest way to test the system on your bus demo video.

Usage:
    python demo.py                              # uses demo_video.mp4 in project root
    python demo.py --video path/to/bus.mp4
    python demo.py --video bus.mp4 --capacity 30   # override bus capacity
    python demo.py --video bus.mp4 --device cuda   # use GPU
    python demo.py --video bus.mp4 --models n s    # use only yolov8n + yolov8s

What it does:
  1. Loads your video file
  2. Runs 4-model YOLOv8 ensemble detection (n + s + m + l)
  3. Applies DeepSORT tracking for stable identity-based counting
  4. Applies temporal smoothing (median filter) to fix count flickering
  5. Displays annotated video with:
       - Per-person bounding boxes + track IDs
       - Live occupancy count (big number)
       - Capacity progress bar
       - Per-model vote breakdown
       - Alert banner + fine amount
  6. Saves annotated output video to output_videos/
  7. Logs all alert events to alerts.db

Keyboard shortcuts in the display window:
  Q / ESC  — quit
  SPACE    — pause / resume
  S        — save snapshot of current frame
"""

import argparse
import sys
import time
from pathlib import Path


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Apple Silicon MPS")
            return "mps"
    except ImportError:
        pass
    print("[INFO] No GPU — running on CPU")
    return "cpu"


def patch_config(args):
    """Override config values from CLI flags before loading inference."""
    import config

    if args.capacity:
        config.MAX_CAPACITY        = args.capacity
        config.WARNING_THRESHOLD   = int(args.capacity * config.WARNING_RATIO)
        config.OVERCROWD_THRESHOLD = args.capacity
        print(f"[CONFIG] Capacity set to {args.capacity}")

    if args.conf:
        config.CONF_THRESHOLD = args.conf
        print(f"[CONFIG] Confidence threshold: {args.conf}")

    if args.models:
        enabled = set(args.models)
        for m in config.MODEL_CONFIGS:
            size = m["name"].replace("yolov8", "").replace(".pt", "")  # 'n','s','m','l'
            m["enabled"] = size in enabled
        active = [m["label"] for m in config.MODEL_CONFIGS if m["enabled"]]
        print(f"[CONFIG] Models enabled: {active}")

    if args.no_save:
        config.SAVE_OUTPUT_VIDEO = False
        config.SAVE_ALERTS_IMG   = False

    if args.strategy:
        config.ENSEMBLE_STRATEGY = args.strategy


def print_banner(video_path: str, device: str, capacity: int):
    active_models = []
    try:
        import config
        active_models = [m["label"] for m in config.MODEL_CONFIGS if m["enabled"]]
    except Exception:
        pass

    print("\n" + "═" * 60)
    print("  BUS OVERCROWDING DETECTION — DEMO")
    print("  4-Model Ensemble | DeepSORT | Temporal Smoothing")
    print("═" * 60)
    print(f"  Video    : {video_path}")
    print(f"  Device   : {device}")
    print(f"  Capacity : {capacity}")
    print(f"  Models   : {', '.join(active_models)}")
    print(f"  Strategy : {__import__('config').ENSEMBLE_STRATEGY}")
    print(f"  Conf     : {__import__('config').CONF_THRESHOLD}")
    print(f"  Smooth   : {__import__('config').STABILIZER_METHOD} "
          f"window={__import__('config').STABILIZER_WINDOW}")
    print("═" * 60)
    print("  Controls: Q=quit  SPACE=pause  S=snapshot")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bus Overcrowding Detection — Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", "-v",
        default="demo_video.mp4",
        help="Path to bus demo video file (default: demo_video.mp4)",
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device",
    )
    parser.add_argument(
        "--capacity", "-c",
        type=int,
        default=None,
        help="Override bus max capacity (default: from config.py)",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["n", "s", "m", "l"],
        default=None,
        help="Which YOLOv8 models to use: n s m l (default: all 4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--strategy",
        choices=["weighted_max", "weighted_mean", "max", "median"],
        default=None,
        help="Ensemble counting strategy",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no window, just save output video)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output video or snapshots",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output video path (auto-named if omitted)",
    )
    args = parser.parse_args()

    # ── Validate video file ────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n[ERROR] Video file not found: {video_path}")
        print("  Place your bus demo video in the project root as 'demo_video.mp4'")
        print("  or pass its path: python demo.py --video /path/to/bus.mp4\n")
        sys.exit(1)

    # ── Patch config BEFORE importing inference ────────────────────────
    patch_config(args)

    # ── Device ────────────────────────────────────────────────────────
    device = detect_device() if args.device == "auto" else args.device

    # ── Print banner ───────────────────────────────────────────────────
    import config
    print_banner(str(video_path), device, config.MAX_CAPACITY)

    # ── Run ────────────────────────────────────────────────────────────
    from inference import run_inference
    from utils import init_db

    conn     = init_db()
    out_path = Path(args.out) if args.out else None

    t0 = time.time()
    run_inference(
        source   = str(video_path),
        device   = device,
        show     = not args.no_display,
        conn     = conn,
        out_path = out_path,
    )
    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────
    from utils import fetch_recent_events
    events     = fetch_recent_events(conn, limit=200)
    warnings   = sum(1 for e in events if e["event_type"] == "WARNING")
    overcrowds = sum(1 for e in events if e["event_type"] == "OVERCROWD")
    fine_total = sum(e["fine_inr"] for e in events)

    print("\n" + "═" * 60)
    print("  DEMO COMPLETE")
    print("═" * 60)
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"  Warning alerts : {warnings}")
    print(f"  Overcrowd alerts: {overcrowds}")
    print(f"  Total fines    : INR {fine_total:,}")
    if config.SAVE_OUTPUT_VIDEO and out_path:
        print(f"  Output video   : {out_path}")
    elif config.SAVE_OUTPUT_VIDEO:
        import os
        vids = sorted(config.OUTPUT_VIDEO_DIR.glob("*.mp4"))
        if vids:
            print(f"  Output video   : {vids[-1]}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
