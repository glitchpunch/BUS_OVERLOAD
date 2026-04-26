# ============================================================
# main.py — Unified Entry Point for Bus Overcrowding Detection
# ============================================================
"""
Starts the full system:
  • Inference thread  — real-time YOLO + DeepSORT pipeline
  • Flask dashboard   — web UI for reviewing alerts (optional)

Usage:
    python main.py                          # inference + dashboard
    python main.py --mode inference         # inference only
    python main.py --mode dashboard         # dashboard only
    python main.py --mode train             # training pipeline
    python main.py --mode preprocess        # preprocessing only
    python main.py --source video.mp4       # use a video file
    python main.py --no-display             # headless (no OpenCV window)
"""

import argparse
import threading
import sys
import signal
from pathlib import Path

import config
from logger import logger
from utils import init_db, fetch_recent_events


# ─────────────────────────────────────────────
# DEVICE AUTO-DETECTION
# ─────────────────────────────────────────────

def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected")
            return "mps"
    except ImportError:
        pass
    logger.info("Running on CPU")
    return "cpu"


# ─────────────────────────────────────────────
# FLASK DASHBOARD
# ─────────────────────────────────────────────

def build_flask_app(conn):
    """
    Construct and return a Flask application for the alert dashboard.
    Routes:
      GET  /            — dashboard home (table of recent alerts)
      GET  /api/events  — JSON endpoint (for AJAX refresh or external monitoring)
      GET  /api/status  — current bus status (latest count + status)
    """
    try:
        from flask import Flask, render_template_string, jsonify
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask")
        return None

    app = Flask(__name__)

    DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bus Overcrowding Monitor — {{ bus_id }}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Courier New', monospace;
      background: #0d0d0d; color: #e0e0e0;
      min-height: 100vh; padding: 24px;
    }
    header {
      display: flex; align-items: center; gap: 16px;
      border-bottom: 2px solid #333; padding-bottom: 16px; margin-bottom: 24px;
    }
    header h1 { font-size: 1.5rem; color: #f5f5f5; letter-spacing: 2px; }
    header span { font-size: 0.85rem; color: #888; }
    .stats {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px; margin-bottom: 28px;
    }
    .card {
      background: #1a1a1a; border: 1px solid #2a2a2a;
      border-radius: 8px; padding: 18px;
    }
    .card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .card .value { font-size: 2rem; font-weight: bold; margin-top: 6px; }
    .card.ok    .value { color: #00c878; }
    .card.warn  .value { color: #ff9f43; }
    .card.alert .value { color: #ff4757; }
    .card.fine  .value { color: #eccc68; }
    table {
      width: 100%; border-collapse: collapse;
      background: #141414; border-radius: 8px; overflow: hidden;
    }
    th {
      background: #1e1e1e; padding: 12px 16px; text-align: left;
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #888;
    }
    td { padding: 12px 16px; border-bottom: 1px solid #1e1e1e; font-size: 0.88rem; }
    tr:last-child td { border-bottom: none; }
    .badge {
      display: inline-block; padding: 3px 10px;
      border-radius: 4px; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px;
    }
    .badge.OK        { background: #0a3d20; color: #00c878; }
    .badge.WARNING   { background: #3d2a0a; color: #ff9f43; }
    .badge.OVERCROWD { background: #3d0a0a; color: #ff4757; }
    .refresh { font-size: 0.75rem; color: #555; margin-bottom: 12px; }
  </style>
  <meta http-equiv="refresh" content="10">
</head>
<body>
  <header>
    <h1>BUS OVERCROWDING MONITOR</h1>
    <span>{{ bus_id }} &nbsp;|&nbsp; Capacity: {{ max_capacity }}</span>
  </header>

  <div class="stats">
    <div class="card ok">
      <div class="label">Total Events</div>
      <div class="value">{{ total }}</div>
    </div>
    <div class="card warn">
      <div class="label">Warnings</div>
      <div class="value">{{ warnings }}</div>
    </div>
    <div class="card alert">
      <div class="label">Overcrowd Alerts</div>
      <div class="value">{{ overcrowds }}</div>
    </div>
    <div class="card fine">
      <div class="label">Total Fines (INR)</div>
      <div class="value">{{ total_fine }}</div>
    </div>
  </div>

  <div class="refresh">Auto-refreshes every 10 s &nbsp;|&nbsp; Showing last 50 events</div>

  <table>
    <thead>
      <tr>
        <th>#</th><th>Timestamp</th><th>Bus</th>
        <th>Type</th><th>Count</th><th>Limit</th><th>Fine (INR)</th><th>Snapshot</th>
      </tr>
    </thead>
    <tbody>
      {% for ev in events %}
      <tr>
        <td>{{ ev.id }}</td>
        <td>{{ ev.timestamp }}</td>
        <td>{{ ev.bus_id }}</td>
        <td><span class="badge {{ ev.event_type }}">{{ ev.event_type }}</span></td>
        <td>{{ ev.person_count }}</td>
        <td>{{ ev.max_capacity }}</td>
        <td>{{ ev.fine_inr if ev.fine_inr else '—' }}</td>
        <td>{{ ev.snapshot_path.split('/')[-1] if ev.snapshot_path else '—' }}</td>
      </tr>
      {% endfor %}
      {% if not events %}
      <tr><td colspan="8" style="text-align:center;color:#555;padding:32px">No alerts logged yet.</td></tr>
      {% endif %}
    </tbody>
  </table>
</body>
</html>
"""

    @app.route("/")
    def dashboard():
        events     = fetch_recent_events(conn, limit=50)
        total      = len(events)
        warnings   = sum(1 for e in events if e["event_type"] == "WARNING")
        overcrowds = sum(1 for e in events if e["event_type"] == "OVERCROWD")
        total_fine = sum(e["fine_inr"] for e in events)
        return render_template_string(
            DASHBOARD_HTML,
            events       = events,
            bus_id       = config.BUS_ID,
            max_capacity = config.MAX_CAPACITY,
            total        = total,
            warnings     = warnings,
            overcrowds   = overcrowds,
            total_fine   = total_fine,
        )

    @app.route("/api/events")
    def api_events():
        return jsonify(fetch_recent_events(conn, limit=100))

    @app.route("/api/status")
    def api_status():
        events = fetch_recent_events(conn, limit=1)
        latest = events[0] if events else {}
        return jsonify({
            "bus_id"      : config.BUS_ID,
            "max_capacity": config.MAX_CAPACITY,
            "latest_event": latest,
        })

    return app


# ─────────────────────────────────────────────
# THREAD RUNNERS
# ─────────────────────────────────────────────

def run_inference_thread(source, device, show, conn, stop_event: threading.Event):
    """Wrapper to run inference in a background thread."""
    from inference import run_inference
    try:
        run_inference(source=source, device=device, show=show, conn=conn)
    except Exception as exc:
        logger.exception(f"Inference thread crashed: {exc}")
    finally:
        stop_event.set()


def run_flask_thread(app, stop_event: threading.Event):
    """Run Flask in a daemon thread."""
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)   # Silence request noise
    try:
        app.run(
            host  = config.FLASK_HOST,
            port  = config.FLASK_PORT,
            debug = False,
            use_reloader = False,
        )
    except Exception as exc:
        logger.exception(f"Flask thread crashed: {exc}")
    finally:
        stop_event.set()


# ─────────────────────────────────────────────
# SYSTEM BANNER
# ─────────────────────────────────────────────

def print_banner(device: str, source, mode: str):
    banner = f"""
╔══════════════════════════════════════════════════════╗
║       BUS OVERCROWDING DETECTION SYSTEM              ║
║       Edge AI — Privacy-Preserving Pipeline          ║
╠══════════════════════════════════════════════════════╣
║  Bus ID    : {config.BUS_ID:<39}║
║  Capacity  : {config.MAX_CAPACITY:<39}║
║  Warning   : {config.WARNING_THRESHOLD:<39}║
║  Device    : {device:<39}║
║  Source    : {str(source):<39}║
║  Mode      : {mode:<39}║
║  Dashboard : http://{config.FLASK_HOST}:{config.FLASK_PORT:<33}║
╚══════════════════════════════════════════════════════╝
"""
    print(banner)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bus Overcrowding Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "dashboard", "both", "train", "preprocess"],
        default="both",
        help="Operating mode",
    )
    parser.add_argument(
        "--source",
        default=str(config.CAMERA_ID),
        help="Video source: camera index, video file, or RTSP URL",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run inference headless (no OpenCV window)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run dataset preprocessing before training",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────
    device = detect_device() if args.device == "auto" else args.device

    # ── Video source ──────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    show   = not args.no_display

    print_banner(device, source, args.mode)

    # ── Preprocessing mode ────────────────────────────────────────────
    if args.mode == "preprocess":
        from preprocessing import run_preprocessing
        run_preprocessing()
        return

    # ── Training mode ─────────────────────────────────────────────────
    if args.mode == "train":
        from training import run_training_pipeline
        run_training_pipeline(
            which      = "both",
            device     = device,
            preprocess = args.preprocess,
        )
        return

    # ── Runtime modes (inference / dashboard / both) ──────────────────
    conn       = init_db()
    stop_event = threading.Event()

    threads = []

    if args.mode in ("inference", "both"):
        inf_thread = threading.Thread(
            target = run_inference_thread,
            args   = (source, device, show, conn, stop_event),
            daemon = True,
            name   = "InferenceThread",
        )
        threads.append(inf_thread)

    if args.mode in ("dashboard", "both"):
        app = build_flask_app(conn)
        if app:
            dash_thread = threading.Thread(
                target = run_flask_thread,
                args   = (app, stop_event),
                daemon = True,
                name   = "DashboardThread",
            )
            threads.append(dash_thread)
            logger.info(
                f"Dashboard available at "
                f"http://{config.FLASK_HOST}:{config.FLASK_PORT}"
            )

    # ── Start threads ─────────────────────────────────────────────────
    for t in threads:
        t.start()
        logger.info(f"Thread started: {t.name}")

    # ── Graceful shutdown on SIGINT / SIGTERM ─────────────────────────
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received. Stopping…")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block main thread until all daemons stop
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass

    logger.info("System shut down cleanly.")


if __name__ == "__main__":
    main()
