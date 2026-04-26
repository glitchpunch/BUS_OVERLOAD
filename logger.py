# ============================================================
# logger.py — Structured Logging for Bus Overcrowding Detection
# ============================================================
"""
Central logger using loguru for colourised console output and
auto-rotating file logging.  Import `logger` from this module
everywhere in the project instead of using Python's stdlib logging.

Usage:
    from logger import logger
    logger.info("System started")
    logger.warning("Occupancy at 90%")
    logger.error("Camera disconnected")
"""

import sys
from pathlib import Path
from loguru import logger as _logger

import config


def _setup_logger() -> None:
    """Configure loguru sinks (console + optional file)."""

    # Remove loguru's default stderr sink so we control the format
    _logger.remove()

    # ── Console sink ──────────────────────────────────────────────────
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )
    _logger.add(
        sys.stderr,
        format=console_fmt,
        level=config.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── File sink (rotating) ──────────────────────────────────────────
    if config.LOG_TO_FILE:
        log_path: Path = config.LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        )
        _logger.add(
            str(log_path),
            format=file_fmt,
            level=config.LOG_LEVEL,
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=False,   # Avoid leaking sensitive locals to disk
            enqueue=True,     # Thread-safe async writes
        )

    _logger.info(
        f"Logger initialised | level={config.LOG_LEVEL} "
        f"| file_logging={config.LOG_TO_FILE}"
    )


# Run setup once at import time
_setup_logger()

# Re-export for project-wide use
logger = _logger


# ─────────────────────────────────────────────
# Optional: performance timing context manager
# ─────────────────────────────────────────────
import time
from contextlib import contextmanager


@contextmanager
def log_time(label: str):
    """Context manager that logs execution time of a code block.

    Example:
        with log_time("YOLOv8 inference"):
            results = model(frame)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"[TIMER] {label} — {elapsed_ms:.1f} ms")
