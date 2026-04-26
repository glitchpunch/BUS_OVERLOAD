# ============================================================
# training.py — YOLOv8 Fine-Tuning for Bus Overcrowding Detection
# ============================================================
"""
Trains (or fine-tunes) one or two YOLOv8 person-detection models on a
custom bus-interior dataset, then validates and exports to ONNX for
edge deployment.

Usage:
    python training.py                        # train both models
    python training.py --model primary        # train only primary (yolov8n)
    python training.py --model secondary      # train only secondary (yolov8s)
    python training.py --export-only          # export existing weights to ONNX
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import config
from logger import logger
from preprocessing import run_preprocessing


# ─────────────────────────────────────────────
# DEVICE AUTO-DETECT
# ─────────────────────────────────────────────

def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on availability."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("GPU detected: using CUDA")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon detected: using MPS")
            return "mps"
    except ImportError:
        logger.warning("torch not importable — defaulting to CPU")
    logger.info("No GPU found: using CPU")
    return "cpu"


# ─────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ─────────────────────────────────────────────

def train_model(
    model_name: str,
    run_name: str,
    data_yaml: Path,
    device: str,
    resume: bool = config.RESUME,
) -> Optional[Path]:
    """
    Fine-tune a single YOLOv8 model.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier, e.g. 'yolov8n.pt'
    run_name   : str
        Human-readable label for this run (used in output folder naming)
    data_yaml  : Path
        Path to the dataset.yaml file
    device     : str
        'cpu' | 'cuda' | 'mps'
    resume     : bool
        Resume from a previous interrupted run if True

    Returns
    -------
    Path to the best.pt weights file, or None on failure.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install ultralytics")
        return None

    logger.info(f"=== Training Start: {run_name} ({model_name}) ===")

    # Load base model (downloads from Ultralytics hub if not cached)
    model_path = config.MODELS_DIR / model_name
    if model_path.exists():
        logger.info(f"Loading local weights: {model_path}")
        model = YOLO(str(model_path))
    else:
        logger.info(f"Downloading base model: {model_name}")
        model = YOLO(model_name)        # downloads to Ultralytics cache

    # ── Training call ─────────────────────────────────────────────────
    results = model.train(
        data      = str(data_yaml),
        epochs    = config.EPOCHS,
        batch     = config.BATCH_SIZE,
        imgsz     = config.TRAIN_IMG_SIZE,
        lr0       = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
        patience  = config.PATIENCE,
        device    = device,
        workers   = config.WORKERS,
        augment   = config.AUGMENT,
        pretrained= config.PRETRAINED,
        resume    = resume,
        project   = config.TRAIN_PROJECT,
        name      = run_name,
        exist_ok  = True,
        verbose   = True,
        save      = True,
        save_period = 5,              # save checkpoint every 5 epochs
        plots     = True,             # generate training curves
        # Class-specific: only train on person (class 0)
        # If using a full COCO-pretrained model, restrict via dataset.yaml nc=1
    )

    # ── Locate best weights ───────────────────────────────────────────
    run_dir   = Path(config.TRAIN_PROJECT) / run_name
    best_pt   = run_dir / "weights" / "best.pt"

    if best_pt.exists():
        # Copy to models/ for easy access
        dest = config.MODELS_DIR / f"{run_name}_best.pt"
        shutil.copy(best_pt, dest)
        logger.info(f"Best weights saved to: {dest}")
        return dest
    else:
        logger.error(f"best.pt not found in {run_dir}")
        return None


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_model(weights_path: Path, data_yaml: Path, device: str) -> dict:
    """
    Run validation on the test split and return metrics dict.
    Logs mAP50, mAP50-95, precision, recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return {}

    logger.info(f"Validating model: {weights_path}")
    model   = YOLO(str(weights_path))
    metrics = model.val(
        data   = str(data_yaml),
        imgsz  = config.TRAIN_IMG_SIZE,
        device = device,
        split  = "test",
        verbose= True,
    )

    results = {
        "mAP50"    : round(metrics.box.map50, 4),
        "mAP50_95" : round(metrics.box.map,   4),
        "precision": round(metrics.box.mp,    4),
        "recall"   : round(metrics.box.mr,    4),
    }
    logger.info(f"Validation results: {results}")
    return results


# ─────────────────────────────────────────────
# ONNX EXPORT  (for edge deployment)
# ─────────────────────────────────────────────

def export_to_onnx(weights_path: Path, device: str) -> Optional[Path]:
    """
    Export a trained .pt model to ONNX format for edge deployment
    (e.g., on Raspberry Pi with ONNX Runtime, or Jetson with TensorRT).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return None

    logger.info(f"Exporting to ONNX: {weights_path}")
    model = YOLO(str(weights_path))
    export_path = model.export(
        format  = "onnx",
        imgsz   = config.IMG_SIZE,
        dynamic = False,        # Static shape — better for edge devices
        simplify= True,         # Simplify ONNX graph
        opset   = 12,           # Good compatibility with ONNX Runtime 1.x
        device  = device,
    )
    onnx_path = Path(export_path)
    logger.info(f"ONNX model saved: {onnx_path}")
    return onnx_path


# ─────────────────────────────────────────────
# OPTIONAL: POSE MODEL
# ─────────────────────────────────────────────

def train_pose_model(data_yaml: Path, device: str) -> Optional[Path]:
    """
    Fine-tune YOLOv8-Pose for keypoint-based density estimation.
    Only use this if USE_POSE=True and you have keypoint annotations.
    """
    if not config.USE_POSE:
        logger.info("Pose estimation disabled in config. Skipping.")
        return None

    logger.info("=== Pose Model Training ===")
    return train_model(
        model_name = config.POSE_MODEL_NAME,
        run_name   = "bus_overcrowd_pose",
        data_yaml  = data_yaml,
        device     = device,
    )


# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────

def run_training_pipeline(
    which: str   = "both",    # 'primary' | 'secondary' | 'both'
    device: str  = "auto",
    preprocess: bool = False,
) -> dict:
    """
    Orchestrate the full training workflow.

    Parameters
    ----------
    which      : which model(s) to train
    device     : compute device (auto-detected if 'auto')
    preprocess : run preprocessing pipeline first

    Returns
    -------
    dict mapping model name to best-weights path (or None on failure)
    """
    if device == "auto":
        device = detect_device()

    # ── Optional: run preprocessing first ─────────────────────────────
    if preprocess:
        data_yaml = run_preprocessing()
    else:
        data_yaml = config.TRAIN_DATA_YAML
        if not data_yaml.exists():
            logger.warning(
                f"dataset.yaml not found at {data_yaml}. "
                "Run preprocessing.py first, or pass --preprocess flag."
            )
            return {}

    results = {}

    # ── Primary model (yolov8n — lightweight, edge-optimised) ─────────
    if which in ("primary", "both"):
        primary_weights = train_model(
            model_name = config.PRIMARY_MODEL_NAME,
            run_name   = f"{config.TRAIN_NAME}_primary",
            data_yaml  = data_yaml,
            device     = device,
        )
        if primary_weights:
            logger.info("--- Primary model validation ---")
            validate_model(primary_weights, data_yaml, device)
            export_to_onnx(primary_weights, device)
            results["primary"] = primary_weights

    # ── Secondary model (yolov8s — larger, for ensemble reliability) ──
    if which in ("secondary", "both") and config.USE_ENSEMBLE:
        secondary_weights = train_model(
            model_name = config.SECONDARY_MODEL_NAME,
            run_name   = f"{config.TRAIN_NAME}_secondary",
            data_yaml  = data_yaml,
            device     = device,
        )
        if secondary_weights:
            logger.info("--- Secondary model validation ---")
            validate_model(secondary_weights, data_yaml, device)
            export_to_onnx(secondary_weights, device)
            results["secondary"] = secondary_weights

    logger.info(f"=== Training Pipeline Complete | Results: {results} ===")
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bus Overcrowding — YOLOv8 Training")
    parser.add_argument(
        "--model",
        choices=["primary", "secondary", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda | mps | cpu | auto",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing pipeline before training",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training; export existing best.pt files to ONNX",
    )
    args = parser.parse_args()

    device = detect_device() if args.device == "auto" else args.device

    if args.export_only:
        for suffix, path in [
            ("primary",   config.MODELS_DIR / f"{config.TRAIN_NAME}_primary_best.pt"),
            ("secondary", config.MODELS_DIR / f"{config.TRAIN_NAME}_secondary_best.pt"),
        ]:
            if path.exists():
                export_to_onnx(path, device)
            else:
                logger.warning(f"Weights not found for export: {path}")
    else:
        run_training_pipeline(
            which      = args.model,
            device     = device,
            preprocess = args.preprocess,
        )
