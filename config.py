from pathlib import Path

BASE_DIR        = Path(__file__).parent.resolve()
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
DATASET_DIR     = DATA_DIR / "dataset"
MODELS_DIR      = BASE_DIR / "models"
RUNS_DIR        = BASE_DIR / "runs"
LOGS_DIR        = BASE_DIR / "logs"
DB_PATH         = BASE_DIR / "alerts.db"

for _d in [RAW_DIR, PROCESSED_DIR, DATASET_DIR, MODELS_DIR, RUNS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

CAMERA_ID       = 0
FRAME_WIDTH     = 900
FRAME_HEIGHT    = 480
DISPLAY_WIDTH   = 900      
DISPLAY_HEIGHT  = 480     
TARGET_FPS      = 30
SKIP_FRAMES     = 2        

BUS_ID          = "BUS-001"
MAX_CAPACITY    = 15
WARNING_RATIO   = 0.9
FINE_AMOUNT_INR = 500

WARNING_THRESHOLD   = int(MAX_CAPACITY * WARNING_RATIO)
OVERCROWD_THRESHOLD = MAX_CAPACITY

# ─────────────────────────────────────────────
# 4-MODEL ENSEMBLE CONFIGURATION
# ─────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "name"   : "yolov8n.pt",
        "path"   : MODELS_DIR / "yolov8n.pt",
        "weight" : 1,
        "label"  : "YOLOv8n",
        "enabled": True,
    },
    {
        "name"   : "yolov8s.pt",
        "path"   : MODELS_DIR / "yolov8s.pt",
        "weight" : 2,
        "label"  : "YOLOv8s",
        "enabled": True,
    },
    {
        "name"   : "yolov8m.pt",
        "path"   : MODELS_DIR / "yolov8m.pt",
        "weight" : 3,
        "label"  : "YOLOv8m",
        "enabled": True,
    },
    {
        "name"   : "yolov8l.pt",
        "path"   : MODELS_DIR / "yolov8l.pt",
        "weight" : 4,
        "label"  : "YOLOv8l",
        "enabled": True,
    },
]

# 'weighted_max' — safety-first (favours higher count)
# 'weighted_mean' — balanced average
# 'max'           — most conservative
# 'median'        — robust to outliers
ENSEMBLE_STRATEGY = "primary"

CONF_THRESHOLD  = 0.25      
IOU_THRESHOLD   = 0.40      
PERSON_CLASS_ID = 0
IMG_SIZE        = 416
DEVICE          = "cuda"

STABILIZER_WINDOW  = 20     
STABILIZER_METHOD  = "median"   
EWM_ALPHA          = 0.30   

DEEPSORT_MAX_AGE         = 60   # Keep tracks alive longer for seated people
DEEPSORT_N_INIT          = 2    # Confirm track after 2 frames (was 3)
DEEPSORT_MAX_IOU_DIST    = 0.8
DEEPSORT_MAX_COSINE_DIST = 0.5
DEEPSORT_NN_BUDGET       = 200

ALERT_COOLDOWN_SEC       = 30
CONSECUTIVE_FRAMES_ALERT = 8

FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False

LOG_LEVEL     = "INFO"
LOG_TO_FILE   = True
LOG_FILE      = LOGS_DIR / "app.log"
LOG_ROTATION  = "10 MB"
LOG_RETENTION = "7 days"

SHOW_LIVE_FEED  = True
SAVE_ALERTS_IMG = True
ALERTS_IMG_DIR  = BASE_DIR / "alert_snapshots"
ALERTS_IMG_DIR.mkdir(exist_ok=True)

COLOR_OK    = (50, 220, 50)
COLOR_WARN  = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)
COLOR_PANEL = (15, 15, 15)

SHOW_MODEL_BREAKDOWN = True     # Show per-model counts in HUD

SAVE_OUTPUT_VIDEO  = True
OUTPUT_VIDEO_DIR   = BASE_DIR / "output_videos"
OUTPUT_VIDEO_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO_FPS   = 20

# ─────────────────────────────────────────────
# PRIVACY
# ─────────────────────────────────────────────
BLUR_FACES      = False
STORE_RAW_VIDEO = False
