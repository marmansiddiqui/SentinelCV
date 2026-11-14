from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolo11n.pt"

IMAGE_SIZE = 512
CONFIDENCE = 0.40

DEVICE = "cpu"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

EVENT_COOLDOWN = 30

DATABASE_PATH = BASE_DIR / "data" / "sentinel.db"

EVENTS_DIR = BASE_DIR / "data" / "events"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"

for directory in [EVENTS_DIR, SNAPSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)