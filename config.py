# config.py
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///db.sqlite")
REDIS_URL = os.environ.get("REDIS_URL", "")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "darkhorse_super_secret_2026")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", 100))
MAX_FILE_AGE_DAYS = int(os.environ.get("MAX_FILE_AGE_DAYS", 30))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads" if os.environ.get("RENDER", "") else "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Optional ML model path (ONNX). If empty, ML detector is disabled.
ML_MODEL_PATH = os.environ.get("ML_MODEL_PATH", "")