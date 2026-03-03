# app.py -- Dark Horse Image Detector (Windows / venv friendly production starter)
from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
from datetime import datetime, timedelta
from io import BytesIO

from flask import Flask, request, jsonify, render_template_string, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from werkzeug.utils import secure_filename

from PIL import Image, ExifTags
import piexif
import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import kurtosis

# -------------------------
# Config (read from env)
# -------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///db.sqlite")
REDIS_URL = os.environ.get("REDIS_URL", "")  # optional; leave empty to use in-memory limiter
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me-to-strong-token")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024))  # 10 MB default
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", 100))  # per hour by default
MAX_FILE_AGE_DAYS = int(os.environ.get("MAX_FILE_AGE_DAYS", 30))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Flask + DB + Security
# -------------------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Talisman sets secure headers (CSP is disabled here to avoid blocking local usage; tune for prod)
Talisman(app, content_security_policy=None)

# -------------------------
# Rate limiter (try Redis; fallback to in-memory)
# -------------------------
if REDIS_URL:
    try:
        limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL, app=app)
    except Exception:
        limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_MAX}/hour"], app=app)
else:
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_MAX}/hour"], app=app)

# -------------------------
# DB model
# -------------------------
class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip = db.Column(db.String(100))
    realness_score = db.Column(db.Float)
    exif_found = db.Column(db.Boolean)
    sensor_std = db.Column(db.Float)
    smoothness_meanvar = db.Column(db.Float)
    frequency_kurtosis = db.Column(db.Float)
    artifact_penalty = db.Column(db.Float)
    reasons = db.Column(db.Text)

with app.app_context():
    db.create_all()

# -------------------------
# Simple homepage UI (use Jinja rendering to safely inject variables)
# -------------------------
INDEX_HTML = """
<!doctype html>
<title>Dark Horse — Image Detector</title>
<style>
  body { font-family: system-ui, -apple-system, Roboto, Arial; padding:20px; max-width:900px }
  .note { color:#666; font-size:0.9em }
  .warning { color:#b00 }
  input[type=file] { margin-right: 8px; }
  img.preview { max-width: 800px; border: 1px solid #ddd; margin-top: 10px; }
  ul { line-height: 1.4; }
</style>
<h1>Dark Horse — Image Authenticity Detector</h1>
<form method="post" enctype="multipart/form-data" action="/detect">
  <input type="file" name="file" accept="image/*" required>
  <input type="submit" value="Upload & Analyze">
</form>
<p class="note">Max file size: <strong>{{ mb }} MB</strong>. Supported: <strong>{{ exts }}</strong>. By uploading you consent to short-term analysis & storage.</p>
<hr>
<p>API: <code>POST /detect</code> form field <code>file</code>. Health: <code>/health</code></p>
<p class="warning">This is a heuristic detector (proof-of-concept). For high-stakes decisions, use additional checks (reverse image search, specialized detectors).</p>
"""

# -------------------------
# Detection functions (heuristics)
# -------------------------
def extract_exif(pil_img):
    try:
        exif_dict = piexif.load(pil_img.info.get('exif', b''))
        out = {}
        for ifd in exif_dict:
            try:
                for k, v in exif_dict[ifd].items():
                    if isinstance(v, bytes):
                        try:
                            v = v.decode(errors='ignore')
                        except:
                            pass
                    out[f"{ifd}:{k}"] = v
            except Exception:
                pass
        return out
    except Exception:
        try:
            raw = pil_img._getexif() or {}
            readable = {}
            for k, v in raw.items():
                name = ExifTags.TAGS.get(k, k)
                readable[name] = v
            return readable
        except Exception:
            return {}

def pil_to_cv2(pil_img):
    img = np.array(pil_img.convert('RGB'))
    img = img[:, :, ::-1].copy()
    return img

def sensor_noise_score(cv2_img_gray):
    img = cv2_img_gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    residual = img - blurred
    std = np.std(residual)
    score = (std - 0.002) / (0.04 - 0.002)
    score = np.clip(score, 0.0, 1.0)
    return float(score), float(std)

def smoothness_score(cv2_img_gray):
    img = cv2_img_gray.astype(np.float32) / 255.0
    mean = cv2.blur(img, (3, 3))
    mean_sq = cv2.blur(img * img, (3, 3))
    var = mean_sq - mean * mean
    mean_var = np.mean(var)
    score = (mean_var - 1e-6) / (0.005 - 1e-6)
    score = np.clip(score, 0.0, 1.0)
    return float(score), float(mean_var)

def frequency_analysis_score(cv2_img_gray):
    img = cv2_img_gray.astype(np.float32)
    h, w = img.shape
    target = 512
    scale = max(1, int(max(h, w) / target))
    if scale > 1:
        img_small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    else:
        img_small = img
    f = fftpack.fft2(img_small)
    fshift = fftpack.fftshift(f)
    magnitude = np.abs(fshift)
    mag = magnitude.flatten()
    log_mag = np.log1p(mag)
    k = float(kurtosis(log_mag, fisher=False, nan_policy='omit'))
    if np.isnan(k):
        k = 0.0
    if k <= 2:
        score = 0.2
    elif k <= 10:
        score = 0.9
    elif k <= 14:
        score = 0.6
    else:
        score = 0.2
    return float(score), float(k)

def artifact_checks(cv2_img_bgr):
    img = cv2_img_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    num_checks = 20
    sims = []
    for i in range(num_checks):
        y = np.random.randint(0, max(1, h - 32))
        x = np.random.randint(0, max(1, w - 32))
        patch = gray[y:y+32, x:x+32]
        y2 = np.random.randint(0, max(1, h - 32))
        x2 = np.random.randint(0, max(1, w - 32))
        patch2 = gray[y2:y2+32, x2:x2+32]
        num = np.sum((patch - patch.mean()) * (patch2 - patch2.mean()))
        den = np.sqrt(np.sum((patch - patch.mean())**2) * np.sum((patch2 - patch2.mean())**2) + 1e-9)
        sims.append(num / (den + 1e-9))
    sims = np.array(sims)
    high_sim_fraction = float(np.mean(sims > 0.8))
    flipped = np.fliplr(gray)
    symmetry = float(np.mean(np.abs(gray - flipped)))
    penalty = 0.0
    if high_sim_fraction > 0.2:
        penalty += min(1.0, high_sim_fraction * 2)
    if symmetry < 0.02:
        penalty += 0.5
    penalty = float(np.clip(penalty, 0.0, 1.0))
    return penalty, high_sim_fraction, symmetry

def compute_final_score(exif_found, sensor_score, smooth_score, freq_score, artifact_penalty):
    w_exif = 0.20
    w_sensor = 0.35
    w_smooth = 0.20
    w_freq = 0.20
    exif_val = 1.0 if exif_found else 0.0
    raw = (w_exif * exif_val) + (w_sensor * sensor_score) + (w_smooth * smooth_score) + (w_freq * freq_score)
    raw = raw * (1.0 - 0.6 * artifact_penalty)
    final = float(np.clip(raw, 0.0, 1.0) * 100.0)
    return final

# -------------------------
# Background cleaner
# -------------------------
def cleanup_old_files():
    while True:
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(days=MAX_FILE_AGE_DAYS)
            for fname in os.listdir(UPLOAD_DIR):
                path = os.path.join(UPLOAD_DIR, fname)
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                    if mtime < cutoff:
                        os.remove(path)
                        try:
                            Upload.query.filter_by(filename=fname).delete()
                            db.session.commit()
                        except Exception:
                            db.session.rollback()
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(3600)

threading.Thread(target=cleanup_old_files, daemon=True).start()

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML, mb=MAX_FILE_SIZE // (1024*1024), exts=", ".join(sorted(ALLOWED_EXT)))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(UPLOAD_DIR, safe, as_attachment=False)

# limiter: RATE_LIMIT_MAX per hour per IP
@limiter.limit(f"{RATE_LIMIT_MAX}/hour")
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    f = request.files["file"]
    filename_raw = secure_filename(f.filename or "")
    _, ext = os.path.splitext(filename_raw.lower())
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "file type not supported", "allowed": list(ALLOWED_EXT)}), 400

    # size check
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "file too large (> allowed bytes)"}), 400

    timestamp_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    save_name = f"{os.path.splitext(filename_raw)[0]}_{timestamp_suffix}{ext}" if filename_raw else f"upload_{timestamp_suffix}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    try:
        f.save(save_path)
    except Exception as e:
        return jsonify({"error": "failed to save file", "detail": str(e)}), 500

    # validate and reopen
    try:
        pil = Image.open(save_path)
        pil.verify()
        pil = Image.open(save_path)
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify({"error": "invalid image", "detail": str(e)}), 400

    exif = extract_exif(pil)
    exif_found = len(exif) > 0
    cv2_img = pil_to_cv2(pil)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    sensor_score, sensor_std = sensor_noise_score(gray)
    smooth_score, mean_var = smoothness_score(gray)
    freq_score, kurt = frequency_analysis_score(gray)
    artifact_penalty, high_sim_fraction, symmetry = artifact_checks(cv2_img)

    final = compute_final_score(exif_found, sensor_score, smooth_score, freq_score, artifact_penalty)

    reasons = []
    if not exif_found:
        reasons.append("No camera EXIF metadata found — suspicious (many AI images miss EXIF).")
    else:
        reasons.append("Camera EXIF present — suggests a real camera capture (but EXIF can be forged).")
    reasons.append(f"Sensor residual std: {sensor_std:.6f} (higher values typically indicate real sensor noise).")
    reasons.append(f"Local variance (mean): {mean_var:.6e} (lower = oversmoothed).")
    reasons.append(f"Frequency kurtosis: {kurt:.3f} (very high values can indicate synthetic frequency spikes).")
    reasons.append(f"Artifact penalty: {artifact_penalty:.3f} (higher = more repeated/symmetric patterns detected).")

    # store in DB (best-effort)
    try:
        row = Upload(
            filename=save_name,
            ip=request.remote_addr or "unknown",
            realness_score=final,
            exif_found=bool(exif_found),
            sensor_std=sensor_std,
            smoothness_meanvar=mean_var,
            frequency_kurtosis=kurt,
            artifact_penalty=artifact_penalty,
            reasons="\n".join(reasons),
        )
        db.session.add(row)
        db.session.commit()
    except Exception:
        db.session.rollback()

    # label
    if final >= 70:
        label = "Likely REAL"
    elif final >= 40:
        label = "Unsure / Possibly Real"
    else:
        label = "Likely AI / Synthetic"

    # result template using Jinja (safe)
    RESULT_TEMPLATE = """
    <!doctype html>
    <title>Result — Dark Horse</title>
    <style>
      body { font-family: system-ui, -apple-system, Roboto, Arial; padding:20px }
      img { max-width:800px; border:1px solid #ddd }
    </style>
    <h1>Dark Horse — Result</h1>
    <h2>Result: <strong>{{ label }}</strong> <small style="color:#666">(score: {{ score }})</small></h2>
    <div><img src="/uploads/{{ save_name }}" alt="uploaded image"></div>
    <h3>Signals / Reasons</h3>
    <ul>{% for r in reasons %}<li>{{ r }}</li>{% endfor %}</ul>
    <h3>Debug metrics</h3>
    <ul>
      <li>exif_found: {{ exif_found }}</li>
      <li>sensor_std: {{ sensor_std }}</li>
      <li>local_variance_mean: {{ mean_var }}</li>
      <li>frequency_kurtosis: {{ kurt }}</li>
      <li>artifact_penalty: {{ artifact_penalty }}</li>
      <li>high_sim_fraction: {{ high_sim_fraction }}</li>
      <li>symmetry_score: {{ symmetry }}</li>
    </ul>
    <p><a href="/">Analyze another image</a></p>
    <p style="color:#888; font-size:0.9em">Note: This tool is a heuristic detector (proof-of-concept). For high-stakes decisions, run additional verification.</p>
    """

    return render_template_string(
        RESULT_TEMPLATE,
        label=label,
        score=round(final, 1),
        save_name=save_name,
        reasons=reasons,
        exif_found=int(bool(exif_found)),
        sensor_std=f"{sensor_std:.6f}",
        mean_var=f"{mean_var:.6e}",
        kurt=f"{kurt:.3f}",
        artifact_penalty=f"{artifact_penalty:.3f}",
        high_sim_fraction=f"{high_sim_fraction:.3f}",
        symmetry=f"{symmetry:.3f}",
    )

# Admin logs export (protected by token)
@app.route("/admin/exports")
def admin_exports():
    token = request.args.get("token", "")
    if token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    rows = Upload.query.order_by(Upload.created_at.desc()).limit(500).all()
    lines = ["id,filename,created_at,ip,realness_score,exif_found,sensor_std,smoothness_meanvar,frequency_kurtosis,artifact_penalty"]
    for r in rows:
        lines.append(",".join([
            str(r.id),
            r.filename,
            r.created_at.isoformat(),
            r.ip,
            str(r.realness_score or ""),
            str(int(bool(r.exif_found))),
            str(r.sensor_std or ""),
            str(r.smoothness_meanvar or ""),
            str(r.frequency_kurtosis or ""),
            str(r.artifact_penalty or ""),
        ]))
    return ("\n".join(lines), 200, {"Content-Type": "text/csv; charset=utf-8"})

# Run note: do NOT use Flask built-in server in production. Use waitress.
if __name__ == "__main__":
    print("Starting Dark Horse dev server. For production use: waitress-serve --host=0.0.0.0 --port=5000 app:app")
    app.run(host="0.0.0.0", port=5000, debug=False)