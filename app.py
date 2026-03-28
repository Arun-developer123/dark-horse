# app.py -- Dark Horse Image Truth Engine (light, colorful UI)
from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
from datetime import datetime, timedelta
import math

from flask import (
    Flask, request, jsonify, render_template_string,
    send_from_directory, abort, url_for
)
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from werkzeug.utils import secure_filename

from PIL import Image, ExifTags, ImageOps
import piexif
import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import kurtosis

# -------------------------
# Config (env)
# -------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///db.sqlite")
REDIS_URL = os.environ.get("REDIS_URL", "")  # optional
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me-to-strong-token")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024))  # 10 MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", 100))
MAX_FILE_AGE_DAYS = int(os.environ.get("MAX_FILE_AGE_DAYS", 30))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads" if os.environ.get("RENDER", "") else "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# App + DB + Security
# -------------------------
app = Flask(__name__, static_folder=None)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + (1024 * 1024)
db = SQLAlchemy(app)

# Talisman: keep default secure headers (CSP None for inline scripts used in this single-file MVP)
Talisman(app, content_security_policy=None)

# Rate limiter
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
    reasons = db.Column(db.Text)


with app.app_context():
    db.create_all()


# -------------------------
# Utility helpers
# -------------------------
def extract_exif(pil_img):
    """Return flattened EXIF dict or {}"""
    try:
        raw_exif = pil_img.info.get('exif', b'')
        if raw_exif:
            exif_dict = piexif.load(raw_exif)
            out = {}
            for ifd in exif_dict:
                try:
                    for k, v in exif_dict[ifd].items():
                        try:
                            if isinstance(v, bytes):
                                v = v.decode(errors='ignore')
                        except Exception:
                            pass
                        out[f"{ifd}:{k}"] = v
                except Exception:
                    pass
            return out
    except Exception:
        pass

    # Fallback to PIL _getexif
    try:
        raw = getattr(pil_img, "_getexif", lambda: {})() or {}
        readable = {}
        for k, v in raw.items():
            name = ExifTags.TAGS.get(k, k)
            readable[name] = v
        return readable
    except Exception:
        return {}


def pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert('RGB'))
    # Return BGR (cv2 default)
    return arr[:, :, ::-1].copy()


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def normalize_pair_diff(a, b, floor=1e-6):
    a = float(a)
    b = float(b)
    return abs(a - b) / (max(abs(a), abs(b), floor))


# -------------------------
# Analysis signals (layered)
# -------------------------
def sensor_noise_score(cv2_gray):
    img = cv2_gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    residual = img - blurred
    std = float(np.std(residual))
    score = (std - 0.002) / (0.04 - 0.002)
    return clamp01(score), std


def smoothness_score(cv2_gray):
    img = cv2_gray.astype(np.float32) / 255.0
    mean = cv2.blur(img, (3, 3))
    mean_sq = cv2.blur(img * img, (3, 3))
    var = mean_sq - mean * mean
    mean_var = float(np.mean(var))
    score = (mean_var - 1e-6) / (0.005 - 1e-6)
    return clamp01(score), mean_var


def frequency_kurtosis_score(cv2_gray):
    img = cv2_gray.astype(np.float32)
    h, w = img.shape
    target = 512
    scale = max(1, int(max(h, w) / target))
    if scale > 1:
        img_small = cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img
    f = fftpack.fft2(img_small)
    fshift = fftpack.fftshift(f)
    mag = np.abs(fshift).flatten()
    log_mag = np.log1p(mag)
    k = float(kurtosis(log_mag, fisher=False, nan_policy='omit') or 0.0)
    if math.isnan(k):
        k = 0.0
    if k <= 2:
        score = 0.25
    elif k <= 10:
        score = 0.95
    elif k <= 14:
        score = 0.7
    else:
        score = 0.25
    return clamp01(score), k


def artifact_penalty(cv2_bgr):
    img = cv2_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    sims = []
    num_checks = 24
    patch = 32
    for _ in range(num_checks):
        if h <= patch or w <= patch:
            break
        y = np.random.randint(0, max(1, h - patch))
        x = np.random.randint(0, max(1, w - patch))
        y2 = np.random.randint(0, max(1, h - patch))
        x2 = np.random.randint(0, max(1, w - patch))
        p1 = gray[y:y + patch, x:x + patch]
        p2 = gray[y2:y2 + patch, x2:x2 + patch]
        num = np.sum((p1 - p1.mean()) * (p2 - p2.mean()))
        den = np.sqrt(np.sum((p1 - p1.mean()) ** 2) * np.sum((p2 - p2.mean()) ** 2) + 1e-9)
        sims.append(num / (den + 1e-9))
    sims = np.array(sims) if sims else np.array([0.0])
    high_sim_frac = float(np.mean(sims > 0.82))
    flipped = np.fliplr(gray)
    symmetry = float(np.mean(np.abs(gray - flipped)))
    penalty = 0.0
    if high_sim_frac > 0.2:
        penalty += min(1.0, high_sim_frac * 2.2)
    if symmetry < 0.02:
        penalty += 0.5
    penalty = float(np.clip(penalty, 0.0, 1.0))
    return penalty, high_sim_frac, symmetry


def edge_density_score(cv2_gray):
    g = cv2_gray.astype(np.uint8)
    v = np.median(g)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(g, lower, upper)
    density = float(np.sum(edges > 0) / (edges.size + 1e-9))
    score = (density - 0.001) / (0.04 - 0.001)
    return clamp01(score), density


def color_hist_kurtosis_score(cv2_bgr):
    chans = cv2.split(cv2_bgr)
    ks = []
    for c in chans:
        hist = cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        log_hist = np.log1p(hist)
        try:
            k = float(kurtosis(log_hist, fisher=False, nan_policy='omit'))
        except Exception:
            k = 0.0
        ks.append(k if not math.isnan(k) else 0.0)
    mean_k = float(np.mean(ks))
    if mean_k <= 2:
        score = 0.3
    elif mean_k <= 10:
        score = 0.9
    elif mean_k <= 14:
        score = 0.6
    else:
        score = 0.25
    return clamp01(score), mean_k


def entropy_score(cv2_gray):
    hist = cv2.calcHist([cv2_gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    ent = float(-np.sum(p * np.log2(p)))
    score = (ent - 3.0) / (7.0 - 3.0)
    return clamp01(score), ent


def jpeg_quality_hint(pil_img):
    try:
        qtables = getattr(pil_img, 'quantization', None)
        if not qtables:
            return None, None
        vals = []
        for _k, v in qtables.items():
            vals.extend(v)
        mean_q = float(np.mean(vals)) if vals else None
        if mean_q is None:
            return None, None
        score = 1.0 - (mean_q / 200.0)
        return clamp01(score), mean_q
    except Exception:
        return None, None


def face_and_eye_analysis(cv2_bgr):
    """Detect faces and eyes, then compute a suspicion score from eye mismatch / absence."""
    try:
        gray = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60))
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        face_count = int(len(faces))
        best = {
            'face_found': False,
            'face_count': face_count,
            'eyes_found': 0,
            'eye_pair_score': 0.0,
            'eye_asymmetry': 0.0,
            'eye_mismatch_score': 0.0,
            'face_bbox': None,
        }

        if face_count == 0:
            # No face detected: not automatically AI, but if a portrait-like image has no eyes/faces,
            # the model should strongly flag it as suspicious.
            return {
                **best,
                'face_found': False,
                'eye_mismatch_score': 0.35,
                'reason': 'No face detected; if this is a portrait, that increases suspicion.'
            }

        x, y, w, h = faces[0]
        best['face_found'] = True
        best['face_bbox'] = [int(x), int(y), int(w), int(h)]
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            best['eye_mismatch_score'] = 0.25
            best['reason'] = 'Face region was too small or empty.'
            return best

        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.06, minNeighbors=8, minSize=(12, 12))
        eye_count = int(len(eyes))
        best['eyes_found'] = eye_count

        if eye_count == 0:
            best['eye_mismatch_score'] = 0.72
            best['reason'] = 'Face found but no eyes detected.'
            return best

        # Use the two best eye detections if available.
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:4]
        centers = []
        for (ex, ey, ew, eh) in eyes:
            centers.append((ex + ew / 2.0, ey + eh / 2.0, ew, eh))

        # Keep two topmost / largest regions for symmetry comparison.
        centers = sorted(centers, key=lambda c: (c[0], c[1]))
        if len(centers) >= 2:
            left_eye = min(centers[:2], key=lambda c: c[0])
            right_eye = max(centers[:2], key=lambda c: c[0])
            dx = abs(right_eye[0] - left_eye[0])
            dy = abs(right_eye[1] - left_eye[1])
            size_diff = normalize_pair_diff(left_eye[2] * left_eye[3], right_eye[2] * right_eye[3])
            y_ratio = dy / max(h, 1)
            x_ratio = dx / max(w, 1)

            # Eye pair should be roughly horizontal and similar sized.
            pair_score = clamp01(1.0 - (0.8 * y_ratio + 0.35 * size_diff))
            asymmetry = clamp01(0.5 * y_ratio + 0.5 * size_diff)
            mismatch_score = clamp01(0.15 + (1.0 - pair_score) * 0.85)

            best['eye_pair_score'] = pair_score
            best['eye_asymmetry'] = asymmetry
            best['eye_mismatch_score'] = mismatch_score
            best['reason'] = f"Face found with {eye_count} eye detection(s); evaluated eye alignment and symmetry."
            return best

        best['eye_mismatch_score'] = 0.55
        best['reason'] = f"Face found with only {eye_count} eye detection(s)."
        return best

    except Exception as e:
        return {
            'face_found': False,
            'face_count': 0,
            'eyes_found': 0,
            'eye_pair_score': 0.0,
            'eye_asymmetry': 0.0,
            'eye_mismatch_score': 0.3,
            'face_bbox': None,
            'reason': f'Face/eye detection failed safely: {str(e)}'
        }


# -------------------------
# Combine signals -> final score
# -------------------------
def compute_final_realness(signals):
    weights = {
        'exif': 0.10,
        'sensor': 0.22,
        'smooth': 0.15,
        'freq': 0.13,
        'edge': 0.09,
        'color': 0.07,
        'entropy': 0.06,
        'eyes': 0.18,
    }
    total_w = 0.0
    acc = 0.0
    for k, w in weights.items():
        if k in signals and signals[k] is not None:
            acc += signals[k] * w
            total_w += w
    if total_w <= 0:
        base = 0.5
    else:
        base = acc / total_w

    penalty = signals.get('artifact_penalty', 0.0)
    raw = base * (1.0 - 0.6 * penalty)
    return float(np.clip(raw, 0.0, 1.0))


# -------------------------
# Background cleaner
# -------------------------
def cleanup_old_files():
    while True:
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(days=MAX_FILE_AGE_DAYS)
            with app.app_context():
                for fname in os.listdir(UPLOAD_DIR):
                    path = os.path.join(UPLOAD_DIR, fname)
                    try:
                        mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                        if mtime < cutoff:
                            try:
                                os.remove(path)
                            except Exception:
                                pass
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
# Beautiful LIGHT UI template (Jinja)
# -------------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dark Horse — Image Truth Engine</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg1: #f8fafc;
      --bg2: #ffffff;
      --card: rgba(15,23,42,0.03);
      --accent1: #7c3aed;
      --accent2: #06b6d4;
      --accent3: #f59e0b;
      --good: #10b981;
      --warn: #f59e0b;
      --bad: #ef4444;
      --text: #0f172a;
      --muted: #475569;
    }
    html,body{ height:100% }
    body{
      background:
        radial-gradient(800px 300px at 10% 8%, rgba(124,58,237,0.06), transparent 6%),
        radial-gradient(600px 250px at 85% 15%, rgba(6,182,212,0.08), transparent 8%),
        linear-gradient(180deg,var(--bg1),var(--bg2));
      color:var(--text);
      font-family: 'Inter', system-ui, -apple-system, Roboto, Arial;
    }
    .container{ max-width:1100px; padding:36px 18px; }
    .brand{ font-weight:800; font-size:1.2rem; letter-spacing:-0.4px; display:flex; gap:8px; align-items:center; }
    .logo-dot{ width:12px; height:12px; border-radius:50%; background:linear-gradient(45deg,var(--accent1),var(--accent2)); box-shadow:0 8px 22px rgba(124,58,237,0.12) }
    .card.glass{ background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,250,0.8)); border:1px solid rgba(15,23,42,0.04); box-shadow: 0 8px 30px rgba(15,23,42,0.06); border-radius:14px; }
    .upload-area{
      border:2px dashed rgba(15,23,42,0.06);
      padding:28px;
      border-radius:16px;
      text-align:center;
      transition:all .18s ease;
      background:linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.4));
      cursor:pointer;
      position:relative;
      overflow:hidden;
    }
    .upload-area::before{
      content:'';
      position:absolute;
      inset:-40px;
      background: radial-gradient(circle at 30% 30%, rgba(124,58,237,0.08), transparent 25%), radial-gradient(circle at 70% 70%, rgba(6,182,212,0.08), transparent 24%);
      pointer-events:none;
      opacity:.8;
    }
    .upload-area > *{ position:relative; z-index:1; }
    .upload-area.dragover{ transform: translateY(-6px) scale(1.01); box-shadow:0 22px 60px rgba(7,16,36,0.06); border-color: rgba(124,58,237,0.9) }
    .meter {
      --size:140px;
      width:var(--size);
      height:var(--size);
      border-radius:999px;
      display:grid;
      place-items:center;
      background: conic-gradient(var(--col, var(--accent2)) var(--pct), rgba(0,0,0,0.06) 0);
      position:relative;
      border:6px solid rgba(9,30,66,0.03);
      transition: background .35s ease, transform .25s ease;
      animation: floaty 5s ease-in-out infinite;
    }
    .meter::after{
      content:'';
      position:absolute;
      inset:10px;
      background:rgba(255,255,255,0.95);
      border-radius:999px;
      border:1px solid rgba(15,23,42,0.04);
      box-shadow: inset 0 2px 10px rgba(15,23,42,0.03);
    }
    .meter .val{ position:relative; z-index:1; font-weight:900; font-size:22px; color:var(--text); }
    .meter-label{ position:relative; z-index:1; font-size:11px; text-transform:uppercase; letter-spacing:0.16em; color:var(--muted); font-weight:700; margin-top:-2px; }
    .layer{ border-left:4px solid rgba(15,23,42,0.04); padding-left:12px; margin-bottom:10px; border-radius:6px; padding-top:8px; padding-bottom:8px; background:rgba(255,255,255,0.45) }
    .score-pill{ padding:6px 10px; border-radius:999px; background: linear-gradient(90deg, rgba(124,58,237,0.08), rgba(6,182,212,0.06)); color:var(--text); font-weight:700; min-width:64px; text-align:center }
    .bar { height:10px; border-radius:999px; background: rgba(15,23,42,0.04); overflow:hidden; }
    .bar > i { display:block; height:100%; border-radius:999px; width:0%; background:linear-gradient(90deg,var(--accent1),var(--accent2)); box-shadow:0 4px 12px rgba(7,16,36,0.06) }
    .muted{ color:var(--muted) }
    .small-muted{ color:var(--muted); font-size:0.95rem }
    footer{ color:var(--muted); margin-top:24px; text-align:center; font-size:0.9rem }
    .img-preview{ max-width:220px; border-radius:12px; border:1px solid rgba(15,23,42,0.04); box-shadow: 0 8px 30px rgba(15,23,42,0.04); }
    .btn-accent{ background: linear-gradient(90deg,var(--accent1),var(--accent2)); border:none; color:white }
    .chip{
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:7px 11px;
      border-radius:999px;
      background:rgba(124,58,237,0.08);
      color:#4c1d95;
      font-weight:700;
      font-size:.9rem;
    }
    .mini-stat{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      border:1px solid rgba(15,23,42,0.06);
      border-radius:14px;
      padding:12px 14px;
      background: rgba(255,255,255,0.62);
    }
    .mini-stat strong{ font-size:1rem; }
    .mini-stat span{ color:var(--muted); font-size:.9rem; }
    .pulse-dot{
      width:10px; height:10px; border-radius:50%;
      background:linear-gradient(45deg,var(--accent1),var(--accent2));
      box-shadow:0 0 0 0 rgba(124,58,237,.35);
      animation:pulse 1.8s infinite;
    }
    @keyframes pulse { 0%{ box-shadow:0 0 0 0 rgba(124,58,237,.35) } 70%{ box-shadow:0 0 0 12px rgba(124,58,237,0) } 100%{ box-shadow:0 0 0 0 rgba(124,58,237,0) } }
    @keyframes floaty { 0%,100%{ transform:translateY(0px) } 50%{ transform:translateY(-4px) } }
    @media (max-width:990px){ .meter{ --size:120px } .img-preview{ max-width:140px } }
  </style>
</head>
<body>
  <div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">
      <div class="d-flex flex-column">
        <div class="brand">
          <div class="logo-dot" aria-hidden></div>
          <div>
            Dark Horse <span class="muted">• Image Truth Engine</span>
          </div>
        </div>
        <div class="small-muted">Explainable layered checks for AI-vs-real image suspicion, with portrait eye analysis and EXIF signals.</div>
      </div>
      <div class="d-flex gap-2">
        <a class="btn btn-sm btn-outline-dark" href="/health">Health</a>
        <a class="btn btn-sm btn-outline-dark" href="/admin/exports?token={{ admin_token }}">Export</a>
      </div>
    </div>

    <div class="card glass p-4 mb-4">
      <div class="row g-3">
        <div class="col-lg-7">
          <div id="uploadArea" class="upload-area">
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none"><path d="M12 3v10" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <h4 class="mt-2">Drag & drop or click to upload</h4>
            <div class="muted">PNG / JPG / WebP / BMP — up to <strong>{{ mb }} MB</strong></div>
            <div class="small-muted mt-2">Checks EXIF, sensor residuals, texture, frequency, edges, color, entropy, and face/eye consistency.</div>
            <input id="fileInput" type="file" accept="image/*" style="display:none;">
          </div>

          <div id="previewCard" class="mt-3 d-none card p-3">
            <div class="d-flex gap-3 align-items-start flex-wrap">
              <img id="previewImage" class="img-preview"/>
              <div class="flex-grow-1">
                <div id="previewName" class="fw-semibold"></div>
                <div id="previewSize" class="small-muted mb-2"></div>
                <div class="d-flex gap-2 flex-wrap">
                  <button id="analyzeBtn" class="btn btn-accent">Analyze</button>
                  <button id="clearBtn" class="btn btn-outline-secondary">Clear</button>
                </div>
                <div class="mt-3 d-flex gap-2 flex-wrap">
                  <div class="chip"><span class="pulse-dot"></span>Interactive analysis</div>
                  <div class="chip">Numbered findings</div>
                  <div class="chip">Portrait eye check</div>
                </div>
              </div>
            </div>
          </div>

          <div id="loading" class="mt-3 d-none">
            <div class="small-muted">Analyzing — one moment...</div>
            <div class="progress mt-2" style="height:8px;"><div id="progBar" class="progress-bar" style="width:0%"></div></div>
          </div>
        </div>

        <div class="col-lg-5">
          <div id="resultCard" class="card p-3 text-center d-none">
            <div class="d-flex flex-column align-items-center">
              <div class="meter mb-2" id="meter" style="--pct:0deg; --col:var(--accent2);">
                <div class="val text-center" id="meterVal">--</div>
                <div class="meter-label">Truth score</div>
              </div>
              <div id="resultLabel" class="h5 mt-2">No result</div>
              <div id="resultSub" class="small-muted mb-2">Upload an image to begin</div>
            </div>

            <div class="row g-2 text-start mt-2 mb-2">
              <div class="col-6">
                <div class="mini-stat"><div><strong id="faceStat">--</strong><br><span>Face(s)</span></div><div class="pulse-dot"></div></div>
              </div>
              <div class="col-6">
                <div class="mini-stat"><div><strong id="eyeStat">--</strong><br><span>Eye detections</span></div><div class="pulse-dot"></div></div>
              </div>
            </div>

            <div class="text-start mt-3">
              <div class="fw-semibold mb-2">Analysis Layers</div>
              <div id="layers"></div>
            </div>
            <div class="mt-3 d-flex justify-content-between flex-wrap gap-2">
              <a id="downloadReport" class="btn btn-sm btn-outline-dark">Download report</a>
              <div>
                <button id="copyScore" class="btn btn-sm btn-outline-dark me-2">Copy score</button>
                <a id="recheckBtn" class="btn btn-sm btn-outline-dark">Analyze another</a>
              </div>
            </div>
          </div>

          <div class="card p-3 mt-3 text-center small-muted">
            <div class="fw-semibold">Tips</div>
            Use original phone photos for best accuracy. Portraits are checked for face and eye consistency; if uncertain, cross-check with reverse-image search and human review.
          </div>
        </div>
      </div>
    </div>

    <footer>Made with ♥ — treat sensitive images with care.</footer>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    const mb = {{ mb }};
    const MAX_FILE_SIZE = {{ MAX_FILE_SIZE }};
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewCard = document.getElementById('previewCard');
    const previewImage = document.getElementById('previewImage');
    const previewName = document.getElementById('previewName');
    const previewSize = document.getElementById('previewSize');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loading = document.getElementById('loading');
    const progBar = document.getElementById('progBar');
    const resultCard = document.getElementById('resultCard');
    const meter = document.getElementById('meter');
    const meterVal = document.getElementById('meterVal');
    const resultLabel = document.getElementById('resultLabel');
    const resultSub = document.getElementById('resultSub');
    const layers = document.getElementById('layers');
    const downloadReport = document.getElementById('downloadReport');
    const recheckBtn = document.getElementById('recheckBtn');
    const copyScore = document.getElementById('copyScore');
    const faceStat = document.getElementById('faceStat');
    const eyeStat = document.getElementById('eyeStat');

    let currentFile = null;
    let lastReport = null;

    fileInput.addEventListener('change', e => {
      const f = e.target.files[0];
      if (!f) return;
      showPreview(f);
    });
    uploadArea.addEventListener('click', ()=> fileInput.click());
    uploadArea.addEventListener('dragover', e=>{ e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', e=>{ uploadArea.classList.remove('dragover'); });
    uploadArea.addEventListener('drop', e=>{ e.preventDefault(); uploadArea.classList.remove('dragover'); const f = e.dataTransfer.files[0]; if(!f) return; fileInput.files = e.dataTransfer.files; showPreview(f); });

    function humanSize(b){ if(b<1024) return b+' B'; let u=['KB','MB','GB']; let i= -1; do{ b/=1024; i++; }while(b>=1024 && i<u.length-1); return b.toFixed(1)+' '+u[i]; }

    function showPreview(file){
      currentFile = file;
      previewImage.src = URL.createObjectURL(file);
      previewName.textContent = file.name;
      previewSize.textContent = humanSize(file.size);
      previewCard.classList.remove('d-none');
      resultCard.classList.add('d-none');
      loading.classList.add('d-none');
      layers.innerHTML = '';
    }

    clearBtn.addEventListener('click', ()=>{
      fileInput.value = '';
      currentFile = null;
      previewImage.src = '';
      previewCard.classList.add('d-none');
      layers.innerHTML = '';
      resultCard.classList.add('d-none');
      lastReport = null;
    });

    analyzeBtn.addEventListener('click', async ()=>{
      if(!currentFile) return alert('Choose an image first');
      if(currentFile.size > MAX_FILE_SIZE) return alert('File too large');
      loading.classList.remove('d-none'); progBar.style.width='6%'; resultCard.classList.add('d-none');
      const fd = new FormData(); fd.append('file', currentFile);
      try{
        const resp = await fetch('/detect', { method:'POST', body: fd, headers: {'X-Requested-With':'XMLHttpRequest'} });
        progBar.style.width = '40%';
        if(!resp.ok){ const err = await resp.json().catch(()=>({error:'server error'})); loading.classList.add('d-none'); return alert('Server error: '+(err.error||JSON.stringify(err))); }
        const data = await resp.json();
        progBar.style.width='85%';
        displayResult(data);
        progBar.style.width='100%';
        lastReport = data;
        setTimeout(()=>loading.classList.add('d-none'), 300);
      }catch(e){ loading.classList.add('d-none'); alert('Network error: '+e.message) }
    });

    function displayResult(d){
      resultCard.classList.remove('d-none');
      const score = Math.max(0, Math.min(100, Number(d.realness_score || 0)));
      const angle = (score/100)*360 + 'deg';
      meter.style.setProperty('--pct', angle);
      let color = 'var(--accent2)';
      if(score >= 70) color = 'var(--good)'; else if(score >= 40) color = 'var(--warn)'; else color = 'var(--bad)';
      meter.style.setProperty('--col', color);
      meterVal.textContent = Math.round(score);
      resultLabel.textContent = d.label || (score>=70 ? 'Likely REAL' : (score>=40 ? 'Unsure / Possibly Real' : 'Likely AI / Synthetic'));
      resultSub.textContent = 'Higher score means more likely a real camera capture';
      faceStat.textContent = d.face_count !== undefined ? d.face_count : '--';
      eyeStat.textContent = d.eyes_found !== undefined ? d.eyes_found : '--';

      layers.innerHTML = '';
      const layerOrder = [
        ['exif','EXIF metadata','exif_found'],
        ['sensor','Sensor noise','sensor_std'],
        ['smooth','Local texture variance','smoothness_meanvar'],
        ['freq','Frequency kurtosis','frequency_kurtosis'],
        ['eyes','Face / eye consistency','eye_mismatch_score'],
        ['artifact','Artifact penalty','artifact_penalty'],
        ['edge','Edge density','edge_density'],
        ['color','Color histogram kurtosis','color_hist_kurtosis'],
        ['entropy','Entropy','entropy']
      ];
      let idx = 1;
      for(const [key,title,field] of layerOrder){
        if(d[field]===undefined) continue;
        const raw = d[field];
        const scorev = (d.signals && d.signals[key]!==undefined) ? Math.round(d.signals[key]*100)/100 : null;
        const barPct = scorev !== null ? Math.round(scorev*100) : 0;
        const desc = buildDescription(key, d, raw);
        const div = document.createElement('div');
        div.className = 'layer';
        div.innerHTML = `
          <div class="d-flex justify-content-between align-items-center gap-3">
            <div>
              <strong>${idx}. ${title}</strong>
              <div class="small-muted">${desc}</div>
            </div>
            <div style="min-width:120px">
              <div class="score-pill mb-1">${(scorev!==null?scorev:'-')}</div>
              <div class="bar"><i style="width:${barPct}%;"></i></div>
            </div>
          </div>
        `;
        layers.appendChild(div);
        idx += 1;
      }

      downloadReport.onclick = ()=> {
        if(!lastReport) return alert('No report');
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastReport, null, 2));
        const a = document.createElement('a'); a.setAttribute('href', dataStr); a.setAttribute('download', 'darkhorse_report_'+Date.now()+'.json'); document.body.appendChild(a); a.click(); a.remove();
      };

      recheckBtn.onclick = ()=>{ fileInput.value=''; currentFile=null; previewImage.src=''; previewCard.classList.add('d-none'); resultCard.classList.add('d-none'); layers.innerHTML=''; lastReport=null; };
      copyScore.onclick = ()=> {
        if(!lastReport) return alert('No report');
        navigator.clipboard.writeText('Dark Horse score: ' + (lastReport.realness_score || '')).then(()=> alert('Score copied to clipboard'));
      };
    }

    function buildDescription(key, d, raw){
      const s = d.signals || {};
      if(key === 'exif'){
        return d.exif_found ? 'EXIF metadata exists. This can support a camera-capture signal, but it can also be forged.' : 'No EXIF metadata found. That raises suspicion, especially for phone-like photos.';
      }
      if(key === 'eyes'){
        const face = d.face_count || 0;
        const eyes = d.eyes_found || 0;
        if(!face) return 'No face detected. If the image is a portrait, that is a strong suspicion cue.';
        if(eyes < 2) return `Face found but only ${eyes} eye detection(s). Portrait realism looks weaker.`;
        if((d.eye_asymmetry || 0) > 0.25) return 'Eyes were detected, but alignment/symmetry looks off enough to raise suspicion.';
        return 'Face and eyes look more consistent with a real photo.';
      }
      if(key === 'sensor') return 'Higher sensor residual detail usually looks more like a camera capture.';
      if(key === 'smooth') return 'Very smooth local texture can be a synthetic-image cue.';
      if(key === 'freq') return 'Frequency-domain shape can reveal over-smoothing or generated patterns.';
      if(key === 'artifact') return 'Repeated or mirrored textures may indicate generation artifacts.';
      if(key === 'edge') return 'Edge density helps compare structural detail against a natural-photo baseline.';
      if(key === 'color') return 'Color histogram shape is another weak cue used in the final blend.';
      if(key === 'entropy') return 'Entropy indicates how diverse the pixel distribution is across the image.';
      return '';
    }
  </script>
</body>
</html>
"""


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    return render_template_string(
        INDEX_HTML,
        mb=MAX_FILE_SIZE // (1024 * 1024),
        exts=", ".join(sorted(ALLOWED_EXT)),
        admin_token=admin_token,
        MAX_FILE_SIZE=MAX_FILE_SIZE
    )


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


# -------------------------
# Detect endpoint
# -------------------------
@limiter.limit(f"{RATE_LIMIT_MAX}/hour")
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    f = request.files["file"]
    filename_raw = secure_filename(f.filename or "")
    _, ext = os.path.splitext(filename_raw.lower())
    if ext not in ALLOWED_EXT:
        return jsonify({'error': 'file type not supported', 'allowed': list(ALLOWED_EXT)}), 400

    # size check
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({'error': 'file too large (> allowed bytes)'}), 400

    timestamp_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    save_name = f"{os.path.splitext(filename_raw)[0]}_{timestamp_suffix}{ext}" if filename_raw else f"upload_{timestamp_suffix}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    try:
        f.save(save_path)
    except Exception as e:
        return jsonify({'error': 'failed to save file', 'detail': str(e)}), 500

    # validate image
    try:
        pil = Image.open(save_path)
        pil.verify()
        # reopen to work with it
        pil = Image.open(save_path)
        # fix orientation if necessary
        try:
            pil = ImageOps.exif_transpose(pil)
        except Exception:
            pass
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify({'error': 'invalid image', 'detail': str(e)}), 400

    # prepare cv2 image and gray
    exif = extract_exif(pil)
    exif_found = len(exif) > 0

    cv2_img = pil_to_cv2(pil)
    if cv2_img is None or cv2_img.size == 0:
        return jsonify({'error': 'failed to decode image'}), 400
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # run signals
    s_sensor, sensor_std = sensor_noise_score(gray)
    s_smooth, mean_var = smoothness_score(gray)
    s_freq, kurt = frequency_kurtosis_score(gray)
    penalty, high_sim_frac, symmetry = artifact_penalty(cv2_img)
    s_edge, edge_density = edge_density_score(gray)
    s_color, color_k = color_hist_kurtosis_score(cv2_img)
    s_entropy, ent = entropy_score(gray)
    s_exif = 1.0 if exif_found else 0.0
    q_score, q_mean = jpeg_quality_hint(pil)
    eye = face_and_eye_analysis(cv2_img)

    # If EXIF is missing, strongly punish suspiciousness instead of letting it be neutral.
    # For genuine phone photos, missing EXIF is a strong negative signal.
    exif_signal = 1.0 if exif_found else 0.15

    # Eye signal: more real-like when face/eyes are consistent; more suspicious when there is mismatch.
    eye_mismatch = float(eye.get('eye_mismatch_score', 0.0) or 0.0)
    eyes_realness = float(np.clip(1.0 - eye_mismatch, 0.0, 1.0))

    signals = {
        'exif': exif_signal,
        'sensor': s_sensor,
        'smooth': s_smooth,
        'freq': s_freq,
        'artifact_penalty': penalty,
        'edge': s_edge,
        'color': s_color,
        'entropy': s_entropy,
        'jpeg_hint': q_score,
        'eyes': eyes_realness,
    }

    final_raw = compute_final_realness(signals)

    # Additional strong penalty when no EXIF AND portrait-like face/eye issues exist.
    if not exif_found:
        final_raw *= 0.83
    if eye.get('face_found') and (eye.get('eyes_found', 0) < 2 or (eye.get('eye_asymmetry', 0.0) > 0.22)):
        final_raw *= 0.87
    if not eye.get('face_found'):
        final_raw *= 0.96

    final_score = float(round(np.clip(final_raw, 0.0, 1.0) * 100.0, 3))

    # Build numbered reasons in a deterministic order.
    reasons = []
    n = 1
    if not exif_found:
        reasons.append(f"{n}. No camera EXIF metadata found — suspicious for many real phone photos.")
    else:
        reasons.append(f"{n}. Camera EXIF present — supports a camera capture signal, but EXIF can still be forged.")
    n += 1

    if eye.get('face_found'):
        if eye.get('eyes_found', 0) >= 2:
            if (eye.get('eye_asymmetry', 0.0) or 0.0) > 0.22:
                reasons.append(f"{n}. Face and eyes were detected, but the eye symmetry/alignment looks off.")
            else:
                reasons.append(f"{n}. Face and eye detections look consistent with a normal portrait.")
        else:
            reasons.append(f"{n}. Face detected, but only {eye.get('eyes_found', 0)} eye detection(s) were found.")
    else:
        reasons.append(f"{n}. No face detected. If this image is a portrait, that is a strong suspicion cue.")
    n += 1

    reasons.append(f"{n}. Sensor residual std: {sensor_std:.6f}")
    n += 1
    reasons.append(f"{n}. Local texture variance (mean): {mean_var:.6e}")
    n += 1
    reasons.append(f"{n}. Frequency kurtosis: {kurt:.3f}")
    n += 1
    reasons.append(f"{n}. Artifact penalty: {penalty:.3f} (high_sim_frac: {high_sim_frac:.3f}, symmetry: {symmetry:.3f})")
    n += 1
    reasons.append(f"{n}. Edge density: {edge_density:.6f}")
    n += 1
    reasons.append(f"{n}. Color histogram mean kurtosis: {color_k:.3f}")
    n += 1
    reasons.append(f"{n}. Entropy: {ent:.3f}")
    n += 1
    if q_score is not None:
        reasons.append(f"{n}. JPEG quantization mean: {q_mean:.2f} (quality hint score: {q_score:.2f})")
        n += 1
    reasons.append(f"{n}. Face count: {eye.get('face_count', 0)} | Eye detections: {eye.get('eyes_found', 0)} | Eye mismatch score: {eye.get('eye_mismatch_score', 0.0):.3f}")

    # DB log (best-effort)
    try:
        row = Upload(filename=save_name, ip=request.remote_addr or 'unknown', realness_score=final_score, reasons="\n".join(reasons))
        db.session.add(row)
        db.session.commit()
    except Exception:
        db.session.rollback()

    # label
    if final_score >= 70:
        label = "Likely REAL"
    elif final_score >= 40:
        label = "Unsure / Possibly Real"
    else:
        label = "Likely AI / Synthetic"

    image_url = url_for('uploaded_file', filename=save_name)

    out = {
        'realness_score': final_score,
        'label': label,
        'reasons': reasons,
        'image_url': image_url,
        'sensor_std': sensor_std,
        'smoothness_meanvar': mean_var,
        'frequency_kurtosis': kurt,
        'artifact_penalty': penalty,
        'high_sim_fraction': high_sim_frac,
        'symmetry': symmetry,
        'edge_density': edge_density,
        'color_hist_kurtosis': color_k,
        'entropy': ent,
        'signals': signals,
        'exif_found': exif_found,
        'face_found': eye.get('face_found', False),
        'face_count': eye.get('face_count', 0),
        'eyes_found': eye.get('eyes_found', 0),
        'eye_pair_score': eye.get('eye_pair_score', 0.0),
        'eye_asymmetry': eye.get('eye_asymmetry', 0.0),
        'eye_mismatch_score': eye.get('eye_mismatch_score', 0.0),
        'face_bbox': eye.get('face_bbox', None),
    }

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.best == 'application/json'
    if is_ajax:
        return jsonify(out)

    # fallback HTML response
    return render_template_string("""
    <!doctype html><title>Result — Dark Horse</title>
    <h1>{{ label }} ({{ score }})</h1>
    <img src="{{ image_url }}" style="max-width:800px"><h3>Signals</h3>
    <ul>{% for r in reasons %}<li>{{ r }}</li>{% endfor %}</ul>
    <p><a href="/">Analyze another</a></p>
    """, label=label, score=final_score, image_url=image_url, reasons=reasons)


# -------------------------
# Admin export
# -------------------------
@app.route("/admin/exports")
def admin_exports():
    token = request.args.get("token", "")
    if token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    rows = Upload.query.order_by(Upload.created_at.desc()).limit(500).all()
    lines = ["id,filename,created_at,ip,realness_score"]
    for r in rows:
        lines.append(",".join([str(r.id), r.filename, r.created_at.isoformat(), r.ip, str(r.realness_score or "")]))
    return ("\n".join(lines), 200, {"Content-Type": "text/csv; charset=utf-8"})


# -------------------------
# Main (bind to PORT)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Dark Horse on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
