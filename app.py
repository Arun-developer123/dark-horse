# app.py -- Dark Horse Image Detector (UI-upgraded, AJAX frontend)
from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
from datetime import datetime, timedelta
from io import BytesIO

from flask import (
    Flask, request, jsonify, render_template_string,
    send_from_directory, abort, url_for
)
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

# Talisman sets secure headers (CSP disabled to allow inline CSS/JS for MVP)
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
# Detection helpers (heuristics)
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
# Beautiful UI template (Bootstrap + custom styles + JS)
# -------------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dark Horse — Image Truth Engine</title>

  <!-- Bootstrap 5 CDN -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg-1: linear-gradient(135deg,#0f172a 0%,#0b1220 100%);
      --card: rgba(255,255,255,0.06);
      --accent: linear-gradient(90deg,#7c3aed,#06b6d4);
      --glass: rgba(255,255,255,0.04);
    }
    body{
      background: var(--bg-1);
      color: #e6eef8;
      font-family: 'Inter', system-ui, -apple-system, Roboto, "Helvetica Neue", Arial;
      min-height:100vh;
    }
    .container{ max-width:1000px; padding-top:36px; padding-bottom:36px; }
    .card.glass{
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
      border: 1px solid rgba(255,255,255,0.04);
      box-shadow: 0 10px 30px rgba(2,6,23,0.6);
      color: #e9f0ff;
    }
    .brand{ font-weight:800; letter-spacing: -0.5px; }
    .accent-text{ background: -webkit-linear-gradient(#7c3aed,#06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .upload-area{
      border: 2px dashed rgba(255,255,255,0.06);
      padding: 28px;
      border-radius: 12px;
      text-align:center;
      background: rgba(255,255,255,0.01);
      transition: all .18s ease;
    }
    .upload-area.dragover{ transform: translateY(-4px); box-shadow: 0 8px 26px rgba(8,10,20,0.6); border-color: rgba(99,102,241,0.6); }
    .muted{ color: #9fb0d6; }
    .meter {
      --size: 120px;
      width: var(--size);
      height: var(--size);
      border-radius: 999px;
      display:grid;
      place-items:center;
      background: conic-gradient(#06b6d4 var(--pct), rgba(255,255,255,0.06) 0);
      box-shadow: inset 0 -6px 18px rgba(0,0,0,0.45);
    }
    .meter .val{ font-weight:700; font-size:20px; color:#fff; text-shadow: 0 2px 8px rgba(2,6,23,0.8); }
    .reason-list li{ margin-bottom:8px; }
    .small-muted{ font-size:0.9rem; color:#9fb0d6; }
    .btn-primary-gradient{
      background-image: linear-gradient(90deg,#7c3aed,#06b6d4);
      border: none;
      box-shadow: 0 8px 30px rgba(12,15,30,0.4);
    }
    footer{ color:#9fb0d6; margin-top:28px; text-align:center; font-size:0.9rem; }
    .img-preview{ max-width:100%; border-radius:8px; border: 1px solid rgba(255,255,255,0.04); }
  </style>
</head>
<body>
  <div class="container">
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div>
        <div class="brand h4 mb-0">Dark Horse <span class="small muted">• Image Truth Engine</span></div>
        <div class="small-muted">Detect whether an image is real or AI-generated — fast, explainable signals.</div>
      </div>
      <div class="text-end">
        <a class="btn btn-sm btn-outline-light" href="/health">Health</a>
        <a class="btn btn-sm btn-outline-light" href="/admin/exports?token={{ admin_token }}">Export</a>
      </div>
    </div>

    <div class="card glass p-4 mb-4">
      <div class="row g-3">
        <div class="col-md-7">
          <div id="uploadArea" class="upload-area" tabindex="0">
            <div class="mb-3">
              <svg width="56" height="56" viewBox="0 0 24 24" fill="none"><path d="M12 3v10" stroke="#9fb0d6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#9fb0d6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="h5 mb-1">Drag & drop or click to upload</div>
            <div class="muted mb-2">PNG / JPG / WebP up to <strong>{{ mb }} MB</strong></div>
            <div class="small-muted">We analyze EXIF, sensor noise, frequency-domain artifacts, and repeated patterns.</div>

            <input id="fileInput" type="file" accept="image/*" style="display:none;">
          </div>

          <div id="previewCard" class="mt-3 d-none">
            <div class="d-flex align-items-start gap-3">
              <img id="previewImage" class="img-preview" alt="preview">
              <div class="flex-grow-1">
                <div id="previewName" class="fw-semibold"></div>
                <div id="previewSize" class="small-muted mb-2"></div>
                <div>
                  <button id="analyzeBtn" class="btn btn-primary-gradient btn-sm">Analyze image</button>
                  <button id="clearBtn" class="btn btn-outline-light btn-sm ms-2">Clear</button>
                </div>
              </div>
            </div>
          </div>

          <div id="loading" class="mt-3 d-none">
            <div class="small-muted">Analyzing — hang tight. This usually takes a few seconds...</div>
            <div class="progress mt-2" style="height:8px;"><div id="progBar" class="progress-bar" style="width:0%"></div></div>
          </div>

        </div>

        <div class="col-md-5">
          <div class="d-flex flex-column align-items-center">
            <div id="resultCard" class="card p-3 w-100 text-center d-none">
              <div class="mb-3">
                <div class="meter mx-auto" id="meter" style="--pct: 0deg;">
                  <div class="val" id="meterVal">--</div>
                </div>
              </div>
              <div id="resultLabel" class="h5 mb-1">No result yet</div>
              <div id="resultSub" class="small-muted mb-2">Upload an image to begin</div>
              <div class="text-start mt-3">
                <div class="fw-semibold mb-2">Signals</div>
                <ul id="reasonsList" class="reason-list small-muted"></ul>
              </div>
            </div>

            <div class="card p-3 w-100 text-center" id="tipsCard">
              <div class="fw-semibold mb-2">Tips</div>
              <div class="small-muted">For best results use original phone photos (no heavy social compression). If unsure, verify with reverse image search.</div>
              <div class="mt-3"><a href="#" id="howItWorks" class="small-muted">How it works</a></div>
            </div>

          </div>
        </div>
      </div>
    </div>

    <footer>Built with ❤ by you — Dark Horse. Keep privacy in mind when uploading sensitive photos.</footer>
  </div>

  <!-- Bootstrap + small helpers -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

  <script>
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
    const reasonsList = document.getElementById('reasonsList');

    let currentFile = null;

    function humanFileSize(bytes) {
      const thresh = 1024;
      if (Math.abs(bytes) < thresh) return bytes + ' B';
      const units = ['KB','MB','GB','TB','PB','EB','ZB','YB'];
      let u = -1;
      do {
        bytes /= thresh;
        ++u;
      } while(Math.abs(bytes) >= thresh && u < units.length - 1);
      return bytes.toFixed(1)+' '+units[u];
    }

    function showPreview(file) {
      currentFile = file;
      const url = URL.createObjectURL(file);
      previewImage.src = url;
      previewName.textContent = file.name;
      previewSize.textContent = humanFileSize(file.size);
      previewCard.classList.remove('d-none');
      resultCard.classList.add('d-none');
      loading.classList.add('d-none');
    }

    function clearPreview() {
      currentFile = null;
      previewImage.src = '';
      previewName.textContent = '';
      previewSize.textContent = '';
      previewCard.classList.add('d-none');
      resultCard.classList.add('d-none');
      loading.classList.add('d-none');
      progBar.style.width = '0%';
      meter.style.setProperty('--pct','0deg');
      meterVal.textContent = '--';
      reasonsList.innerHTML = '';
      resultLabel.textContent = 'No result yet';
      resultSub.textContent = 'Upload an image to begin';
    }

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (!f) return;
      showPreview(f);
    });

    // drag/drop
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', (e) => {
      uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      const f = e.dataTransfer.files[0];
      if (!f) return;
      fileInput.files = e.dataTransfer.files;
      showPreview(f);
    });

    clearBtn.addEventListener('click', () => {
      fileInput.value = '';
      clearPreview();
    });

    analyzeBtn.addEventListener('click', async () => {
      if (!currentFile) return;
      // basic client-side size check
      if (currentFile.size > {{ MAX_FILE_SIZE | default(10485760) }}) {
        alert('File too large. Max {{ mb }} MB allowed.');
        return;
      }
      loading.classList.remove('d-none');
      progBar.style.width = '8%';
      resultCard.classList.add('d-none');

      const fd = new FormData();
      fd.append('file', currentFile);

      try {
        // AJAX request
        const resp = await fetch('/detect', {
          method: 'POST',
          body: fd,
          headers: { 'X-Requested-With': 'XMLHttpRequest' }
        });

        progBar.style.width = '40%';

        if (!resp.ok) {
          const err = await resp.json().catch(()=>({error:'server error'}));
          loading.classList.add('d-none');
          alert('Server error: ' + (err.error || JSON.stringify(err)));
          return;
        }

        const data = await resp.json();
        progBar.style.width = '85%';
        // response structure:
        // { realness_score, label, reasons: [...], image_url, sensor_std, smoothness_meanvar, frequency_kurtosis, artifact_penalty }
        displayResult(data);
        progBar.style.width = '100%';
        setTimeout(()=>loading.classList.add('d-none'), 400);
      } catch (e) {
        loading.classList.add('d-none');
        alert('Network error: ' + e.message);
      }
    });

    function displayResult(data){
      resultCard.classList.remove('d-none');
      // clamp and map to angle for meter (0-100 -> 0deg-360deg)
      const score = Math.max(0, Math.min(100, Math.round((data.realness_score||0)*10)/10));
      const angle = (score / 100) * 360;
      meter.style.setProperty('--pct', angle + 'deg');
      meterVal.textContent = score;

      // label mapping (server also returns label)
      resultLabel.textContent = data.label || (score >= 70 ? 'Likely REAL' : (score >= 40 ? 'Unsure / Possibly Real' : 'Likely AI / Synthetic'));
      resultSub.textContent = 'Confidence score — higher means more likely a real camera capture.';

      // reasons
      reasonsList.innerHTML = '';
      if (Array.isArray(data.reasons)) {
        data.reasons.forEach(r => {
          const li = document.createElement('li');
          li.textContent = r;
          reasonsList.appendChild(li);
        });
      }

      // show preview image replaced by server URL (if available)
      if (data.image_url) {
        previewImage.src = data.image_url;
        previewCard.classList.remove('d-none');
      }
    }

    // "How it works" small modal behavior
    document.getElementById('howItWorks').addEventListener('click', (e)=>{
      e.preventDefault();
      const html = `
        EXIF metadata, natural sensor residual noise, local texture variance, and frequency-domain signatures are combined to estimate "realness". This is heuristic — always cross-check with reverse image search and human review.
      `;
      alert(html);
    });

    // On load: clear any previous UI state
    clearPreview();
  </script>
</body>
</html>
"""

# -------------------------
# Routes and endpoints
# -------------------------
@app.route("/")
def index():
    # pass a safe admin token link only if set (helps in UI - you should not expose token publicly in real prod)
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    return render_template_string(INDEX_HTML, mb=MAX_FILE_SIZE // (1024*1024), exts=", ".join(sorted(ALLOWED_EXT)), admin_token=admin_token, MAX_FILE_SIZE=MAX_FILE_SIZE)

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
# Detect route (returns HTML for normal form, returns JSON for AJAX)
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
        return jsonify({'error': 'invalid image', 'detail': str(e)}), 400

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
        reasons.append('No camera EXIF metadata found — suspicious (many AI images miss EXIF).')
    else:
        reasons.append('Camera EXIF present — suggests a real camera capture (but EXIF can be forged).')
    reasons.append(f'Sensor residual std: {sensor_std:.6f} (higher values typically indicate real sensor noise).')
    reasons.append(f'Local variance (mean): {mean_var:.6e} (lower = oversmoothed).')
    reasons.append(f'Frequency kurtosis: {kurt:.3f} (very high values can indicate synthetic frequency spikes).')
    reasons.append(f'Artifact penalty: {artifact_penalty:.3f} (higher = more repeated/symmetric patterns detected).')

    # store to DB (best-effort)
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

    label = "Likely REAL" if final >= 70 else ("Unsure / Possibly Real" if final >= 40 else "Likely AI / Synthetic")
    image_url = url_for('uploaded_file', filename=save_name)

    # Build JSON-friendly result
    out = {
        'realness_score': float(round(final, 3)),
        'label': label,
        'reasons': reasons,
        'image_url': image_url,
        'sensor_std': float(sensor_std),
        'smoothness_meanvar': float(mean_var),
        'frequency_kurtosis': float(kurt),
        'artifact_penalty': float(artifact_penalty),
    }

    # If request is AJAX (fetch) return JSON; otherwise render HTML page for compatibility
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.best == 'application/json'
    if is_ajax:
        return jsonify(out)

    # fallback HTML result page (same info)
    RESULT_TEMPLATE = """
    <!doctype html>
    <title>Result — Dark Horse</title>
    <style>body{font-family:system-ui, -apple-system, Roboto, Arial; padding:20px} img{max-width:800px; border:1px solid #ddd}</style>
    <h1>Result — {{ label }} ({{ score }})</h1>
    <img src="{{ image_url }}" alt="uploaded image"><br>
    <h3>Reasons</h3>
    <ul>{% for r in reasons %}<li>{{ r }}</li>{% endfor %}</ul>
    <p><a href="/">Analyze another image</a></p>
    """
    return render_template_string(RESULT_TEMPLATE, label=label, score=round(final,1), image_url=image_url, reasons=reasons)

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

# Run note: for local prod on Windows use waitress or deploy on Render
if __name__ == "__main__":
    print("Starting Dark Horse dev server. For production use: waitress-serve --host=0.0.0.0 --port=5000 app:app")
    app.run(host="0.0.0.0", port=5000, debug=False)