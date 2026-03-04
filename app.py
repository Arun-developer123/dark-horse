# app.py -- Dark Horse Image Truth Engine (final UI + layered analysis, render-friendly)
from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
from datetime import datetime, timedelta
from io import BytesIO
import math
import json

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

# -------------------------
# Analysis signals (layered)
# -------------------------
def sensor_noise_score(cv2_gray):
    img = cv2_gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    residual = img - blurred
    std = float(np.std(residual))
    score = (std - 0.002) / (0.04 - 0.002)
    return clamp01(score), std

def smoothness_score(cv2_gray):
    img = cv2_gray.astype(np.float32) / 255.0
    mean = cv2.blur(img, (3,3))
    mean_sq = cv2.blur(img*img, (3,3))
    var = mean_sq - mean*mean
    mean_var = float(np.mean(var))
    score = (mean_var - 1e-6) / (0.005 - 1e-6)
    return clamp01(score), mean_var

def frequency_kurtosis_score(cv2_gray):
    img = cv2_gray.astype(np.float32)
    h, w = img.shape
    target = 512
    scale = max(1, int(max(h,w)/target))
    if scale > 1:
        img_small = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
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
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    h, w = gray.shape
    sims = []
    num_checks = 24
    for _ in range(num_checks):
        y = np.random.randint(0, max(1, h-32))
        x = np.random.randint(0, max(1, w-32))
        y2 = np.random.randint(0, max(1, h-32))
        x2 = np.random.randint(0, max(1, w-32))
        p1 = gray[y:y+32, x:x+32]
        p2 = gray[y2:y2+32, x2:x2+32]
        num = np.sum((p1-p1.mean())*(p2-p2.mean()))
        den = np.sqrt(np.sum((p1-p1.mean())**2)*np.sum((p2-p2.mean())**2)+1e-9)
        sims.append(num/(den+1e-9))
    sims = np.array(sims)
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
    density = float(np.sum(edges>0) / (edges.size + 1e-9))
    score = (density - 0.001) / (0.04 - 0.001)
    return clamp01(score), density

def color_hist_kurtosis_score(cv2_bgr):
    chans = cv2.split(cv2_bgr)
    ks = []
    for c in chans:
        hist = cv2.calcHist([c], [0], None, [256], [0,256]).flatten()
        hist = hist / (hist.sum()+1e-9)
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
    hist = cv2.calcHist([cv2_gray.astype(np.uint8)], [0], None, [256], [0,256]).flatten()
    p = hist / (hist.sum()+1e-9)
    p = p[p>0]
    ent = float(-np.sum(p * np.log2(p)))
    score = (ent - 3.0) / (7.0 - 3.0)
    return clamp01(score), ent

def jpeg_quality_hint(pil_img):
    try:
        qtables = getattr(pil_img, 'quantization', None)
        if not qtables:
            return None, None
        vals = []
        for k,v in qtables.items():
            vals.extend(v)
        mean_q = float(np.mean(vals)) if vals else None
        if mean_q is None:
            return None, None
        score = 1.0 - (mean_q / 200.0)
        return clamp01(score), mean_q
    except Exception:
        return None, None

# -------------------------
# Combine signals -> final score
# -------------------------
def compute_final_realness(signals):
    weights = {
        'exif': 0.12,
        'sensor': 0.25,
        'smooth': 0.18,
        'freq': 0.15,
        'edge': 0.10,
        'color': 0.08,
        'entropy': 0.07,
    }
    total_w = 0.0
    acc = 0.0
    for k,w in weights.items():
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
# Beautiful UI template (Jinja)
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
      --bg1: #041025;
      --bg2: #07172b;
      --card: rgba(255,255,255,0.03);
      --accent1: #7c3aed;
      --accent2: #06b6d4;
      --good: #10b981;
      --warn: #f59e0b;
      --bad: #ef4444;
    }
    html,body{ height:100% }
    body{ background: radial-gradient(1200px 600px at 10% 10%, rgba(124,58,237,0.06), transparent 10%), linear-gradient(135deg,var(--bg1),var(--bg2)); color:#e6eef8; font-family: 'Inter', system-ui, -apple-system, Roboto, Arial; }
    .container{ max-width:1100px; padding:36px 18px; }
    .brand{ font-weight:800; font-size:1.2rem; letter-spacing:-0.4px; display:flex; gap:8px; align-items:center; }
    .logo-dot{ width:12px; height:12px; border-radius:50%; background:linear-gradient(45deg,var(--accent1),var(--accent2)); box-shadow:0 6px 18px rgba(124,58,237,0.16) }
    .card.glass{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); box-shadow: 0 12px 40px rgba(2,6,23,0.6); border-radius:14px; }
    .upload-area{ border:2px dashed rgba(255,255,255,0.04); padding:28px; border-radius:12px; text-align:center; transition:all .14s ease; background:rgba(255,255,255,0.008) }
    .upload-area.dragover{ transform: translateY(-6px); box-shadow:0 20px 60px rgba(2,6,23,0.7); border-color: rgba(124,58,237,0.9) }
    .meter { --size:140px; width:var(--size); height:var(--size); border-radius:999px; display:grid; place-items:center; background: conic-gradient(var(--col, #06b6d4) var(--pct), rgba(255,255,255,0.04) 0); position:relative }
    .meter .val{ font-weight:800; font-size:20px; color:#fff; text-shadow:0 2px 8px rgba(0,0,0,0.6) }
    .layer{ border-left:4px solid rgba(255,255,255,0.03); padding-left:12px; margin-bottom:10px; border-radius:6px; padding-top:8px; padding-bottom:8px }
    .score-pill{ padding:6px 10px; border-radius:999px; background: rgba(255,255,255,0.03); color:#eaf5ff; font-weight:700; min-width:64px; text-align:center }
    .bar { height:10px; border-radius:999px; background: rgba(255,255,255,0.03); overflow:hidden; }
    .bar > i { display:block; height:100%; border-radius:999px; width:0%; background:linear-gradient(90deg,var(--accent1),var(--accent2)); box-shadow:0 4px 18px rgba(7,16,36,0.6) }
    .muted{ color:#9fb0d6 }
    .small-muted{ color:#9fb0d6; font-size:0.95rem }
    footer{ color:#9fb0d6; margin-top:24px; text-align:center; font-size:0.9rem }
    .img-preview{ max-width:220px; border-radius:12px; border:1px solid rgba(255,255,255,0.04); box-shadow: 0 8px 40px rgba(2,6,23,0.5); }
    @media (max-width:990px){ .meter{ --size:120px } .img-preview{ max-width:140px } }
  </style>
</head>
<body>
  <div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div class="d-flex flex-column">
        <div class="brand">
          <div class="logo-dot" aria-hidden></div>
          <div>
            Dark Horse <span class="muted">• Image Truth Engine</span>
          </div>
        </div>
        <div class="small-muted">Explainable heuristics to flag likely AI-generated images — layered analysis & friendly UI.</div>
      </div>
      <div class="d-flex gap-2">
        <a class="btn btn-sm btn-outline-light" href="/health">Health</a>
        <a class="btn btn-sm btn-outline-light" href="/admin/exports?token={{ admin_token }}">Export</a>
      </div>
    </div>

    <div class="card glass p-4 mb-4">
      <div class="row g-3">
        <div class="col-lg-7">
          <div id="uploadArea" class="upload-area">
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none"><path d="M12 3v10" stroke="#9fb0d6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#9fb0d6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <h4 class="mt-2">Drag & drop or click to upload</h4>
            <div class="muted">PNG / JPG / WebP / BMP — up to <strong>{{ mb }} MB</strong></div>
            <div class="small-muted mt-2">We run EXIF, sensor residuals, texture & frequency analyses, edge & color checks, and more.</div>
            <input id="fileInput" type="file" accept="image/*" style="display:none;">
          </div>

          <div id="previewCard" class="mt-3 d-none card p-3">
            <div class="d-flex gap-3 align-items-start">
              <img id="previewImage" class="img-preview"/>
              <div class="flex-grow-1">
                <div id="previewName" class="fw-semibold"></div>
                <div id="previewSize" class="small-muted mb-2"></div>
                <div>
                  <button id="analyzeBtn" class="btn btn-primary" style="background:linear-gradient(90deg,var(--accent1),var(--accent2)); border:none">Analyze</button>
                  <button id="clearBtn" class="btn btn-outline-light ms-2">Clear</button>
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
              <div class="meter mb-2" id="meter" style="--pct:0deg; --col:#06b6d4;"><div class="val" id="meterVal">--</div></div>
              <div id="resultLabel" class="h5">No result</div>
              <div id="resultSub" class="small-muted mb-2">Upload an image to begin</div>
            </div>

            <div class="text-start mt-3">
              <div class="fw-semibold mb-2">Analysis Layers</div>
              <div id="layers"></div>
            </div>
            <div class="mt-3 d-flex justify-content-between">
              <a id="downloadReport" class="btn btn-sm btn-outline-light">Download report</a>
              <div>
                <button id="copyScore" class="btn btn-sm btn-outline-light me-2">Copy score</button>
                <a id="recheckBtn" class="btn btn-sm btn-outline-light">Analyze another</a>
              </div>
            </div>
          </div>

          <div class="card p-3 mt-3 text-center small-muted">
            <div class="fw-semibold">Tips</div>
            Use original phone photos for best accuracy. If uncertain, cross-check with reverse-image search and human review.
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
    }

    clearBtn.addEventListener('click', ()=>{
      fileInput.value = '';
      currentFile = null;
      previewImage.src = '';
      previewCard.classList.add('d-none');
      layers.innerHTML = '';
      resultCard.classList.add('d-none');
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
      const score = Math.max(0, Math.min(100, Math.round((d.realness_score||0)*10)/10));
      const angle = (score/100)*360 + 'deg';
      meter.style.setProperty('--pct', angle);
      let color = '#06b6d4';
      if(score >= 70) color = '#10b981'; else if(score >= 40) color = '#f59e0b'; else color = '#ef4444';
      meter.style.setProperty('--col', color);
      meterVal.textContent = score;
      resultLabel.textContent = d.label || (score>=70 ? 'Likely REAL' : (score>=40 ? 'Unsure / Possibly Real' : 'Likely AI / Synthetic'));
      resultSub.textContent = 'Confidence — higher means more likely a real camera capture';

      layers.innerHTML = '';
      const layerOrder = [
        ['exif','EXIF metadata','exif_found'],
        ['sensor','Sensor noise','sensor_std'],
        ['smooth','Local texture variance','smoothness_meanvar'],
        ['freq','Frequency kurtosis','frequency_kurtosis'],
        ['artifact','Artifact penalty','artifact_penalty'],
        ['edge','Edge density','edge_density'],
        ['color','Color histogram kurtosis','color_hist_kurtosis'],
        ['entropy','Entropy','entropy']
      ];
      for(const [key,title,field] of layerOrder){
        if(d[field]===undefined) continue;
        const val = d[field];
        const scorev = (d.signals && d.signals[key]!==undefined) ? Math.round(d.signals[key]*100)/100 : null;
        const barPct = scorev !== null ? Math.round((scorev)*100) : 0;
        const div = document.createElement('div');
        div.className = 'layer';
        div.innerHTML = `
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <strong>${title}</strong>
              <div class="small-muted">${String(field).replaceAll('_',' ')}</div>
            </div>
            <div style="min-width:120px">
              <div class="score-pill mb-1">${(scorev!==null?scorev:'-')}</div>
              <div class="bar"><i style="width:${barPct}%;"></i></div>
            </div>
          </div>
          <div class="small-muted mt-1">${Array.isArray(d.reasons)? '' : ''}</div>
        `;
        layers.appendChild(div);
      }

      downloadReport.onclick = ()=> {
        if(!lastReport) return alert('No report');
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastReport, null, 2));
        const a = document.createElement('a'); a.setAttribute('href', dataStr); a.setAttribute('download', 'darkhorse_report_'+Date.now()+'.json'); document.body.appendChild(a); a.click(); a.remove();
      };

      recheckBtn.onclick = ()=>{ fileInput.value=''; currentFile=null; previewImage.src=''; previewCard.classList.add('d-none'); resultCard.classList.add('d-none'); layers.innerHTML=''; };
      copyScore.onclick = ()=> {
        if(!lastReport) return alert('No report');
        navigator.clipboard.writeText('Dark Horse score: ' + (lastReport.realness_score || '')).then(()=> alert('Score copied to clipboard'));
      };
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
        mb=MAX_FILE_SIZE // (1024*1024),
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
        return jsonify({'error':'no file provided'}), 400
    f = request.files["file"]
    filename_raw = secure_filename(f.filename or "")
    _, ext = os.path.splitext(filename_raw.lower())
    if ext not in ALLOWED_EXT:
        return jsonify({'error':'file type not supported','allowed': list(ALLOWED_EXT)}), 400

    # size check
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({'error':'file too large (> allowed bytes)'}), 400

    timestamp_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    save_name = f"{os.path.splitext(filename_raw)[0]}_{timestamp_suffix}{ext}" if filename_raw else f"upload_{timestamp_suffix}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    try:
        f.save(save_path)
    except Exception as e:
        return jsonify({'error':'failed to save file','detail':str(e)}), 500

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
        return jsonify({'error':'invalid image','detail':str(e)}), 400

    # prepare cv2 image and gray
    exif = extract_exif(pil)
    exif_found = len(exif) > 0

    cv2_img = pil_to_cv2(pil)
    if cv2_img is None or cv2_img.size == 0:
        return jsonify({'error':'failed to decode image'}), 400
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

    signals = {
        'exif': s_exif,
        'sensor': s_sensor,
        'smooth': s_smooth,
        'freq': s_freq,
        'artifact_penalty': penalty,
        'edge': s_edge,
        'color': s_color,
        'entropy': s_entropy,
        'jpeg_hint': q_score
    }

    final_raw = compute_final_realness(signals)
    final_score = float(round(final_raw * 100.0, 3))

    reasons = []
    if not exif_found:
        reasons.append("No camera EXIF metadata found — suspicious (many AI images miss EXIF).")
    else:
        reasons.append("Camera EXIF present — suggests a camera capture (EXIF can be forged).")
    reasons.append(f"Sensor residual std: {sensor_std:.6f}")
    reasons.append(f"Local texture variance (mean): {mean_var:.6e}")
    reasons.append(f"Frequency kurtosis: {kurt:.3f}")
    reasons.append(f"Artifact penalty: {penalty:.3f} (high_sim_frac: {high_sim_frac:.3f}, symmetry: {symmetry:.3f})")
    reasons.append(f"Edge density: {edge_density:.6f}")
    reasons.append(f"Color histogram mean kurtosis: {color_k:.3f}")
    reasons.append(f"Entropy: {ent:.3f}")
    if q_score is not None:
        reasons.append(f"JPEG quantization mean: {q_mean:.2f} (quality hint score: {q_score:.2f})")

    # DB log (best-effort)
    try:
        row = Upload(filename=save_name, ip=request.remote_addr or 'unknown', realness_score=final_score, reasons="\n".join(reasons))
        db.session.add(row); db.session.commit()
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
        'exif_found': exif_found
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
        return jsonify({"error":"unauthorized"}), 401
    rows = Upload.query.order_by(Upload.created_at.desc()).limit(500).all()
    lines = ["id,filename,created_at,ip,realness_score"]
    for r in rows:
        lines.append(",".join([str(r.id), r.filename, r.created_at.isoformat(), r.ip, str(r.realness_score or "")]))
    return ("\n".join(lines), 200, {"Content-Type":"text/csv; charset=utf-8"})

# -------------------------
# Main (bind to PORT)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Dark Horse on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)