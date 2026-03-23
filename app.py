# app.py -- Dark Horse Image Truth Engine (light, colorful UI)
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
      --bg1: #f8fafc; /* soft sky */
      --bg2: #ffffff; /* paper */
      --card: rgba(15,23,42,0.03);
      --accent1: #7c3aed; /* purple */
      --accent2: #06b6d4; /* teal */
      --accent3: #f59e0b; /* warm */
      --good: #10b981;
      --warn: #f59e0b;
      --bad: #ef4444;
      --text: #0f172a;
      --muted: #475569;
    }
    html,body{ height:100% }
    body{ background: radial-gradient(800px 300px at 10% 8%, rgba(124,58,237,0.06), transparent 6%), linear-gradient(180deg,var(--bg1),var(--bg2)); color:var(--text); font-family: 'Inter', system-ui, -apple-system, Roboto, Arial; }
    .container{ max-width:1100px; padding:36px 18px; }
    .brand{ font-weight:800; font-size:1.2rem; letter-spacing:-0.4px; display:flex; gap:8px; align-items:center; }
    .logo-dot{ width:12px; height:12px; border-radius:50%; background:linear-gradient(45deg,var(--accent1),var(--accent2)); box-shadow:0 8px 22px rgba(124,58,237,0.12) }
    .card.glass{ background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,250,0.8)); border:1px solid rgba(15,23,42,0.04); box-shadow: 0 8px 30px rgba(15,23,42,0.06); border-radius:14px; }
    .upload-area{ border:2px dashed rgba(15,23,42,0.06); padding:28px; border-radius:12px; text-align:center; transition:all .14s ease; background:linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.4)); }
    .upload-area.dragover{ transform: translateY(-6px); box-shadow:0 22px 60px rgba(7,16,36,0.06); border-color: rgba(124,58,237,0.9) }
    .meter { --size:140px; width:var(--size); height:var(--size); border-radius:999px; display:grid; place-items:center; background: conic-gradient(var(--col, var(--accent2)) var(--pct), rgba(0,0,0,0.06) 0); position:relative; border:6px solid rgba(9,30,66,0.03) }
    .meter .val{ font-weight:800; font-size:20px; color:var(--text); text-shadow:none }
    .layer{ border-left:4px solid rgba(15,23,42,0.04); padding-left:12px; margin-bottom:10px; border-radius:6px; padding-top:8px; padding-bottom:8px }
    .score-pill{ padding:6px 10px; border-radius:999px; background: linear-gradient(90deg, rgba(124,58,237,0.08), rgba(6,182,212,0.06)); color:var(--text); font-weight:700; min-width:64px; text-align:center }
    .bar { height:10px; border-radius:999px; background: rgba(15,23,42,0.04); overflow:hidden; }
    .bar > i { display:block; height:100%; border-radius:999px; width:0%; background:linear-gradient(90deg,var(--accent1),var(--accent2)); box-shadow:0 4px 12px rgba(7,16,36,0.06) }
    .muted{ color:var(--muted) }
    .small-muted{ color:var(--muted); font-size:0.95rem }
    footer{ color:var(--muted); margin-top:24px; text-align:center; font-size:0.9rem }
    .img-preview{ max-width:220px; border-radius:12px; border:1px solid rgba(15,23,42,0.04); box-shadow: 0 8px 30px rgba(15,23,42,0.04); }
    .btn-accent{ background: linear-gradient(90deg,var(--accent1),var(--accent2)); border:none; color:white }
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
                  <button id="analyzeBtn" class="btn btn-accent">Analyze</button>
                  <button id="clearBtn" class="btn btn-outline-secondary ms-2">Clear</button>
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
              <div class="meter mb-2" id="meter" style="--pct:0deg; --col:var(--accent2);"><div class="val" id="meterVal">--</div></div>
              <div id="resultLabel" class="h5">No result</div>
              <div id="resultSub" class="small-muted mb-2">Upload an image to begin</div>
            </div>

            <div class="text-start mt-3">
              <div class="fw-semibold mb-2">Analysis Layers</div>
              <div id="layers"></div>
            </div>
            <div class="mt-3 d-flex justify-content-between">
              <a id="downloadReport" class="btn btn-sm btn-outline-dark">Download report</a>
              <div>
                <button id="copyScore" class="btn btn-sm btn-outline-dark me-2">Copy score</button>
                <a id="recheckBtn" class="btn btn-sm btn-outline-dark">Analyze another</a>
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
      let color = 'var(--accent2)';
      if(score >= 70) color = 'var(--good)'; else if(score >= 40) color = 'var(--warn)'; else color = 'var(--bad)';
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
# app.py
# Dark Horse — Production-style AI Image Truth Engine
# Direct REAL / FAKE output via trained binary classifier

from __future__ import annotations

import os
import io
import csv
import time
import json
import math
import uuid
import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    send_from_directory,
    abort,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from werkzeug.utils import secure_filename

from PIL import Image, ExifTags, ImageOps
import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import kurtosis

try:
    import joblib
except Exception:
    joblib = None


# =========================================================
# Config
# =========================================================
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///db.sqlite3")
REDIS_URL = os.environ.get("REDIS_URL", "")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me-now")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "60"))
MAX_FILE_AGE_DAYS = int(os.environ.get("MAX_FILE_AGE_DAYS", "30"))

UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    "/tmp/darkhorse_uploads" if os.environ.get("RENDER") else "uploads"
)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/darkhorse_detector.joblib")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)


# =========================================================
# App / DB / Security
# =========================================================
app = Flask(__name__, static_folder=None)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + 1024  # hard Flask limit

db = SQLAlchemy(app)

# Security headers for a public-facing app
Talisman(
    app,
    content_security_policy=None,
    force_https=False,   # set True behind HTTPS proxy in production
)

if REDIS_URL:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=REDIS_URL,
        default_limits=[f"{RATE_LIMIT_MAX}/hour"],
        app=app,
    )
else:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{RATE_LIMIT_MAX}/hour"],
        app=app,
    )


# =========================================================
# DB Model
# =========================================================
class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(512), nullable=False)
    original_name = db.Column(db.String(512))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip = db.Column(db.String(100))
    label = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    score_real = db.Column(db.Float)
    meta = db.Column(db.Text)

with app.app_context():
    db.create_all()


# =========================================================
# Model loading
# =========================================================
@dataclass
class DetectorBundle:
    model: Any
    feature_names: List[str]
    threshold: float = 0.50
    scaler: Any = None


_DETECTOR: Optional[DetectorBundle] = None


def load_detector() -> Optional[DetectorBundle]:
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    if joblib is None:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    bundle = joblib.load(MODEL_PATH)

    # Supported formats:
    # 1) dict with keys: model, feature_names, threshold, scaler
    # 2) plain estimator with predict_proba
    if isinstance(bundle, dict):
        model = bundle.get("model")
        feature_names = bundle.get("feature_names", [])
        threshold = float(bundle.get("threshold", 0.50))
        scaler = bundle.get("scaler", None)
        _DETECTOR = DetectorBundle(
            model=model,
            feature_names=feature_names,
            threshold=threshold,
            scaler=scaler,
        )
    else:
        _DETECTOR = DetectorBundle(
            model=bundle,
            feature_names=[],
            threshold=0.50,
            scaler=None,
        )
    return _DETECTOR


# =========================================================
# Utility
# =========================================================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def image_to_cv2_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def image_to_gray(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("L"))


def get_file_ext(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_save_upload(file_storage, save_path: str) -> None:
    file_storage.save(save_path)


def extract_exif(pil_img: Image.Image) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        raw_exif = pil_img.info.get("exif", b"")
        if raw_exif:
            import piexif
            exif_dict = piexif.load(raw_exif)
            for ifd_name, ifd in exif_dict.items():
                if not isinstance(ifd, dict):
                    continue
                for k, v in ifd.items():
                    if isinstance(v, bytes):
                        try:
                            v = v.decode(errors="ignore")
                        except Exception:
                            v = str(v)
                    out[f"{ifd_name}:{k}"] = v
            return out
    except Exception:
        pass

    try:
        raw = getattr(pil_img, "_getexif", lambda: {})() or {}
        for k, v in raw.items():
            name = ExifTags.TAGS.get(k, k)
            out[str(name)] = v
    except Exception:
        pass
    return out


def jpeg_quality_hint(pil_img: Image.Image) -> Tuple[Optional[float], Optional[float]]:
    try:
        qtables = getattr(pil_img, "quantization", None)
        if not qtables:
            return None, None
        vals = []
        for _, table in qtables.items():
            vals.extend(list(table))
        if not vals:
            return None, None
        mean_q = float(np.mean(vals))
        score = clamp01(1.0 - (mean_q / 200.0))
        return score, mean_q
    except Exception:
        return None, None


def entropy_score(gray: np.ndarray) -> Tuple[float, float]:
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    ent = float(-np.sum(p * np.log2(p + 1e-12)))
    score = clamp01((ent - 3.0) / (7.0 - 3.0))
    return score, ent


def edge_density(gray: np.ndarray) -> Tuple[float, float]:
    g = gray.astype(np.uint8)
    med = float(np.median(g))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(g, lower, upper)
    density = float(np.mean(edges > 0))
    score = clamp01((density - 0.001) / (0.04 - 0.001))
    return score, density


def laplacian_variance(gray: np.ndarray) -> Tuple[float, float]:
    var = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    score = clamp01(var / 1200.0)
    return score, var


def residual_noise(gray: np.ndarray) -> Tuple[float, float]:
    img = gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    residual = img - blurred
    std = float(np.std(residual))
    score = clamp01((std - 0.002) / (0.04 - 0.002))
    return score, std


def local_smoothness(gray: np.ndarray) -> Tuple[float, float]:
    img = gray.astype(np.float32) / 255.0
    mean = cv2.blur(img, (3, 3))
    mean_sq = cv2.blur(img * img, (3, 3))
    var = mean_sq - mean * mean
    mean_var = float(np.mean(var))
    score = clamp01((mean_var - 1e-6) / (0.005 - 1e-6))
    return score, mean_var


def frequency_features(gray: np.ndarray) -> Tuple[float, float, float]:
    img = gray.astype(np.float32)
    h, w = img.shape[:2]
    target = 512
    scale = max(1, int(max(h, w) / target))
    if scale > 1:
        img = cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)

    f = fftpack.fft2(img)
    fshift = fftpack.fftshift(f)
    mag = np.abs(fshift).flatten()
    log_mag = np.log1p(mag)
    k = float(kurtosis(log_mag, fisher=False, nan_policy="omit") or 0.0)
    if math.isnan(k):
        k = 0.0

    # Low-mid score is more "camera-like"; this is only a feature for the model.
    if k <= 2:
        score = 0.25
    elif k <= 10:
        score = 0.95
    elif k <= 14:
        score = 0.70
    else:
        score = 0.25

    hf_ratio = float(np.mean(log_mag > np.percentile(log_mag, 85)))
    hf_ratio = clamp01(hf_ratio)
    return clamp01(score), k, hf_ratio


def color_stats(bgr: np.ndarray) -> Tuple[List[float], List[float], float]:
    chans = cv2.split(bgr)
    means = [float(np.mean(c)) / 255.0 for c in chans]
    stds = [float(np.std(c)) / 255.0 for c in chans]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation_mean = float(np.mean(hsv[:, :, 1])) / 255.0

    return means, stds, saturation_mean


def symmetry_score(gray: np.ndarray) -> float:
    flipped = np.fliplr(gray)
    diff = float(np.mean(np.abs(gray.astype(np.float32) - flipped.astype(np.float32)))) / 255.0
    # lower difference => more symmetric
    return clamp01(1.0 - diff / 0.35)


def patch_self_similarity(gray: np.ndarray) -> float:
    # deterministic grid-based self similarity
    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return 0.0

    patch = 32
    ys = np.linspace(0, h - patch, 4).astype(int)
    xs = np.linspace(0, w - patch, 4).astype(int)

    patches = []
    for y in ys:
        for x in xs:
            p = gray[y:y + patch, x:x + patch].astype(np.float32)
            p = p - p.mean()
            denom = np.sqrt(np.sum(p * p) + 1e-9)
            patches.append(p / denom)

    sims = []
    for i in range(len(patches) - 1):
        a = patches[i].flatten()
        b = patches[i + 1].flatten()
        sims.append(float(np.dot(a, b)))

    if not sims:
        return 0.0

    high = float(np.mean(np.array(sims) > 0.80))
    return clamp01(high)


def extract_features(pil_img: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
    pil_img = ImageOps.exif_transpose(pil_img)
    bgr = image_to_cv2_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    exif = extract_exif(pil_img)
    exif_found = len(exif) > 0

    h, w = gray.shape[:2]
    megapixels = float((h * w) / 1_000_000.0)
    aspect_ratio = float(w / max(1, h))

    s_sensor, sensor_std = residual_noise(gray)
    s_smooth, smooth_var = local_smoothness(gray)
    s_freq, freq_kurt, hf_ratio = frequency_features(gray)
    s_edge, edge_den = edge_density(gray)
    s_entropy, ent = entropy_score(gray)
    s_lap, lap_var = laplacian_variance(gray)

    means, stds, sat_mean = color_stats(bgr)
    sym = symmetry_score(gray)
    self_sim = patch_self_similarity(gray)
    q_score, q_mean = jpeg_quality_hint(pil_img)

    # Feature vector for the trained classifier.
    # Keep this order stable and use the same order in training.
    feats = np.array([
        1.0 if exif_found else 0.0,
        float(w),
        float(h),
        megapixels,
        aspect_ratio,
        float(gray.mean()) / 255.0,
        float(gray.std()) / 255.0,
        ent,
        edge_den,
        lap_var,
        sensor_std,
        smooth_var,
        freq_kurt,
        hf_ratio,
        means[0], means[1], means[2],
        stds[0], stds[1], stds[2],
        sat_mean,
        sym,
        self_sim,
        0.0 if q_score is None else float(q_score),
        0.0 if q_mean is None else float(q_mean) / 200.0,
    ], dtype=np.float32)

    details = {
        "exif_found": exif_found,
        "width": w,
        "height": h,
        "megapixels": megapixels,
        "aspect_ratio": aspect_ratio,
        "sensor_std": sensor_std,
        "smoothness_meanvar": smooth_var,
        "frequency_kurtosis": freq_kurt,
        "high_freq_ratio": hf_ratio,
        "edge_density": edge_den,
        "entropy": ent,
        "laplacian_variance": lap_var,
        "color_means": means,
        "color_stds": stds,
        "saturation_mean": sat_mean,
        "symmetry": sym,
        "self_similarity": self_sim,
        "jpeg_quality_score": q_score,
        "jpeg_quant_mean": q_mean,
    }
    return feats, details


def predict_real_fake(features: np.ndarray) -> Tuple[str, float, float, Dict[str, Any]]:
    detector = load_detector()
    if detector is None or detector.model is None:
        raise RuntimeError(
            f"Model not loaded. Put a trained binary classifier at: {MODEL_PATH}"
        )

    x = features.reshape(1, -1)

    if detector.scaler is not None:
        x = detector.scaler.transform(x)

    model = detector.model

    # Probability for REAL class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if len(proba) == 2:
            # Try to map class '1' to REAL
            if 1 in classes:
                idx = classes.index(1)
                prob_real = float(proba[idx])
            else:
                prob_real = float(proba[1])
        else:
            prob_real = float(proba[-1])
    else:
        # fallback for margin-based models
        pred = model.predict(x)[0]
        prob_real = 1.0 if int(pred) == 1 else 0.0

    threshold = float(detector.threshold)
    label = "REAL" if prob_real >= threshold else "FAKE"
    confidence = prob_real if label == "REAL" else (1.0 - prob_real)
    score_real = round(prob_real * 100.0, 3)

    meta = {
        "threshold": threshold,
        "feature_count": int(features.shape[0]),
        "feature_names": detector.feature_names,
    }
    return label, confidence, score_real, meta


# =========================================================
# Cleanup thread
# =========================================================
def cleanup_old_files():
    while True:
        try:
            cutoff = datetime.utcnow() - timedelta(days=MAX_FILE_AGE_DAYS)
            for fname in os.listdir(UPLOAD_DIR):
                path = os.path.join(UPLOAD_DIR, fname)
                try:
                    if not os.path.isfile(path):
                        continue
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


# =========================================================
# UI
# =========================================================
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dark Horse — Image Truth Engine</title>
  <style>
    body { font-family: system-ui, Arial; background: #f8fafc; color: #0f172a; margin: 0; }
    .wrap { max-width: 1000px; margin: 0 auto; padding: 28px 18px; }
    .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px; box-shadow: 0 8px 30px rgba(15,23,42,.06); }
    .row { display: grid; grid-template-columns: 1.2fr .8fr; gap: 16px; }
    @media (max-width: 900px){ .row { grid-template-columns: 1fr; } }
    .drop { border: 2px dashed #cbd5e1; border-radius: 16px; padding: 26px; text-align: center; cursor: pointer; background: linear-gradient(180deg,#fff,#f8fafc); }
    .drop.drag { border-color: #7c3aed; transform: translateY(-2px); }
    .muted { color: #64748b; }
    .btn { background: linear-gradient(90deg,#7c3aed,#06b6d4); color: white; border: none; border-radius: 999px; padding: 10px 16px; cursor: pointer; }
    .btn2 { background: #e2e8f0; border: none; border-radius: 999px; padding: 10px 16px; cursor: pointer; }
    .bar { height: 10px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
    .bar > div { height: 100%; width: 0%; background: linear-gradient(90deg,#7c3aed,#06b6d4); }
    code { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }
    img.preview { max-width: 100%; max-height: 260px; border-radius: 14px; border: 1px solid #e2e8f0; }
    ul { padding-left: 18px; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card" style="margin-bottom:16px;">
    <h1 style="margin:0 0 8px;">Dark Horse</h1>
    <div class="muted">Production-style image truth engine. Direct <b>REAL</b> / <b>FAKE</b> output from a trained binary model.</div>
  </div>

  <div class="row">
    <div class="card">
      <div id="drop" class="drop">
        <h3 style="margin-top:0;">Upload image</h3>
        <div class="muted">PNG / JPG / WebP / BMP / TIFF — up to {{ mb }} MB</div>
        <input id="file" type="file" accept="image/*" style="display:none">
        <div style="margin-top:14px;">
          <button id="pick" class="btn">Choose image</button>
        </div>
      </div>

      <div id="previewWrap" style="display:none; margin-top:16px;">
        <img id="preview" class="preview">
        <div style="margin-top:10px;">
          <div><b id="fname"></b></div>
          <div class="muted" id="fsize"></div>
        </div>
        <div style="margin-top:14px;">
          <button id="analyze" class="btn">Analyze</button>
          <button id="clear" class="btn2">Clear</button>
        </div>
      </div>

      <div id="loading" style="display:none; margin-top:16px;">
        <div class="muted">Analyzing...</div>
        <div class="bar" style="margin-top:8px;"><div id="bar"></div></div>
      </div>
    </div>

    <div class="card">
      <div class="pill" id="label">No result</div>
      <h2 id="headline" style="margin:10px 0 8px;">Upload an image</h2>
      <div class="muted" id="sub">Model output will appear here.</div>

      <div style="margin:16px 0 10px;">
        <div class="muted" style="margin-bottom:6px;">Confidence</div>
        <div class="bar"><div id="confbar"></div></div>
      </div>

      <div style="margin-top:16px;">
        <div><b>Details</b></div>
        <ul id="details"></ul>
      </div>

      <div style="margin-top:16px;">
        <a id="download" href="#" style="display:none;">Download JSON report</a>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <b>Health:</b> <a href="/health">/health</a> |
    <b>Export:</b> <a href="/admin/exports?token={{ token }}">CSV export</a>
    <div class="muted" style="margin-top:8px;">
      If the model file is missing, /detect returns a clear error instead of guessing.
    </div>
  </div>
</div>

<script>
const file = document.getElementById('file');
const pick = document.getElementById('pick');
const drop = document.getElementById('drop');
const previewWrap = document.getElementById('previewWrap');
const preview = document.getElementById('preview');
const fname = document.getElementById('fname');
const fsize = document.getElementById('fsize');
const analyze = document.getElementById('analyze');
const clearBtn = document.getElementById('clear');
const loading = document.getElementById('loading');
const bar = document.getElementById('bar');
const label = document.getElementById('label');
const headline = document.getElementById('headline');
const sub = document.getElementById('sub');
const confbar = document.getElementById('confbar');
const details = document.getElementById('details');
const download = document.getElementById('download');
let currentFile = null;
let lastReport = null;

function humanSize(b){
  if(b < 1024) return b + ' B';
  const units = ['KB','MB','GB'];
  let i = -1;
  do { b /= 1024; i++; } while(b >= 1024 && i < units.length-1);
  return b.toFixed(1) + ' ' + units[i];
}

pick.onclick = () => file.click();
drop.addEventListener('click', () => file.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('drag');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    file.files = e.dataTransfer.files;
    showPreview(e.dataTransfer.files[0]);
  }
});

file.addEventListener('change', () => {
  if (file.files && file.files[0]) showPreview(file.files[0]);
});

function showPreview(f){
  currentFile = f;
  preview.src = URL.createObjectURL(f);
  fname.textContent = f.name;
  fsize.textContent = humanSize(f.size);
  previewWrap.style.display = 'block';
  details.innerHTML = '';
  download.style.display = 'none';
  label.textContent = 'Ready';
  headline.textContent = 'Ready to analyze';
  sub.textContent = 'Press Analyze to get direct REAL / FAKE output.';
}

clearBtn.onclick = () => {
  file.value = '';
  currentFile = null;
  previewWrap.style.display = 'none';
  loading.style.display = 'none';
  details.innerHTML = '';
  download.style.display = 'none';
  label.textContent = 'No result';
  headline.textContent = 'Upload an image';
  sub.textContent = 'Model output will appear here.';
  confbar.style.width = '0%';
};

analyze.onclick = async () => {
  if(!currentFile) return alert('Choose an image first');
  loading.style.display = 'block';
  bar.style.width = '20%';

  const fd = new FormData();
  fd.append('file', currentFile);

  try {
    const resp = await fetch('/detect', {
      method: 'POST',
      body: fd,
      headers: {'X-Requested-With':'XMLHttpRequest'}
    });
    bar.style.width = '60%';
    const data = await resp.json();
    if(!resp.ok){
      loading.style.display = 'none';
      alert(data.error || 'Server error');
      return;
    }
    lastReport = data;
    renderResult(data);
    bar.style.width = '100%';
    setTimeout(() => loading.style.display = 'none', 250);
  } catch (e) {
    loading.style.display = 'none';
    alert('Network error: ' + e.message);
  }
};

function renderResult(d){
  label.textContent = d.label;
  headline.textContent = d.label === 'REAL' ? 'Likely authentic camera image' : 'Likely synthetic image';
  sub.textContent = 'Confidence: ' + (Math.round(d.confidence * 1000) / 10) + '% | Real probability: ' + (Math.round(d.score_real * 10)/10) + '%';

  const c = Math.max(0, Math.min(100, Math.round(d.confidence * 100)));
  confbar.style.width = c + '%';

  details.innerHTML = '';
  (d.details || []).forEach(line => {
    const li = document.createElement('li');
    li.textContent = line;
    details.appendChild(li);
  });

  download.style.display = 'inline';
  download.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(d, null, 2));
  download.download = 'darkhorse_report_' + Date.now() + '.json';
  download.textContent = 'Download JSON report';
}
</script>
</body>
</html>
"""


# =========================================================
# Routes
# =========================================================
@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML,
        mb=MAX_FILE_SIZE // (1024 * 1024),
        token=ADMIN_TOKEN
    )


@app.route("/health")
def health():
    detector = load_detector()
    return jsonify({
        "status": "ok",
        "time_utc": datetime.utcnow().isoformat(),
        "model_loaded": detector is not None and detector.model is not None,
        "model_path": MODEL_PATH,
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(UPLOAD_DIR, safe, as_attachment=False)


@app.route("/detect", methods=["POST"])
@limiter.limit(f"{RATE_LIMIT_MAX}/hour")
def detect():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    f = request.files["file"]
    original_name = secure_filename(f.filename or "")
    ext = get_file_ext(original_name)

    if ext not in ALLOWED_EXT:
        return jsonify({
            "error": "file type not supported",
            "allowed": sorted(list(ALLOWED_EXT))
        }), 400

    # file size check
    try:
        pos = f.stream.tell()
        f.stream.seek(0, io.SEEK_END)
        size = f.stream.tell()
        f.stream.seek(pos, io.SEEK_SET)
    except Exception:
        size = None

    if size is not None and size > MAX_FILE_SIZE:
        return jsonify({"error": "file too large"}), 400

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    base = os.path.splitext(original_name)[0] or "upload"
    save_name = f"{base}_{timestamp}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    try:
        safe_save_upload(f, save_path)
    except Exception as e:
        return jsonify({"error": "failed to save file", "detail": str(e)}), 500

    try:
        pil = Image.open(save_path)
        pil.verify()
        pil = Image.open(save_path)
        pil = ImageOps.exif_transpose(pil)
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify({"error": "invalid image", "detail": str(e)}), 400

    try:
        features, details = extract_features(pil)
        label, confidence, score_real, meta = predict_real_fake(features)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "analysis failed", "detail": str(e)}), 500

    reasons = [
        f"Image size: {details['width']}x{details['height']} ({details['megapixels']:.2f} MP)",
        f"EXIF present: {'yes' if details['exif_found'] else 'no'}",
        f"Sensor residual std: {details['sensor_std']:.6f}",
        f"Laplacian variance: {details['laplacian_variance']:.2f}",
        f"Texture variance: {details['smoothness_meanvar']:.6e}",
        f"Frequency kurtosis: {details['frequency_kurtosis']:.3f}",
        f"Edge density: {details['edge_density']:.6f}",
        f"Entropy: {details['entropy']:.3f}",
        f"Symmetry score: {details['symmetry']:.3f}",
        f"Self similarity: {details['self_similarity']:.3f}",
    ]
    if details["jpeg_quality_score"] is not None:
        reasons.append(
            f"JPEG quality hint: {details['jpeg_quality_score']:.3f} "
            f"(mean table {details['jpeg_quant_mean']:.2f})"
        )

    # save analytics
    try:
        row = Upload(
            filename=save_name,
            original_name=original_name,
            ip=request.remote_addr or "unknown",
            label=label,
            confidence=float(confidence),
            score_real=float(score_real),
            meta=json.dumps({
                "details": details,
                "meta": meta,
                "features_len": int(features.shape[0]),
            }, ensure_ascii=False),
        )
        db.session.add(row)
        db.session.commit()
    except Exception:
        db.session.rollback()

    image_url = url_for("uploaded_file", filename=save_name)

    out = {
        "label": label,
        "confidence": round(float(confidence), 4),
        "score_real": round(float(score_real), 3),
        "image_url": image_url,
        "details": reasons,
        "details_raw": details,
        "meta": meta,
    }
    return jsonify(out)


@app.route("/admin/exports")
def admin_exports():
    token = request.args.get("token", "")
    if token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    rows = Upload.query.order_by(Upload.created_at.desc()).limit(1000).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "filename", "original_name", "created_at", "ip", "label", "confidence", "score_real"])
    for r in rows:
        writer.writerow([
            r.id,
            r.filename,
            r.original_name or "",
            r.created_at.isoformat(),
            r.ip or "",
            r.label or "",
            r.confidence if r.confidence is not None else "",
            r.score_real if r.score_real is not None else "",
        ])

    return output.getvalue(), 200, {"Content-Type": "text/csv; charset=utf-8"}


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting Dark Horse on 0.0.0.0:{port}")
    print(f"Model path: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=False)